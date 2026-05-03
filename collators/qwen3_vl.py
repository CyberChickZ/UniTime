"""
Qwen3-VL data collator for UniTime moment retrieval.

Uses per-frame IMAGE path (not video): each frame is an independent image with
its own <|image_pad|> tokens, allowing timestamp text between frames.
Processor handles token expansion automatically.

Requires transformers >= 5.5.0 (UniTime-gemma4 env).
"""
import os
from typing import Dict, List, Optional, Sequence

import decord
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator

PAD_IDX = -100


def find_segments(sample_timestamps, gt_window):
    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    closest_start = max(candidates_start) if candidates_start else sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)
    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    closest_end = max(candidates_end) if candidates_end else sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)
    return start_idx, end_idx


@register_collator("qwen3-vl")
class Qwen3VLDataCollator(BaseDataCollator):
    def __init__(
        self,
        config: Optional[AutoConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[AutoProcessor] = None,
        mask_question_tokens: bool = True,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.num_frames = 32

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def _load_frames(self, video_path: str):
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        fps = vr.get_avg_fps()
        n = min(self.num_frames, total)
        idx = torch.linspace(0, total - 1, n).round().long().tolist()
        frames = vr.get_batch(idx).asnumpy()
        timestamps = [round(i / fps, 1) for i in idx]
        return [Image.fromarray(f) for f in frames], timestamps

    def build_target_text(self, sampled_timestamps, windows):
        hit = []
        for window in windows:
            s_idx, e_idx = find_segments(sampled_timestamps, window)
            hit.extend(sampled_timestamps[s_idx : e_idx + 1])
        seen = set()
        ordered = []
        for t in hit:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        if not ordered:
            return "(no timestamps)"
        return ", ".join(f"{t} seconds" for t in ordered) + "."

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_mm_token_type_ids = []

        for inst in instances:
            mode = inst["mode"]
            if mode != "mr_seg":
                raise NotImplementedError(f"Qwen3VLDataCollator only supports mr_seg, got '{mode}'")
            temporal_window = inst["temporal_window"]

            video_path = None
            for msg in inst["message"]:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "video":
                            video_path = item.get("video")
                            break
                if video_path is not None:
                    break

            query_text = None
            for msg in inst["message"]:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text", "")
                            if txt.startswith("Query:"):
                                query_text = txt.split("Query:", 1)[1].split("\nAnswer", 1)[0].strip()
                                break
                if query_text is not None:
                    break

            if video_path is None or query_text is None:
                raise ValueError(f"Missing video_path or query in instance qid={inst.get('qid')}")

            windows = temporal_window[0]
            pil_frames, sampled_timestamps = self._load_frames(video_path)
            target_text = self.build_target_text(sampled_timestamps, windows)

            user_content = [
                {"type": "text", "text": "This is a sequence interleaved with timestamps and frames. "
                 "Your task is to identify the specific timestamp(s) when the given query appears.\n"},
            ]
            for t in sampled_timestamps:
                user_content.append({"type": "text", "text": f"timestamp: {t} seconds "})
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": f"\nQuery: {query_text}\nAnswer:"})

            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
            prompt_messages = [{"role": "user", "content": user_content}]

            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            full_inputs = self.processor(
                text=[full_text], images=pil_frames, return_tensors="pt", padding=False
            )
            prompt_inputs = self.processor(
                text=[prompt_text], images=pil_frames, return_tensors="pt", padding=False
            )

            full_ids = full_inputs["input_ids"][0]
            prompt_len = prompt_inputs["input_ids"].shape[1]

            labels = full_ids.clone()
            labels[:prompt_len] = PAD_IDX

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            if "pixel_values" in full_inputs:
                all_pixel_values.append(full_inputs["pixel_values"])
            if "image_grid_thw" in full_inputs:
                all_image_grid_thw.append(full_inputs["image_grid_thw"])
            if "mm_token_type_ids" in full_inputs:
                all_mm_token_type_ids.append(full_inputs["mm_token_type_ids"][0])

        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_t = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_t = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attn_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_t[i, : len(ids)] = ids
            labels_t[i, : len(lbls)] = lbls
            attn_mask[i, : len(ids)] = 1

        mm_token_type_ids_t = None
        if all_mm_token_type_ids:
            mm_token_type_ids_t = torch.zeros((len(all_mm_token_type_ids), max_len), dtype=torch.long)
            for i, mm_ids in enumerate(all_mm_token_type_ids):
                mm_token_type_ids_t[i, : len(mm_ids)] = mm_ids

        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        image_grid_thw = torch.cat(all_image_grid_thw, dim=0) if all_image_grid_thw else None

        result = dict(
            input_ids=input_ids_t,
            attention_mask=attn_mask,
            labels=labels_t,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if mm_token_type_ids_t is not None:
            result["mm_token_type_ids"] = mm_token_type_ids_t
        return result
