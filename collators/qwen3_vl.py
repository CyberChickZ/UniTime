"""
Qwen3-VL data collator for UniTime moment retrieval.

Pixel-values path: loads video frames at each training step via decord,
builds timestamp-interleaved prompt, uses Qwen3VLProcessor for tokenization.
Same approach as collators/gemma4_vl.py but adapted to Qwen3-VL's chat template.
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
        self.im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
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
        all_pixel_values_videos = []
        all_video_grid_thw = []

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

            user_text = (
                "This is a sequence interleaved with timestamps and frames. "
                "Your task is to identify the specific timestamp(s) when the given query appears.\n"
            )
            for t in sampled_timestamps:
                user_text += f"timestamp: {t} seconds "
                user_text += "<|vision_start|><|video_pad|><|vision_end|>"
            user_text += f"\nQuery: {query_text}\nAnswer:"

            full_text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{target_text}<|im_end|>\n"
            )
            prompt_text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            video_inputs = self.processor.video_processor(
                pil_frames, return_tensors="pt"
            )

            full_ids = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prompt_len = prompt_ids.shape[0]

            labels = full_ids.clone()
            labels[:prompt_len] = PAD_IDX

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            if "pixel_values_videos" in video_inputs:
                all_pixel_values_videos.append(video_inputs["pixel_values_videos"])
            if "video_grid_thw" in video_inputs:
                all_video_grid_thw.append(video_inputs["video_grid_thw"])

        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_t = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_t = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attn_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_t[i, : len(ids)] = ids
            labels_t[i, : len(lbls)] = lbls
            attn_mask[i, : len(ids)] = 1

        pixel_values_videos = torch.cat(all_pixel_values_videos, dim=0) if all_pixel_values_videos else None
        video_grid_thw = torch.cat(all_video_grid_thw, dim=0) if all_video_grid_thw else None

        return dict(
            input_ids=input_ids_t,
            attention_mask=attn_mask,
            labels=labels_t,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
