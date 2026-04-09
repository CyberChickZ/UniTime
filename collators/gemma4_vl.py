"""
Gemma4-VL data collator for UniTime moment retrieval.

PIXEL-VALUES PATH (no pre-extracted features): loads PIL images from disk at
each training step, passes them alongside the tokenized prompt to the standard
Gemma4ForConditionalGeneration.forward. The model runs its own frozen vision
tower + PLE + language model. This avoids the PLE + inputs_embeds OOM issue.

Each (video, query) instance:
  1. Load 32 uniform video frames as PIL images
  2. Build prompt text with timestamp + <|image> markers interleaved + query
  3. Call Gemma4Processor to get input_ids + pixel_values + image_position_ids
  4. Build mr_seg target and concat as labels
  5. Return batch dict that Gemma4ForConditionalGeneration.forward accepts directly
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


def find_segments(sample_timestamps: List[float], gt_window: List[float]):
    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    closest_start = max(candidates_start) if candidates_start else sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)
    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    closest_end = max(candidates_end) if candidates_end else sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)
    return start_idx, end_idx


@register_collator("gemma4")
class Gemma4DataCollator(BaseDataCollator):
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

        self.boi_token = getattr(tokenizer, "boi_token", "<|image>")
        self.start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        self.end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self.num_frames = 32

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def _load_frames(self, video_path: str) -> List[Image.Image]:
        """Load num_frames uniform frames as PIL images."""
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        fps = vr.get_avg_fps()
        n = min(self.num_frames, total)
        idx = torch.linspace(0, total - 1, n).round().long().tolist()
        frames = vr.get_batch(idx).asnumpy()
        timestamps = [round(i / fps, 1) for i in idx]
        return [Image.fromarray(f) for f in frames], timestamps

    def build_target_text(
        self, sampled_timestamps: List[float], windows: List[List[float]]
    ) -> str:
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
        all_image_position_ids = []

        for inst in instances:
            mode = inst["mode"]
            if mode != "mr_seg":
                raise NotImplementedError(f"Gemma4DataCollator only supports mr_seg, got '{mode}'")
            temporal_window = inst["temporal_window"]

            # Find video path from message
            video_path = None
            for msg in inst["message"]:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "video":
                            video_path = item.get("video")
                            break
                if video_path is not None:
                    break

            # Find query
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

            # Load frames
            pil_frames, sampled_timestamps = self._load_frames(video_path)

            # Build user content: interleaved timestamp text + image markers
            user_content = [
                {"type": "text", "text": "This is a sequence interleaved with timestamps and frames. "
                 "Your task is to identify the specific timestamp(s) when the given query appears.\n"},
            ]
            for t, _ in zip(sampled_timestamps, pil_frames):
                user_content.append({"type": "text", "text": f"timestamp: {t} seconds "})
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": f"\nQuery: {query_text}\nAnswer:"})

            # Build target
            target_text = self.build_target_text(sampled_timestamps, windows)

            # Build messages
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]

            # Use processor to build input_ids + pixel_values
            # apply_chat_template returns text with <|image> markers replaced
            prompt_with_target = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Now split into prompt and target for label masking
            prompt_only_msgs = [
                {"role": "user", "content": user_content},
            ]
            prompt_only_text = self.processor.apply_chat_template(
                prompt_only_msgs, tokenize=False, add_generation_prompt=True
            )

            # Process full text (prompt + target) with images
            full_inputs = self.processor(
                text=[prompt_with_target],
                images=pil_frames,
                return_tensors="pt",
                padding=False,
            )
            # Process prompt only (for label masking: find where prompt ends)
            prompt_inputs = self.processor(
                text=[prompt_only_text],
                images=pil_frames,
                return_tensors="pt",
                padding=False,
            )

            full_ids = full_inputs["input_ids"][0]
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Labels: -100 for prompt, real ids for target
            labels = full_ids.clone()
            labels[:prompt_len] = PAD_IDX

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_pixel_values.append(full_inputs["pixel_values"])
            if "image_position_ids" in full_inputs:
                all_image_position_ids.append(full_inputs["image_position_ids"])

        # Pad to longest
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_t = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_t = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attn_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_t[i, :len(ids)] = ids
            labels_t[i, :len(lbls)] = lbls
            attn_mask[i, :len(ids)] = 1

        # Concat pixel_values across batch
        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        image_position_ids = (
            torch.cat(all_image_position_ids, dim=0) if all_image_position_ids else None
        )

        result = dict(
            input_ids=input_ids_t,
            attention_mask=attn_mask,
            labels=labels_t,
            pixel_values=pixel_values,
        )
        if image_position_ids is not None:
            result["image_position_ids"] = image_position_ids
        return result
