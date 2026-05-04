"""
Qwen3-VL data collator for UniTime moment retrieval.

Feature-inputs path (aligned with UniTime upstream):
  - Pre-extracted features [T, H', W', D] from extract_qwen3_features.py
  - combine_timestamps for coarse-grained timestamp grouping
  - Reuses gemma_vision_process.py for feature loading + combine

Requires transformers >= 5.5.0 (UniTime-gemma4 env).
"""
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator
from .gemma_vision_process import process_vision_info_gemma3

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
        self.video_token_id = config.video_token_id

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

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

    def build_user_text(self, sampled_timestamps, query, combine_t_list, tokens_per_image):
        instruction = (
            "This is a sequence interleaved with timestamps and frames. "
            "Your task is to identify the specific timestamp(s) when the given query appears.\n"
        )
        # Qwen3-VL uses <|vision_start|><|video_pad|>×N<|vision_end|> for video tokens
        pad_token = "<|video_pad|>"
        vision_seq = "<|vision_start|>" + (pad_token * tokens_per_image) + "<|vision_end|>"
        parts = [instruction]
        if combine_t_list is None:
            combine_t_list = [1] * len(sampled_timestamps)
        for t, n_frames in zip(sampled_timestamps, combine_t_list):
            parts.append(f"timestamp: {t} seconds; feature: ")
            for _ in range(n_frames):
                parts.append(vision_seq)
        parts.append(f"\nQuery: {query}\nAnswer:")
        return "".join(parts)

    def build_full_prompt(self, user_text, target_text):
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        target_with_end = f"{target_text}<|im_end|>\n"
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_with_end, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [PAD_IDX] * len(prompt_ids) + list(target_ids)
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        feature_inputs_list, sampled_ts_list, combine_t_lists = process_vision_info_gemma3(
            [inst["message"] for inst in instances]
        )
        if feature_inputs_list is None:
            raise RuntimeError(
                "Qwen3VLDataCollator requires pre-extracted features. "
                "Run extract_qwen3_features.py first."
            )

        all_input_ids = []
        all_labels = []
        all_features = []

        for inst, feature, sampled_timestamps, combine_t_list in zip(
            instances, feature_inputs_list, sampled_ts_list, combine_t_lists
        ):
            mode = inst["mode"]
            if mode != "mr_seg":
                raise NotImplementedError(f"Only mr_seg supported, got '{mode}'")
            temporal_window = inst["temporal_window"]

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
            if query_text is None:
                raise ValueError(f"could not find query in instance {inst.get('qid')}")

            windows = temporal_window[0]

            if feature.dim() == 4:
                tokens_per_image = feature.shape[1] * feature.shape[2]
            else:
                tokens_per_image = feature.shape[1]

            user_text = self.build_user_text(sampled_timestamps, query_text, combine_t_list, tokens_per_image)
            target_text = self.build_target_text(sampled_timestamps, windows)
            input_ids, labels = self.build_full_prompt(user_text, target_text)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_features.append(feature)

        max_len = max(len(x) for x in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_tensor = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_tensor = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attention_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels_tensor[i, : len(lbls)] = torch.tensor(lbls, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        feature_inputs_concat = torch.cat(
            [f.reshape(-1, f.shape[-1]) for f in all_features], dim=0
        )

        n_video_slots = (input_ids_tensor == self.video_token_id).sum().item()
        if n_video_slots != feature_inputs_concat.shape[0]:
            raise RuntimeError(
                f"video-token slot count {n_video_slots} != feature row count "
                f"{feature_inputs_concat.shape[0]}. Feature shapes: "
                f"{[tuple(f.shape) for f in all_features]}"
            )

        return dict(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            labels=labels_tensor,
            feature_inputs=feature_inputs_concat,
            multi_qa=False,
            attention_mask_multiqa=None,
            combine_t_list=None,
        )
