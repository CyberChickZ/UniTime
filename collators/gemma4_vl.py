"""
Gemma4-VL data collator for UniTime moment retrieval.

Phase 1: SINGLE-QUERY mr_seg only. Same logic as collators/gemma3_vl.py with
Gemma 4 special tokens swapped in:
  Gemma 3: <start_of_image>...<image_soft_token>×N...<end_of_image>
  Gemma 4: <|image>...<|image|>×N...<image|>

mm_tokens_per_image is read dynamically from the loaded feature .pt files
(feature.shape[1]) so we don't have to hardcode 280 (the Gemma 4 image
processor default).
"""
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator
from .gemma_vision_process import process_vision_info_gemma3  # same loader works for Gemma 4


PAD_IDX = -100


def find_segments(sample_timestamps: List[float], gt_window: List[float]):
    """Mirror of collators/qwen2_vl.py:find_segments and gemma3_vl.py:find_segments."""
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

        # Gemma 4 special tokens (different from Gemma 3)
        self.boi_token = "<|image>"     # begin of image, id ~255999
        self.eoi_token = "<image|>"     # end of image
        self.image_token = "<|image|>"  # soft image token, id ~258880
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.start_of_turn_id = self.tokenizer.convert_tokens_to_ids("<start_of_turn>")
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")

        # Will be set on first batch from feature.shape[1]
        self._mm_tokens_per_image = None

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def _full_image_sequence(self, mm_tokens_per_image: int) -> str:
        return (
            "\n\n"
            + self.boi_token
            + (self.image_token * mm_tokens_per_image)
            + self.eoi_token
            + "\n\n"
        )

    def build_user_text(
        self,
        sampled_timestamps: List[float],
        query: str,
        mm_tokens_per_image: int,
    ) -> str:
        full_image_seq = self._full_image_sequence(mm_tokens_per_image)
        parts = [
            "This is a sequence interleaved with timestamps and frames. "
            "Your task is to identify the specific timestamp(s) when the given query appears.\n",
        ]
        for t in sampled_timestamps:
            parts.append(f"timestamp: {t} seconds")
            parts.append(full_image_seq)
        parts.append(f"\nQuery: {query}\nAnswer:")
        return "".join(parts)

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

    def build_full_prompt(self, user_text: str, target_text: str):
        bos = self.tokenizer.bos_token  # <bos>
        prompt_text = (
            f"{bos}\n"
            f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        target_with_eot = f"{target_text}<end_of_turn>\n"

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, return_tensors=None
        )["input_ids"]
        target_ids = self.tokenizer(
            target_with_eot, add_special_tokens=False, return_tensors=None
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [PAD_IDX] * len(prompt_ids) + list(target_ids)
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        feature_inputs_list, sampled_ts_list = process_vision_info_gemma3(
            [inst["message"] for inst in instances]
        )
        if feature_inputs_list is None:
            raise RuntimeError(
                "Gemma4DataCollator phase 1 requires pre-extracted features in the "
                "video item under key 'feature'. Run extract_gemma4_features.py first."
            )

        # Lock in mm_tokens_per_image from the first feature seen.
        if self._mm_tokens_per_image is None:
            first_feat = feature_inputs_list[0]
            if first_feat.dim() != 3:
                raise ValueError(
                    f"feature shape {tuple(first_feat.shape)} unexpected; "
                    f"expected [T, mm_tokens_per_image, hidden_dim]"
                )
            self._mm_tokens_per_image = int(first_feat.shape[1])

        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []
        all_features: List[torch.Tensor] = []

        for inst, feature, sampled_timestamps in zip(instances, feature_inputs_list, sampled_ts_list):
            mode = inst["mode"]
            if mode != "mr_seg":
                raise NotImplementedError(
                    f"Gemma4DataCollator phase 1 only supports mode='mr_seg', got '{mode}'"
                )
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

            user_text = self.build_user_text(sampled_timestamps, query_text, self._mm_tokens_per_image)
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

        n_image_slots = (input_ids_tensor == self.image_token_id).sum().item()
        if n_image_slots != feature_inputs_concat.shape[0]:
            raise RuntimeError(
                f"image-token slot count {n_image_slots} != feature row count "
                f"{feature_inputs_concat.shape[0]}. Check that build_user_text inserts "
                f"{self._mm_tokens_per_image} image tokens per frame and that "
                f"extract_gemma4_features.py produced features with the same shape."
            )

        return dict(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            labels=labels_tensor,
            feature_inputs=feature_inputs_concat,
            multi_qa=False,
            attention_mask_multiqa=None,
            combine_t_list=None,
            pixel_values=None,
            pixel_values_videos=None,
        )
