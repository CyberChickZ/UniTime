"""
Gemma4-VL data collator for UniTime moment retrieval.

Feature-inputs path (aligned with UniTime upstream):
  - Pre-extracted features [T, H', W', D] from extract_gemma4_features.py
  - combine_timestamps for coarse-grained timestamp grouping
  - Same pipeline as Gemma3 collator, reuses gemma_vision_process.py

Requires transformers >= 5.0 (UniTime-gemma4 env).
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

        self.image_token_id = config.image_token_id
        self.boi_token = tokenizer.convert_ids_to_tokens(config.boi_token_id)
        self.eoi_token = tokenizer.convert_ids_to_tokens(config.eoi_token_id)
        self.image_token = tokenizer.convert_ids_to_tokens(config.image_token_id)

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def _make_image_sequence(self, tokens_per_image: int) -> str:
        return (
            "\n\n" + self.boi_token
            + (self.image_token * tokens_per_image)
            + self.eoi_token + "\n\n"
        )

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
        img_seq = self._make_image_sequence(tokens_per_image)
        parts = [instruction]
        if combine_t_list is None:
            combine_t_list = [1] * len(sampled_timestamps)
        for t, n_frames in zip(sampled_timestamps, combine_t_list):
            parts.append(f"timestamp: {t} seconds")
            for _ in range(n_frames):
                parts.append(img_seq)
        parts.append(f"\nQuery: {query}\nAnswer:")
        return "".join(parts)

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def build_input_ids_direct(self, sampled_timestamps, query, combine_t_list,
                                tokens_per_image, target_text):
        """Build input_ids by inserting image token IDs directly as integers.

        Avoids string-concatenation of <image_soft_token>×N which the tokenizer
        may not faithfully round-trip back to exactly N token IDs.
        """
        bos = self.tokenizer.bos_token or ""
        instruction = (
            "This is a sequence interleaved with timestamps and frames. "
            "Your task is to identify the specific timestamp(s) when the given query appears.\n"
        )
        header_ids = self._tokenize(f"{bos}<|turn>user\n{instruction}")

        boi_id = self.config.boi_token_id
        eoi_id = self.config.eoi_token_id
        img_tok = self.image_token_id
        newline_ids = self._tokenize("\n\n")

        body_ids = []
        if combine_t_list is None:
            combine_t_list = [1] * len(sampled_timestamps)
        for t, n_frames in zip(sampled_timestamps, combine_t_list):
            body_ids.extend(self._tokenize(f"timestamp: {t} seconds"))
            for _ in range(n_frames):
                body_ids.extend(newline_ids)
                body_ids.append(boi_id)
                body_ids.extend([img_tok] * tokens_per_image)
                body_ids.append(eoi_id)
                body_ids.extend(newline_ids)

        query_ids = self._tokenize(f"\nQuery: {query}\nAnswer:")
        turn_end_ids = self._tokenize("<turn|>\n<|turn>model\n")
        target_ids = self._tokenize(f"{target_text}<turn|>\n")

        prompt_ids = header_ids + body_ids + query_ids + turn_end_ids
        input_ids = prompt_ids + target_ids
        labels = [PAD_IDX] * len(prompt_ids) + list(target_ids)
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        feature_inputs_list, sampled_ts_list, combine_t_lists = process_vision_info_gemma3(
            [inst["message"] for inst in instances]
        )
        if feature_inputs_list is None:
            raise RuntimeError(
                "Gemma4DataCollator requires pre-extracted features. "
                "Run extract_gemma4_features.py first."
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

            target_text = self.build_target_text(sampled_timestamps, windows)
            input_ids, labels = self.build_input_ids_direct(
                sampled_timestamps, query_text, combine_t_list,
                tokens_per_image, target_text,
            )

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
            pixel_values=None,
            pixel_values_videos=None,
        )
