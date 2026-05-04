"""
Gemma3-VL data collator for UniTime moment retrieval.

Phase 2: SINGLE-QUERY mr_seg only. No multi-qa optimization (each (video,
action_class) pair is one training example, not multiple queries packed into
one video forward).

Approach (per (video, query) instance):
  1. Load cached Gemma3 features for the video — shape [T, 256, hidden].
  2. Build the prompt by INTERLEAVING `timestamp: X seconds` text and
     `<start_of_image>` markers (one per frame), then expanding each marker
     into the full Gemma3 image-token sequence
     `\n\n<start_of_image><image_soft_token>×256<end_of_image>\n\n`.
  3. Construct the mr_seg target — list of sampled timestamps that fall inside
     ANY of the windows for the query (mirrors collators/qwen2_vl.py:91-102).
  4. Tokenize user-prompt + target separately, concat, build labels (mask the
     prompt tokens with -100 so loss is only computed on the answer span).
  5. Concatenate per-instance feature tensors into a single feature_inputs
     tensor for the wrapper's forward path.

This mirrors `paper_notes/01_unitime.md:55` model-agnostic claim and the
mr_seg multi-window target construction at
`UniTime/collators/qwen2_vl.py:91-102`.
"""
import os
from typing import Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator
from .gemma_vision_process import process_vision_info_gemma3


PAD_IDX = -100


def find_segments(sample_timestamps: List[float], gt_window: List[float]):
    """Mirror of collators/qwen2_vl.py:find_segments."""
    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    closest_start = max(candidates_start) if candidates_start else sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)

    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    closest_end = max(candidates_end) if candidates_end else sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)
    return start_idx, end_idx


@register_collator("gemma3")
class Gemma3DataCollator(BaseDataCollator):
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

        # Cache the special token strings + ids once.
        # Target mode: "timestamp" (default, UniTime original) or "binary" (per-frame 0/1 mask)
        # Set via env var UNITIME_TARGET_MODE=binary in train script
        self.target_mode = os.environ.get("UNITIME_TARGET_MODE", "timestamp")

        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"
        self.image_token = "<image_soft_token>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.start_of_turn_id = self.tokenizer.convert_tokens_to_ids("<start_of_turn>")
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")

        self.mm_tokens_per_image_config = (
            getattr(self.config, "mm_tokens_per_image", None)
            or 256
        )

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    # ---- per-instance prompt construction ----

    def _make_image_sequence(self, tokens_per_image: int) -> str:
        return (
            "\n\n" + self.boi_token
            + (self.image_token * tokens_per_image)
            + self.eoi_token + "\n\n"
        )

    def build_user_text(self, sampled_timestamps: List[float], query: str,
                        combine_t_list: Optional[List[int]] = None,
                        tokens_per_image: int = 256) -> str:
        if self.target_mode == "binary":
            instruction = (
                "This is a sequence of video frames with timestamps. "
                "For the described action, output a binary string where each character "
                "corresponds to one frame: '1' if the action is happening in that frame, "
                "'0' if not.\n"
            )
        else:
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

    def build_target_text(
        self, sampled_timestamps: List[float], windows: List[List[float]]
    ) -> str:
        """Build the target string based on target_mode.

        'timestamp' mode (default): list of sampled timestamps inside any window.
            Mirrors collators/qwen2_vl.py:91-102.
        'binary' mode: per-frame 0/1 string. Each character = one sampled frame.
            '1' if the frame is inside ANY window, '0' otherwise.
            Dense per-frame supervision — much stronger signal than sparse timestamps.
        """
        if self.target_mode == "binary":
            return self._build_binary_target(sampled_timestamps, windows)
        return self._build_timestamp_target(sampled_timestamps, windows)

    def _build_timestamp_target(
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

    def _build_binary_target(
        self, sampled_timestamps: List[float], windows: List[List[float]]
    ) -> str:
        """Per-frame binary mask: '1' if timestamp is inside any window, '0' otherwise."""
        bits = []
        for t in sampled_timestamps:
            inside = any(w[0] <= t <= w[1] for w in windows)
            bits.append("1" if inside else "0")
        return "".join(bits)

    def build_full_prompt(self, user_text: str, target_text: str):
        """Apply Gemma3 chat template structure manually and tokenize.

        Returns (input_ids, labels) for one instance.
        """
        # We bypass apply_chat_template to keep label-mask construction trivial:
        # we know exactly where the target tokens start.
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

    # ---- batch entry point ----

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Phase 2: enforce single-query per instance.
        # Each instance must have exactly one anno (one query, one window list).
        feature_inputs_list, sampled_ts_list, combine_t_lists = process_vision_info_gemma3(
            [inst["message"] for inst in instances]
        )
        if feature_inputs_list is None:
            raise RuntimeError(
                "Gemma3DataCollator phase 2 requires pre-extracted features in the "
                "video item under key 'feature'. Run extract_gemma_features.py first."
            )

        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []
        all_features: List[torch.Tensor] = []

        for inst, feature, sampled_timestamps, combine_t_list in zip(instances, feature_inputs_list, sampled_ts_list, combine_t_lists):
            # Extract query + window from the (Qwen-format) message wrapper.
            # The dataset packs everything in the second user-turn after the
            # video item; we read the original temporal_window/mode that the
            # dataset already returned for us.
            mode = inst["mode"]
            if mode != "mr_seg":
                raise NotImplementedError(
                    f"Gemma3DataCollator phase 2 only supports mode='mr_seg', got '{mode}'"
                )
            temporal_window = inst["temporal_window"]  # list of [list of [s,e]]
            # The dataset's `query` lives inside the message structure (second user turn).
            # Walk the message to find it.
            query_text = None
            for msg in inst["message"]:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text", "")
                            if txt.startswith("Query:"):
                                # `Query:{q}\nAnswer: `
                                query_text = txt.split("Query:", 1)[1].split("\nAnswer", 1)[0].strip()
                                break
                if query_text is not None:
                    break
            if query_text is None:
                raise ValueError(f"could not find query in instance {inst.get('qid')}")

            # phase-2 single-query: only the first window list is used.
            windows = temporal_window[0]

            # Derive tokens_per_image from feature shape
            # 4D [T, H', W', D] → H'*W'; 3D [T, 256, D] → 256
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

        n_image_slots = (input_ids_tensor == self.image_token_id).sum().item()
        if n_image_slots != feature_inputs_concat.shape[0]:
            raise RuntimeError(
                f"image-token slot count {n_image_slots} != feature row count "
                f"{feature_inputs_concat.shape[0]}. Feature shape per video: "
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
            video_grid_thw=None,
            pixel_values_videos=None,
        )
