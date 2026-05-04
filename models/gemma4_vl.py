"""
Gemma4 wrapper for UniTime moment retrieval.

Mirrors models/gemma3_vl.py but adapted to Gemma 4's slightly deeper class
hierarchy (Gemma4ForConditionalGeneration only owns self.model + self.lm_head;
all multimodal merging lives in Gemma4Model).

Approach: pre-merge cached features into inputs_embeds, then call the parent
forward with input_ids=None and pixel_values=None so Gemma4Model skips its own
vision merging step (image_mask comes back empty against the pre-merged
embeddings, which is what we want).

This file imports `transformers.Gemma4ForConditionalGeneration`, which only
exists in transformers >= 5.0. The collators/__init__ and loaders/__init__
guard the imports of this module so that the older UniTime env (transformers
4.51 + Gemma 3 wrapper) keeps working.
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Gemma4ForConditionalGeneration  # noqa: F401  — requires tf>=5.0
from transformers.cache_utils import Cache
from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast


class Gemma4VLMRForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma4-VL with UniTime additions: cached features + multi-qa mask."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        video_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # ---- UniTime extras ----
        feature_inputs: Optional[torch.Tensor] = None,
        multi_qa: bool = False,
        attention_mask_multiqa: Optional[torch.Tensor] = None,
        combine_t_list=None,  # noqa: ARG002 — kept for collator interface symmetry
        **kwargs,
    ) -> Gemma4CausalLMOutputWithPast:
        # Cached-features path: pre-merge features into inputs_embeds, then call
        # super().forward with input_ids=None so Gemma4Model.get_placeholder_mask
        # returns an empty mask (no double-merging) and the vision_tower call is
        # skipped (pixel_values=None).
        if feature_inputs is not None:
            if input_ids is None:
                raise ValueError(
                    "feature_inputs path requires input_ids (to locate image_token "
                    "positions for masked_scatter)."
                )
            # Embed text tokens
            inputs_embeds_local = self.get_input_embeddings()(input_ids)
            image_token_id = self.config.image_token_id
            image_mask = (
                (input_ids == image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds_local)
                .to(inputs_embeds_local.device)
            )
            features = feature_inputs.to(inputs_embeds_local.device, inputs_embeds_local.dtype)
            if inputs_embeds_local[image_mask].numel() != features.numel():
                n_text_image_tokens = image_mask.sum().item() // inputs_embeds_local.shape[-1]
                raise ValueError(
                    f"feature_inputs numel ({features.numel()}) does not match the number of "
                    f"image-token slots in input_ids ({n_text_image_tokens} tokens × "
                    f"{inputs_embeds_local.shape[-1]} hidden_dim). Check that your collator "
                    f"inserts mm_tokens_per_image placeholder tokens per frame matching the "
                    f"shape of features written by extract_gemma4_features.py."
                )
            inputs_embeds_local = inputs_embeds_local.masked_scatter(image_mask, features)

            # Use the explicit multi-qa mask if provided, else let the base class
            # build the standard causal+bidirectional one.
            attn_mask_to_pass = (
                attention_mask_multiqa.to(inputs_embeds_local.device, inputs_embeds_local.dtype)
                if (attention_mask_multiqa is not None and multi_qa)
                else attention_mask
            )

            # Build mm_token_type_ids for PLE: 0=text, 1=image
            if mm_token_type_ids is None:
                mm_token_type_ids = (input_ids == image_token_id).long()

            return super().forward(
                input_ids=None,
                inputs_embeds=inputs_embeds_local,
                pixel_values=None,
                pixel_values_videos=None,
                input_features=None,
                attention_mask=attn_mask_to_pass,
                input_features_mask=None,
                position_ids=position_ids,
                image_position_ids=None,
                video_position_ids=None,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # Pass-through (raw pixel_values path) — let the base do its full merge.
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Processor wrapper
# ---------------------------------------------------------------------------
from transformers import Gemma4Processor


class Gemma4VLMRProcessor(Gemma4Processor):
    """No-op subclass that accepts the UniTime collator extras as kwargs."""

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        audio=None,
        # UniTime extras (no-op, accepted for symmetry with Gemma3 / Qwen2VL processors)
        features=None,  # noqa: ARG002
        timestamps=None,  # noqa: ARG002
        combine_t_list=None,  # noqa: ARG002
        **kwargs,
    ):
        return super().__call__(
            images=images,
            text=text,
            videos=videos,
            audio=audio,
            **kwargs,
        )
