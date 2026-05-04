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
        # Cached-features path: replicate the base class flow but inject our
        # pre-extracted features instead of running the vision tower.
        #
        # Key constraint: Gemma4's PLE (Per-Layer Embeddings) tries to reverse
        # the embedding lookup when input_ids=None, causing an O(seq×vocab×hidden)
        # OOM. We must compute PLE from input_ids BEFORE scattering features.
        if feature_inputs is not None:
            if input_ids is None:
                raise ValueError(
                    "feature_inputs path requires input_ids (to locate image_token "
                    "positions for masked_scatter)."
                )
            image_token_id = self.config.image_token_id
            pad_token_id = self.config.text_config.pad_token_id
            image_mask_1d = (input_ids == image_token_id)

            # 1) Replace image tokens with PAD for clean embedding + PLE
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask_1d] = pad_token_id
            inputs_embeds_local = self.get_input_embeddings()(llm_input_ids)

            # 2) Compute PLE from the PAD-replaced IDs (matches base class logic)
            #    self.model = Gemma4MultiModalModel
            #    self.model.language_model = Gemma4Model (text backbone with PLE)
            lm = self.model.language_model
            per_layer_inputs = None
            if getattr(lm, "hidden_size_per_layer_input", 0):
                per_layer_inputs = lm.get_per_layer_inputs(llm_input_ids, inputs_embeds_local)
                per_layer_inputs = lm.project_per_layer_inputs(inputs_embeds_local, per_layer_inputs)

            # 3) Scatter cached features into the embedding
            image_mask = image_mask_1d.unsqueeze(-1).expand_as(inputs_embeds_local)
            features = feature_inputs.to(inputs_embeds_local.device, inputs_embeds_local.dtype)
            if inputs_embeds_local[image_mask].numel() != features.numel():
                n_img = image_mask_1d.sum().item()
                raise ValueError(
                    f"feature_inputs numel ({features.numel()}) != image-token slots "
                    f"({n_img} tokens × {inputs_embeds_local.shape[-1]} dim)."
                )
            inputs_embeds_local = inputs_embeds_local.masked_scatter(image_mask, features)

            attn_mask_to_pass = (
                attention_mask_multiqa.to(inputs_embeds_local.device, inputs_embeds_local.dtype)
                if (attention_mask_multiqa is not None and multi_qa)
                else attention_mask
            )

            # 4) Call Gemma4Model directly (bypass MultiModalModel + ConditionalGen
            #    layers that enforce XOR checks and do their own PLE)
            lm_out = lm(
                input_ids=None,
                inputs_embeds=inputs_embeds_local,
                attention_mask=attn_mask_to_pass,
                position_ids=position_ids,
                past_key_values=past_key_values,
                per_layer_inputs=per_layer_inputs,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = lm_out.last_hidden_state

            loss = None
            logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return Gemma4CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=lm_out.past_key_values,
            )

        # Pass-through (raw pixel_values path)
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        feature_inputs=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if is_first_iteration or not use_cache:
            model_inputs["feature_inputs"] = feature_inputs
        else:
            model_inputs["feature_inputs"] = None
        return model_inputs


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
