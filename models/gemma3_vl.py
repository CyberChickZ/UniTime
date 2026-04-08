"""
Gemma3 wrapper for UniTime moment retrieval.

Mirrors models/qwen2_vl.py:Qwen2VLMRForConditionalGeneration but adapted to
Gemma3's simpler architecture (no mRoPE, fixed-size SigLIP image tokens).

Compared to the Qwen2VL wrapper:
  - No `get_rope_index_multiqa` (Gemma3 uses 1D rotary, not 3D)
  - No `encode_video_chunk` (vision tower handled by base get_image_features
    when pixel_values present; for cached features we go through feature_inputs)
  - No `image_grid_thw` / `video_grid_thw` (SigLIP outputs fixed mm_tokens_per_image)
  - Forward signature additions (over base Gemma3ForConditionalGeneration):
      feature_inputs: pre-extracted image features in LLM hidden space
      multi_qa: whether to use the multi-query attention mask path
      attention_mask_multiqa: explicit 4D mask built by the collator
      combine_t_list: ignored, kept for collator-interface symmetry

The forward implementation is a port of Gemma3ForConditionalGeneration.forward
with two extra branches:
  1. feature_inputs path: skip vision tower entirely, masked_scatter the cached
     features into inputs_embeds at image_token_index positions.
  2. attention_mask_multiqa path: bypass _update_causal_mask, use the explicit
     mask the collator built (mirrors Qwen2VL wrapper line 310).
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Gemma3ForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast


class Gemma3VLMRForConditionalGeneration(Gemma3ForConditionalGeneration):
    """Gemma3-VL with UniTime additions: cached features + multi-qa mask."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # ---- UniTime extras ----
        feature_inputs: Optional[torch.Tensor] = None,
        multi_qa: bool = False,
        attention_mask_multiqa: Optional[torch.Tensor] = None,
        combine_t_list=None,  # noqa: ARG002 — kept for collator interface symmetry
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        is_training = token_type_ids is not None and labels is not None

        # Replace OOV image_token with PAD to safely embed (mirrors base impl)
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # ---- merge image features into inputs_embeds ----
        # Three paths, in priority order:
        #   1. cached features (UniTime offline path)
        #   2. raw pixel_values (standard Gemma3 path)
        #   3. neither (text-only forward)
        image_features = None
        if feature_inputs is not None:
            # feature_inputs shape options the collator may produce:
            #   (sum_tokens, hidden) — already flat (Qwen-style concatenation)
            #   (num_images, mm_tokens_per_image, hidden) — per-image
            # masked_scatter only requires matching numel, so any flat-equivalent shape works.
            image_features = feature_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_text_image_tokens = special_image_mask.sum().item() // inputs_embeds.shape[-1]
                raise ValueError(
                    f"feature_inputs numel ({image_features.numel()}) does not match the number of "
                    f"image-token slots in input_ids ({n_text_image_tokens} tokens × "
                    f"{inputs_embeds.shape[-1]} hidden_dim). Check that your collator inserts "
                    f"mm_tokens_per_image={self.config.mm_tokens_per_image} placeholder tokens "
                    f"per frame and that extract_gemma_features.py produced features with the "
                    f"same dim."
                )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        elif pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels (BC, mirrors base)
        if labels is not None and self.pad_token_id in labels:
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        # ---- attention mask: multi-qa explicit OR base causal-bidir ----
        if attention_mask_multiqa is not None and multi_qa:
            causal_mask = attention_mask_multiqa.to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                inputs_embeds,
                is_training,
            )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None and attention_mask.ndim == 2:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Mirror the base impl, but forward the UniTime extras through kwargs.
        feature_inputs = kwargs.pop("feature_inputs", None)
        multi_qa = kwargs.pop("multi_qa", False)
        attention_mask_multiqa = kwargs.pop("attention_mask_multiqa", None)
        combine_t_list = kwargs.pop("combine_t_list", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            **kwargs,
        )

        # Cached image features should only be injected on the first decoding step.
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["feature_inputs"] = feature_inputs
        else:
            model_inputs["feature_inputs"] = None

        model_inputs["multi_qa"] = multi_qa
        model_inputs["attention_mask_multiqa"] = attention_mask_multiqa
        model_inputs["combine_t_list"] = combine_t_list

        return model_inputs


# ---------------------------------------------------------------------------
# Processor wrapper
# ---------------------------------------------------------------------------
# Gemma3Processor builds the prompt by replacing each `<boi>` placeholder with
# the full image-token sequence (`<boi><img>×256<eoi>`). For UniTime we want to
# interleave timestamp text BETWEEN frames, so the collator pre-builds a prompt
# that already contains many `<boi>` markers and per-frame timestamp strings.
# The base Gemma3Processor.__call__ already handles multi-image prompts, so we
# only need a thin subclass for symmetry with Qwen2VLMRProcessor and to expose
# the same kwargs (`features=`, `timestamps=`, `combine_t_list=`) as the
# Qwen2VL processor — those kwargs are no-ops here because the prompt is fully
# pre-baked, but accepting them keeps the collator code path identical.

from transformers import Gemma3Processor


class Gemma3VLMRProcessor(Gemma3Processor):
    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        audio=None,
        # UniTime extras (no-op, accepted for symmetry with Qwen2VL processor)
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
