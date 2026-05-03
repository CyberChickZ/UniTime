"""
Qwen3-VL wrapper for UniTime moment retrieval.

Qwen3-VL uses pixel_values path (model runs its own vision tower each step).
The forward adds UniTime extras: feature_inputs, multi_qa, attention_mask_multiqa,
combine_t_list.

Requires transformers >= 5.5.0 (UniTime-gemma4 env).
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast


class Qwen3VLMRForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # UniTime extras
        feature_inputs=None,
        multi_qa: bool = False,
        attention_mask_multiqa=None,
        combine_t_list=None,
        **kwargs,
    ) -> Qwen3VLCausalLMOutputWithPast:
        if feature_inputs is not None and input_ids is not None:
            text_embeds = self.model.get_input_embeddings()(input_ids)
            video_token_id = self.config.video_token_id
            video_mask = (
                (input_ids == video_token_id)
                .unsqueeze(-1)
                .expand_as(text_embeds)
                .to(text_embeds.device)
            )
            features = feature_inputs.to(text_embeds.device, text_embeds.dtype)
            text_embeds = text_embeds.masked_scatter(video_mask, features)

            if attention_mask_multiqa is not None and multi_qa:
                attention_mask = attention_mask_multiqa.to(text_embeds.device, text_embeds.dtype)

            return super().forward(
                input_ids=None,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if attention_mask_multiqa is not None and multi_qa:
            attention_mask = attention_mask_multiqa.to(
                (inputs_embeds if inputs_embeds is not None else input_ids).device
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


from transformers import Qwen3VLProcessor


class Qwen3VLMRProcessor(Qwen3VLProcessor):
    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        features=None,
        timestamps=None,
        combine_t_list=None,
        **kwargs,
    ):
        return super().__call__(
            images=images,
            text=text,
            videos=videos,
            **kwargs,
        )
