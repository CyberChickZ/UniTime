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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Gemma4ForConditionalGeneration  # noqa: F401  — requires tf>=5.0
from transformers.cache_utils import Cache
from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast


def _find_hw(n):
    best = (1, n)
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            best = (i, n // i)
    return best


class Gemma4VLMRForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma4-VL with UniTime additions: cached features + multi-qa mask + spatial pooling."""

    spatial_pool_grid: Optional[Tuple[int, int]] = None
    _mm_model = None  # 保存 Gemma4MultiModalModel 的直接引用, 绕过 peft 劫持

    def set_spatial_pool(self, grid: Optional[Tuple[int, int]]):
        """Set spatial pooling target grid. None = no pooling (default).

        Patches Gemma4Model.get_image_features so pooling happens inside
        self.model.forward() where the base class calls it.
        必须在 peft 包装之后调用 (peft 会劫持 self.model).
        """
        self.spatial_pool_grid = grid
        # 找到真正的 Gemma4MultiModalModel, 不管是否被 peft 包装
        mm = self.model
        while hasattr(mm, 'model') and not hasattr(mm, 'vision_tower'):
            mm = mm.model
        self._mm_model = mm

        if grid is not None:
            _original = mm.get_image_features.__func__
            _self_ref = self

            def _patched_get_image_features(model_self, pixel_values, image_position_ids=None, **kwargs):
                out = _original(model_self, pixel_values, image_position_ids, **kwargs)
                tgt_h, tgt_w = _self_ref.spatial_pool_grid
                tgt_tokens = tgt_h * tgt_w
                pooler = out.pooler_output
                D = pooler.shape[-1]
                total = pooler.shape[0]
                n_images = pixel_values.shape[0] if pixel_values.dim() >= 2 else 1
                tpi = total // n_images
                src_h, src_w = _find_hw(tpi)
                chunks = pooler.reshape(n_images, src_h, src_w, D)
                chunks = chunks.permute(0, 3, 1, 2)
                pooled = F.interpolate(chunks, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
                pooled = pooled.permute(0, 2, 3, 1)
                out.pooler_output = pooled.reshape(n_images * tgt_tokens, D)
                return out

            import types
            self.model.get_image_features = types.MethodType(_patched_get_image_features, self.model)

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
            # Cached-features path: features 已 spatial pooling (collator 做的).
            # scatter 到 inputs_embeds 后, 手动算 PLE, 然后走 peft-wrapped LLM.
            if input_ids is None:
                raise ValueError("feature_inputs path requires input_ids.")
            image_token_id = self.config.image_token_id
            pad_token_id = self.config.text_config.pad_token_id
            image_mask_1d = (input_ids == image_token_id)

            # 1) Embedding (用 PAD 替换 image tokens)
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask_1d] = pad_token_id
            inputs_embeds_local = self.get_input_embeddings()(llm_input_ids)

            # 2) PLE (Per-Layer Embeddings) — 从 PAD-replaced input_ids 算
            mm = self._mm_model if self._mm_model else self.model
            lm_raw = mm.language_model  # 原始 Gemma4TextModel (不经过 peft)
            per_layer_inputs = None
            if getattr(lm_raw, "hidden_size_per_layer_input", 0):
                per_layer_inputs = lm_raw.get_per_layer_inputs(llm_input_ids, inputs_embeds_local)
                per_layer_inputs = lm_raw.project_per_layer_inputs(inputs_embeds_local, per_layer_inputs)

            # 3) Scatter features
            features = feature_inputs.to(inputs_embeds_local.device, inputs_embeds_local.dtype)
            image_mask = image_mask_1d.unsqueeze(-1).expand_as(inputs_embeds_local)
            if inputs_embeds_local[image_mask].numel() != features.numel():
                n_img = image_mask_1d.sum().item()
                raise ValueError(
                    f"feature numel ({features.numel()}) != slots "
                    f"({n_img} × {inputs_embeds_local.shape[-1]})"
                )
            inputs_embeds_local = inputs_embeds_local.masked_scatter(image_mask, features)

            # 4) LLM forward — 通过 peft 包装后的 language_model (LoRA 生效)
            #    peft 包装后, self.model 内部的 language_model 已经有 LoRA layers
            lm_out = lm_raw(
                input_ids=None,
                inputs_embeds=inputs_embeds_local,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                per_layer_inputs=per_layer_inputs,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = lm_out.last_hidden_state

            # 5) Sparse loss: 只在 GT 部分算 logits
            loss = None
            if labels is not None:
                n_gt = (labels[0] != -100).sum().item()
                gt_hidden = hidden_states[:, -(n_gt + 1):-1, :]
                gt_logits = self.lm_head(gt_hidden)
                gt_labels = labels[:, -n_gt:]
                loss = nn.functional.cross_entropy(
                    gt_logits.reshape(-1, gt_logits.size(-1)),
                    gt_labels.reshape(-1),
                    ignore_index=-100,
                )
                logits = gt_logits[:, -1:, :]
            else:
                logits = self.lm_head(hidden_states[:, -1:, :])

            return Gemma4CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=lm_out.past_key_values,
            )

        # Pixel-values path: 走 _mm_model (Gemma4MultiModalModel) 做 vision + LLM forward，
        # 但不用 base class 的 loss（它对整个 seq_len 算 logits 会 OOM）。
        # 只对 labels != -100 的 GT 部分算 lm_head + cross-entropy。
        # 用 _mm_model 直接引用原始 model, 绕过 peft 对 self.model 的劫持。
        if labels is not None and pixel_values is not None and self._mm_model is not None:
            outputs = self._mm_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                image_position_ids=image_position_ids,
                video_position_ids=video_position_ids,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state

            # 只在有标签的位置算 logits (省 ~13 GB 显存)
            label_mask = (labels[0] != -100)
            n_gt = label_mask.sum().item()
            gt_hidden = hidden_states[:, -(n_gt + 1):-1, :]
            gt_logits = self.lm_head(gt_hidden)
            gt_labels = labels[:, -n_gt:]
            loss = nn.functional.cross_entropy(
                gt_logits.reshape(-1, gt_logits.size(-1)),
                gt_labels.reshape(-1),
                ignore_index=-100,
            )

            return Gemma4CausalLMOutputWithPast(
                loss=loss,
                logits=gt_logits[:, -1:, :],
                past_key_values=outputs.past_key_values,
            )

        # Pass-through (无 labels 的推理路径)
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
