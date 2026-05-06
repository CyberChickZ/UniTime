"""Gemma4 Collator — Temporal Action Segmentation 滑窗训练.

两种路径:
  1. Cached features: 从 .pt 加载预提取的 [T, 2520, 2560] features, 按 sample_indices 切片
  2. Online pixels: decord 读帧 → image_processor (fallback, 需要更多显存)

模型的 forward (feature_inputs path) 负责 spatial pooling.

Token 序列结构:
  <bos><|turn>user
  Previous actions: take, open, put
  0.0s <boi>[img×tpi]<eoi> 0.3s <boi>[img×tpi]<eoi> ... 29.7s <boi>[img×tpi]<eoi>
  List all action segments with start and end timestamps.
  <turn|><|turn>model
  take 0.0 1.7, open 2.3 6.0, ...<turn|>

tpi = spatial_pool_grid[0] * spatial_pool_grid[1] (如 12×12=144).
"""
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator

PAD_IDX = -100


@register_collator("gemma4_tas")
class Gemma4TASDataCollator(BaseDataCollator):
    def __init__(
        self,
        config: Optional[AutoConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[AutoProcessor] = None,
        mask_question_tokens: bool = True,
        spatial_pool_grid: Optional[tuple] = None,  # (h, w) 或 None 表示不 pool
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.spatial_pool_grid = spatial_pool_grid

        self.image_token_id = config.image_token_id

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _load_features(self, feat_path: str, sample_indices: List[int]) -> torch.Tensor:
        """从 .pt 加载预提取 features, 按 sample_indices 切片.
        如果设置了 spatial_pool_grid, 在 CPU 上做 bilinear pooling.
        返回 [n_frames, tpi, D] (tpi = pool 后的 tokens per image).
        """
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        feature = data["feature"]  # [T_total, raw_tpi, D]
        sliced = feature[sample_indices]  # [n_frames, raw_tpi, D]

        if self.spatial_pool_grid is not None:
            import torch.nn.functional as Fn
            import math
            tgt_h, tgt_w = self.spatial_pool_grid
            n, raw_tpi, D = sliced.shape
            # 找 raw_tpi 的最接近正方形的因数分解
            src_h, src_w = 1, raw_tpi
            for i in range(2, int(math.sqrt(raw_tpi)) + 1):
                if raw_tpi % i == 0:
                    src_h, src_w = i, raw_tpi // i
            sliced = sliced.reshape(n, src_h, src_w, D).permute(0, 3, 1, 2).float()
            sliced = Fn.interpolate(sliced, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
            sliced = sliced.permute(0, 2, 3, 1).reshape(n, tgt_h * tgt_w, D).to(feature.dtype)

        return sliced

    def _process_single(self, inst: Dict):
        """处理单条数据: 加载 features → 构造 input_ids + labels."""
        context = inst["context"]
        gt_text = inst["gt_text"]
        timestamps = inst["frame_timestamps"]
        feat_path = inst["feat_path"]
        sample_indices = inst["sample_indices"]

        # 加载预提取 features
        features = self._load_features(feat_path, sample_indices)  # [90, 2520, D]
        tokens_per_image_raw = features.shape[1]  # 2520

        # 每帧在 input_ids 中占的 image token 数 (pool 后)
        if self.spatial_pool_grid is not None:
            tokens_per_image = self.spatial_pool_grid[0] * self.spatial_pool_grid[1]
        else:
            tokens_per_image = tokens_per_image_raw

        # 构造 token 序列
        bos = self.tokenizer.bos_token or ""
        boi_id = self.config.boi_token_id
        eoi_id = self.config.eoi_token_id
        img_tok = self.image_token_id
        newline_ids = self._tokenize("\n\n")

        # Header: 前文 context
        if context:
            context_str = ", ".join(context)
            header_text = f"{bos}<|turn>user\nPrevious actions: {context_str}\n"
        else:
            header_text = f"{bos}<|turn>user\n"
        header_ids = self._tokenize(header_text)

        # Body: 交错 timestamp + image tokens (每帧前标时间)
        body_ids = []
        for t in timestamps:
            body_ids.extend(self._tokenize(f"{t}s "))
            body_ids.extend(newline_ids)
            body_ids.append(boi_id)
            body_ids.extend([img_tok] * tokens_per_image)
            body_ids.append(eoi_id)
            body_ids.extend(newline_ids)

        # Instruction + turn tokens
        instr_ids = self._tokenize(
            "List all action segments with start and end timestamps.\n"
        )
        turn_end_ids = self._tokenize("<turn|>\n<|turn>model\n")

        # GT target
        target_ids = self._tokenize(f"{gt_text}<turn|>\n")

        # 拼接: prompt (masked) + target (计算 loss)
        prompt_ids = header_ids + body_ids + instr_ids + turn_end_ids
        input_ids = prompt_ids + target_ids
        labels = [PAD_IDX] * len(prompt_ids) + list(target_ids)

        return input_ids, labels, features, tokens_per_image

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_features = []  # 每个 instance 的 [90, 2520, D]
        tpi = None

        for inst in instances:
            input_ids, labels, features, tokens_per_image = self._process_single(inst)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_features.append(features)
            if tpi is None:
                tpi = tokens_per_image

        # Pad input_ids / labels 到 batch 内最长
        max_len = max(len(x) for x in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_tensor = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_tensor = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attention_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels_tensor[i, :len(lbls)] = torch.tensor(lbls, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        # feature_inputs: concat 所有帧的 features → [total_frames * 2520, D]
        # 模型 forward 的 feature_inputs path 期望 2D [N_tokens, D]
        # spatial pooling 在 forward 里做 (需要知道每帧 2520 tokens 来 reshape)
        feature_concat = torch.cat(
            [f.reshape(-1, f.shape[-1]) for f in all_features], dim=0
        )

        # 验证: input_ids 中的 image token 数
        n_image_slots = (input_ids_tensor == self.image_token_id).sum().item()
        total_frames = sum(f.shape[0] for f in all_features)
        expected_slots = total_frames * tpi
        if n_image_slots != expected_slots:
            raise RuntimeError(
                f"image-token slots {n_image_slots} != expected {expected_slots} "
                f"({total_frames} frames x {tpi} tokens/frame)"
            )

        return dict(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            labels=labels_tensor,
            feature_inputs=feature_concat,
        )
