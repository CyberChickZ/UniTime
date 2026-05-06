"""Gemma4 Collator — Temporal Action Segmentation 滑窗训练.

两种路径:
  1. Cached features: 从 .pt 加载预提取 features → feature_inputs → model 内部 spatial pool
  2. Online pixels: decord 读帧 → image_processor → pixel_values → model 自己过 vision encoder

选择逻辑: inst["feat_path"] 存在则用 cached, 否则用 online.
"""
from typing import Dict, List, Optional, Sequence
import math

import decord
import torch
import torch.nn.functional as Fn
from PIL import Image
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
        spatial_pool_grid: Optional[tuple] = None,  # (h, w) 或 None
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

    def _read_frames(self, video_path: str, indices: List[int]) -> List[Image.Image]:
        """用 decord 按帧索引读取视频帧."""
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]

    def _spatial_pool(self, features: torch.Tensor) -> torch.Tensor:
        """CPU 上对 features [n, raw_tpi, D] 做 bilinear spatial pooling."""
        if self.spatial_pool_grid is None:
            return features
        tgt_h, tgt_w = self.spatial_pool_grid
        n, raw_tpi, D = features.shape
        src_h, src_w = 1, raw_tpi
        for i in range(2, int(math.sqrt(raw_tpi)) + 1):
            if raw_tpi % i == 0:
                src_h, src_w = i, raw_tpi // i
        pooled = features.reshape(n, src_h, src_w, D).permute(0, 3, 1, 2).float()
        pooled = Fn.interpolate(pooled, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        return pooled.permute(0, 2, 3, 1).reshape(n, tgt_h * tgt_w, D).to(features.dtype)

    def _load_features(self, feat_path: str, sample_indices: List[int]) -> torch.Tensor:
        """从 .pt 加载预提取 features, 按 sample_indices 切片 + spatial pool."""
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        sliced = data["feature"][sample_indices]
        return self._spatial_pool(sliced)

    def _load_pixels(self, video_path: str, sample_indices: List[int]):
        """读视频帧 → image_processor → pixel_values + position_ids."""
        frames = self._read_frames(video_path, sample_indices)
        inputs = self.processor.image_processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        position_ids = inputs.get("image_position_ids")
        raw_tpi = pixel_values.shape[1]  # 原始 vision encoder 输入 patch 数
        return pixel_values, position_ids, raw_tpi

    def _build_token_sequence(self, context, timestamps, tokens_per_image, gt_text):
        """构造 input_ids + labels."""
        bos = self.tokenizer.bos_token or ""
        boi_id = self.config.boi_token_id
        eoi_id = self.config.eoi_token_id
        img_tok = self.image_token_id
        newline_ids = self._tokenize("\n\n")

        if context:
            header = self._tokenize(f"{bos}<|turn>user\nPrevious actions: {', '.join(context)}\n")
        else:
            header = self._tokenize(f"{bos}<|turn>user\n")

        body = []
        for t in timestamps:
            body.extend(self._tokenize(f"{t}s "))
            body.extend(newline_ids)
            body.append(boi_id)
            body.extend([img_tok] * tokens_per_image)
            body.append(eoi_id)
            body.extend(newline_ids)

        instr = self._tokenize("List all action segments with start and end timestamps.\n")
        turn = self._tokenize("<turn|>\n<|turn>model\n")
        target = self._tokenize(f"{gt_text}<turn|>\n")

        prompt = header + body + instr + turn
        input_ids = prompt + target
        labels = [PAD_IDX] * len(prompt) + list(target)
        return input_ids, labels

    def _process_single(self, inst: Dict):
        """处理单条: cached features 或 online pixels."""
        feat_path = inst.get("feat_path")
        use_cached = feat_path is not None

        if use_cached:
            # Cached features 路径
            features = self._load_features(feat_path, inst["sample_indices"])
            tpi = features.shape[1]
            input_ids, labels = self._build_token_sequence(
                inst["context"], inst["frame_timestamps"], tpi, inst["gt_text"]
            )
            return {"input_ids": input_ids, "labels": labels, "feature_inputs": features, "mode": "cached"}
        else:
            # Online pixels 路径
            pixel_values, position_ids, raw_tpi = self._load_pixels(
                inst["video_path"], inst["sample_indices"]
            )
            # pool 后的 tpi 决定 input_ids 中的 image token 数
            if self.spatial_pool_grid is not None:
                tpi = self.spatial_pool_grid[0] * self.spatial_pool_grid[1]
            else:
                # 在线路径 pooler_output 是 264 tokens (vision tower 内部 pool)
                # 但 input_ids 需要知道最终每帧多少 token
                # Gemma4 base forward 自己处理, 每帧 pooler_output = 264
                tpi = 264  # Gemma4 VisionPooler 输出, 720x404 → 264 tokens
            input_ids, labels = self._build_token_sequence(
                inst["context"], inst["frame_timestamps"], tpi, inst["gt_text"]
            )
            result = {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values, "mode": "online"}
            if position_ids is not None:
                result["image_position_ids"] = position_ids
            return result

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed = [self._process_single(inst) for inst in instances]
        mode = processed[0]["mode"]

        # Pad input_ids / labels
        all_ids = [p["input_ids"] for p in processed]
        all_labels = [p["labels"] for p in processed]
        max_len = max(len(x) for x in all_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(all_ids), max_len), PAD_IDX, dtype=torch.long)
        attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(all_ids, all_labels)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        result = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if mode == "cached":
            # feature_inputs: [total_tokens, D]
            feats = [p["feature_inputs"] for p in processed]
            result["feature_inputs"] = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)
        else:
            # pixel_values: [total_frames, patches, patch_dim]
            result["pixel_values"] = torch.cat([p["pixel_values"] for p in processed], dim=0)
            if "image_position_ids" in processed[0]:
                result["image_position_ids"] = torch.cat([p["image_position_ids"] for p in processed], dim=0)

        return result
