"""Gemma4 Collator — Temporal Action Segmentation 滑窗训练.

在线 pixel-values 路径: 用 decord 读视频帧 → 传原始像素给模型.
模型的 forward 负责 vision encoding + 可选 spatial pooling.

Token 序列结构:
  <bos><|turn>user
  Previous actions: take, open, put
  0.0s <boi>[img×tpi]<eoi> 0.3s <boi>[img×tpi]<eoi> ... 29.7s <boi>[img×tpi]<eoi>
  List all action segments with start and end timestamps.
  <turn|><|turn>model
  take 0.0 1.7, open 2.3 6.0, ...<turn|>

其中 tpi = spatial_pool_grid[0] * spatial_pool_grid[1] (如 12×12=144),
若未设置 spatial pooling 则 tpi = 原始 vision encoder 输出的 token 数.
"""
from typing import Dict, List, Optional, Sequence

import decord
import torch
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

    def _read_frames(self, video_path: str, indices: List[int]) -> List[Image.Image]:
        """用 decord 按帧索引读取视频帧, 返回 PIL Image 列表."""
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]

    def _process_single(self, inst: Dict):
        """处理单条数据: 读帧 → image_processor → 构造 input_ids + labels."""
        context = inst["context"]
        gt_text = inst["gt_text"]
        timestamps = inst["frame_timestamps"]
        video_path = inst["video_path"]
        sample_indices = inst["sample_indices"]

        # 读视频帧
        frames = self._read_frames(video_path, sample_indices)

        # 逐帧过 image_processor, 得到 pixel_values + position_ids
        all_pixel_values = []
        all_position_ids = []
        raw_tokens_per_image = None
        for img in frames:
            inputs = self.processor.image_processor([img], return_tensors="pt")
            pv = inputs["pixel_values"].squeeze(0)   # [n_tokens, patch_dim]
            all_pixel_values.append(pv)
            pos = inputs.get("image_position_ids")
            if pos is not None:
                all_position_ids.append(pos.squeeze(0))  # [n_tokens, 2]
            if raw_tokens_per_image is None:
                raw_tokens_per_image = pv.shape[0]

        # 每帧在 input_ids 中占的 image token 数
        # 如果有 spatial pooling, 用 pool 后的 token 数; 否则用原始数
        if self.spatial_pool_grid is not None:
            tokens_per_image = self.spatial_pool_grid[0] * self.spatial_pool_grid[1]
        else:
            tokens_per_image = raw_tokens_per_image

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

        return input_ids, labels, all_pixel_values, all_position_ids, tokens_per_image

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_pv = []       # 所有帧的 pixel_values, 展平
        all_pos = []      # 所有帧的 position_ids, 展平
        tpi = None

        for inst in instances:
            input_ids, labels, pvs, pos_ids, tokens_per_image = self._process_single(inst)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_pv.extend(pvs)
            all_pos.extend(pos_ids)
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

        # pixel_values: [总帧数, 原始token数, patch_dim]
        # 传给模型, vision encoder + spatial pooling 在 forward 里做
        pixel_values = torch.stack(all_pv, dim=0)

        # 验证: input_ids 中的 image token 数 == 帧数 × pool后每帧token数
        n_image_slots = (input_ids_tensor == self.image_token_id).sum().item()
        expected_slots = len(all_pv) * tpi
        if n_image_slots != expected_slots:
            raise RuntimeError(
                f"image-token slots {n_image_slots} != expected {expected_slots} "
                f"({len(all_pv)} frames x {tpi} tokens/frame)"
            )

        result = dict(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            labels=labels_tensor,
            pixel_values=pixel_values,
        )
        if all_pos:
            result["image_position_ids"] = torch.stack(all_pos, dim=0)
        return result
