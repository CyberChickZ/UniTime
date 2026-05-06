"""Gemma4 collator for temporal action segmentation with sliding window.

Online pixel-values path: reads video frames via decord, passes raw pixels
to the model. The model's forward handles vision encoding + spatial pooling.
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
        spatial_pool_grid: Optional[tuple] = None,
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
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]

    def _process_single(self, inst: Dict):
        context = inst["context"]
        gt_text = inst["gt_text"]
        timestamps = inst["frame_timestamps"]
        video_path = inst["video_path"]
        sample_indices = inst["sample_indices"]

        frames = self._read_frames(video_path, sample_indices)

        all_pixel_values = []
        all_position_ids = []
        raw_tokens_per_image = None
        for img in frames:
            inputs = self.processor.image_processor([img], return_tensors="pt")
            pv = inputs["pixel_values"].squeeze(0)
            all_pixel_values.append(pv)
            pos = inputs.get("image_position_ids")
            if pos is not None:
                all_position_ids.append(pos.squeeze(0))
            if raw_tokens_per_image is None:
                raw_tokens_per_image = pv.shape[0]

        if self.spatial_pool_grid is not None:
            tokens_per_image = self.spatial_pool_grid[0] * self.spatial_pool_grid[1]
        else:
            tokens_per_image = raw_tokens_per_image

        bos = self.tokenizer.bos_token or ""
        boi_id = self.config.boi_token_id
        eoi_id = self.config.eoi_token_id
        img_tok = self.image_token_id
        newline_ids = self._tokenize("\n\n")

        if context:
            context_str = ", ".join(context)
            header_text = f"{bos}<|turn>user\nPrevious actions: {context_str}\n"
        else:
            header_text = f"{bos}<|turn>user\n"
        header_ids = self._tokenize(header_text)

        body_ids = []
        for t in timestamps:
            body_ids.extend(self._tokenize(f"{t}s "))
            body_ids.extend(newline_ids)
            body_ids.append(boi_id)
            body_ids.extend([img_tok] * tokens_per_image)
            body_ids.append(eoi_id)
            body_ids.extend(newline_ids)

        instr_ids = self._tokenize(
            "List all action segments with start and end timestamps.\n"
        )
        turn_end_ids = self._tokenize("<turn|>\n<|turn>model\n")
        target_ids = self._tokenize(f"{gt_text}<turn|>\n")

        prompt_ids = header_ids + body_ids + instr_ids + turn_end_ids
        input_ids = prompt_ids + target_ids
        labels = [PAD_IDX] * len(prompt_ids) + list(target_ids)

        return input_ids, labels, all_pixel_values, all_position_ids, tokens_per_image

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_pv = []
        all_pos = []
        tpi = None

        for inst in instances:
            input_ids, labels, pvs, pos_ids, tokens_per_image = self._process_single(inst)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_pv.extend(pvs)
            all_pos.extend(pos_ids)
            if tpi is None:
                tpi = tokens_per_image

        max_len = max(len(x) for x in all_input_ids)
        pad_id = self.PAD_TOKEN_ID

        input_ids_tensor = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        labels_tensor = torch.full((len(all_input_ids), max_len), PAD_IDX, dtype=torch.long)
        attention_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            input_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels_tensor[i, :len(lbls)] = torch.tensor(lbls, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        pixel_values = torch.stack(all_pv, dim=0)

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
