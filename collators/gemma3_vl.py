"""
Data collator for Gemma3-VL UniTime training.

Phase-1 STUB: registers the "gemma3" family with COLLATORS so that
supported_models.py's sanity check passes and the loader/wrapper imports work.
The actual collator logic (mr_seg multi-window target construction + multi-qa
attention mask) is identical in spirit to collators/qwen2_vl.py:Qwen2VLDataCollator
and will be filled in in phase 2 once the Gemma3 vision-process module
(collators/gemma_vision_process.py) and feature extraction script land.

For now this collator will raise NotImplementedError if actually invoked, so
training cannot accidentally start with a half-baked pipeline.
"""
from typing import Dict, Optional, Sequence

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

from . import register_collator
from .base import BaseDataCollator


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
        self.default_instances = None

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Gemma3DataCollator is a phase-1 stub. The mr_seg multi-window target "
            "construction + multi-qa attention mask logic still needs to be ported "
            "from collators/qwen2_vl.py:Qwen2VLDataCollator. See gemma3 port plan in "
            "docs/research_journal.md."
        )
