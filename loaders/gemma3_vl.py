from typing import Tuple

from transformers import AutoConfig

from . import register_loader
from .base import BaseModelLoader
from models.gemma3_vl import Gemma3VLMRForConditionalGeneration, Gemma3VLMRProcessor


@register_loader("gemma3")
class Gemma3ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True):
        if load_model:
            if self.model_finetune_path is None:
                model = Gemma3VLMRForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                )
            else:
                model = Gemma3VLMRForConditionalGeneration.from_pretrained(
                    self.model_finetune_path,
                    **self.loading_kwargs,
                )
        processor = Gemma3VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config
