from transformers import AutoConfig

from . import register_loader
from .base import BaseModelLoader
from models.qwen3_vl import Qwen3VLMRForConditionalGeneration, Qwen3VLMRProcessor


@register_loader("qwen3-vl")
class Qwen3VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True):
        if load_model:
            if self.model_finetune_path is None:
                model = Qwen3VLMRForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                )
            else:
                model = Qwen3VLMRForConditionalGeneration.from_pretrained(
                    self.model_finetune_path,
                    **self.loading_kwargs,
                )
        processor = Qwen3VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config
