from transformers import AutoConfig

from . import register_loader
from .base import BaseModelLoader
from models.gemma4_vl import Gemma4VLMRForConditionalGeneration, Gemma4VLMRProcessor


@register_loader("gemma4")
class Gemma4ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True):
        if load_model:
            if self.model_finetune_path is None:
                model = Gemma4VLMRForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                )
            else:
                model = Gemma4VLMRForConditionalGeneration.from_pretrained(
                    self.model_finetune_path,
                    **self.loading_kwargs,
                )
        processor = Gemma4VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config
