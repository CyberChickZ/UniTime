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
        # NOTE: do NOT disable PLE. With the pixel_values path (collator provides
        # raw images, model runs vision tower itself), PLE works correctly:
        # input_ids has image placeholder tokens → PLE gives them neutral per-layer
        # signals → vision tower merges features into inputs_embeds afterwards.
        # PLE only breaks with the inputs_embeds bypass (input_ids=None), which we
        # no longer use for Gemma 4.

        processor = Gemma4VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config
