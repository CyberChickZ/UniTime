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
        # Disable Per-Layer Embeddings (PLE). Gemma 4's PLE requires input_ids
        # for a compact per-layer token lookup. Our UniTime wrapper passes
        # input_ids=None + inputs_embeds (pre-merged features), which makes PLE
        # fall back to broadcasting full embeddings across all 42 layers → 320 GB
        # OOM on 80 GB H100. Setting hidden_size_per_layer_input=0 makes
        # Gemma4Model skip PLE entirely (per_layer_inputs=None). Quality trade-off
        # is small for LoRA finetuning with pre-extracted features.
        if hasattr(model.config, "text_config"):
            model.config.text_config.hidden_size_per_layer_input = 0
        if hasattr(model, "model") and hasattr(model.model, "vocab_size_per_layer_input"):
            model.model.vocab_size_per_layer_input = 0

        processor = Gemma4VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        # Also disable PLE in the returned config so train.py doesn't re-enable
        if hasattr(config, "text_config"):
            config.text_config.hidden_size_per_layer_input = 0
        return model, tokenizer, processor, config
