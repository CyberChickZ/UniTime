from typing import Dict, List
from collections import OrderedDict

from collators import COLLATORS
from loaders import LOADERS


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "qwen2-vl": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    },
    "qwen2.5-vl": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    },
    "gemma3": {
        # Gemma3ForConditionalGeneration submodule layout:
        #   self.vision_tower            -> SigLIP-2
        #   self.multi_modal_projector   -> Gemma3MultiModalProjector
        #   self.language_model          -> Gemma3ForCausalLM (text LLM)
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "qwen3-vl": {
        "vision_encoder": ["model.visual"],
        "vision_projector": [],
        "llm": ["model.language_model"]
    },
    "gemma4": {
        # Gemma4ForConditionalGeneration → self.model = Gemma4Model, which has
        #   self.model.vision_tower            -> Gemma4VisionModel
        #   self.model.embed_vision            -> Gemma4MultimodalEmbedder (projector)
        #   self.model.language_model          -> Gemma4TextModel (LLM)
        # The lm_head lives at self.lm_head (not inside self.model).
        "vision_encoder": ["model.vision_tower"],
        "vision_projector": ["model.embed_vision"],
        "llm": ["model.language_model"]
    }
}


MODEL_HF_PATH = OrderedDict()

MODEL_FAMILIES = OrderedDict()


def register_model(model_id: str, model_family_id: str, model_hf_path: str) -> None:
    if model_id in MODEL_HF_PATH or model_id in MODEL_FAMILIES:
        raise ValueError(f"Duplicate model_id: {model_id}")
    MODEL_HF_PATH[model_id] = model_hf_path
    MODEL_FAMILIES[model_id] = model_family_id


#=============================================================
# qwen2-vl ---------------------------------------------------
register_model(
    model_id="qwen2-vl-2b-instruct",
    model_family_id="qwen2-vl",
    model_hf_path="Qwen/Qwen2-VL-2B-Instruct"
)

register_model(
    model_id="qwen2-vl-7b-instruct",
    model_family_id="qwen2-vl",
    model_hf_path="Qwen/Qwen2-VL-7B-Instruct"
)

#=============================================================
# gemma3 -----------------------------------------------------
register_model(
    model_id="gemma3-4b-it",
    model_family_id="gemma3",
    model_hf_path="google/gemma-3-4b-it"
)

register_model(
    model_id="gemma3-12b-it",
    model_family_id="gemma3",
    model_hf_path="google/gemma-3-12b-it"
)

#=============================================================
# gemma4 -----------------------------------------------------
# Only register if the gemma4 collator + loader actually loaded (i.e. the
# current env has transformers >= 5.0). In the default UniTime env (tf 4.51.3)
# the gemma4 imports are guarded out and these registrations are skipped, so
# the sanity-check loop below doesn't fire on missing collators.
if "qwen3-vl" in COLLATORS and "qwen3-vl" in LOADERS:
    register_model(
        model_id="qwen3-vl-2b-instruct",
        model_family_id="qwen3-vl",
        model_hf_path="Qwen/Qwen3-VL-2B-Instruct"
    )

if "gemma4" in COLLATORS and "gemma4" in LOADERS:
    register_model(
        model_id="gemma4-e4b-it",
        model_family_id="gemma4",
        model_hf_path="google/gemma-4-E4B-it"
    )

# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MODULE_KEYWORDS, f"Module keywords not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
