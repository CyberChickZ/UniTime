COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

from .qwen2_vl import Qwen2VLDataCollator
from .qwen_vision_process import process_vision_info
from .gemma3_vl import Gemma3DataCollator

# Gemma4 is only importable when transformers >= 5.0 (UniTime-gemma4 env).
# In the default UniTime env (transformers 4.51.3), the import fails — guard
# it so the older paths keep working.
try:
    from .gemma4_vl import Gemma4DataCollator  # noqa: F401
except ImportError:
    pass

try:
    from .qwen3_vl import Qwen3VLDataCollator  # noqa: F401
except ImportError:
    pass

# from .qwen2_5_vl import Qwen2_5_VLDataCollator