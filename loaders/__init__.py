LOADERS = {}

def register_loader(name):
    def register_loader_cls(cls):
        if name in LOADERS:
            return LOADERS[name]
        LOADERS[name] = cls
        return cls
    return register_loader_cls

from .qwen2_vl import Qwen2VLModelLoader
from .gemma3_vl import Gemma3ModelLoader

# Gemma4 only importable in UniTime-gemma4 env (transformers >= 5.0). Guard it.
try:
    from .gemma4_vl import Gemma4ModelLoader  # noqa: F401
except ImportError:
    pass

try:
    from .qwen3_vl import Qwen3VLModelLoader  # noqa: F401
except ImportError:
    pass