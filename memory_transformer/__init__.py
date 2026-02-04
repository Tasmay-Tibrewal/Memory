"""
Memory-Augmented Transformer

A PyTorch implementation of memory-augmented transformers with 
learnable cross-attention memory banks.
"""

from .config import MemoryConfig, TrainingConfig, ModelConfig, load_config
from .memory_bank import (
    MemoryBank,
    StandardMemoryBank,
    FactorizedMemoryBank,
    ReducedDimMemoryBank,
)
from .memory_attention import MemoryCrossAttention
from .memory_block import MemoryTransformerBlock
from .router import ChapterRouter

__version__ = "0.1.0"

# Lazy imports for classes that require transformers
def __getattr__(name):
    if name == "MemoryTransformer":
        from .model import MemoryTransformer
        return MemoryTransformer
    elif name == "MemoryAdapter":
        from .adapter import MemoryAdapter
        return MemoryAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MemoryConfig",
    "TrainingConfig", 
    "ModelConfig",
    "load_config",
    "MemoryBank",
    "StandardMemoryBank",
    "FactorizedMemoryBank",
    "ReducedDimMemoryBank",
    "MemoryCrossAttention",
    "MemoryTransformerBlock",
    "ChapterRouter",
    "MemoryTransformer",
    "MemoryAdapter",
]
