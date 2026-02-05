# Memory Transformer Core Package

This package contains all core components for the memory-augmented transformer architecture.

## Module Overview

```
memory_transformer/
├── config.py           # Centralized configuration system
├── memory_bank.py      # Memory bank implementations
├── memory_attention.py # Cross-attention for memory access
├── memory_block.py     # Transformer blocks with memory
├── router.py           # Chapter-based routing (MoE-style)
├── lora.py             # Standard LoRA implementation
├── model.py            # Full MemoryTransformer model
├── adapter.py          # Memory adapter for pretrained models
├── quantization.py     # Memory bank quantization
└── utils.py            # Utilities and helpers
```

---

## Detailed Module Documentation

### `config.py` - Configuration System

**Purpose**: Centralized configuration management using Python dataclasses with YAML loading.

**Classes**:

| Class | Description |
|-------|-------------|
| `MemoryConfig` | Memory bank settings, routing, LoRA, quantization |
| `ModelConfig` | Transformer architecture settings |
| `TrainingConfig` | Training hyperparameters, dataset, distributed settings |
| `Config` | Main config combining all sub-configs |

**Notable Fields**:
- `model.tokenizer_name`: Ensures tokenizer/vocab alignment for from-scratch training.
- `memory.routing_window_size`: Window size for rolling/hybrid routing during generation.

**Key Functions**:
```python
load_config(path) -> Config           # Load from YAML
save_config(config, path)             # Save to YAML
get_memory_layer_indices(config)      # Compute which layers get memory
get_memory_bank_assignments(config)   # Compute bank-to-layer mapping
```

**Example Usage**:
```python
from memory_transformer import load_config
config = load_config("configs/adapter_qwen2.5_1.5b.yaml")
print(config.memory.num_memory_tokens)  # 2048
```

---

### `memory_bank.py` - Memory Bank Implementations

**Purpose**: Learnable memory token storage with multiple compression options.

**Classes**:

| Class | Description | Parameters |
|-------|-------------|------------|
| `MemoryBank` | Abstract base class | - |
| `StandardMemoryBank` | Full N×d memory | N_m × d |
| `FactorizedMemoryBank` | M = A × B^T decomposition | (N_m × r) + (d × r) |
| `ReducedDimMemoryBank` | Store in reduced dimension | N_m × r |
| `ChapteredMemoryBank` | Wrapper adding chapter structure | Wraps any bank |

**Factory Function**:
```python
create_memory_bank(
    num_tokens: int,
    dim: int,
    use_low_rank: bool = False,
    rank: int = 64,
    low_rank_mode: str = "factorized",  # or "reduced_dim"
    init_std: float = 0.02,
) -> MemoryBank
```

**Memory Compression Comparison**:
```
Standard:    N_m × d = 4096 × 4096 = 67M params
Factorized:  (N_m + d) × r = 8192 × 512 = 4.2M params (16× less)
ReducedDim:  N_m × r = 4096 × 512 = 2.1M params (32× less)
```

---

### `memory_attention.py` - Memory Cross-Attention

**Purpose**: Cross-attention layer where queries come from hidden states and keys/values come from memory.

**Classes**:

| Class | Description |
|-------|-------------|
| `MemoryCrossAttention` | Standard memory cross-attention |
| `MemoryCrossAttentionWithRouting` | *(dead code)* - routing handled externally |

**Key Features**:
- Multi-head attention with configurable heads
- Low-rank projection options (`use_low_rank_projections`)
- Reduced-dimension mode (`reduced_dim_mode`)
- Flash Attention support
- **Gradient Checkpointing** support (for non-FlashAttention)
- **Zero-initialized W_o** for stable training (adapter and from-scratch)

**Equation**:
```
Q = H @ W_q          # Queries from hidden states
K = M @ W_k          # Keys from memory
V = M @ W_v          # Values from memory
Output = softmax(QK^T / √d) @ V @ W_o
```

---

### `memory_block.py` - Transformer Blocks

**Purpose**: Complete transformer blocks with optional memory integration.

**Classes**:

| Class | Description |
|-------|-------------|
| `RMSNorm` | Root Mean Square Layer Normalization |
| `RotaryPositionalEmbedding` | RoPE implementation |
| `SelfAttention` | Standard self-attention |
| `MLP` | SwiGLU feed-forward network |
| `MemoryTransformerBlock` | Block with optional memory cross-attention |
| `VanillaTransformerBlock` | Standard block without memory |

**Block Variants**:
```
Variant A (default):
    Self-Attn → Memory Cross-Attn → MLP

Variant B:
    Self-Attn → MLP → Memory Cross-Attn → MLP
```

**Usage**:
```python
block = MemoryTransformerBlock(
    hidden_dim=768,
    num_heads=12,
    has_memory=True,
    memory_block_variant="A",
    wo_init_zero=True,
)
output = block(hidden_states, memory=memory_tokens)
```

---

### `router.py` - Chapter Routing

**Purpose**: MoE-inspired routing for selecting relevant memory chapters.

**Classes**:

| Class | Description |
|-------|-------------|
| `ChapterRouter` | Sequence-level chapter routing |
| `TokenLevelRouter` | Per-token routing (generation only) |
| `RollingRouter` | Rolling window routing |

**Router Losses** (from MoE literature):

| Loss | Purpose | Coefficient |
|------|---------|-------------|
| Load Balance | Encourage uniform chapter usage | 0.01 |
| Auxiliary | Penalize over/under-utilization | 0.01 |
| Z-Loss | Regularize router logits | 0.001 |

**Example**:
```python
router = ChapterRouter(
    hidden_dim=768,
    num_chapters=16,
    top_k=4,
)
chapter_indices, weights, losses = router(hidden_states, return_losses=True)
```

---

### `lora.py` - Low-Rank Adaptation

**Purpose**: Standard LoRA implementation for comparison experiments.

**Classes**:

| Class | Description |
|-------|-------------|
| `LoRALinear` | Linear layer with LoRA adapters |

**Key Functions**:
```python
apply_lora_to_model(model, targets, rank, alpha)  # Add LoRA to model
get_lora_parameters(model)                         # Get trainable params
merge_lora_weights(model)                          # Merge for inference
unmerge_lora_weights(model)                        # Unmerge for training
```

**Comparison Modes**:
```yaml
# LoRA only
use_lora: true
use_memory_adapter: false

# Memory only
use_lora: false
use_memory_adapter: true

# Combined
use_both_memory_and_lora: true
```

---

### `model.py` - Full MemoryTransformer

**Purpose**: Complete model for from-scratch training.

**Class**: `MemoryTransformer`

**Features**:
- Token and positional embeddings
- N transformer blocks with optional memory
- Shared or per-layer memory banks
- Chapter routing
- Language modeling head

**Key Methods**:
```python
model = MemoryTransformer(config)

# Forward pass
outputs = model(input_ids, attention_mask, labels)
# Returns: {"logits": ..., "loss": ..., "router_losses": [...]}

# Parameter counting (use utility function)
from memory_transformer.utils import count_parameters  # Bug 20 fix: correct import path
count_parameters(model)  # Returns param count
```

---

### `adapter.py` - Memory Adapter

**Purpose**: Inject memory into pretrained models (Qwen, Llama, Mistral).

**Class**: `MemoryAdapter`

**Supported Architectures**:
- Qwen 2.5 / Qwen 3 series
- Llama 2 / Llama 3 series
- Mistral series

**How It Works**:
1. Loads pretrained model from HuggingFace
2. Freezes base model parameters (optional)
3. Creates memory banks and adapters
4. Uses PyTorch hooks to inject memory after each layer

**Key Methods**:
```python
adapter = MemoryAdapter(config)

# Forward (same interface as model)
outputs = adapter(input_ids, labels=labels)

# Get parameter groups for different LRs
groups = adapter.get_parameter_groups()
# [{"params": memory_params, "lr": 2e-4},
#  {"params": lora_params, "lr": 1e-4}]
```

---

### `quantization.py` - Memory Quantization

**Purpose**: Reduce memory footprint via quantization.

**Classes**:
- `QuantizedMemoryBank`: 8-bit or 4-bit storage

**Functions**:
```python
quantize_memory_bank(tensor, bits=8)   # Returns (quantized, scales)
dequantize_memory_bank(quantized, scales)  # Returns float tensor
```

**Note**: Quantization during training requires gradient handling. Currently best for inference.

---

### `utils.py` - Utilities

**Functions**:

| Function | Description |
|----------|-------------|
| `set_seed(seed)` | Set random seeds |
| `count_parameters(model)` | Count trainable params |
| `format_params(count)` | Format as "1.2B", "350M" |
| `get_model_size_mb(model)` | Memory footprint in MB |
| `print_model_info(model, config)` | Print summary |
| `save_checkpoint(model, path, ...)` | Save training state |
| `load_checkpoint(path)` | Load training state |
| `get_cosine_schedule(...)` | Cosine LR scheduler |
| `get_linear_schedule(...)` | Linear LR scheduler |

---

## Import Examples

```python
# Import config
from memory_transformer import MemoryConfig, load_config

# Import memory components
from memory_transformer import MemoryBank, StandardMemoryBank, FactorizedMemoryBank

# Import attention
from memory_transformer import MemoryCrossAttention

# Import models (lazy loaded)
from memory_transformer import MemoryTransformer, MemoryAdapter

# Import router
from memory_transformer import ChapterRouter
```

---

## Dependencies

- `torch`: Core tensor operations
- `transformers`: For pretrained model loading (adapter mode)
- `omegaconf`: YAML config loading
- `flash_attn` (optional): Flash Attention acceleration
