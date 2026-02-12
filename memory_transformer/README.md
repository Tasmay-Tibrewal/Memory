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
├── token_routing_kernel.py # Token-level sparse kernel loader (v1/v2/v3)
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

| Class            | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `MemoryConfig`   | Memory bank settings, routing, LoRA, quantization       |
| `ModelConfig`    | Transformer architecture settings                       |
| `TrainingConfig` | Training hyperparameters, dataset, distributed settings |
| `Config`         | Main config combining all sub-configs                   |

**Notable Fields**:

- `model.tokenizer_name`: Ensures tokenizer/vocab alignment for from-scratch training.
- `model.num_kv_heads`: Enables GQA in from-scratch self-attention/memory attention (`null` => no grouping).
- `memory.{memory_num_heads,memory_num_kv_heads}`: Optional overrides for memory cross-attention heads (`null` => reuse model/base heads).
- `model.hidden_activation`: Selects MLP activation (`swiglu`, `silu`, `relu`, `gelu`, `sigmoid`, `tanh`).
- `model.initializer_range`: Std used for from-scratch `nn.Linear`/`nn.Embedding` initialization.
- `model.self_attn_wo_init_std`: Optional std override for self-attn output projection `W_o` init (`null` => `initializer_range`).
- `model.mlp_down_proj_init_std`: Optional std override for `MLP.down_proj` init (`null` => `initializer_range`).
- `model.tie_embeddings`: Tie or untie token embeddings and LM head.
- `model.{bos,eos,pad}_token_id`: Optional tokenizer special-ID overrides.
- `memory.num_shared_chapters`: Always include first N chapters in chaptered memory routing.
- `memory.routed_scaling_factor`: Scale routed chapter weights relative to shared chapter weights.
- `memory.normalize_shared_routed_before_mixing`: Normalize shared/routed memory vectors separately before weighted mixing.
- `memory.shared_routed_norm_type`: Branch-vector normalization type (`rms` or `layernorm`).
- `memory.shared_routed_norm_eps`: Epsilon for shared/routed vector normalization.
- `training.save_total_limit`: Checkpoint retention limit (`null` => disable cleanup / keep all).
- `training.scheduler`: Supports `cosine`, `linear`, `constant`, and `wsd`.
- `memory.routing_window_size`: Window size for rolling/hybrid routing during generation.
- `memory.routing_strategy_{train,inference}`: Supports `sequence`, `sequence-rolling`, and `token` (plus `rolling`/`hybrid` inference modes).
- `memory.token_routing_kernel_version`: Sparse token-routing kernel version (`v1`, `v2`, `v3`; default `v2`).

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

| Class                  | Description                      | Parameters          |
| ---------------------- | -------------------------------- | ------------------- |
| `MemoryBank`           | Abstract base class              | -                   |
| `StandardMemoryBank`   | Full N×d memory                  | N_m × d             |
| `FactorizedMemoryBank` | M = A × B^T decomposition        | (N_m × r) + (d × r) |
| `ReducedDimMemoryBank` | Store in reduced dimension       | N_m × r             |
| `ChapteredMemoryBank`  | Wrapper adding chapter structure | Wraps any bank      |

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

| Class                             | Description                                |
| --------------------------------- | ------------------------------------------ |
| `MemoryCrossAttention`            | Standard memory cross-attention            |
| `MemoryCrossAttentionWithRouting` | _(dead code)_ - routing handled externally |

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

**Token-Level Routing Path (Implemented)**:

- Triggered when model/adapter passes `token_routing_state` (used by `routing_strategy_*: token`).
- Shared chapters: dense cross-attention branch (FlashAttention if available, else PyTorch attention). Always weighted at **1.0** (no router gating).
- Routed chapters: sparse top-k chapter branch via `kernels-final` (`v1/v2/v3`) with `FSA_topk_sparse_attention_bthd`.
- **Weighted combination (MoE-style)**: each chapter gets its own independent softmax; outputs weighted by router probabilities and summed. This ensures the router directly controls chapter contribution (joint softmax would let Q·K similarity override weighting).
- **CUDA parallelism**: per-chapter kernel calls launch on separate CUDA streams; event-based sync (`torch.cuda.Event` + `wait_event`) before weighted accumulation.
- Combined output: `shared_output + routed_scaling_factor * routed_output`, then projected by `W_o`.
- Fallback: emulated PyTorch sparse path when kernels unavailable.
- **Benchmark**: `kernels-final/benchmark_kernels_final.py` verifies correctness (forward + dQ/dK/dV/dW backward) and times both modes.

---

### `memory_block.py` - Transformer Blocks

**Purpose**: Complete transformer blocks with optional memory integration.

**Classes**:

| Class                       | Description                                |
| --------------------------- | ------------------------------------------ |
| `RMSNorm`                   | Root Mean Square Layer Normalization       |
| `RotaryPositionalEmbedding` | RoPE implementation                        |
| `SelfAttention`             | Standard self-attention                    |
| `MLP`                       | SwiGLU feed-forward network                |
| `MemoryTransformerBlock`    | Block with optional memory cross-attention |
| `VanillaTransformerBlock`   | Standard block without memory              |

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
    memory_dropout=0.0,
    wo_init_zero=True,
)
output = block(hidden_states, memory=memory_tokens)
```

---

### `router.py` - Chapter Routing

**Purpose**: MoE-inspired routing for selecting relevant memory chapters.

**Classes**:

| Class              | Description                    |
| ------------------ | ------------------------------ |
| `ChapterRouter`    | Sequence-level chapter routing |
| `TokenLevelRouter` | Per-token routing helper       |
| `RollingRouter`    | Rolling window routing         |

**Weight Normalization**: All routers apply `softmax → top-k → renormalize`. The renormalization uses `.clamp_min(1e-12)` to guard against division-by-zero in fp16 edge cases.

**Router Losses** (from MoE literature):

| Loss         | Purpose                         | Coefficient |
| ------------ | ------------------------------- | ----------- |
| Load Balance | Encourage uniform chapter usage | 0.01        |
| Auxiliary    | Penalize over/under-utilization | 0.01        |
| Z-Loss       | Regularize router logits        | 0.001       |

`ChapterRouter` also emits `entropy` as a monitoring metric (not added to training loss unless explicitly wired).

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

| Class        | Description                     |
| ------------ | ------------------------------- |
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

**Gradient Checkpointing (Adapter Mode)**:

- `MemoryAdapter` injects memory using **persistent** forward hooks (registered lazily on
  first `forward()`, never removed). This is compatible with `GradientCheckpointingLayer`
  backward recomputation (`qwen2`, `qwen3`, `llama`, `mistral`, etc.).
- Hooks preserve the original output type (tuple vs tensor). HF layers return 1-tuples;
  returning a bare tensor would break `layer_outputs[0]` in the model loop.
- Per-forward state is stashed on `self._fwd_*`; side effects (router losses, cache mutation)
  are suppressed during recompute via `self._fwd_processed_layers`.
- **Limitation 1**: assumes one forward → one backward per micro-step (standard training).
  Multiple forwards before a single backward is not supported with adapter GC.
- **Limitation 2**: the out-of-band guard (`_fwd_all_router_losses is None`) is best-effort.
  After `adapter.forward()`, `_fwd_*` state stays alive for backward GC recompute, so direct
  `self.base_model(...)` calls will still trigger memory hooks. Always use `adapter.forward()`.
- Full rationale: `docs/design.md` ("Adapter Hooks + Gradient Checkpointing").

**How It Works**:

1. Loads pretrained model from HuggingFace
2. Freezes base model parameters (optional)
3. Creates memory banks and adapters
4. Registers persistent forward hooks on decoder layers (lazily on first forward)
5. Hooks inject memory cross-attention after each layer, surviving GC recompute

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
quantize_memory_bank(tensor, quant_bits=8)         # -> (int8 (num_tokens, dim), scales)
dequantize_memory_bank(quantized, scales, quant_bits=8)  # -> float (num_tokens, dim)

# 4-bit is packed (two signed INT4 values per byte), dim must be even:
quantize_memory_bank(tensor, quant_bits=4)         # -> (int8 (num_tokens, dim//2), scales)
dequantize_memory_bank(quantized, scales, quant_bits=4)  # -> float (num_tokens, dim)
```

**Note**: Quantization during training requires gradient handling. Currently best for inference.

---

### `utils.py` - Utilities

**Functions**:

| Function                            | Description              |
| ----------------------------------- | ------------------------ |
| `set_seed(seed)`                    | Set random seeds         |
| `count_parameters(model)`           | Count trainable params   |
| `format_params(count)`              | Format as "1.2B", "350M" |
| `get_model_size_mb(model)`          | Memory footprint in MB   |
| `print_model_info(model, config)`   | Print summary            |
| `save_checkpoint(model, path, ...)` | Save training state      |
| `load_checkpoint(path)`             | Load training state      |
| `get_cosine_schedule(...)`          | Cosine LR scheduler      |
| `get_linear_schedule(...)`          | Linear LR scheduler      |

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
