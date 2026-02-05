# Memory-Augmented Transformer Architecture

This document explains the architecture of the Memory-Augmented Transformer in detail.

## Overview

The Memory-Augmented Transformer extends standard transformers with a **learnable memory bank** accessed via **cross-attention**. Unlike typical memory solutions (RAG, KV-cache), this is an **architectural** addition where memory is learned end-to-end during training.

```
┌─────────────────────────────────────────────────────┐
│                  Input Tokens                       │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│                 Token Embedding                      │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│           Transformer Block (×N layers)             │
│  ┌─────────────────────────────────────────────┐   │
│  │           Self-Attention + RoPE              │   │
│  └──────────────────────┬──────────────────────┘   │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │     Memory Cross-Attention (optional)        │◄──┼── Memory Bank
│  └──────────────────────┬──────────────────────┘   │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │              MLP (SwiGLU)                    │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│                   LM Head                           │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Memory Bank

The memory bank is a set of **learnable latent tokens** that act as persistent knowledge storage.

```
Memory Bank M ∈ ℝ^(N_m × d)
- N_m: Number of memory tokens (e.g., 1024-100,000)
- d: Hidden dimension (matches model dimension)
```

**Types:**
- **StandardMemoryBank**: Full parameters (`N_m × d`)
- **FactorizedMemoryBank**: `M = A × B^T` where A: `(N_m, r)`, B: `(d, r)` - saves `(N_m + d)r` vs `N_m × d`
- **ReducedDimMemoryBank**: Store M as `(N_m, r)`, do attention in reduced space

### 2. Memory Cross-Attention

Standard cross-attention where:
- **Queries (Q)**: Come from input hidden states H
- **Keys (K)** and **Values (V)**: Come from memory bank M

```python
Q = H @ W_q     # (batch, seq_len, d) → (batch, seq_len, d)
K = M @ W_k     # (N_m, d) → (N_m, d)  
V = M @ W_v     # (N_m, d) → (N_m, d)

Attention = softmax(Q @ K^T / √d_k) @ V
Output = Attention @ W_o
```

**Key Design: Zero-initialized W_o**
- Output projection starts at zero
- Model starts as if no memory exists
- Gradual learning of when/how to use memory
- Critical for stable adapter training

### 3. Block Variants

**Variant A** (Default):
```
Self-Attention → Memory Cross-Attention → MLP
```

**Variant B** (Extra MLP):
```
Self-Attention → MLP → Memory Cross-Attention → MLP
```

Variant B provides additional nonlinear processing after memory retrieval.

### 4. Memory Layer Placement

Memory layers can be placed:
- `all`: Every layer has memory
- `first_k`: First k layers only
- `last_k`: Last k layers only  
- `every_n`: Every n-th layer
- `custom`: Explicit list of layer indices

### 5. Memory Sharing

- `shared`: One memory bank for all layers (more capacity per access)
- `per_layer`: Each layer has own bank (layer-specific information)
- `every_k_layers`: Groups of k layers share a bank

**Note**: With shared memory, different layers operate in different vector spaces. The per-layer K/V projections learn both the key/value mapping AND the manifold transformation.

## Chapter-Based Routing

For large memory banks (100k+ tokens), attending to all tokens is expensive. We use **MoE-inspired routing**:

```
Memory organized into C chapters, each with N_c = N_m/C tokens

Router Input: Mean-pooled hidden states
Router Output: Top-k chapter selections

Attention only on selected chapters → O(L × k × N_c) instead of O(L × N_m)
```

### Router Architecture

```python
pooled = hidden_states.mean(dim=1)           # (batch, d)
logits = W_router @ pooled + b               # (batch, num_chapters)
probs = softmax(logits)
top_k_indices, top_k_weights = topk(probs, k)
```

### Router Losses

1. **Load Balance Loss**: Encourages uniform chapter usage
   ```
   L_balance = C × Σ(f_i × P_i)
   ```
   where f_i = fraction routed to chapter i, P_i = avg probability for chapter i

2. **Auxiliary Loss**: Penalizes variance in chapter usage

3. **Z-Loss**: Regularizes router logits to prevent divergence
   ```
   L_z = mean(log²(Σ exp(logits)))
   ```

## Memory Adapter Mode

For pretrained models, we inject memory as adapters:

```
┌─────────────────────────────────────────┐
│        Pretrained Transformer            │
│  ┌────────────────────────────────────┐ │
│  │      Original Layer (frozen)        │ │
│  └──────────────────┬─────────────────┘ │
│                     ▼                    │
│  ┌────────────────────────────────────┐ │
│  │   Memory Adapter (trainable)        │ │← New memory bank + projections
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Trainable parameters:**
- Memory bank M
- Memory projections (W_q, W_k, W_v, W_o)
- Optional: LoRA on attention (W_q, W_v)
- Chapter routers (if using)

## Low-Rank Options

### Memory Bank Factorization
```
M = A @ B^T
A: (N_m, r)  -- tokens
B: (d, r)    -- basis

Parameters: (N_m + d) × r  vs  N_m × d
Compression: ~8× for r=512, d=4096
```

### Reduced Dimension Mode
Entire attention happens in reduced space:
```
W_q: d → r    (project queries down)
W_k: r → r    (operate in reduced space)
W_v: r → r
W_o: r → d    (project back up)
```

### Projection Factorization
```
W = W_down @ W_up
W_down: d → r
W_up: r → d
```

## Comparison: Memory vs LoRA

| Aspect | Memory Adapter | LoRA |
|--------|---------------|------|
| What it adds | New cross-attention | Modifies existing attention |
| Parameters | M + projections | A, B matrices per layer |
| Information | Explicit memory tokens | Implicit in weight modifications |
| Mechanism | Attend to stored knowledge | Adapt computation |
| Combination | ✓ Can use together | ✓ Can use together |

## Initialization

- **Token embeddings**: Normal(0, 0.02)
- **Memory bank**: Normal(0, 0.02)  
- **All projections**: Normal(0, 0.02)
- **W_o (output)**: **Zero** (critical for stable adapter training)
- **LoRA B**: Zero (standard)

## Token-Level vs Sequence-Level Routing

**Sequence-level** (implemented):
- Mean-pool sequence → route → same chapters for all tokens
- Memory efficient: K is (batch, selected_tokens, d)

**Token-level** (future work):
- Each token routes independently
- Memory prohibitive: K would be (batch × seq_len, selected_tokens, d)
- Feasible during generation (seq_len = 1)

## Summary

The Memory-Augmented Transformer provides:
1. **Explicit memory** via learnable latent tokens
2. **Flexible placement** of memory across layers
3. **Scalable access** via chapter routing
4. **Parameter efficiency** via low-rank variants
5. **Easy adaptation** of pretrained models

---

## Implementation Mapping

### Where Each Component Lives

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Memory Bank | `memory_bank.py` | `StandardMemoryBank`, `FactorizedMemoryBank`, `ReducedDimMemoryBank`, `ChapteredMemoryBank` |
| Cross-Attention | `memory_attention.py` | `MemoryCrossAttention`, `MemoryCrossAttentionWithRouting` |
| Transformer Block | `memory_block.py` | `MemoryTransformerBlock`, `VanillaTransformerBlock`, `RMSNorm`, `RotaryPositionalEmbedding`, `SelfAttention`, `MLP` |
| Chapter Router | `router.py` | `ChapterRouter`, `TokenLevelRouter`, `RollingRouter` |
| Full Model | `model.py` | `MemoryTransformer` |
| Pretrained Adapter | `adapter.py` | `MemoryAdapter`, `MemoryAdapterLayer` |
| LoRA | `lora.py` | `LoRALinear`, `apply_lora_to_model` |
| Quantization | `quantization.py` | `QuantizedMemoryBank` |
| Configuration | `config.py` | `MemoryConfig`, `ModelConfig`, `TrainingConfig`, `Config` |

### Key Implementation Details

#### Memory Initialization
```python
# memory_bank.py line 67
nn.init.normal_(self.memory, mean=0.0, std=self.init_std)  # Default 0.02
```

#### Zero Output Projection Initialization
```python
# memory_attention.py line 112
if wo_init_zero:
    nn.init.zeros_(self.o_proj.weight)
```

#### Block Variant Selection
```python
# memory_block.py lines 370-400
if self.memory_block_variant == "A":
    # Self-Attn → Memory → MLP
    hidden = self.self_attn(hidden)
    hidden = self.memory_attn(hidden, memory)
    hidden = self.mlp(hidden)
elif self.memory_block_variant == "B":
    # Self-Attn → MLP1 → Memory → MLP2
    hidden = self.self_attn(hidden)
    hidden = self.mlp1(hidden)
    hidden = self.memory_attn(hidden, memory)
    hidden = self.mlp2(hidden)
```

#### Hook-Based Adapter Injection
```python
# adapter.py lines 310-356
def create_hook(layer_idx):
    def hook(module, input, output):
        # Apply memory cross-attention after layer
        if layer_idx in self.memory_layer_indices:
            memory = self.get_memory_for_layer(layer_idx)
            hidden_states = self.memory_adapters[str(layer_idx)](output, memory)
            return hidden_states
    return hook
```

---

## Parameter Counts

### Memory Bank Parameters
| Size | Standard | Factorized (r=256) | ReducedDim (r=256) |
|------|----------|--------------------|--------------------|
| 1K tokens, d=768 | 786K | 262K (3× less) | 262K |
| 4K tokens, d=768 | 3.1M | 264K (12× less) | 1M |
| 16K tokens, d=768 | 12.3M | 270K (45× less) | 4.1M |
| 4K tokens, d=4096 | 16.7M | 1.1M (15× less) | 1M |

### Total Trainable Parameters (Adapter Mode)
```
Memory adapter on Qwen2.5-1.5B:
- Memory bank: 2048 × 256 (factorized) = 0.5M
- Memory projections: 10 layers × 4 × 256 × 1536 = 15.7M
- Chapter routers: 10 × (1536 × 8) = 0.1M
- Total memory params: ~16.3M

vs Full model: 1.5B → 1% of parameters trainable
```

---

## Computational Complexity

### Attention Costs
| Component | Without Chapters | With Chapters (k=4, C=16) |
|-----------|------------------|---------------------------|
| Self-Attention | O(L²d) | O(L²d) |
| Memory Cross-Attention | O(L × N_m × d) | O(L × k × N_c × d) |

For L=2048, N_m=16K, C=16, k=4:
- Without chapters: 2048 × 16384 = 33.5M ops per head
- With chapters: 2048 × 4 × 1024 = 8.4M ops per head (4× faster)

---

## Configuration Quick Reference

### Essential Flags
| Flag | Purpose | Typical Values |
|------|---------|----------------|
| `vanilla_mode` | Disable memory | `false` (true for control) |
| `num_memory_tokens` | Memory size | 512-16384 |
| `memory_layer_placement` | Which layers | `all`, `custom` |
| `memory_block_variant` | Block structure | `A` (default) |
| `use_chapters` | Enable routing | `true` if N_m > 4K |
| `wo_init_zero` | Zero init W_o | `true` (adapter and from-scratch) |

### Memory Compression Flags
| Flag | Purpose | When to Use |
|------|---------|-------------|
| `use_low_rank_memory` | Factorize M | Large memory, adapter mode |
| `memory_rank` | Factorization rank | 128-512 |
| `low_rank_mode` | Factorization type | `factorized` or `reduced_dim` |
| `use_low_rank_projections` | Factorize W | Very large models |

---

## Related Work References

- **Transformer-XL**: Segment-level recurrence (different approach)
- **Memorizing Transformers**: KV-cache as memory (retrieval-based)
- **RETRO**: Retrieve from external corpus
- **RAG**: Retrieve-augment-generate
- **Our approach**: Learnable internal memory with cross-attention

Key difference: Our memory is **learned end-to-end** and **internal to the model**, not retrieved from external sources.

