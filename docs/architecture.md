# Memory-Augmented Transformer / Mixture of Chapters Architecture

This document explains the architecture of the Memory-Augmented Transformer in detail. The configuration that matches the ICLR 2026 NFAM workshop paper ([`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf)) lives in [`configs/base_small_run2.yaml`](../configs/base_small_run2.yaml).

For a single-image overview see [`idea/MoC Arch Diagram Excalidraw.png`](../idea/MoC%20Arch%20Diagram%20Excalidraw.png), reproduced in the root [`README.md`](../README.md).

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
K = M @ W_k     # (N_m, d) → (N_m, d_kv)
V = M @ W_v     # (N_m, d) → (N_m, d_kv)

Attention = softmax(Q @ K^T / √d_h) @ V
Output = Attention @ W_o
```

The memory cross-attention supports **independent head counts** for query and KV (`memory_num_heads` / `memory_num_kv_heads`). Setting `memory_num_kv_heads < memory_num_heads` enables grouped-query attention on the memory side. The workshop paper config uses 12 query heads and 12 KV heads (no GQA on memory) on top of a backbone that uses 12-query / 4-KV GQA on self-attention.

**Key Design: Zero-initialized W_o**

- Output projection starts at zero
- Model starts as if no memory exists
- Gradual learning of when/how to use memory
- Critical for stable training (adapter and from-scratch)

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

For large memory banks (the workshop paper uses 262,208 tokens), attending to all tokens is expensive. We use **MoE-inspired routing**:

```
Memory organized into C chapters, each with N_c = N_m/C tokens

Router Input: Mean-pooled hidden states (or rolling window / per-token, see "Token-Level vs Sequence-Level Routing")
Router Output: Top-k chapter selections per input

Attention only on selected chapters → O(L × k × N_c) instead of O(L × N_m)
```

**Workshop paper instance**: `N_m = 262,208` memory tokens partitioned into `C = 4,097` chapters of `N_c = 64` tokens each, with `top-k = 64` routed chapters and `1` always-on shared chapter — so memory attention touches only `(64 + 1) × 64 = 4,160` tokens (~1.6 % of the bank) per sequence, while the bank itself sits in VRAM at full capacity.

### Router Architecture

```python
pooled = hidden_states.mean(dim=1)           # (batch, d)
logits = W_router @ pooled + b               # (batch, num_chapters)
probs = softmax(logits)
top_k_indices, top_k_weights = topk(probs, k)
```

### Shared Chapters and Routed Scaling

The model can reserve the first `num_shared_chapters` chapters as **always-on shared knowledge** (e.g., a 1-chapter prefix the router never chooses against). After top-k selection of routed chapters, the final chapter set is:

```
chapter_indices = [shared_0, ..., shared_{S-1},  routed_top_1, ..., routed_top_k]
chapter_weights = [    1, ..., 1,        scale × p_1, ..., scale × p_k]   (then normalised to sum to 1)
```

where `S = num_shared_chapters`, `scale = routed_scaling_factor`, and `p_i` are the renormalised top-k router probabilities. The workshop paper uses `S = 1` shared chapter and `scale = 2.5×` so routed chapters dominate the mix. Optionally, `normalize_shared_routed_before_mixing: true` normalises the shared and routed branches separately (RMSNorm or LayerNorm on the token vectors) before weighted combination — this prevents one branch from dominating purely by raw magnitude.

### Router Losses

1. **Load Balance Loss**: Encourages uniform chapter usage

   ```
   L_balance = C × Σ(f_i × P_i)
   ```

   where f_i = fraction routed to chapter i, P_i = avg probability for chapter i

2. **Auxiliary Loss**: Penalizes variance in chapter usage (squared probabilities vs uniform target).

3. **Z-Loss**: Regularizes router logits to prevent divergence
   ```
   L_z = mean(log²(Σ exp(logits)))
   ```

4. **Entropy** (monitoring only, not added to training loss): tracks router sharpness / collapse.

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

| Aspect       | Memory Adapter             | LoRA                             |
| ------------ | -------------------------- | -------------------------------- |
| What it adds | New cross-attention        | Modifies existing attention      |
| Parameters   | M + projections            | A, B matrices per layer          |
| Information  | Explicit memory tokens     | Implicit in weight modifications |
| Mechanism    | Attend to stored knowledge | Adapt computation                |
| Combination  | ✓ Can use together         | ✓ Can use together               |

## Initialization

- **Token embeddings**: Normal(0, 0.02)
- **Memory bank**: Normal(0, 0.02)
- **All projections**: Normal(0, 0.02)
- **W_o (output)**: **Zero** (critical for stable training — adapter and from-scratch)
- **LoRA B**: Zero (standard)

## Token-Level vs Sequence-Level Routing

**Sequence-level** (default):

- Mean-pool sequence → route → same chapters for all tokens
- Memory tokens are **pre-multiplied by router weights** before attention (soft chapter mixing)
- Memory efficient: K is (batch, selected_tokens, d)

**Token-level** (activated by `routing_strategy_*: token`):

- Each token routes independently via `TokenLevelRouter` or `ChapterRouter.route_token_level()`
- Router outputs `(B, T, top_k)` indices and **normalized** weights (softmax → top-k → renormalize with `clamp_min(1e-12)` safety)

### Weighted Combination (MoE-Style)

When router weights are provided, each chapter is attended **independently** with its own softmax, then outputs are combined with router weights:

```
For each chapter i in top-k:
    output_i = softmax(Q @ K_i^T / √d) @ V_i    ← independent softmax

final = Σ_i  w_i · output_i                      ← weighted combination
```

This is critical: **joint softmax** (all chapters in one attention call) would let Q·K similarity override the router's weighting. Independent softmax ensures router weights **directly control** each chapter's contribution.

### Branch Architecture

1. **Shared chapters** (prefix chapters): dense cross-attention, always weighted at **1.0** (no router gating)
2. **Routed chapters** (per-token top-k): sparse attention via `kernels-final` (`v1/v2/v3` for unweighted, `v4` for exact MoE-weighted fused, `v5` for joint-bias approximation). Each chapter gets independent softmax, weighted by router probabilities (in the unweighted path, weights are applied externally on per-chapter outputs; in the weighted-fused `v4` path, weights enter the kernel directly)
3. **Output merge**: `shared_output + routed_scaling_factor × routed_output`, then projected by `W_o`

### CUDA Stream Parallelism

On GPU with `top_k > 1`, per-chapter attention calls are launched on separate CUDA streams for pipeline overlap. Event-based synchronization (`torch.cuda.Event` + `wait_event`) is used to safely synchronize before weighted accumulation.

Fallback: sequential computation on CPU or when `top_k = 1`.

### Emulated Fallback

If the sparse kernel is unavailable (unsupported device/dtype/shape), the model falls back to an emulated PyTorch path that correctly implements the same independent-softmax-per-chapter logic.

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

| Component                  | File                                       | Key Classes/Functions                                                                                               |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Memory Bank                | `memory_bank.py`                           | `StandardMemoryBank`, `FactorizedMemoryBank`, `ReducedDimMemoryBank`, `ChapteredMemoryBank`                         |
| Cross-Attention            | `memory_attention.py`                      | `MemoryCrossAttention`, `MemoryCrossAttentionWithRouting` (dead code - routing handled externally)                  |
| Transformer Block          | `memory_block.py`                          | `MemoryTransformerBlock`, `VanillaTransformerBlock`, `RMSNorm`, `RotaryPositionalEmbedding`, `SelfAttention`, `MLP` |
| Chapter Router             | `router.py`                                | `ChapterRouter`, `TokenLevelRouter`, `RollingRouter`                                                                |
| Token Sparse Kernel Loader | `token_routing_kernel.py`                  | `normalize_kernel_version`, `get_token_routing_kernel_fn`                                                           |
| Full Model                 | `model.py`                                 | `MemoryTransformer`                                                                                                 |
| Pretrained Adapter         | `adapter.py`                               | `MemoryAdapter`, `MemoryAdapterLayer`                                                                               |
| LoRA                       | `lora.py`                                  | `LoRALinear`, `apply_lora_to_model`                                                                                 |
| Quantization               | `quantization.py`                          | `QuantizedMemoryBank`                                                                                               |
| Configuration              | `config.py`                                | `MemoryConfig`, `ModelConfig`, `TrainingConfig`, `Config`                                                           |
| Sparse Routing Kernels     | `kernels-final/kernel_v{1..5}.py`          | v1 reference / v2 default / v3 alternate / v4 exact MoE-weighted fused / v5 joint-bias approximation                |
| Kernel Benchmark           | `kernels-final/benchmark_kernels_final.py` | Correctness + timing for v1/v2/v3 (unweighted joint-softmax) and weighted MoE-style (per-chapter independent softmax + dW gradient checks for v4) |

### Key Implementation Details

#### Memory Initialization

```python
# memory_bank.py line 67
nn.init.normal_(self.memory, mean=0.0, std=self.init_std)  # Default 0.02
```

#### Targeted Init Overrides (From-Scratch Path)

```python
# model.py (optional overrides)
model.self_attn_wo_init_std      # null => initializer_range
model.mlp_down_proj_init_std     # null => initializer_range
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

#### Persistent Hook-Based Adapter Injection

```python
# adapter.py — hooks registered lazily on first forward(), never removed.
# This ensures hooks survive GradientCheckpointingLayer backward recompute.
def _create_memory_hook(self, layer_idx):
    def hook(module, input, output):
        if self._fwd_all_router_losses is None:
            return  # out-of-band call — pass through (best-effort; see note below)
        if layer_idx not in self.memory_layer_indices:
            if layer_idx not in self._fwd_processed_layers:
                self._fwd_processed_layers.add(layer_idx)
            return  # non-memory layer — don't modify output
        # Preserve original output type: HF layers return 1-tuples like
        # (hidden_states,); returning a bare tensor would break layer_outputs[0].
        output_was_tuple = isinstance(output, tuple)
        hidden_states = output[0] if output_was_tuple else output
        rest = output[1:] if output_was_tuple else ()
        is_recompute = layer_idx in self._fwd_processed_layers
        memory = self.get_memory_for_layer(layer_idx)
        # Router + chapter selection (side effects suppressed on recompute)
        # ...
        if memory is not None:
            hidden_states = self.memory_adapters[str(layer_idx)](hidden_states, memory)
        if not is_recompute:
            self._fwd_processed_layers.add(layer_idx)
        return (hidden_states,) + rest if output_was_tuple else hidden_states
    return hook
```

**Note on out-of-band guard**: The `_fwd_all_router_losses is None` check is best-effort.
After an `adapter.forward()` call, `_fwd_all_router_losses` is intentionally **not** cleared
(backward GC recompute still needs it). Direct `self.base_model(...)` calls after a forward
will therefore still trigger memory injection. Always use `adapter.forward()` as the API
surface; do not call `self.base_model(...)` directly.

---

## Parameter Counts

### Memory Bank Parameters

| Size              | Standard | Factorized (r=256) | ReducedDim (r=256) |
| ----------------- | -------- | ------------------ | ------------------ |
| 1K tokens, d=768  | 786K     | 262K (3× less)     | 262K               |
| 4K tokens, d=768  | 3.1M     | 264K (12× less)    | 1M                 |
| 16K tokens, d=768 | 12.3M    | 270K (45× less)    | 4.1M               |
| 4K tokens, d=4096 | 16.7M    | 1.1M (15× less)    | 1M                 |

### Total Trainable Parameters (Adapter Mode)

```
Memory adapter on Qwen2.5-1.5B (configs/adapter_qwen2.5_1.5b.yaml):
- Memory bank (factorized rank 256): (N_m + d) × r = (2048 + 1536) × 256 ≈ 0.92M
- Memory projections (low-rank, projection_rank 128, 4 projections × 2 matrices each):
    10 layers × 8 × 1536 × 128                                              ≈ 15.7M
- Chapter routers: 10 layers × (1536 × 8 + 8)                               ≈ 0.12M
- Total memory params:                                                       ≈ 16.7M

vs Full model: 1.5B → ~1.1 % of parameters trainable
```

### Workshop Paper Configuration (From-Scratch)

```
Mixture of Chapters (configs/base_small_run2.yaml):
- Backbone: 16 layers × ~9.2M ≈ 147.87M
  (hidden_dim=768, num_heads=12, num_kv_heads=4, intermediate_dim=2304, vocab=49152)
- Memory cross-attention (4 memory layers × ~5.5M)               ≈  22.04M
- Memory bank: 262,208 × 768 (full rank, shared across 4 layers) ≈ 201.38M
- Total                                                          ≈ 371.29M

Iso-FLOP dense baseline (configs/vanilla_control_run2.yaml):
- 24 layers of the same backbone block                            ≈ 202.94M
```

---

## Computational Complexity

### Attention Costs

| Component              | Without Chapters | With Chapters (k=4, C=16) |
| ---------------------- | ---------------- | ------------------------- |
| Self-Attention         | O(L²d)           | O(L²d)                    |
| Memory Cross-Attention | O(L × N_m × d)   | O(L × k × N_c × d)        |

For L=2048, N_m=16K, C=16, k=4:

- Without chapters: 2048 × 16384 = 33.5M ops per head
- With chapters: 2048 × 4 × 1024 = 8.4M ops per head (4× faster)

For the workshop paper config (L=1024, N_m=262208, C=4097, T=64, k=64+1 shared):

- Without chapters: 1024 × 262208 ≈ 268.4M ops per head
- With chapters: 1024 × 65 × 64 ≈ 4.26M ops per head (~63× faster)

---

## Configuration Quick Reference

### Essential Flags

| Flag                                 | Purpose                     | Typical Values                          |
| ------------------------------------ | --------------------------- | --------------------------------------- |
| `vanilla_mode`                       | Disable memory              | `false` (true for control)              |
| `num_memory_tokens`                  | Memory size                 | 512-16384                               |
| `memory_layer_placement`             | Which layers                | `all`, `custom`                         |
| `memory_block_variant`               | Block structure             | `A` (default)                           |
| `use_chapters`                       | Enable routing              | `true` if N_m > 4K                      |
| `routing_strategy_{train,inference}` | Chapter routing granularity | `sequence`, `sequence-rolling`, `token` |
| `token_routing_kernel_version`       | Sparse token-routing kernel | `v2` (default), `v1`, `v3`, `v4`, `v5`  |
| `wo_init_zero`                       | Zero init W_o               | `true` (adapter and from-scratch)       |

### Memory Compression Flags

| Flag                       | Purpose            | When to Use                   |
| -------------------------- | ------------------ | ----------------------------- |
| `use_low_rank_memory`      | Factorize M        | Large memory, adapter mode    |
| `memory_rank`              | Factorization rank | 128-512                       |
| `low_rank_mode`            | Factorization type | `factorized` or `reduced_dim` |
| `use_low_rank_projections` | Factorize W        | Very large models             |

---

## Related Work References

Closest comparisons (most relevant for direct benchmarking):

- **Memory Layers at Scale** (Berges et al., 2025): trainable key-value memory layers as sparse FFN replacements with strong factual gains. Our work differs in using cross-attention to a learnable latent-token bank (rather than key-value FFN substitutes) and in scaling access via *chapter-level* routing rather than product-key addressing.
- **Product Key Memory (PKM)** (Lample et al., 2019): efficient large key-value memory via product keys. Same scaling motivation; different access pattern.
- **Memformer** (Wu et al., 2020): cross-attention to a memory bank with gated dynamic updates. Most architecturally similar to our base read; our memory is *static after training* and we add chapter routing on top.

Other memory mechanisms (more distant):

- **Transformer-XL** (Dai et al., 2019): segment-level recurrence over hidden states.
- **Memorizing Transformers** (Wu et al., 2022): non-parametric kNN over cached KV.
- **RETRO** (Borgeaud et al., 2021), **RAG** (Lewis et al., 2020): retrieval over external text corpora.
- **Titans** (Behrouz et al., 2025), **MemoryLLM / M+** (Wang et al., 2024 / 2025): test-time updatable memory.
- **Routing Transformer** (Roy et al., 2021), **Switch / MoE** (Shazeer et al., 2017; Fedus et al., 2022): sparse routing over computation, the conceptual template we adapt for memory selection.

Key distinction: our memory is **learned end-to-end**, **internal to the model**, **static after training**, and **scaled via attention-based chapter routing**, not retrieved from external sources or updated at test time. See the workshop paper's "Related Work" and Appendix A.1 (Table 3) for the full literature map.
