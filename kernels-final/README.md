# Kernels Final (Stable Set)

This folder contains the **stable kernel variants** selected from the larger experimentation workspace in `kernels/`.

The intent is simple:

- `kernels/` is for exploration and benchmarking.
- `kernels-final/` is the curated set that was fast and stable enough to keep using.

## What's In Here

The five curated kernel variants are:

- `kernel_v1.py`: unoptimized baseline (local FSA lineage).
- `kernel_v2.py`: older optimized version (overall most stable in practice — **default**).
- `kernel_v3.py`: old optimized version (stable alternate).
- `kernel_v4.py`: **exact MoE-weighted fused kernel**. Single launch with forward + dQ/dK/dV/dW backward. Each routed slot keeps its own softmax normalisation domain so routing weights `w_s` directly control chapter contribution: `output = Σ_s w_s · softmax(Q · K_s^T / √d) · V_s`. Backward uses custom Triton kernels for dQ, dK/dV (via inverted-index or chunked partial-buffer schedules), and dW.
- `kernel_v5.py`: **joint-bias weighted approximation**. Adds `log(w_s)` as a per-slot logit bias and runs a single joint softmax across all selected chapters. Not exact MoE semantics, but stays close to raw v1/v2/v3 throughput while still reflecting router preferences.

Mapping from `kernels/` sources (for the unweighted variants):

- v1 ⇐ `kernels/fsa_topk_sparse_attention_local.py`
- v2 ⇐ `kernels/fsa_topk_sparse_attention_local_optimized_older.py`
- v3 ⇐ `kernels/fsa_topk_sparse_attention_local_optimized_old.py`

`v4` and `v5` are new weighted variants developed for this project; they delegate the unweighted code path back to v1's `FSA_topk_sparse_attention_bthd` and add a new `FSA_topk_sparse_attention_weighted_bthd` entry point for the weighted path.

## What Was Not Selected (And Why)

Not included as "final":

1. Newest local optimized kernel (`kernels/fsa_topk_sparse_attention_local_optimized.py`).
   - Observed instability in benchmark/correctness runs (`CUDA illegal memory access`).
   - Also not consistently faster enough to justify instability.
2. Local compatibility/varlen and other reference paths.
   - Some showed correctness mismatches/NaNs depending on environment.

In plain terms:

- The newest branch had the most ambitious optimization surface, but was not stable enough yet.
- The "local varlen compatibility" path was closer to upstream behavior, but it did not match correctness reliably in the documented checks.

## NSA Note (Why It Wasn't the Default)

Native Sparse Attention (NSA) style kernels showed:

- Forward could be faster in `G>=16` (example: ~11 ms forward at `D=1024, HQ=16, G=16`).
- But backward was very slow (example: ~357 ms), and it does not support `G<16` as a general solution.

So even if NSA wins forward in some regimes, it loses end-to-end training step time.

## Recommended Default

Use `kernel_v2.py` as the default starting point:

- Most stable overall in practice.
- Good end-to-end performance relative to alternatives.

Keep v1 and v3 as:

- v1: simplest fallback/reference.
- v3: stable alternate that can win on some shapes.

## Repository Integration (Implemented)

Token-level routing is now integrated in the main model stack (`memory_transformer/`), not just benchmark scripts.

When `memory.routing_strategy_train` or `memory.routing_strategy_inference` is set to `token`, the attention path is:

1. Shared chapters (prefix chapters) -> dense cross-attention path (FlashAttention when available; PyTorch fallback otherwise).
2. Routed chapters (per-token top-k) -> sparse routed path via selected kernel from this folder.
3. Final output -> `shared_output + routed_scaling_factor * routed_output`.

**Note on router weights**: The kernels themselves do **not** handle router weight application. Weights are applied externally in Python (MoE-style: independent kernel call per chapter, weighted output accumulation). This keeps the kernel code unchanged and maintainable.

Kernel selection is controlled by:

- `memory.token_routing_kernel_version: v1|v2|v3|v4|v5` (default `v2`).

Current wiring uses:

- v1 → `kernels-final/kernel_v1.py`
- v2 → `kernels-final/kernel_v2.py` (default)
- v3 → `kernels-final/kernel_v3.py`
- v4 → `kernels-final/kernel_v4.py` (weighted-fused MoE; falls back to v1 for unweighted calls)
- v5 → `kernels-final/kernel_v5.py` (joint-bias weighted approximation; falls back to v1 for unweighted calls)

If the sparse kernel path is unavailable for the current runtime (for example unsupported device/dtype/shape), the model falls back to an emulated sparse PyTorch path for functional correctness.

## What "v1/v2/v3/v4/v5" Means Here

These are not "semantic versions" of the whole repository. They are just a naming scheme for the kernel variants that were kept as practical options:

- **v1**: baseline (useful for sanity and as a reference point) — unweighted only.
- **v2**: stability-first optimized choice — **default**, unweighted only.
- **v3**: older large optimized alternative — unweighted only.
- **v4**: exact MoE-weighted fused kernel with full backward (dQ + dK/dV + dW). Use this when the router weights need to directly control per-chapter contribution and you want gradient flow back to the router from a single fused kernel.
- **v5**: joint-bias single-softmax approximation. Adds `log(weight)` as a logit bias and runs one joint softmax. Faster than v4 but not exact MoE.

Practical guidance:

- For pure unweighted joint-softmax routing (the workshop paper path), v2 is the default.
- For end-to-end MoE-style training where the router weights must drive contribution exactly, v4 is the preferred kernel.
- v5 is a useful middle ground when v4's per-chapter independent-softmax launch overhead matters more than exact MoE semantics.

## Benchmarking

### Kernels-Final Benchmark (Primary)

`benchmark_kernels_final.py` in this folder is the primary benchmark for the stable kernel set. It tests the unweighted variants (v1, v2, v3) and the weighted variants (v4, v5) with two operation modes:

1. **Unweighted** (joint softmax): Single kernel call with all top-k chapters — cross-chapter softmax normalisation.
2. **Weighted** (MoE-style): Per-chapter independent kernel calls on CUDA streams with event-based synchronisation, followed by router-weighted output accumulation. This matches the production path in `memory_attention.py`.

**What it checks:**

| Check                         | Unweighted | Weighted (MoE) |
| ----------------------------- | ---------- | -------------- |
| Forward output                | ✓          | ✓              |
| dQ (query gradient)           | ✓          | ✓              |
| dK (key gradient)             | ✓          | ✓              |
| dV (value gradient)           | ✓          | ✓              |
| dW (chapter_weights gradient) | N/A        | ✓              |

Correctness is verified against naive Python reference implementations.

**Usage examples:**

```bash
# Full suite: correctness + timing for all kernels
python benchmark_kernels_final.py --mode all

# Correctness only, 3 trials
python benchmark_kernels_final.py --mode correctness --num-checks 3

# Timing only, forward-only, specific kernels
python benchmark_kernels_final.py --mode timing --kernels v2,v3 --fwd-only

# Skip weighted mode (unweighted only)
python benchmark_kernels_final.py --no-weighted

# Use preset config matching Qwen-7B shapes
python benchmark_kernels_final.py --preset qwen-7b --mode timing

# Custom shapes
python benchmark_kernels_final.py --mem-tokens 8192 --chapter-size 128 --topk 4 \
    --seq-len 2048 --hidden-dim 3584 --heads 28 --kv-heads 4
```

### Legacy Benchmarks

Broader benchmarks including NSA baselines, FlexAttention, and experimental kernel variants live in `kernels/`:

- `kernels/benchmark_memory_xattn_optimized_import.py` (main legacy benchmark)
- `kernels/benchmark_memory_xattn_optimized_import__newly_changed.py` (experimental)

## Safety Note

If you see `CUDA illegal memory access`:

- Restart the runtime before continuing.
- Do not trust any subsequent correctness/timing results in the same process.
