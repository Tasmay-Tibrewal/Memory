# Kernels Final (Stable Set)

This folder contains the **stable kernel variants** selected from the larger experimentation workspace in `kernels/`.

The intent is simple:

- `kernels/` is for exploration and benchmarking.
- `kernels-final/` is the curated set that was fast and stable enough to keep using.

## What's In Here

The three versions are:

- `kernel_v1.py`: unoptimized baseline (local FSA lineage).
- `kernel_v2.py`: older optimized version (overall most stable in practice).
- `kernel_v3.py`: old optimized version (stable alternate).

Mapping from `kernels/` sources:

- v1 <= `kernels/fsa_topk_sparse_attention_local.py`
- v2 <= `kernels/fsa_topk_sparse_attention_local_optimized_older.py`
- v3 <= `kernels/fsa_topk_sparse_attention_local_optimized_old.py`

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

- `memory.token_routing_kernel_version: v1|v2|v3` (default `v2`).

Current wiring uses:

- v1 -> `kernels-final/kernel_v1.py`
- v2 -> `kernels-final/kernel_v2.py` (default)
- v3 -> `kernels-final/kernel_v3.py`

If the sparse kernel path is unavailable for the current runtime (for example unsupported device/dtype/shape), the model falls back to an emulated sparse PyTorch path for functional correctness.

## What "v1/v2/v3" Means Here

These are not "semantic versions" of the whole repository. They are just a naming scheme for the kernel variants that were kept as practical options:

- v1: baseline (useful for sanity and as a reference point)
- v2: stability-first optimized choice (default)
- v3: older large optimized alternative

## Benchmarking

### Kernels-Final Benchmark (Primary)

`benchmark_kernels_final.py` in this folder is the primary benchmark for the stable kernel set. It tests all three versions (v1, v2, v3) with two operation modes:

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
