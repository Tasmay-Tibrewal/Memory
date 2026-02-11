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

## How This Relates to Benchmarks

Benchmarks live in `kernels/`:

- Main benchmark: `kernels/benchmark_memory_xattn_optimized_import.py`
- Experimental benchmark branch: `kernels/benchmark_memory_xattn_optimized_import__newly_changed.py`

The convention used in this project:

- Keep the older working benchmark file as "main".
- Keep the "newly changed" variant for experiments.

## Safety Note

If you see `CUDA illegal memory access`:

- Restart the runtime before continuing.
- Do not trust any subsequent correctness/timing results in the same process.
