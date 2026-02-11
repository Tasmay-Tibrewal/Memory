# Kernels (Exploration + Documentation)

This folder contains the kernel engineering work done to make **token-level routing into a large memory bank** practical for train/prefill.

Important scope note:

- The repository is primarily about **memory-augmented transformers** (`memory_transformer/`, `docs/`, `idea/`).
- This `kernels/` folder is an **engineering workspace**: experiments, benchmarks, reports, and multiple kernel variants.
- The working repository code is intended to rely on the curated stable kernels in `kernels-final/` (v1/v2/v3).

## Why Token-Level Routing Kernels Were Needed

Token-level routing means each query token can route to different memory chapters.

Why this is hard with naive PyTorch attention:

1. If you try to keep the operation dense/batched, you typically end up duplicating selected KV for each token or group so shapes line up. That creates a VRAM/bandwidth blow-up.
2. If you avoid duplication by de-batching into many small operations, performance collapses due to launch overhead and poor GPU utilization.

There is also a modeling/causality motivation:

- Sequence-level / rolling / hybrid routing can incorporate signals from future tokens into route choices (even if next-token inputs are not explicitly given), which can create a look-ahead risk through chapter selection.
- Token-level routing is the safer and more expressive direction for retrieval.

## High-Level Journey (What Happened)

This folder captures the progression:

1. Started by trying to reuse FlashAttention KV-cache style paths.
   - Not a good fit for train/prefill token-level routing: missing/awkward backward for this setting and too slow end-to-end.
2. Built custom Triton kernels (`memory_cross_attn.py`).
   - Big gains vs naive approaches, but still too slow in early iterations.
3. Took inspiration from NSA/FSA-style kernels and metadata orchestration.
   - Reached practical territory for large memory banks.

Representative performance observations from this work (as reported during the project):

- Target scenario: memory ~65k tokens, `B=32`, `L=8192`, `topk=8`, chapter size `128`, activated memory tokens ~1024, `D=512`, `HQ/HK=8/8 (G=1)`.
- With FSA/NSA-inspired improvements: around ~40 ms forward and around ~40-45 ms backward.
- Larger `G` (e.g. 2/4) often helped due to fewer effective memory accesses.
- NSA forward could be faster for `G>=16` (example: ~11 ms forward at `D=1024, HQ=16, G=16`), but backward was far slower (example: ~357 ms) and it does not support `G<16` as a general solution.

## What This Folder Is (And Is Not)

This folder is:

- A comprehensive archive of the engineering attempts (successful and unsuccessful).
- A place to keep benchmark harnesses and investigative scripts.
- A place to keep handoff notes so the kernel work can be resumed cleanly.

This folder is not:

- The primary repo interface for training/inference (that remains `scripts/` + `memory_transformer/`).
- A "clean" production API surface. For practical usage, prefer `kernels-final/`.

## Folder Structure and Intent

- `kernels/` contains everything (all variants, reports, notebook, benchmark branches).
- `kernels-final/` contains the stable kernels selected for practical usage:
  - v1: unoptimized baseline
  - v2: older optimized (overall most stable)
  - v3: old optimized

## Key Documents

- `kernel-architecture.md`
  - Detailed explanation of the methods explored and how the kernels are structured.
- `TOKEN_LEVEL_MEMORY_ROUTING_HANDOFF.md`
  - Restart/handoff context: how to reproduce, what's stable/unstable, what's left.
- `FSA_LOCAL_OPTIMIZATION_REPORT.md`
  - Optimization checklist / roadmap and what was implemented.
- `TOKEN_LEVEL_ROUTING_NSA_REPORT.md`
  - Plain-language explanation of NSA-style reshaping and why it appears in sparse attention kernels.

## What We Actually Benchmarked

The benchmark scripts in this folder were used to compare:

1. Dense reference baselines (FlashAttention + gather style).
2. Local FSA variants (unoptimized / older / old / newest).
3. "Compatibility" baselines (including local-varlen compatibility paths).
4. NSA selected-attention baseline (only in regimes where it supports the head-group ratio).
5. Chapter-routed MoE-inspired method experiments.

Example shape families that appeared in the runs documented in this repo:

- "G=1" regime:
  - `HQ=8, HK=8, G=1`, `hidden_dim=512`, head dim `D=64`, `dtype=bf16`
  - `B=32, L=8192` (so `TQ=262144`)
  - `TK=65536`, `block_size=128`, `topk=8`
- "G=16" regime:
  - `HQ=16, HK=1, G=16`, `hidden_dim=1024`, head dim `D=64`, `dtype=bf16`
  - NSA often had strong forward in this family, but backward was too slow.

Operational detail seen in runs:

- Cold-start mode often used `MEM_XATTN_FAST_START=1` with a configured `TRITON_CACHE_DIR`.

## Benchmarks

This folder includes two benchmark drivers:

- `benchmark_memory_xattn_optimized_import.py`
  - The main benchmark script used for most comparisons.
- `benchmark_memory_xattn_optimized_import__newly_changed.py`
  - A newer experimental benchmark variant (kept for iteration; not the "main" reference).

There is also an older baseline script:

- `benchmark_memory_xattn.py`
  - Earlier benchmark harness used when iterating on `memory_cross_attn.py` variants.

Notebook artifact:

- `benchmark-memory-kernel.ipynb`
  - Exploratory notebook used during development.

Benchmark baselines commonly compared here:

- Dense FlashAttention + gather baselines
- FSA local variants
- NSA selected-attention baseline when valid (`G>=16`)
- Compatibility/reference variants (some of which were unstable depending on environment)

Stability note:

- If you hit `CUDA illegal memory access`, the CUDA context is typically poisoned. Restart the runtime before trusting subsequent timings or correctness checks.

## Kernel Variants (What Each File Is)

This list is intentionally explicit because the folder contains multiple lineages at once.

## Complete File List

This table enumerates every file currently in `kernels/` and what it is for.

| File | Type | Purpose |
|---|---|---|
| `FSA_LOCAL_OPTIMIZATION_REPORT.md` | doc | Optimization roadmap and status notes for the local FSA lineage |
| `TOKEN_LEVEL_MEMORY_ROUTING_HANDOFF.md` | doc | Session restart/handoff context for token-level routed memory attention |
| `TOKEN_LEVEL_ROUTING_NSA_REPORT.md` | doc | Plain-language explanation of NSA-style reshaping and motivation |
| `kernel-architecture.md` | doc | Detailed kernel-method architecture and engineering narrative |
| `benchmark_memory_xattn.py` | benchmark | Older benchmark harness for early `memory_cross_attn.py` iterations |
| `benchmark_memory_xattn_optimized_import.py` | benchmark | Main benchmark driver used for current comparisons |
| `benchmark_memory_xattn_optimized_import__newly_changed.py` | benchmark | Experimental benchmark variant used during iteration |
| `benchmark-memory-kernel.ipynb` | notebook | Exploratory notebook used during kernel work |
| `flash_sparse_attn_triton_local_nomask.py` | kernel | Reference flash-sparse style Triton implementation without masking |
| `memory_cross_attn.py` | kernel | Custom Triton chapter-routed cross-attention (multiple dK/dV strategies) |
| `memory_cross_attn_fsa_opt.py` | kernel | FSA-inspired backend variant over `memory_cross_attn` |
| `fsa_topk_sparse_attention_chapter_routed.py` | kernel | MoE-inspired chapter-routed method experiment (group by chapter, dense per group, gather back) |
| `fsa_topk_sparse_attention_local.py` | kernel | Local unoptimized baseline (upstream-like) |
| `fsa_topk_sparse_attention_local_optimized_older.py` | kernel | "Older" optimized local variant (maps to v2 in `kernels-final/`) |
| `fsa_topk_sparse_attention_local_optimized_old.py` | kernel | "Old" optimized local variant (maps to v3 in `kernels-final/`) |
| `fsa_topk_sparse_attention_local_optimized.py` | kernel | Newest optimization-heavy local variant (not promoted to `kernels-final/` due to instability in documented runs) |

### Core custom kernels

- `memory_cross_attn.py`
  - Custom Triton implementation for chapter-routed cross-attention with multiple dK/dV strategy families.
- `memory_cross_attn_fsa_opt.py`
  - FSA-inspired backend variant: forward/dQ reuse + specialized dK/dV scheduling.

### FSA local lineage (copied/adapted)

- `fsa_topk_sparse_attention_local.py`
  - Unoptimized local baseline (upstream-like).
- `fsa_topk_sparse_attention_local_optimized_older.py`
  - "Older" optimized variant (became v2 in `kernels-final/`).
- `fsa_topk_sparse_attention_local_optimized_old.py`
  - "Old" optimized variant (became v3 in `kernels-final/`).
- `fsa_topk_sparse_attention_local_optimized.py`
  - Newest optimization-heavy branch with broad env/policy matrix.
  - Observed instability in some real benchmark runs (illegal memory access), so not in `kernels-final/` yet.

### Additional experiments / reference

- `flash_sparse_attn_triton_local_nomask.py`
  - Flash-sparse style local Triton reference without masking.
- `fsa_topk_sparse_attention_chapter_routed.py`
  - MoE-inspired "chapter-routed" method exploration (group by chapter, run dense ops per group, gather back).

## Known Failure Modes (From This Folder's Runs)

These are included so people don't waste time rediscovering them:

1. `CUDA illegal memory access`
   - Once this happens, CUDA context is usually poisoned. Restart the runtime.
2. Correctness mismatches / NaNs in some compatibility baselines
   - In some runs, the local-varlen compatibility path produced forward mismatches with NaNs.
3. "Fast forward" is not enough if backward dominates
   - NSA forward can win in `G>=16`, but the backward penalty dominated end-to-end training step time.

## What Is "Stable" vs "Exploratory"

- Stable kernels used for practical work: `kernels-final/` (v1/v2/v3).
- Exploratory work and detailed engineering notes: this folder.

## How This Fits the Main Project

These kernels exist to enable token-level routing for the memory bank in scenarios where:

- training/prefill requires batching, and
- each token can route differently, and
- naive dense attention would force KV duplication or would require de-batching.

Decoding/inference note:

- During decoding (`seq_len = 1`), token-level routing is much easier to do without huge memory blow-ups.
- The kernels are primarily motivated by train/prefill, but using them at inference time can still help squeeze throughput.

## Next Reading

1. `kernel-architecture.md`
2. `TOKEN_LEVEL_MEMORY_ROUTING_HANDOFF.md`
3. `kernels-final/README.md`
