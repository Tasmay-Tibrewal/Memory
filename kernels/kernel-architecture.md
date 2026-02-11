# Token-Level Routing Kernel Architecture (`kernels/`)

This document explains the kernel engineering explored in `kernels/` to support token-level memory routing for long-sequence train/prefill.

It complements the main repository architecture (`docs/architecture.md`) and focuses only on this folder's kernel work.

---

## 1) Core Problem

Goal: compute chapter-routed memory cross-attention where each query token can select different memory chapters (`topk` blocks) without duplicating KV memory per token.

### Why default batched attention is not enough

For token-level routing, naive dense batching generally leads to one of two bad options:

1. Duplicate selected memory per token (or subgroup) so attention can stay dense.
   - Causes VRAM explosion and bandwidth bottlenecks.
2. Break batching into many small calls.
   - Causes large launch/host overhead and poor utilization.

So the kernel work builds sparse-selected attention directly over routed chapter indices.

---

## 2) Data Model Used Across Implementations

Common tensor model:

- `q`: `[B, TQ, HQ, D]`
- `k, v`: `[B, TK, HK, D]`
- `block_indices`: `[B, TQ, HK, S]`
  - chapter IDs for each token/KV-head route
  - `S = topk`
- `block_size = BS`
- number of chapters `M = TK / BS`
- grouped-query relation `G = HQ / HK`

Semantics:

- Each `(token, kv-head)` attends only selected chapter blocks.
- Global result must match dense attention over the selected tokens.

---

## 3) Method Families Explored

### A) Dense/Gather Baselines

Used as anchors:

1. Dense FlashAttention with gathered sparse KV subsets.
2. FlashAttention KV-cache style sparse emulation.

Purpose: establish reference latency and correctness envelopes.

---

### B) Early Custom Triton Family (`memory_cross_attn.py`)

Architecture:

1. Forward sparse-selected online softmax over routed blocks.
2. Backward dQ + dK/dV with several backend strategies:
   - A: KV-block parallel, scan-style
   - B: query-chunk x KV-block with atomics
   - C: inverted-index strategy
   - D: chunked inverted-index with partial buffers + reduce

Outcome:

- Major speedups over early sparse baselines.
- Still not sufficient at target training regime scale.

---

### C) FSA-Inspired Path (`memory_cross_attn_fsa_opt.py`)

Architecture:

1. Reuse validated forward + dQ from early family.
2. Replace dK/dV with KV-block-outer schedule and GPU-built inverted index.

Outcome:

- Useful intermediate step.
- In some bf16 environments, Triton dtype compile mismatch appeared in dK/dV path.

---

### D) Local FSA Variants

Files:

1. `fsa_topk_sparse_attention_local.py` (unoptimized baseline)
2. `fsa_topk_sparse_attention_local_optimized_older.py` (older optimized)
3. `fsa_topk_sparse_attention_local_optimized_old.py` (old optimized)
4. `fsa_topk_sparse_attention_local_optimized.py` (latest optimization-heavy branch)

Core architecture elements in this family:

1. Multiple forward modes:
   - NSA-style selected-attention forward
   - full-deserialized forward
2. Multiple backward modes:
   - atomic dQ
   - staged dQ reduce
   - fused GQA dK/dV
   - worklist and persistent queue schedules
3. Metadata pipeline and policies:
   - active block map/worklist construction
   - optional route sorting
   - arch policy buckets (`sm90/sm80/generic`)
   - packed-GQA controls
   - block pruning/sanitization knobs

Outcome:

- Best overall backward quality among explored families.
- Latest branch added broad optimization/policy machinery.
- But latest branch showed runtime instability in tested high-load runs (`illegal memory access`), so stable deployment set was frozen separately.

---

### E) NSA Selected-Attention Baseline

Observed behavior:

1. Strong forward performance in `G>=16` configs (with `tl.dot` path).
2. Backward significantly slower for this use case.
3. Limited support for `G<16`.

Interpretation:

- Useful forward reference and inspiration.
- Not the best end-to-end training kernel for current target workloads.

---

### F) Chapter-Routed MoE-Inspired Path (`fsa_topk_sparse_attention_chapter_routed.py`)

This path explores a more explicit "token routes to chapter, perform chapter-wise dense-like ops, then gather back" structure inspired by MoE grouping ideas.

It includes:

1. PyTorch-oriented execution path.
2. Triton-enabled execution path.
3. Routing/gather/scatter orchestration experiments.

Outcome:

- Valuable conceptual route.
- Still experimental relative to stable local-FSA variants.

---

## 4) Optimization Themes Implemented in the Latest Local Branch

The latest local optimized kernel (`fsa_topk_sparse_attention_local_optimized.py`) includes extensive engineering from the optimization roadmap, including:

1. Reduced host orchestration (batching scalar transfers, metadata path changes).
2. Precomputed/reused permutation metadata where possible.
3. Vectorized or GPU-first metadata prep in several hotspots.
4. Worklist/persistent scheduling options for dK/dV.
5. Packed-GQA handling in forward and backward hotspots.
6. Block-pruning policy integration with sanitization.
7. Sequence-parallel/flat-multi-seq backward pathways.
8. Hopper-aware pipeline controls exposed via environment policies.

Even with this, real-world stability and universal speedup closure remained incomplete for the newest branch, which is why stable rollout focuses on curated older variants.

---

## 5) Why Token-Level Routing Matters Architecturally

Beyond speed, there is a modeling reason:

1. Sequence-level, rolling, and hybrid routing can use future tokens (or mixed-window signals) when deciding chapter routes.
2. This can leak future information into route choices and memory selection behavior.
3. Token-level routing enforces more causal, per-token decisions and can be more expressive for localized memory retrieval.

So this kernel work is both:

1. a systems optimization effort, and
2. a modeling-correctness/causality effort.

---

## 6) Performance Narrative (Reported During This Work)

Representative progression reported in project sessions:

1. Initial sparse emulation paths: too slow for practical training.
2. Early custom kernels: large improvement but still around hundreds of ms forward and multi-second backward in earlier stages.
3. FSA/NSA-inspired improvements:
   - around 40 ms forward and around 40-45 ms backward at representative target setup (`TQ=32*8192`, memory around 65k tokens, `topk=8`, chapter size 128, `G=1`, `D=512`).
4. NSA forward hotspot at `G=16`:
   - around 11 ms forward for `D=1024, HQ=16, G=16`.
   - backward much slower (around 357 ms), reducing end-to-end utility.

## 6.1) Why Long-Sequence Scaling Helps

One practical observation behind doing this engineering:

- Self-attention cost grows roughly with sequence length squared.
- Memory cross-attention into a fixed or slowly growing activated memory subset grows roughly linearly with sequence length.

So as training moves from 8k to 64k sequences:

- self-attention can become dramatically more expensive,
- while memory attention overhead grows much more slowly (even if activated memory grows, it is still linear in the activated subset),
- which means the relative overhead of memory routing kernels can shrink in the regimes that matter most.

## 6.2) Train/Prefill vs Decoding

These kernels were primarily motivated by train/prefill:

- In decoding, KV is already per-sample and sequence length per step is 1, so token-level routing is far easier to do without KV duplication.
- In train/prefill, batching and per-token route diversity create the hard systems problem.

Even so, using the kernels during inference can still be useful to maximize throughput when the memory subsystem is a significant part of the overall architecture cost.

---

## 7) Stability Outcome and Final Selection

Final practical decision:

1. Keep full experimental history in `kernels/`.
2. Use curated stable set in `kernels-final/`:
   - v1: unoptimized local
   - v2: older optimized local (overall stability winner)
   - v3: old optimized local

Latest local optimized and local varlen compatibility paths remain future-fix candidates (especially around illegal memory access and reliability under broad shape/env matrices).

---

## 8) Benchmarking Architecture in This Folder

Primary benchmark harness:

- `benchmark_memory_xattn_optimized_import.py` (main working script)

Experimental branch harness:

- `benchmark_memory_xattn_optimized_import__newly_changed.py`

Typical benchmark compares:

1. Dense + gather references.
2. FSA local variants (current/old/older/unoptimized).
3. FSA varlen compatibility path.
4. Chapter-routed method.
5. NSA selected-attention baseline when valid (`G>=16`).

Operational caveat:

- After any kernel illegal access, CUDA context is typically poisoned and requires runtime restart for trustworthy follow-up measurements.

---

## 9) Remaining High-Value Technical Directions

1. Stabilize newest local optimized kernel under aggressive workloads and broad G/head/block regimes.
2. Further reduce remaining metadata/control overhead in backward orchestration.
3. Improve forward math path for `G=2/4/8` with better `tl.dot`-equivalent behavior.
4. Consider targeted CUDA custom kernels for hotspots that are difficult to close in Triton.
5. Re-run shape-family policy retuning with profiler-guided thresholds on production GPUs.

---

## 10) Relationship to Main Repository

Main repository architecture remains memory-augmented transformers and routing design in `memory_transformer/` and `docs/`.

This kernel architecture is a specialized implementation track to make token-level routing feasible at scale, especially for train/prefill where default framework attention strategies are insufficient.

### 10.1 Current Integrated Token-Routing Path

This repository now implements token-level routing end-to-end in model code:

1. Router produces per-token routed chapter indices (`[B, T, topk]`) with optional shared chapter prefix exclusion.
2. Memory bank is split into:
   - shared chapter tokens (always-on dense branch),
   - routed chapter tokens (sparse branch).
3. Dense shared branch uses memory cross-attention with:
   - FlashAttention when available,
   - PyTorch attention fallback otherwise.
4. Routed sparse branch uses `FSA_topk_sparse_attention_bthd` from:
   - `kernels-final/kernel_v1.py`, or
   - `kernels-final/kernel_v2.py` (default), or
   - `kernels-final/kernel_v3.py`.
5. Branches are combined as:
   - `output = shared_output + routed_scaling_factor * routed_output`.

Config knobs used by this path:

- `memory.routing_strategy_train` / `memory.routing_strategy_inference`: set `token` to enable token-level routing.
- `memory.token_routing_kernel_version`: `v1|v2|v3` (default `v2`).
- `memory.num_shared_chapters`: shared dense prefix chapters.
- `memory.routed_scaling_factor`: routed-branch mixing scale.

Runtime fallback behavior:

- If sparse kernel execution is unavailable for current device/dtype/shape, model code falls back to an emulated sparse PyTorch path for functional correctness.
