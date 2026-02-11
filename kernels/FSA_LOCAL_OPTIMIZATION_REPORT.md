# FSA Local Optimization Report (Current-State, Impact-Ordered)

## 1) Scope And Method

This report was rebuilt from direct code inspection of:

- `fsa_topk_sparse_attention_local_optimized.py`
- `benchmark_memory_xattn_optimized_import.py`
- `flash-sparse-attention-flash-algo` Triton stack:
  - `flash_sparse_attn/flash_dmattn_triton.py`
  - `flash_sparse_attn/ops/triton/flash_fwd.py`
  - `flash_sparse_attn/ops/triton/utils.py`
  - `flash_sparse_attn/flash_sparse_attn_triton.py`

It focuses on **net training-step impact**, not just kernel microbenchmarks.

---

## 2) What Is Already Implemented (And Real)

These are present and functional in the optimized local kernel:

- Forward modes:
  - NSA-style forward path: `_topk_sparse_attention_fwd_nsa_style`
  - Full-deserialized forward path: `_topk_sparse_attention_fwd_opt_per_seq_all_heads`
  - Small-`G` modes: `pad`, `fma`, `torch`, `fallback`
- Backward modes:
  - dQ atomic accumulation path (`dq_compute_atomic_kernel`, `FSA_LOCAL_DQ_ACCUM_MODE`)
  - dQ staged path + reduce path
  - dK/dV fused GQA kernels (`backward_dkdv_gqa_fused`)
  - dK/dV worklist kernel (`backward_dkdv_gqa_fused_worklist`)
  - dK/dV persistent queue kernel (`backward_dkdv_gqa_fused_persistent_queue`)
  - optional two-pass dK/dV launch mode
- Routing and launch helpers:
  - active KV block compaction map and worklist
  - optional routed-query sorting
  - internal block-size cap (`FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE`)

So the current stagnation is **not** because advanced kernels are missing.

---

## 3) Why Performance Is Still Stagnant

### A) Host-side metadata path is still heavy in hot loop

Main examples:

- `_maybe_sort_reordered_topk_q_idx` uses CPU boundary extraction and Python per-segment loop.
- `_build_active_kv_block_map` uses CPU boolean map and Python nested loops over `(batch, kv_head)`.
- active-range detection and seqlen operations still trigger host transfers in multiple places.

Effect:

- GPU kernels may be fast, but end-to-end op time remains dominated by CPU orchestration/sync overhead.

### B) Fast forward path has fallback gates and path instability

Full-deser forward requires strict preconditions and otherwise falls back to legacy behavior. In large dynamic cases, fallback frequency can be high enough to erase expected gains.

### C) dQ is still orchestration-heavy even with atomic mode

Atomic dQ kernel exists, but metadata prep and per-sequence/per-head orchestration still carries significant host overhead.

### D) Prefix/no-prefix path coexistence still causes inconsistent execution

Benchmark logs show path fragility (`grad_fn` absent, `NoneType` failures, fallback ambiguity). This pollutes measured gains and blocks stable tuning.

### E) Environment matrix is too broad for stable "fast default"

Many mode switches are active simultaneously. Auto policies can route to suboptimal combinations for specific shape families.

---

## 4) Lessons Borrowable From flash-sparse-attention-flash-algo

Validated high-value borrow points:

### 1. GPU preprocess compaction stage (`_fwd_preprocess`)

From `flash_dmattn_triton.py`:

- Pre-gathers sparse-selected K/V (+ optional mask/bias) into compact contiguous buffers.
- Main fused attention kernel then runs on compacted memory, reducing random-gather overhead.

Relevance:

- Directly applicable to your routed chapter/top-k pipeline.

### 2. PACK-GQA pointer model (`flash_fwd.py`)

From `ops/triton/flash_fwd.py`:

- Avoids naive head replication by packed pointer addressing.

Relevance:

- Useful for forward and dQ/dK/dV dataflow simplification under GQA/MQA.

### 3. Architecture-aware autotune policy (`utils.py`)

From `ops/triton/utils.py`:

- Explicit per-architecture config strategy (SM80/SM90/...).

Relevance:

- Better than broad generic heuristics for your shape-sensitive mixed paths.

### 4. Mask/block pruning structure (`block_info.py`, `mask.py`)

- Early block-range pruning and skip when no active contribution.

Relevance:

- Can further reduce dead compute around sparse block iteration and long timelines.

### 5. Sequence-parallel backward pattern (`flash_sparse_attn_triton.py`)

- Uses sequence-parallel variants and atomic reduction strategy.

Relevance:

- Aligns with long-sequence backward bottlenecks in your pipeline.

---

## 5) Comprehensive Remaining Work (Priority By Net Impact)

### Implementation Status (as of latest patch)

- `[x]` **P0.1 (implemented for core metadata bottlenecks)**  
  Implemented:
  - `_build_active_kv_block_map` moved to GPU-vectorized construction (`nonzero` + `bucketize` + `bincount` + ranked scatter).
  - `_maybe_sort_reordered_topk_q_idx` no longer does CPU segmented loops; uses GPU segmented stable sort per KV head.
  - Active-range tensor API is used consistently (`active_starts`, `active_counts`) in forward/dQ callsites.
  - Removed hot-path metadata `.cpu()` / `.tolist()` conversions in optimized local kernel.
  Remaining caveat:
  - Residual scalar `.item()` control decisions still exist (shape/scheduling decisions), but the major CPU metadata bottlenecks from this item are addressed.

- `[x]` **P0.2 (implemented for fast-path gating)**  
  Implemented:
  - Full-deser forward no longer requires identical active ranges across KV heads.
  - Union-range handling added (`min(start)`, `max(end)` over routed KV heads).
  - Fallback now only for invalid `HQ % HK` in this gate.
  - Telemetry counters are active (`attempts/success/fallback_hq_hk/expanded_active_range`).
  Remaining caveat:
  - This does not by itself guarantee best performance; it guarantees path stability for the previous range-mismatch fallback reason.

- `[x]` **P0.3 (implemented)**  
  Implemented:
  - dQ atomic path now consumes prebuilt metadata in one de-serialized path over query-head tiles (`valid_topk_idx_concat`, per-head offsets/lens/start).
  - Hot-path per-KV-head dQ orchestration is bypassed when atomic mode is enabled.
  - Atomic mode is the default/forced path under full de-serialization controls.
  Remaining caveat:
  - A few scalar shape/scheduling decisions still use `.item()`; these are control-plane, not per-token metadata loops.

- `[x]` **P1.1 (implemented)**  
  Implemented:
  - Preprocess-compaction now exists as a reusable stage (`_maybe_precompact_kv_for_seq`) and is integrated into:
    - NSA-style forward path
    - full-deserialized forward path
    - legacy per-sequence forward path
  - Chapter-id remap is applied before routing metadata and fused forward math, so the whole forward pipeline benefits.
  - Safety guard added for non chapter-aligned `TK` (`TK % block_size != 0` -> skip compaction), avoiding partial-chapter OOB gather risk.
  Remaining caveat:
  - Compaction currently uses union-selected chapters across KV heads/queries for the sequence; it is not yet a finer-grained head-specific compactor.

- `[x]` **P1.2 (implemented)**  
  Implemented:
  - Shape-family policy hooks for dK/dV mode selection, two-pass enablement, and schedule (`grid/worklist/persistent`) are active.
  - Policy inputs now include workload/context signals (e.g., active ratio, batch size, share-heads, block size, head dim).
  Remaining caveat:
  - Policy quality still depends on tuning thresholds per workload family.

- `[x]` **P1.3 (implemented for benchmark/runtime reliability)**  
  Implemented:
  - No-prefix benchmark path now guards against detached/invalid outputs and auto-falls back to grad-valid prefix execution.
  - Query-loss slicing is shape-driven (`TQ` vs `TK+TQ`) instead of mode-assumption-only.
  - FLA local compatibility paths now use the same detached-output guard/fallback model.
  Remaining caveat:
  - This is a robustness fix-path and can alter measured speed when fallback is triggered.

- `[x]` **P2.3 (implemented)**  
  Implemented:
  - Added architecture + shape-bucket policy table (`sm90`, `sm80`, `generic`) for:
    - `dkdv_bq`, `dq_bq`, `dq_num_q_blocks`
    - `dkdv_two_pass`, `dkdv_schedule_auto`
    - launch tuples per op (`index_map`, `fwd_qk`, `fwd_qkv`, `fwd_reduce`, `bwd_delta`, `bwd_dq`, `bwd_dkdv`)
  - Added unified launch resolver and wired it into forward/backward launch sites.
  Remaining caveat:
  - Policy values are now fully wired, but still need workload-specific retuning to maximize gains.

- `[x]` **P2.1 (implemented: persistent-queue tuning closure)**  
  Implemented:
  - Extended schedule auto-policy with explicit persistent admission gates:
    - active ratio
    - minimum active-Q workload
    - minimum active work-items
    - minimum Q-per-work-item density
    - minimum batch
  - Added dynamic persistent dequeue chunk policy using active-ratio + per-item workload density.
  - Added worker-count policy that balances:
    - workers-per-SM factor
    - target items per worker
    - minimum items per worker
    - max-workers guard
  - Added schedule/persistent-rejection counters for runtime tuning telemetry.
  Remaining caveat:
  - Final thresholds are policy-complete but still workload-tunable for peak perf.

- `[x]` **P2.2 (implemented: packed-GQA across hotspots)**  
  Implemented:
  - Packed-GQA resolver is now shared across forward and dQ codepaths.
  - NSA-style forward hotspot now supports explicit packed-GQA execution path (`FSA_LOCAL_NSA_PACKED_GQA` + scope control) including small-`G`.
  - dQ atomic packed-GQA mode uses unified resolver and policy fallback.
  - BTHD wrapper now validates/collapses HQ-route inputs for GQA with explicit policy (`FSA_LOCAL_GQA_ROUTE_COLLAPSE`), removing silent mismatch behavior.
  Remaining caveat:
  - If HQ routes differ inside GQA groups and collapse policy is permissive (`auto/first`), behavior is best-effort by design.

- `[x]` **Section (4).4 mapping closure (implemented)**  
  Implemented:
  - Added exhaustive block-id sanitization before forward/backward metadata builds (`FSA_LOCAL_BLOCK_PRUNING_SANITIZE`).
  - Effective-block pruning now respects real KV block bounds, avoiding pseudo-tail blocks beyond real memory.
  - Pruning/sanitization telemetry is exposed for verification.

- `[x]` **Section (4).5 mapping closure (implemented)**  
  Implemented:
  - Dedicated sequence-parallel backward path is active with stream-policy resolver and dispatch integration.

- `[x]` **P3.1 (implemented in-kernel for dK/dV family)**  
  Implemented:
  - Hopper-aware launch pipelining control added via env:
    - `FSA_LOCAL_HOPPER_ASYNC_PIPELINE`
    - `FSA_LOCAL_HOPPER_PIPELINE_STAGES`
    - `FSA_LOCAL_HOPPER_PIPELINE_CHUNKS`
  - This is applied through the same launch resolver used by forward/backward kernels.
  - dK/dV kernels now use chunk-unrolled inner-loop pipelining (`PIPELINE_CHUNKS`) with staged `tl.range` dispatch.
  - All dK/dV launch paths are wired (legacy, fused, worklist, persistent queue; one-pass/two-pass).
  Remaining caveat:
  - This is Triton in-kernel pipelining (stage + chunk overlap), not a hand-written CUDA `cp.async` microkernel.

- `[x]` **P3.2 (implemented for benchmark compatibility path)**  
  Implemented:
  - Forced local-compat mode now deterministically chooses available compatibility backend:
    - local varlen if available
    - else local bthd
    - else explicit skip (no hidden upstream attempt)
  - Reduced noisy upstream-pointer-tuple retry path in forced-compat runs.
  Remaining caveat:
  - Upstream Triton pointer-tuple issue remains an upstream codegen issue; this fix improves benchmark stability, not upstream kernel internals.

## P0 (Do First, Biggest Net Payoff)

### P0.1 Build a GPU-only metadata pipeline

Goal:

- Remove CPU/Python loops from:
  - active block map build
  - routed query sort segmentation
  - per-head start/count metadata shaping

What to change:

- Replace `_build_active_kv_block_map` CPU path with GPU prefix-sum/scatter kernels.
- Replace `_maybe_sort_reordered_topk_q_idx` CPU segmented sort with GPU segmented sort or radix + segment boundaries.
- Ensure all routing metadata remains on device.

Expected impact:

- **1.25x to 1.9x total** (high confidence).

### P0.2 Make full-deser forward deterministic and fallback-free for primary target regime

Goal:

- For your primary benchmark regimes, never drop to legacy per-KV-head path.

What to change:

- Remove or rework active-range-equality fallback by supporting per-KV-head ranges in one batched flow.
- Add one-time explicit telemetry counters for fallback reason frequency.
- Enforce a single "fast default" profile for benchmark path.

Expected impact:

- **1.3x to 2.5x forward** where fallback currently occurs (high confidence).

### P0.3 dQ path: eliminate host orchestration around atomic kernel

Goal:

- Keep atomic compute, but move dQ metadata feed and scheduling to GPU-first orchestration.

What to change:

- No Python/per-head loops for dQ token maps in hot path.
- One batched metadata build per sequence batch.

Expected impact:

- **1.2x to 1.7x backward** (medium-high confidence).

---

## P1 (High Impact, Next)

### P1.1 Integrate preprocess-compaction stage (flash-sparse style)

Goal:

- Compact selected K/V (and optional score controls) once, then run fused kernels on compact buffers.

Expected impact:

- **1.15x to 1.7x forward**, **1.1x to 1.35x backward** (medium confidence).

### P1.2 Replace heuristic schedule with shape-family policy

Goal:

- Formal policy for selecting:
  - dK/dV mode (`gqa_fused` vs fallback)
  - schedule (`grid`, `worklist`, `persistent`)
  - two-pass enablement
  - block sizes

Expected impact:

- **1.1x to 1.35x total** via better path selection stability (medium confidence).

### P1.3 Tighten no-prefix first-class path

Goal:

- No-prefix should be equally stable as prefix mode and not trigger grad/path failures.

Expected impact:

- Primarily reliability, plus **~1.05x to 1.2x** from reduced timeline overhead in some regimes.

---

## P2 (Medium Impact)

### P2.1 Persistent queue tuning (implemented)

- Multi-item dequeue per worker.
- Better worker count policy per shape and occupancy.
- Explicit persistent admission gates by workload density and active-map statistics.

Expected impact:

- **1.05x to 1.25x backward** on skewed routing.

### P2.2 Complete packed-GQA dataflow in all hotspots (implemented)

- Extend packed pointer treatment to all forward/backward prep and staging paths.
- NSA hotspot now has explicit packed-GQA path with configurable scope.
- Wrapper now validates/collapses HQ routes for packed GQA consistently.

Expected impact:

- **1.05x to 1.25x** in GQA-heavy cases.

### P2.3 Arch-specialized autotune table

- Introduce architecture + shape bucket presets instead of broad auto heuristics.

Expected impact:

- **5% to 20%** depending on GPU/shape.

---

## P3 (Lower Priority / Higher Complexity)

### P3.1 Hopper async pipelining

- cp.async-style overlap for gather/compute.

Expected impact:

- **~1.05x to 1.2x** unless memory latency dominates heavily.

### P3.2 Upstream FLA compatibility cleanup

- Good for cleaner baseline comparison, low direct impact on your kernel speed.

---

## 6) Items To Deprioritize (Low ROI Right Now)

- More small-`G` fallback variants as primary strategy.
- CPU/PyTorch-host dot experiments for main fused path.
- Additional kernel variants without removing host metadata bottlenecks first.

---

## 7) Updated "What's Left" Checklist (Concrete)

Remaining practical work:

1. Workload-specific threshold retuning for policy tables (`P1.2/P2.1/P2.3`) to maximize perf deltas.
2. Extended A/B validation matrix for no-prefix production regimes on target GPUs.
3. Optional explicit CUDA `cp.async` microkernel rewrite (outside current Triton-first scope).

---

## 10) Section (4) -> Section (5) Mapping Check

Not all section (4) borrowable items were explicitly represented before; mapping is now:

1. Section (4).1 GPU preprocess compaction -> Section (5) `P1.1` (explicit).
2. Section (4).2 PACK-GQA pointer model -> Section (5) `P2.2` (explicit).
3. Section (4).3 Architecture-aware autotune -> Section (5) `P2.3` (explicit).
4. Section (4).4 Mask/block pruning -> explicit policy + exhaustive sanitization integrated.
5. Section (4).5 Sequence-parallel backward -> dedicated track implemented and wired.

---

## 8) Realistic Optimization Potential From Current Baseline

If P0 is executed correctly:

- **~1.5x to 2.3x net** is realistic.

If P0 + P1 are executed correctly:

- **~2.2x to 3.2x net** is realistic in your long-sequence sparse regime.

Beyond that, gains are still possible but become increasingly profile-sensitive.

---

## 9) Recommended Next Execution Order

1. GPU-only metadata pipeline (remove host loops/sync first).
2. Forward fast-path stabilization (eliminate fallback for target shapes).
3. dQ orchestration rewrite around atomic path.
4. Preprocess-compaction integration.
5. Dynamic schedule policy + arch-aware tuning tables.

This ordering maximizes probability of breaking the current "stagnant despite many features" state.
