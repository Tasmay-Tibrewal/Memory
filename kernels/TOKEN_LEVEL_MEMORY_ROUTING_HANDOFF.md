# Token-Level Memory Routing Handoff (FSA Local / Triton)

## 1) Problem Statement

We are building a **token-level routed memory cross-attention** system for long-context inference/training.

Core idea:
- Query tokens (`TQ`) do **not** attend to full memory (`TK`) densely.
- A router selects top-k memory chapters per token/head (`block_indices`).
- Attention compute should be performed only on routed memory blocks, while matching dense semantics on selected tokens.

Target objective:
- Achieve sparse memory attention with much lower wall-clock than full dense alternatives, especially for large `TQ` and `TK`.
- Keep backward pass stable and fast enough for training use.

Current pain point:
- Forward has improved significantly in some configs.
- Backward still tends to dominate in many regimes.
- Performance can stagnate when metadata/orchestration overhead dominates kernel math.

---

## 2) What This Repo Is Doing

Primary working optimized path:
- `kernels/fsa_topk_sparse_attention_local_optimized.py`
- Benchmark driver:
  - `kernels/benchmark_memory_xattn_optimized_import.py`

Other reference paths included in benchmark:
- Dense FlashAttention + gathered KV baseline
- `flash_attn_with_kvcache` sparse emulation baseline
- FLA upstream compatibility baseline (with local fallback path)
- NSA baseline (only valid for some `G` regimes)

---

## 3) Memory Routing Formulation (Token-Level)

Given:
- `q`: `[B, TQ, HQ, D]`
- `k, v`: `[B, TK, HK, D]`
- `block_indices`: `[B, TQ, HK, S]` chapter IDs (`S = topk`)
- `block_size = BS`, `M = TK / BS`

For each `(token t, kv-head h_k)`:
- Router provides `S` chapters.
- Effective selected KV length = `S * BS` tokens (sparse subset, token-level routing).

For GQA:
- `G = HQ / HK` query heads per KV head group.
- Kernel must handle both `G >= 16` (tensor-core friendly) and `G < 16` small-G path robustly.

---

## 4) Current Status Snapshot

### Implemented and Active (high level)
- GPU-side metadata/preprocess improvements (active block mapping, worklist support, compaction path)
- GQA-aware fused dK/dV modes with schedule options
- One-pass and two-pass dK/dV controls
- dQ atomic accumulation mode and deserialization controls
- Packed-GQA route for forward/NSA/dQ, with safe auto-gating for `G=1` (`num_share_q_heads <= 1`)
- Arch-aware launch policy (`sm90/sm80/generic`) with shape buckets
- Hopper-oriented pipeline controls:
  - `FSA_LOCAL_HOPPER_ASYNC_PIPELINE`
  - `FSA_LOCAL_HOPPER_PIPELINE_STAGES`
  - `FSA_LOCAL_HOPPER_PIPELINE_CHUNKS`
- Strong block-pruning integration:
  - explicit pruning policy mode
  - bounded/sanitized block IDs (`FSA_LOCAL_BLOCK_PRUNING_SANITIZE`)
  - forward + backward metadata/reorder integration
- `P3.1` in-kernel dK/dV chunk-pipelining wired across:
  - legacy
  - fused grid
  - worklist
  - persistent queue paths
- Persistent queue policy closure:
  - schedule admission now also gates on min work-items and min q-per-item
  - dynamic persistent chunk/workers policy from workload density
  - schedule telemetry counters for acceptance/rejection causes

### Recent stability/compatibility fixes (already landed)
- Fixed `G=1` packed-dQ failure (`valid_lens_qh is None`) by hard-disabling packed-GQA for `num_share_q_heads <= 1`.
- Full-deserialize forward route handling is now robust to per-KV-head routed-query differences (union-range handling).
- Benchmark import path now deterministically prefers local optimized file and prints loaded module file path.
- Benchmark failure diagnostics now print full traceback by default (`FSA_BENCH_VERBOSE_ERRORS=1`).

### Known recurring runtime failure modes
- `CUDA illegal memory access` when a prior kernel fault poisons context
- Grad detach failures in some fallback/compat paths:
  - `element 0 of tensors does not require grad and does not have a grad_fn`
- Upstream FLA Triton pointer-tuple compile issue in some environments

---

## 5) Key Files to Read First

1. `kernels/fsa_topk_sparse_attention_local_optimized.py`
- Main optimized Triton kernels + orchestration
- Env knobs and scheduling logic live here

2. `kernels/benchmark_memory_xattn_optimized_import.py`
- All benchmark comparisons and path selection
- Good place to verify active code path and printed tuning knobs

3. `kernels/FSA_LOCAL_OPTIMIZATION_REPORT.md`
- Priority roadmap and implementation status tracking

---

## 6) What Is Left (Priority-Ordered)

This list is practical priority, not just feature checklist.

### Explicit status by label (for session handoff)
- Implemented (core):
  - `P0` workstream (`P0.1`, `P0.2`, `P0.3`) implemented
  - `P1` workstream (`P1.1`, `P1.2`, `P1.3`) implemented
  - `P2.1` persistent queue tuning closure (policy/threshold/chunk/workers refinement)
  - `P2.2` packed-GQA dataflow completion across forward/dQ/NSA hotspots
  - `P2.3` (arch/shape launch policy wiring)
  - Section `(4).4` mask/block pruning explicit integration (including exhaustive block-id sanitization)
  - Section `(4).5` sequence-parallel backward dedicated track
  - `P3.1` (Triton in-kernel dK/dV chunk-pipelining + launch wiring)
  - `P3.2` (benchmark compatibility routing cleanup)
- Still left / not fully completed (measurement-driven, not core missing plumbing):
  - Workload-family retuning of policy thresholds (`P1.2`, `P2.1`, `P2.3`) using real benchmark/profiler data
  - Further two-pass dK/dV specialization by shape family (policy refinement)
  - Optional explicit CUDA `cp.async` microkernel rewrite (outside Triton-first scope)

### Remaining priorities (current)
1. Workload-family threshold retuning for launch/schedule policy (`P1.2`/`P2.1`/`P2.3`) with measured occupancy + memory-traffic data.
2. Further two-pass dK/dV policy specialization by shape family.
3. Optional explicit CUDA `cp.async` microkernel rewrite (outside Triton-first scope).

---

## 7) Recommended Env Configs (Starting Point)

Use this as baseline for H200/H100 tuning runs:

```python
import os
os.environ["MEM_XATTN_FAST_START"] = "1"
os.environ["FSA_LOCAL_USE_NSA_STYLE_FWD"] = "1"
os.environ["FSA_LOCAL_FORCE_NSA_STYLE_FWD_SMALL_G"] = "0"
os.environ["FSA_LOCAL_SMALL_G_MODE"] = "fallback"
os.environ["FSA_LOCAL_PAD_G_TO_16"] = "1"
os.environ["FSA_LOCAL_DQ_ACCUM_MODE"] = "atomic"
os.environ["FSA_LOCAL_DQ_FORCE_ATOMIC"] = "1"
os.environ["FSA_LOCAL_DQ_FULL_DESERIALIZE"] = "1"
os.environ["FSA_LOCAL_COMPACT_ACTIVE_BLOCKS"] = "auto"
os.environ["FSA_LOCAL_DKDV_MODE"] = "auto"
os.environ["FSA_LOCAL_DKDV_TWO_PASS"] = "auto"
os.environ["FSA_LOCAL_DKDV_SCHEDULE"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_AUTO"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_ACTIVE_RATIO"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_MIN_ACTIVE_Q"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_MIN_WORK_ITEMS"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_MIN_Q_PER_ITEM"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_CHUNK"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_WORKERS_FACTOR"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_TARGET_ITEMS_PER_WORKER"] = "auto"
os.environ["FSA_LOCAL_DKDV_PERSISTENT_MIN_ITEMS_PER_WORKER"] = "auto"
os.environ["FSA_LOCAL_FWD_PACKED_GQA"] = "auto"
os.environ["FSA_LOCAL_NSA_PACKED_GQA"] = "auto"
os.environ["FSA_LOCAL_NSA_PACKED_GQA_SCOPE"] = "small_g"
os.environ["FSA_LOCAL_DQ_PACKED_GQA"] = "auto"
os.environ["FSA_LOCAL_GQA_ROUTE_COLLAPSE"] = "auto"
os.environ["FSA_LOCAL_BLOCK_PRUNING_MODE"] = "auto"
os.environ["FSA_LOCAL_BLOCK_PRUNING_SANITIZE"] = "1"
os.environ["FSA_LOCAL_ACTIVE_MAP_RATIO_THRESHOLD"] = "auto"
os.environ["FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE"] = "128"
os.environ["FSA_LOCAL_USE_ARCH_POLICY"] = "1"
os.environ["FSA_LOCAL_HOPPER_ASYNC_PIPELINE"] = "auto"
os.environ["FSA_LOCAL_HOPPER_PIPELINE_STAGES"] = "auto"
os.environ["FSA_LOCAL_HOPPER_PIPELINE_CHUNKS"] = "auto"
os.environ["FSA_BENCH_VERBOSE_ERRORS"] = "1"
```

If you hit illegal memory access:
- restart runtime/context first
- rerun with:
  - `CUDA_LAUNCH_BLOCKING=1`
  - optionally reduce aggressive schedule knobs to deterministic defaults

---

## 8) Benchmark Interpretation Notes

Important:
- Operator timings and full training-step timings differ.
- Some baselines include gather externally or internally; compare like-for-like paths.
- Prefix timeline vs no-prefix path changes both math and orchestration cost.
- Benchmark now prints `fsa_local module file: ...`; always confirm this matches your intended local file.

Do not compare only headline ms without confirming:
- same `TQ/TK/BS/topk/HQ/HK/D`
- same path (`prefix` vs `no-prefix`)
- same fallback status
- same warmup/iters and cache behavior

---

## 9) Suggested Next Session Execution Plan

1. Run one stable baseline matrix (small + medium + target large).
2. Verify active path telemetry from benchmark printouts (no silent fallback).
3. Profile top 3 kernels with Nsight for a single large config.
4. Tune schedule thresholds using collected occupancy/memory metrics.
5. Update `kernels/FSA_LOCAL_OPTIMIZATION_REPORT.md` with measured impact deltas only.

---

## 10) Handoff Checklist

Before handing to another session:
- confirm benchmark uses `kernels/benchmark_memory_xattn_optimized_import.py`
- confirm optimized kernel import path resolves to:
  - `kernels/fsa_topk_sparse_attention_local_optimized.py`
- confirm benchmark output includes `fsa_local module file: ...` and points to expected file (no stale `/root` copy mismatch)
- capture one full benchmark log with env block + config block
- note any crash signature and first failing strategy
- if `G=1`, keep packed-GQA knobs on `auto` and verify no packed-dQ path is force-enabled

This file is intended to be the single-session restart context for token-level memory routing work.
