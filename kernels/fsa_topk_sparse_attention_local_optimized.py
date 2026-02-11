# Copyright 2025 Ran Yan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific
import math
import os
import importlib
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import triton
import triton.language as tl

# Keep upstream source untouched: this local copy wires import paths at runtime.
_ROOT = Path(__file__).resolve().parent


def _ensure_nsa_ref_importable() -> None:
    try:
        importlib.import_module("nsa_ref.ops.topk_sparse_attention")
        return
    except Exception:
        pass

    candidates = []
    env_root = os.getenv("FSA_SRC_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        candidates.extend([p, p / "Flash-Sparse-Attention"])

    base_dirs = [_ROOT, Path.cwd(), *_ROOT.parents]
    for base in base_dirs:
        candidates.extend([base, base / "Flash-Sparse-Attention", base / "flash-sparse-attention"])

    seen = set()
    valid = []
    for cand in candidates:
        try:
            c = cand.resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if (c / "nsa_ref" / "ops" / "topk_sparse_attention.py").exists():
            valid.append(c)
            pstr = str(c)
            if pstr not in sys.path:
                sys.path.insert(0, pstr)

    try:
        importlib.import_module("nsa_ref.ops.topk_sparse_attention")
    except Exception as e:
        searched = "\n".join(str(x) for x in valid) if valid else "(no candidate with nsa_ref/ops found)"
        raise ModuleNotFoundError(
            "Could not import nsa_ref for local FSA copy. "
            "Set FSA_SRC_ROOT to your Flash-Sparse-Attention root.\n"
            f"Discovered candidates:\n{searched}"
        ) from e


_ensure_nsa_ref_importable()

from nsa_ref.ops.topk_sparse_attention import (backward_sum_o_do,
                                               reorder_topk_idx)
from nsa_ref.ops.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()
_FWD_FULL_DESER_NOTICE_PRINTED = False
_FWD_FULL_DESER_STATS = {
    "attempts": 0,
    "success": 0,
    "fallback_hq_hk": 0,
    "expanded_active_range": 0,
}
_FWD_PRECOMPACT_STATS = {
    "attempts": 0,
    "applied": 0,
    "skipped": 0,
}
_BLOCK_PRUNE_STATS = {
    "sanitize_calls": 0,
    "sanitize_applied": 0,
    "sanitize_dropped": 0,
    "effective_calls": 0,
    "tail_pruned_blocks": 0,
}
_DKDV_SCHEDULE_STATS = {
    "grid": 0,
    "worklist": 0,
    "persistent": 0,
    "persistent_rejected_ratio": 0,
    "persistent_rejected_min_active_q": 0,
    "persistent_rejected_min_batch": 0,
    "persistent_rejected_min_work_items": 0,
    "persistent_rejected_min_q_per_item": 0,
}
_GQA_ROUTE_COLLAPSE_WARNED = False

# OPT-5: Workspace buffer cache for backward pass.
# Avoids re-allocating large tensors every backward call when shapes don't change.
_BWD_WORKSPACE: dict = {}


def _is_default_cuda_stream(device: torch.device) -> bool:
    """Check if current CUDA stream is the default stream (safe for buffer reuse)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    try:
        return torch.cuda.current_stream(device) == torch.cuda.default_stream(device)
    except Exception:
        return False


def _get_cached_zeros(key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached zero-filled tensor, allocating only when shape/dtype/device change.
    Skips cache on non-default CUDA streams to avoid multi-stream data races."""
    if _is_default_cuda_stream(device):
        cached = _BWD_WORKSPACE.get(key)
        if cached is not None and cached.shape == shape and cached.dtype == dtype and cached.device == device:
            cached.zero_()
            return cached
    t = torch.zeros(shape, dtype=dtype, device=device)
    if _is_default_cuda_stream(device):
        _BWD_WORKSPACE[key] = t
    return t


def _get_cached_empty(key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached empty tensor, allocating only when shape/dtype/device change.
    Skips cache on non-default CUDA streams to avoid multi-stream data races."""
    if _is_default_cuda_stream(device):
        cached = _BWD_WORKSPACE.get(key)
        if cached is not None and cached.shape == shape and cached.dtype == dtype and cached.device == device:
            return cached
    t = torch.empty(shape, dtype=dtype, device=device)
    if _is_default_cuda_stream(device):
        _BWD_WORKSPACE[key] = t
    return t


def clear_fsa_workspace_cache() -> None:
    """Explicitly free all cached workspace buffers."""
    _BWD_WORKSPACE.clear()


def _use_workspace_cache_enabled() -> bool:
    return os.getenv("FSA_LOCAL_WORKSPACE_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def _workspace_empty(key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if _use_workspace_cache_enabled():
        return _get_cached_empty(key, shape, dtype, device)
    return torch.empty(shape, dtype=dtype, device=device)


def _workspace_zeros(key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if _use_workspace_cache_enabled():
        return _get_cached_zeros(key, shape, dtype, device)
    return torch.zeros(shape, dtype=dtype, device=device)


def _maybe_print_fwd_full_deser_notice(msg: str):
    """
    Print a one-time notice for full-de-serialized forward path decisions.
    Controlled by env `FSA_LOCAL_FWD_FULL_DESERIALIZE_VERBOSE` (default: on).
    """
    global _FWD_FULL_DESER_NOTICE_PRINTED
    verbose = os.getenv("FSA_LOCAL_FWD_FULL_DESERIALIZE_VERBOSE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if verbose and not _FWD_FULL_DESER_NOTICE_PRINTED:
        print(msg)
        _FWD_FULL_DESER_NOTICE_PRINTED = True


def _record_fwd_full_deser_stat(key: str, inc: int = 1):
    _FWD_FULL_DESER_STATS[key] = int(_FWD_FULL_DESER_STATS.get(key, 0)) + int(inc)
    every_raw = os.getenv("FSA_LOCAL_FWD_FULL_DESERIALIZE_STATS_EVERY", "0").strip().lower()
    try:
        every = int(every_raw) if every_raw not in ("", "auto") else 0
    except Exception:
        every = 0
    attempts = int(_FWD_FULL_DESER_STATS.get("attempts", 0))
    if every > 0 and attempts > 0 and (attempts % every == 0):
        print(
            "FSA full-deser stats:",
            f"attempts={_FWD_FULL_DESER_STATS.get('attempts', 0)},",
            f"success={_FWD_FULL_DESER_STATS.get('success', 0)},",
            f"fallback_hq_hk={_FWD_FULL_DESER_STATS.get('fallback_hq_hk', 0)},",
            f"expanded_active_range={_FWD_FULL_DESER_STATS.get('expanded_active_range', 0)}",
        )


def get_fsa_local_fwd_full_deser_stats() -> dict:
    return dict(_FWD_FULL_DESER_STATS)


def get_fsa_local_fwd_precompact_stats() -> dict:
    return dict(_FWD_PRECOMPACT_STATS)


def get_fsa_local_block_prune_stats() -> dict:
    return dict(_BLOCK_PRUNE_STATS)


def get_fsa_local_dkdv_schedule_stats() -> dict:
    return dict(_DKDV_SCHEDULE_STATS)


def _record_block_prune_stat(key: str, inc: int = 1) -> None:
    _BLOCK_PRUNE_STATS[key] = int(_BLOCK_PRUNE_STATS.get(key, 0)) + int(inc)


def _record_dkdv_schedule_stat(key: str, inc: int = 1) -> None:
    _DKDV_SCHEDULE_STATS[key] = int(_DKDV_SCHEDULE_STATS.get(key, 0)) + int(inc)


def _ensure_dq_atomic_metadata(permute_results: dict, num_kv_heads: int, num_blocks: int, device: torch.device) -> dict:
    """
    Ensure precomputed metadata required by atomic dQ path is present.

    Fields created when missing:
      - valid_topk_idx_concat
      - valid_topk_idx_offsets
      - valid_lens_stack
      - valid_start_indices_stack
    """
    if (
        permute_results.get("valid_topk_idx_concat", None) is not None
        and permute_results.get("valid_topk_idx_offsets", None) is not None
        and permute_results.get("valid_lens_stack", None) is not None
        and permute_results.get("valid_start_indices_stack", None) is not None
    ):
        return permute_results

    valid_lens = permute_results.get("valid_lens", [])
    valid_start_indices = permute_results.get("valid_start_indices", [])
    valid_topk_idx_permuted_tile = permute_results.get("valid_topk_idx_permuted_tile", [])

    if len(valid_lens) == 0 or len(valid_start_indices) == 0 or len(valid_topk_idx_permuted_tile) == 0:
        permute_results["valid_topk_idx_concat"] = torch.empty((0,), dtype=torch.int32, device=device)
        permute_results["valid_topk_idx_offsets"] = torch.zeros((num_kv_heads,), dtype=torch.int32, device=device)
        permute_results["valid_lens_stack"] = torch.zeros((num_kv_heads, num_blocks), dtype=torch.int32, device=device)
        permute_results["valid_start_indices_stack"] = torch.zeros((num_kv_heads, num_blocks), dtype=torch.int32, device=device)
        return permute_results

    valid_lens_stack = torch.stack(valid_lens, dim=0)
    valid_start_indices_stack = torch.stack(valid_start_indices, dim=0)
    per_kh_counts = valid_lens_stack.sum(dim=1).to(torch.int32)
    offsets = torch.zeros((num_kv_heads,), dtype=torch.int32, device=device)
    if num_kv_heads > 1:
        offsets[1:] = torch.cumsum(per_kh_counts, dim=0)[:-1]
    total_sel = int(per_kh_counts.sum().to(dtype=torch.int64).cpu().tolist())
    if total_sel > 0:
        concat = torch.cat(valid_topk_idx_permuted_tile, dim=0)
    else:
        concat = torch.empty((0,), dtype=torch.int32, device=device)

    permute_results["valid_topk_idx_concat"] = concat
    permute_results["valid_topk_idx_offsets"] = offsets
    permute_results["valid_lens_stack"] = valid_lens_stack
    permute_results["valid_start_indices_stack"] = valid_start_indices_stack
    return permute_results

_NATIVE_NSA_PARALLEL_FWD = None
_NATIVE_NSA_IMPORT_ERROR = None
_NATIVE_NSA_IMPORT_DONE = False


def _try_get_native_nsa_parallel_fwd():
    """
    Best-effort import for native-sparse-attention selected forward kernel.
    Returns callable or None.
    """
    global _NATIVE_NSA_PARALLEL_FWD, _NATIVE_NSA_IMPORT_ERROR, _NATIVE_NSA_IMPORT_DONE
    if _NATIVE_NSA_IMPORT_DONE:
        return _NATIVE_NSA_PARALLEL_FWD
    _NATIVE_NSA_IMPORT_DONE = True

    try:
        from native_sparse_attention.ops.parallel import parallel_nsa_fwd as _parallel_nsa_fwd
        _NATIVE_NSA_PARALLEL_FWD = _parallel_nsa_fwd
        return _NATIVE_NSA_PARALLEL_FWD
    except Exception as first_error:
        try:
            candidates = []
            here = _ROOT
            candidates.append(here / "native-sparse-attention" / "native_sparse_attention")
            candidates.append(here / "native_sparse_attention")
            for p in sys.path:
                candidates.append(Path(p) / "native_sparse_attention")

            pkg_dir = None
            for cand in candidates:
                if (cand / "ops" / "parallel.py").exists():
                    pkg_dir = cand
                    break
            if pkg_dir is None:
                raise RuntimeError("could not locate native_sparse_attention package directory")

            for name in list(sys.modules.keys()):
                if name == "native_sparse_attention" or name.startswith("native_sparse_attention."):
                    sys.modules.pop(name, None)

            nsa_pkg = types.ModuleType("native_sparse_attention")
            nsa_pkg.__path__ = [str(pkg_dir)]
            ops_pkg = types.ModuleType("native_sparse_attention.ops")
            ops_pkg.__path__ = [str(pkg_dir / "ops")]
            sys.modules["native_sparse_attention"] = nsa_pkg
            sys.modules["native_sparse_attention.ops"] = ops_pkg

            parallel_mod = importlib.import_module("native_sparse_attention.ops.parallel")
            _NATIVE_NSA_PARALLEL_FWD = parallel_mod.parallel_nsa_fwd
            return _NATIVE_NSA_PARALLEL_FWD
        except Exception as fallback_error:
            _NATIVE_NSA_IMPORT_ERROR = f"{first_error}; fallback failed: {fallback_error}"
            _NATIVE_NSA_PARALLEL_FWD = None
            return None


@triton.jit
def fused_fill_kernel(ptr_tile, ptr_m_i_cur_tiles, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    tl.store(ptr_tile + offsets, -1, mask=mask)  # fill int32 with -1
    tl.store(ptr_m_i_cur_tiles + offsets, float("-inf"), mask=mask)


def fused_fill(topk_idx_permuted_tile: torch.Tensor, m_i_cur_tiles):

    numel = topk_idx_permuted_tile.numel()
    BLOCK_SIZE = 1024

    # Flatten for pointer access
    tile_flat = topk_idx_permuted_tile.view(-1)

    m_i_cur_tiles_flat = m_i_cur_tiles.view(-1)
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="index_map",
        head_dim=64,
        block_size=BLOCK_SIZE,
        default_warps=1,
        default_stages=3,
    )

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    fused_fill_kernel[grid](
        tile_flat,
        m_i_cur_tiles_flat,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _detect_active_token_range(topk_idx_tile: torch.Tensor) -> tuple[int, int]:
    """
    Detect contiguous active tail tokens [start, total_len) for prefix layouts.
    Falls back to full range when active tokens are not a suffix.
    """
    total_len = int(topk_idx_tile.shape[1])  # topk_idx_tile: [head_tile, total_len, topk]
    token_has_route = (topk_idx_tile >= 0).any(dim=0).any(dim=-1)  # [total_len]
    nz = torch.nonzero(token_has_route, as_tuple=False).flatten()
    if nz.numel() == 0:
        return 0, 0
    first = int(nz[0].to(dtype=torch.int32).cpu().tolist())
    count = int(nz.numel())
    if count == (total_len - first):
        return first, count
    return 0, total_len


def _detect_active_token_ranges_per_kv_head(topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-first active-range detection for all KV heads.

    Returns:
      starts: int32 [H]
      counts: int32 [H]

    Semantics per head:
      - no routed tokens -> (0, 0)
      - routed suffix [s, N) -> (s, N-s)
      - non-suffix routing -> conservative full range (0, N)
    """
    if topk_idx.ndim != 3:
        raise ValueError(f"topk_idx must be 3D [H,N,S], got shape={tuple(topk_idx.shape)}")

    _, total_len, _ = topk_idx.shape
    has_route = (topk_idx >= 0).any(dim=-1)  # [H, N]
    counts = has_route.sum(dim=-1).to(torch.int32)  # [H]
    first_idx = has_route.to(torch.int32).argmax(dim=-1).to(torch.int32)  # [H]
    total_len_t = torch.full_like(first_idx, int(total_len), dtype=torch.int32)
    is_suffix = counts == (total_len_t - first_idx)

    starts = torch.where(
        counts > 0,
        torch.where(is_suffix, first_idx, torch.zeros_like(first_idx)),
        torch.zeros_like(first_idx),
    ).to(torch.int32)
    full_counts = torch.where(
        counts > 0,
        torch.where(is_suffix, total_len_t - starts, total_len_t),
        torch.zeros_like(counts),
    ).to(torch.int32)
    return starts, full_counts


def _cu_seqlens_to_ranges(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    """
    Convert int32 cu_seqlens [B+1] into Python (start, end) ranges.
    Uses a single host transfer for CUDA tensors to avoid repeated scalar syncs.
    """
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape={tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() < 2:
        return []
    seq_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.int32).tolist()
    return [(int(seq_cpu[i]), int(seq_cpu[i + 1])) for i in range(len(seq_cpu) - 1)]


def _build_seq_dispatch_meta(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device: torch.device,
) -> tuple[list[tuple[int, int, int, int, int, int]], torch.Tensor, torch.Tensor]:
    """
    Build per-sequence dispatch metadata once:
      - Python ranges and lengths
      - local [0, seqlen] cu_seqlens rows for q/k (shape [nseq, 2], int32)
    This avoids per-sequence torch.tensor allocations in wrapper hot paths.
    """
    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) != len(k_ranges):
        raise RuntimeError(
            f"Mismatched sequence partitions: len(q_ranges)={len(q_ranges)} vs len(k_ranges)={len(k_ranges)}."
        )

    seq_meta: list[tuple[int, int, int, int, int, int]] = []
    if len(q_ranges) == 0:
        empty = torch.empty((0, 2), dtype=torch.int32, device=device)
        return seq_meta, empty, empty

    q_lens = []
    k_lens = []
    for (q_start, q_end), (k_start, k_end) in zip(q_ranges, k_ranges):
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        seq_meta.append((int(q_start), int(q_end), int(k_start), int(k_end), q_len, k_len))
        q_lens.append(q_len)
        k_lens.append(k_len)

    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device=device)
    k_lens_t = torch.tensor(k_lens, dtype=torch.int32, device=device)
    cu_q_local = torch.stack((torch.zeros_like(q_lens_t), q_lens_t), dim=1)
    cu_k_local = torch.stack((torch.zeros_like(k_lens_t), k_lens_t), dim=1)
    return seq_meta, cu_q_local, cu_k_local


def _globalize_topk_idx_by_seq_offsets(
    topk_idx: torch.Tensor,        # [HK, TQ_total, S], local block ids per sequence
    cu_seqlens_q: torch.Tensor,    # [B+1]
    cu_seqlens_k: torch.Tensor,    # [B+1]
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert per-sequence-local block ids to flattened global block ids.

    Returns:
      topk_idx_global: [HK, TQ_total, S] with sequence block-offset added to valid ids
      cu_seqblocks:    [B+1] cumulative KV block offsets per sequence
    """
    if topk_idx.ndim != 3:
        raise ValueError(f"topk_idx must be [H,N,S], got {tuple(topk_idx.shape)}")
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(dtype=torch.int32)
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(dtype=torch.int32)
    if q_lens.numel() != k_lens.numel():
        raise RuntimeError("Mismatched q/k sequence partitions for topk globalization.")

    seq_blocks = torch.div(
        k_lens + int(block_size) - 1,
        int(block_size),
        rounding_mode="floor",
    ).to(dtype=torch.int32)
    cu_seqblocks = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(seq_blocks, dim=0),
        ),
        dim=0,
    )

    total_q = int(topk_idx.shape[1])
    if total_q <= 0:
        return topk_idx, cu_seqblocks

    q_tok = torch.arange(total_q, device=topk_idx.device, dtype=torch.int64)
    q_seq = torch.bucketize(q_tok, cu_seqlens_q[1:].to(dtype=torch.int64), right=True)
    blk_off = cu_seqblocks.index_select(0, q_seq).to(dtype=topk_idx.dtype).view(1, total_q, 1)
    topk_idx_global = torch.where(
        topk_idx >= 0,
        topk_idx + blk_off,
        torch.full_like(topk_idx, -1),
    )
    return topk_idx_global, cu_seqblocks


def _pack_varlen_unified_timeline(
    q: torch.Tensor,                # [TQ, HQ, D]
    k: torch.Tensor,                # [TK, HK, D]
    v: torch.Tensor,                # [TK, HK, D]
    topk_idx: torch.Tensor,         # [HK, TQ, S] local-per-seq block ids
    cu_seqlens_q: torch.Tensor,     # [B+1]
    cu_seqlens_k: torch.Tensor,     # [B+1]
    block_size: int,
    o: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
) -> Optional[dict]:
    """
    Universal multi-sequence packing into a single block-aligned timeline.

    Per sequence i:
      - unified token base is u_start_i = (sum_{j<i} ceil(max(q_len_j, k_len_j)/BS)) * BS
      - q/k/v are copied into [u_start_i : u_start_i + q_len_i/k_len_i]
      - topk block ids are shifted by unified block offset

    This guarantees local causal alignment in flattened execution even when q/k lengths differ.
    """
    seq_meta, _cu_q_local, _cu_k_local = _build_seq_dispatch_meta(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        device=q.device,
    )
    nseq = len(seq_meta)
    if nseq <= 1:
        return None

    hq = int(q.shape[1])
    d = int(q.shape[2])
    hk = int(k.shape[1])
    topk = int(topk_idx.shape[-1])

    seq_blocks = []
    for _q_start, _q_end, _k_start, _k_end, q_len, k_len in seq_meta:
        seg_tokens = max(int(q_len), int(k_len))
        seg_blocks = (seg_tokens + int(block_size) - 1) // int(block_size)
        seq_blocks.append(max(1, int(seg_blocks)) if seg_tokens > 0 else 0)

    seq_blocks_t = torch.tensor(seq_blocks, dtype=torch.int32, device=q.device)
    cu_u_blocks = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=q.device),
            torch.cumsum(seq_blocks_t, dim=0),
        ),
        dim=0,
    )
    u_total_blocks = int(cu_u_blocks[-1].to(dtype=torch.int32).cpu().tolist()) if cu_u_blocks.numel() > 0 else 0
    u_total_tokens = u_total_blocks * int(block_size)
    if u_total_tokens <= 0:
        return None

    q_u = torch.zeros((u_total_tokens, hq, d), dtype=q.dtype, device=q.device)
    k_u = torch.zeros((u_total_tokens, hk, d), dtype=k.dtype, device=k.device)
    v_u = torch.zeros((u_total_tokens, hk, d), dtype=v.dtype, device=v.device)
    topk_u = torch.full((hk, u_total_tokens, topk), -1, dtype=topk_idx.dtype, device=topk_idx.device)

    o_u = torch.zeros((u_total_tokens, hq, d), dtype=o.dtype, device=o.device) if o is not None else None
    do_u = torch.zeros((u_total_tokens, hq, d), dtype=do.dtype, device=do.device) if do is not None else None
    lse_u = torch.full((hq, u_total_tokens), float("-inf"), dtype=lse.dtype, device=lse.device) if lse is not None else None
    delta_u = torch.zeros((hq, u_total_tokens), dtype=delta.dtype, device=delta.device) if delta is not None else None

    packed_meta: list[tuple[int, int, int, int, int, int, int]] = []
    for i, (q_start, q_end, k_start, k_end, q_len, k_len) in enumerate(seq_meta):
        q_len_i = int(q_len)
        k_len_i = int(k_len)
        u_blk_off = int(cu_u_blocks[i].to(dtype=torch.int32).cpu().tolist())
        u_start = u_blk_off * int(block_size)
        u_q_end = u_start + q_len_i
        u_k_end = u_start + k_len_i

        if q_len_i > 0:
            q_u[u_start:u_q_end] = q[q_start:q_end]
            topk_local = topk_idx[:, q_start:q_end, :]
            topk_u[:, u_start:u_q_end, :] = torch.where(
                topk_local >= 0,
                topk_local + u_blk_off,
                torch.full_like(topk_local, -1),
            )
            if o_u is not None:
                o_u[u_start:u_q_end] = o[q_start:q_end]
            if do_u is not None:
                do_u[u_start:u_q_end] = do[q_start:q_end]
            if lse_u is not None:
                lse_u[:, u_start:u_q_end] = lse[:, q_start:q_end]
            if delta_u is not None:
                delta_u[:, u_start:u_q_end] = delta[:, q_start:q_end]
        if k_len_i > 0:
            k_u[u_start:u_k_end] = k[k_start:k_end]
            v_u[u_start:u_k_end] = v[k_start:k_end]

        packed_meta.append((q_start, q_end, k_start, k_end, u_start, q_len_i, k_len_i))

    cu_u = torch.tensor([0, u_total_tokens], dtype=torch.int32, device=q.device)
    return {
        "q_u": q_u,
        "k_u": k_u,
        "v_u": v_u,
        "topk_u": topk_u,
        "o_u": o_u,
        "do_u": do_u,
        "lse_u": lse_u,
        "delta_u": delta_u,
        "cu_u": cu_u,
        "u_total_tokens": u_total_tokens,
        "packed_meta": packed_meta,
    }


def _unpack_varlen_unified_q(
    packed_meta: list[tuple[int, int, int, int, int, int, int]],
    src_u: torch.Tensor,   # [U, ...] or [H, U]
    dst: torch.Tensor,
    by_head_first: bool = False,
) -> torch.Tensor:
    """
    Unpack query-timeline slices from unified timeline back to original varlen layout.
    """
    if by_head_first:
        for q_start, q_end, _k_start, _k_end, u_start, q_len_i, _k_len_i in packed_meta:
            if q_len_i <= 0:
                continue
            dst[:, q_start:q_end] = src_u[:, u_start:u_start + q_len_i]
    else:
        for q_start, q_end, _k_start, _k_end, u_start, q_len_i, _k_len_i in packed_meta:
            if q_len_i <= 0:
                continue
            dst[q_start:q_end] = src_u[u_start:u_start + q_len_i]
    return dst


def _unpack_varlen_unified_kv(
    packed_meta: list[tuple[int, int, int, int, int, int, int]],
    dk_u: torch.Tensor,   # [U, HK, D]
    dv_u: torch.Tensor,   # [U, HK, D]
    dk_dst: torch.Tensor, # [TK, HK, D]
    dv_dst: torch.Tensor, # [TK, HK, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack K/V-timeline slices from unified timeline back to original varlen layout.
    """
    for _q_start, _q_end, k_start, k_end, u_start, _q_len_i, k_len_i in packed_meta:
        if k_len_i <= 0:
            continue
        dk_dst[k_start:k_end] = dk_u[u_start:u_start + k_len_i]
        dv_dst[k_start:k_end] = dv_u[u_start:u_start + k_len_i]
    return dk_dst, dv_dst


def _sanitize_topk_block_indices(
    topk_idx: torch.Tensor,
    real_num_blocks: int,
) -> tuple[torch.Tensor, int]:
    """
    Exhaustive block-id sanitization for routed metadata.
    Any block id outside [0, real_num_blocks) is invalidated to -1.
    """
    _record_block_prune_stat("sanitize_calls")
    mode = os.getenv("FSA_LOCAL_BLOCK_PRUNING_SANITIZE", "1").strip().lower()
    if mode in ("0", "false", "no", "off", ""):
        return topk_idx, 0
    if topk_idx.numel() == 0:
        return topk_idx, 0

    real_num_blocks = max(0, int(real_num_blocks))
    if real_num_blocks <= 0:
        invalid = topk_idx >= 0
        dropped = int(invalid.to(dtype=torch.int64).sum().cpu().tolist())
        if dropped <= 0:
            return topk_idx, 0
        out = torch.full_like(topk_idx, -1)
        _record_block_prune_stat("sanitize_applied")
        _record_block_prune_stat("sanitize_dropped", dropped)
        return out, dropped

    valid = (topk_idx >= 0) & (topk_idx < real_num_blocks)
    invalid = (topk_idx >= 0) & (~valid)
    dropped = int(invalid.to(dtype=torch.int64).sum().cpu().tolist())
    if dropped <= 0:
        return topk_idx, 0

    out = torch.where(valid, topk_idx, torch.full_like(topk_idx, -1))
    # Keep deterministic sorted order after sanitization.
    out = out.sort(dim=-1).values
    _record_block_prune_stat("sanitize_applied")
    _record_block_prune_stat("sanitize_dropped", dropped)
    return out, dropped


def _get_arch_bucket() -> str:
    """
    Runtime architecture bucket for policy selection.
    """
    if not torch.cuda.is_available():
        return "generic"
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return "generic"
    if major >= 9:
        return "sm90"
    if major >= 8:
        return "sm80"
    return "generic"


def _get_shape_bucket(head_dim: int, block_size: int) -> str:
    """
    Lightweight shape family bucketing for policy lookup.
    """
    if block_size >= 512:
        return "bs_ge_512"
    if block_size >= 256:
        return "bs_256"
    if head_dim >= 128:
        return "hd_ge_128"
    return "default"


_ARCH_POLICY_TABLE = {
    "sm90": {
        "default": {
            "dkdv_bq": 128,
            "dq_bq": 128,
            "dq_num_q_blocks": 8,
            "dkdv_two_pass": True,
            "dkdv_schedule_auto": "worklist",
            "active_map_ratio_threshold": 0.94,
            "persistent_auto_enable": True,
            "persistent_auto_active_ratio": 0.55,
            "persistent_auto_min_active_q": 262144,
            "persistent_auto_min_work_items": 1024,
            "persistent_auto_min_q_per_item": 16.0,
            "persistent_chunk": 2,
            "persistent_workers_per_sm": 2,
            "persistent_target_items_per_worker": 16,
            "persistent_min_items_per_worker": 2,
            "block_prune_min_tail": 8,
            "block_prune_min_ratio": 0.04,
            "fwd_packed_gqa": True,
            "nsa_packed_gqa": True,
            "bwd_seq_parallel": True,
            "bwd_seq_parallel_streams": 4,
            "launch": {
                "index_map": (2, 3),
                "fwd_qk": (8, 4),
                "fwd_qkv": (4, 4),
                "fwd_reduce": (1, 2),
                "bwd_delta": (4, 3),
                "bwd_dq": (8, 4),
                "bwd_dkdv": (8, 4),
            },
        },
        "hd_ge_128": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 8,
            "dkdv_two_pass": True,
        },
        "bs_256": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 8,
            "dkdv_two_pass": True,
        },
        "bs_ge_512": {
            "dkdv_bq": 32,
            "dq_bq": 32,
            "dq_num_q_blocks": 4,
            "dkdv_two_pass": True,
            "launch": {
                "bwd_dq": (8, 5),
                "bwd_dkdv": (8, 5),
            },
        },
    },
    "sm80": {
        "default": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 4,
            "dkdv_two_pass": False,
            "dkdv_schedule_auto": "grid",
            "active_map_ratio_threshold": 0.92,
            "persistent_auto_enable": False,
            "persistent_chunk": 1,
            "persistent_workers_per_sm": 1,
            "persistent_auto_min_work_items": 4096,
            "persistent_auto_min_q_per_item": 24.0,
            "persistent_target_items_per_worker": 24,
            "persistent_min_items_per_worker": 4,
            "block_prune_min_tail": 8,
            "block_prune_min_ratio": 0.05,
            "fwd_packed_gqa": True,
            "nsa_packed_gqa": True,
            "bwd_seq_parallel": True,
            "bwd_seq_parallel_streams": 2,
            "launch": {
                "index_map": (2, 3),
                "fwd_qk": (8, 3),
                "fwd_qkv": (4, 3),
                "fwd_reduce": (1, 2),
                "bwd_delta": (4, 3),
                "bwd_dq": (4, 3),
                "bwd_dkdv": (4, 3),
            },
        },
        "hd_ge_128": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 4,
            "dkdv_two_pass": True,
        },
        "bs_256": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 4,
            "dkdv_two_pass": False,
        },
        "bs_ge_512": {
            "dkdv_bq": 32,
            "dq_bq": 32,
            "dq_num_q_blocks": 2,
            "dkdv_two_pass": False,
        },
    },
    "generic": {
        "default": {
            "dkdv_bq": 64,
            "dq_bq": 64,
            "dq_num_q_blocks": 4,
            "dkdv_two_pass": False,
            "dkdv_schedule_auto": "grid",
            "active_map_ratio_threshold": 0.92,
            "persistent_auto_enable": False,
            "persistent_chunk": 1,
            "persistent_workers_per_sm": 1,
            "persistent_auto_min_work_items": 8192,
            "persistent_auto_min_q_per_item": 24.0,
            "persistent_target_items_per_worker": 32,
            "persistent_min_items_per_worker": 4,
            "block_prune_min_tail": 8,
            "block_prune_min_ratio": 0.05,
            "fwd_packed_gqa": True,
            "nsa_packed_gqa": False,
            "bwd_seq_parallel": False,
            "bwd_seq_parallel_streams": 1,
            "launch": {
                "index_map": (2, 3),
                "fwd_qk": (4, 3),
                "fwd_qkv": (4, 3),
                "fwd_reduce": (1, 2),
                "bwd_delta": (4, 3),
                "bwd_dq": (4, 3),
                "bwd_dkdv": (4, 3),
            },
        },
    },
}


def _use_arch_policy() -> bool:
    return os.getenv("FSA_LOCAL_USE_ARCH_POLICY", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def _policy_get(
    key: str,
    head_dim: int,
    block_size: int,
    default=None,
):
    """
    Arch+shape policy lookup with shape bucket override on top of default bucket.
    """
    if not _use_arch_policy():
        return default
    arch = _get_arch_bucket()
    shape = _get_shape_bucket(head_dim=head_dim, block_size=block_size)
    arch_tbl = _ARCH_POLICY_TABLE.get(arch, _ARCH_POLICY_TABLE["generic"])
    base = arch_tbl.get("default", {})
    shape_tbl = arch_tbl.get(shape, {})
    if key in shape_tbl:
        return shape_tbl[key]
    if key in base:
        return base[key]
    return default


def _resolve_launch_warps_stages(
    op: str,
    head_dim: int,
    block_size: int,
    default_warps: int,
    default_stages: int,
) -> tuple[int, int]:
    """
    P2.3 + P3.1:
      - arch-specialized launch table
      - optional Hopper deeper pipelining
    """
    launch = _policy_get("launch", head_dim=head_dim, block_size=block_size, default=None)
    warps, stages = int(default_warps), int(default_stages)
    if isinstance(launch, dict) and op in launch:
        try:
            w, s = launch[op]
            warps, stages = int(w), int(s)
        except Exception:
            pass

    hopper_async = os.getenv("FSA_LOCAL_HOPPER_ASYNC_PIPELINE", "auto").strip().lower()
    enable_async = False
    if hopper_async in ("1", "true", "yes", "on"):
        enable_async = True
    elif hopper_async in ("0", "false", "no", "off"):
        enable_async = False
    else:
        enable_async = (_get_arch_bucket() == "sm90")
    if enable_async and _get_arch_bucket() == "sm90":
        ov = os.getenv("FSA_LOCAL_HOPPER_PIPELINE_STAGES", "auto").strip().lower()
        if ov not in ("", "auto"):
            try:
                stages = max(stages, int(ov))
            except Exception:
                stages = max(stages, 4)
        else:
            stages = max(stages, 4)
    stages = max(1, min(stages, 8))
    warps = max(1, min(warps, 8))
    return warps, stages


def _resolve_hopper_pipeline_chunks(head_dim: int, block_size: int) -> int:
    """
    Resolve inner-loop chunk unrolling for dK/dV kernels.

    Env:
      FSA_LOCAL_HOPPER_PIPELINE_CHUNKS:
        - auto (default): 2 on sm90 for moderate tiles, else 1
        - positive int: clamped to [1, 4]
    """
    raw = os.getenv("FSA_LOCAL_HOPPER_PIPELINE_CHUNKS", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            v = int(raw)
            return max(1, min(v, 4))
        except Exception:
            return 1
    if _get_arch_bucket() != "sm90":
        return 1
    # Avoid high register pressure for very large tiles.
    if block_size >= 512 or head_dim >= 256:
        return 1
    return 2


def _resolve_head_tile(num_share_q_heads: int) -> int:
    """
    Resolve query-head tiling factor for forward/dQ local paths.

    Env:
      FSA_LOCAL_HEAD_TILE:
        - "auto" (default): largest power-of-two <= min(num_share_q_heads, 8)
        - integer >= 1: clamped to [1, num_share_q_heads]
    """
    if num_share_q_heads <= 1:
        return 1
    raw = os.getenv("FSA_LOCAL_HEAD_TILE", "auto").strip().lower()
    if raw in ("", "auto"):
        # Allow larger head batching by default; 16 is still safe on recent GPUs for these kernels.
        cap = min(num_share_q_heads, 16)
        tile = 1
        while tile * 2 <= cap:
            tile *= 2
        return max(1, tile)
    try:
        tile = int(raw)
    except Exception:
        tile = 1
    tile = max(1, min(tile, num_share_q_heads))
    # Keep it power-of-two for more stable Triton autotune/cache behavior.
    pow2 = 1
    while pow2 * 2 <= tile:
        pow2 *= 2
    return max(1, pow2)


def _resolve_bwd_dkdv_bq(head_dim: int, block_size: int) -> int:
    """
    Resolve BLOCK_SIZE_Q for dK/dV backward.
    Env: FSA_LOCAL_BWD_DKDV_BQ in {32,64,128,256} or 'auto'.
    """
    raw = os.getenv("FSA_LOCAL_BWD_DKDV_BQ", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            val = int(raw)
            if val in (32, 64, 128, 256):
                return val
        except Exception:
            pass
    policy_val = _policy_get("dkdv_bq", head_dim=head_dim, block_size=block_size, default=None)
    if policy_val in (32, 64, 128, 256):
        return int(policy_val)
    if block_size >= 512:
        return 32
    if block_size >= 256:
        return 64
    if head_dim >= 128:
        return 64
    return 128 if IS_HOPPER_GPU else 64


def _resolve_dkdv_mode(num_share_q_heads: int, head_dim: int, block_size: int) -> str:
    """
    Resolve dK/dV kernel mode.

    Env:
      FSA_LOCAL_DKDV_MODE:
        - auto (default): use gqa_fused when G>1, else legacy
        - legacy: previous per-qhead kernel (dk/dv over share-head dim then sum)
        - gqa_fused: one kernel instance per (batch, kv_head, kv_block), loops over
                     all shared q-heads and accumulates directly into dk/dv
    """
    raw = os.getenv("FSA_LOCAL_DKDV_MODE", "auto").strip().lower()
    if raw in ("legacy", "old", "baseline"):
        return "legacy"
    if raw in ("gqa", "gqa_fused", "gqa-fused", "fused"):
        return "gqa_fused"
    if raw in ("", "auto"):
        # P1.2 shape-family policy:
        # - keep fused mode for almost all production shapes
        # - allow optional tiny-shape legacy fallback for debugging/ablation
        tiny_legacy = os.getenv("FSA_LOCAL_POLICY_TINY_LEGACY", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if tiny_legacy and num_share_q_heads <= 1 and head_dim <= 64 and block_size <= 64:
            return "legacy"
        return "gqa_fused"
    return "gqa_fused"


def _resolve_dkdv_two_pass(num_share_q_heads: int, head_dim: int, block_size: int) -> bool:
    """
    Resolve whether to run dK/dV as two separate passes in fused GQA mode.

    Env:
      FSA_LOCAL_DKDV_TWO_PASS:
        - auto (default): enabled
        - 1/true/on: enabled
        - 0/false/off: disabled
    """
    raw = os.getenv("FSA_LOCAL_DKDV_TWO_PASS", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    policy_val = _policy_get("dkdv_two_pass", head_dim=head_dim, block_size=block_size, default=None)
    if isinstance(policy_val, bool):
        return bool(policy_val)
    # P1.2 shape-family policy:
    # - Hopper: prefer two-pass for larger tiles/groups to reduce register pressure.
    # - Others: default one-pass unless shape is clearly pressure-heavy.
    if IS_HOPPER_GPU:
        return bool((block_size >= 128) or (num_share_q_heads >= 4) or (head_dim >= 128))
    return bool((block_size >= 256) and (head_dim >= 128))


def _resolve_dkdv_schedule_mode(
    use_active_map: bool,
    total_active_q: int,
    active_ratio: float,
    active_work_items: int,
    batch_size: int,
    num_share_q_heads: int,
    head_dim: int,
    block_size: int,
) -> str:
    """
    Resolve dK/dV scheduling mode.

    Env:
      FSA_LOCAL_DKDV_SCHEDULE:
        - auto (default): policy-guided
          * use_active_map=False -> grid
          * use_active_map=True  -> worklist, with optional persistent auto-promotion
            when sparse/skewed workload thresholds are met
        - worklist: use explicit active worklist
        - persistent: persistent workers pop active worklist items via atomic queue
        - grid: static grid launch
    """
    raw = os.getenv("FSA_LOCAL_DKDV_SCHEDULE", "auto").strip().lower()
    if raw in ("persistent", "persist", "queue", "atomic_queue"):
        _record_dkdv_schedule_stat("persistent")
        return "persistent"
    if raw in ("worklist", "wl"):
        _record_dkdv_schedule_stat("worklist")
        return "worklist"
    if raw in ("grid", "static"):
        _record_dkdv_schedule_stat("grid")
        return "grid"
    policy_sched = _policy_get("dkdv_schedule_auto", head_dim=head_dim, block_size=block_size, default=None)
    # Only honor policy when it is explicit and valid.
    if policy_sched in ("grid", "worklist", "persistent"):
        if policy_sched == "persistent" and not use_active_map:
            _record_dkdv_schedule_stat("grid")
            return "grid"
        if policy_sched == "worklist" and not use_active_map:
            _record_dkdv_schedule_stat("grid")
            return "grid"
        _record_dkdv_schedule_stat(policy_sched)
        return policy_sched
    if use_active_map:
        # Conservative persistent auto-enable for skewed active-block workloads.
        # Explicit env FSA_LOCAL_DKDV_SCHEDULE still has highest priority.
        persist_auto_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_AUTO", "auto").strip().lower()
        if persist_auto_raw in ("1", "true", "yes", "on"):
            persist_auto_enable = True
        elif persist_auto_raw in ("0", "false", "no", "off"):
            persist_auto_enable = False
        else:
            policy_persist_auto = _policy_get(
                "persistent_auto_enable",
                head_dim=head_dim,
                block_size=block_size,
                default=None,
            )
            if isinstance(policy_persist_auto, bool):
                persist_auto_enable = bool(policy_persist_auto)
            else:
                persist_auto_enable = (_get_arch_bucket() == "sm90")

        if persist_auto_enable:
            ratio_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_ACTIVE_RATIO", "auto").strip().lower()
            minq_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_ACTIVE_Q", "auto").strip().lower()
            minw_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_WORK_ITEMS", "auto").strip().lower()
            minqpw_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_Q_PER_ITEM", "auto").strip().lower()
            minb_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_BATCH", "1").strip().lower()

            if ratio_raw not in ("", "auto"):
                try:
                    ratio_th = float(ratio_raw)
                except Exception:
                    ratio_th = 0.55
            else:
                ratio_pol = _policy_get(
                    "persistent_auto_active_ratio",
                    head_dim=head_dim,
                    block_size=block_size,
                    default=0.55,
                )
                try:
                    ratio_th = float(ratio_pol)
                except Exception:
                    ratio_th = 0.55
            ratio_th = max(0.0, min(1.0, ratio_th))

            if minq_raw not in ("", "auto"):
                try:
                    min_active_q = int(minq_raw)
                except Exception:
                    min_active_q = 262144
            else:
                minq_pol = _policy_get(
                    "persistent_auto_min_active_q",
                    head_dim=head_dim,
                    block_size=block_size,
                    default=(262144 if _get_arch_bucket() == "sm90" else 1048576),
                )
                try:
                    min_active_q = int(minq_pol)
                except Exception:
                    min_active_q = 262144 if _get_arch_bucket() == "sm90" else 1048576
            min_active_q = max(1, min_active_q)

            if minw_raw not in ("", "auto"):
                try:
                    min_work_items = int(minw_raw)
                except Exception:
                    min_work_items = 1024 if _get_arch_bucket() == "sm90" else 4096
            else:
                minw_pol = _policy_get(
                    "persistent_auto_min_work_items",
                    head_dim=head_dim,
                    block_size=block_size,
                    default=(1024 if _get_arch_bucket() == "sm90" else 4096),
                )
                try:
                    min_work_items = int(minw_pol)
                except Exception:
                    min_work_items = 1024 if _get_arch_bucket() == "sm90" else 4096
            min_work_items = max(1, min_work_items)

            if minqpw_raw not in ("", "auto"):
                try:
                    min_q_per_item = float(minqpw_raw)
                except Exception:
                    min_q_per_item = 16.0 if _get_arch_bucket() == "sm90" else 24.0
            else:
                minqpw_pol = _policy_get(
                    "persistent_auto_min_q_per_item",
                    head_dim=head_dim,
                    block_size=block_size,
                    default=(16.0 if _get_arch_bucket() == "sm90" else 24.0),
                )
                try:
                    min_q_per_item = float(minqpw_pol)
                except Exception:
                    min_q_per_item = 16.0 if _get_arch_bucket() == "sm90" else 24.0
            min_q_per_item = max(1.0, min_q_per_item)

            try:
                min_batch = max(1, int(minb_raw))
            except Exception:
                min_batch = 1

            active_work_items = max(0, int(active_work_items))
            if use_active_map and active_work_items <= 0:
                # Conservative estimate when worklist has not been built yet.
                active_work_items = max(1, int(total_active_q // max(1, num_share_q_heads)))
            q_per_item = float(total_active_q) / float(max(1, active_work_items))

            if active_ratio > ratio_th:
                _record_dkdv_schedule_stat("persistent_rejected_ratio")
            elif total_active_q < min_active_q:
                _record_dkdv_schedule_stat("persistent_rejected_min_active_q")
            elif batch_size < min_batch:
                _record_dkdv_schedule_stat("persistent_rejected_min_batch")
            elif active_work_items < min_work_items:
                _record_dkdv_schedule_stat("persistent_rejected_min_work_items")
            elif q_per_item < min_q_per_item:
                _record_dkdv_schedule_stat("persistent_rejected_min_q_per_item")
            else:
                _record_dkdv_schedule_stat("persistent")
                return "persistent"
        _record_dkdv_schedule_stat("worklist")
        return "worklist"
    _record_dkdv_schedule_stat("grid")
    return "grid"


def _resolve_fwd_packed_gqa(num_share_q_heads: int, head_dim: int, block_size: int) -> bool:
    """
    Resolve whether to use packed-GQA grouped forward execution (avoid head replication).
    """
    if int(num_share_q_heads) <= 1:
        return False
    raw = os.getenv("FSA_LOCAL_FWD_PACKED_GQA", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    pol = _policy_get("fwd_packed_gqa", head_dim=head_dim, block_size=block_size, default=None)
    if isinstance(pol, bool):
        return bool(pol)
    return num_share_q_heads > 1


def _resolve_nsa_packed_gqa(num_share_q_heads: int, head_dim: int, block_size: int) -> bool:
    """
    Resolve packed-GQA mode for NSA-style forward hotspot.
    """
    if int(num_share_q_heads) <= 1:
        return False
    raw = os.getenv("FSA_LOCAL_NSA_PACKED_GQA", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    pol = _policy_get("nsa_packed_gqa", head_dim=head_dim, block_size=block_size, default=None)
    if isinstance(pol, bool):
        return bool(pol)
    return _resolve_fwd_packed_gqa(
        num_share_q_heads=num_share_q_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


def _resolve_dq_packed_gqa(num_share_q_heads: int, head_dim: int, block_size: int) -> bool:
    """
    Resolve packed-GQA mode for dQ hotspot.
    """
    if int(num_share_q_heads) <= 1:
        return False
    raw = os.getenv("FSA_LOCAL_DQ_PACKED_GQA", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return _resolve_fwd_packed_gqa(
        num_share_q_heads=num_share_q_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


def _resolve_effective_num_blocks(
    valid_lens_all: torch.Tensor,
    num_blocks_full: int,
    real_num_blocks: int,
    head_dim: int,
    block_size: int,
) -> int:
    """
    Exact block-tail pruning policy:
    only trims trailing KV blocks with zero routed queries across all KV heads.
    """
    _record_block_prune_stat("effective_calls")
    if num_blocks_full <= 1 or valid_lens_all.numel() == 0:
        return max(1, int(num_blocks_full))

    mode = os.getenv("FSA_LOCAL_BLOCK_PRUNING_MODE", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return int(num_blocks_full)

    real_num_blocks = max(1, min(int(real_num_blocks), int(num_blocks_full)))
    valid_view = valid_lens_all[:, :real_num_blocks]
    active_any = torch.any(valid_view > 0, dim=0)
    nz = torch.nonzero(active_any, as_tuple=False)
    if int(nz.numel()) <= 0:
        eff = 1
    else:
        # `nonzero(as_tuple=False)` returns shape [N, 1] for 1D inputs; index explicitly.
        eff = int(nz[-1, 0].to(dtype=torch.int32).cpu().item()) + 1
    eff = max(1, min(real_num_blocks, eff))

    tail_pruned = max(0, int(num_blocks_full) - int(eff))
    if tail_pruned > 0:
        _record_block_prune_stat("tail_pruned_blocks", tail_pruned)

    if mode in ("1", "true", "yes", "on"):
        return eff

    tail = int(num_blocks_full) - int(eff)
    if tail <= 0:
        return int(real_num_blocks)

    min_tail_raw = os.getenv("FSA_LOCAL_BLOCK_PRUNE_MIN_TAIL", "auto").strip().lower()
    min_ratio_raw = os.getenv("FSA_LOCAL_BLOCK_PRUNE_MIN_RATIO", "auto").strip().lower()

    if min_tail_raw not in ("", "auto"):
        try:
            min_tail = max(0, int(min_tail_raw))
        except Exception:
            min_tail = 8
    else:
        min_tail_pol = _policy_get("block_prune_min_tail", head_dim=head_dim, block_size=block_size, default=8)
        try:
            min_tail = max(0, int(min_tail_pol))
        except Exception:
            min_tail = 8

    if min_ratio_raw not in ("", "auto"):
        try:
            min_ratio = float(min_ratio_raw)
        except Exception:
            min_ratio = 0.05
    else:
        min_ratio_pol = _policy_get("block_prune_min_ratio", head_dim=head_dim, block_size=block_size, default=0.05)
        try:
            min_ratio = float(min_ratio_pol)
        except Exception:
            min_ratio = 0.05
    min_ratio = max(0.0, min(1.0, min_ratio))

    if tail < min_tail and (tail / float(max(1, num_blocks_full))) < min_ratio:
        return int(real_num_blocks)
    return int(eff)


def _resolve_bwd_sequence_parallel(
    expected_seqs: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    block_size: int,
) -> bool:
    """
    Resolve dedicated sequence-parallel backward mode.
    """
    if expected_seqs <= 1:
        return False

    raw = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True

    pol = _policy_get("bwd_seq_parallel", head_dim=head_dim, block_size=block_size, default=None)
    if isinstance(pol, bool) and not pol:
        return False

    min_seqs_raw = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL_MIN_SEQS", "2").strip().lower()
    min_tokens_raw = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL_MIN_TOKENS", "2048").strip().lower()
    try:
        min_seqs = max(2, int(min_seqs_raw))
    except Exception:
        min_seqs = 2
    try:
        min_tokens = max(1, int(min_tokens_raw))
    except Exception:
        min_tokens = 2048
    if expected_seqs < min_seqs:
        return False
    return max(int(max_seqlen_q), int(max_seqlen_k)) >= min_tokens


def _resolve_bwd_sequence_parallel_streams(
    q_device: torch.device,
    expected_seqs: int,
    head_dim: int,
    block_size: int,
) -> int:
    """
    Resolve number of CUDA streams for sequence-parallel backward.
    """
    if expected_seqs <= 1:
        return 1
    raw = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL_STREAMS", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            return max(1, min(int(raw), expected_seqs))
        except Exception:
            return 1

    pol = _policy_get("bwd_seq_parallel_streams", head_dim=head_dim, block_size=block_size, default=None)
    if isinstance(pol, int) and pol > 0:
        return max(1, min(int(pol), expected_seqs))

    try:
        sm_count = int(torch.cuda.get_device_properties(q_device).multi_processor_count)
    except Exception:
        sm_count = 80
    if sm_count >= 120:
        return max(1, min(expected_seqs, 4))
    if sm_count >= 80:
        return max(1, min(expected_seqs, 3))
    return max(1, min(expected_seqs, 2))


def _resolve_active_map_ratio_threshold(head_dim: int, block_size: int) -> float:
    """
    Resolve active-map enable threshold for auto mode.
    Lower threshold means stronger pruning before enabling active-map path.
    """
    raw = os.getenv("FSA_LOCAL_ACTIVE_MAP_RATIO_THRESHOLD", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            v = float(raw)
            return max(0.0, min(1.0, v))
        except Exception:
            pass
    pol = _policy_get(
        "active_map_ratio_threshold",
        head_dim=head_dim,
        block_size=block_size,
        default=0.92,
    )
    try:
        return max(0.0, min(1.0, float(pol)))
    except Exception:
        return 0.92


def _resolve_dkdv_persistent_chunk(
    head_dim: int,
    block_size: int,
    active_ratio: float,
    avg_active_q_per_item: float,
    num_work_items: int,
) -> int:
    """
    Resolve per-worker dequeue chunk for persistent queue kernel.
    """
    raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_CHUNK", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            v = int(raw)
            return max(1, min(v, 8))
        except Exception:
            return 1

    pol = _policy_get(
        "persistent_chunk",
        head_dim=head_dim,
        block_size=block_size,
        default=None,
    )
    if isinstance(pol, int) and pol > 0:
        return max(1, min(int(pol), 8))

    if _get_arch_bucket() == "sm90":
        if active_ratio < 0.30 and avg_active_q_per_item >= 32.0 and num_work_items >= 4096:
            return 4
        if active_ratio < 0.55 and avg_active_q_per_item >= 16.0 and num_work_items >= 1024:
            return 2
        if active_ratio < 0.75 and avg_active_q_per_item >= 8.0 and num_work_items >= 512:
            return 2
    return 1


def _resolve_dkdv_persistent_workers(
    q_device: torch.device,
    num_work_items: int,
    avg_active_q_per_item: float,
    head_dim: int,
    block_size: int,
) -> int:
    """
    Resolve persistent worker count from env/policy and GPU SM count.
    """
    workers_env = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_WORKERS", "auto").strip().lower()
    if workers_env not in ("", "auto"):
        try:
            return max(1, min(int(workers_env), max(1, num_work_items)))
        except Exception:
            return max(1, min(1, max(1, num_work_items)))

    try:
        sm_count = int(torch.cuda.get_device_properties(q_device).multi_processor_count)
    except Exception:
        sm_count = 80

    factor_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_WORKERS_FACTOR", "auto").strip().lower()
    if factor_raw not in ("", "auto"):
        try:
            factor = float(factor_raw)
        except Exception:
            factor = 2.0 if _get_arch_bucket() == "sm90" else 1.0
    else:
        pol = _policy_get(
            "persistent_workers_per_sm",
            head_dim=head_dim,
            block_size=block_size,
            default=(2 if _get_arch_bucket() == "sm90" else 1),
        )
        try:
            factor = float(pol)
        except Exception:
            factor = 2.0 if _get_arch_bucket() == "sm90" else 1.0
    factor = max(0.5, min(factor, 8.0))

    max_workers_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MAX_WORKERS", "auto").strip().lower()
    if max_workers_raw not in ("", "auto"):
        try:
            max_workers = max(1, int(max_workers_raw))
        except Exception:
            max_workers = num_work_items
    else:
        max_workers = num_work_items

    target_items_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_TARGET_ITEMS_PER_WORKER", "auto").strip().lower()
    if target_items_raw not in ("", "auto"):
        try:
            target_items = max(1, int(target_items_raw))
        except Exception:
            target_items = 16 if _get_arch_bucket() == "sm90" else 24
    else:
        target_items_pol = _policy_get(
            "persistent_target_items_per_worker",
            head_dim=head_dim,
            block_size=block_size,
            default=(16 if _get_arch_bucket() == "sm90" else 24),
        )
        try:
            target_items = max(1, int(target_items_pol))
        except Exception:
            target_items = 16 if _get_arch_bucket() == "sm90" else 24
    if avg_active_q_per_item < 8.0:
        target_items = max(target_items, 32)

    min_items_raw = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_ITEMS_PER_WORKER", "auto").strip().lower()
    if min_items_raw not in ("", "auto"):
        try:
            min_items_per_worker = max(1, int(min_items_raw))
        except Exception:
            min_items_per_worker = 2 if _get_arch_bucket() == "sm90" else 4
    else:
        min_items_pol = _policy_get(
            "persistent_min_items_per_worker",
            head_dim=head_dim,
            block_size=block_size,
            default=(2 if _get_arch_bucket() == "sm90" else 4),
        )
        try:
            min_items_per_worker = max(1, int(min_items_pol))
        except Exception:
            min_items_per_worker = 2 if _get_arch_bucket() == "sm90" else 4

    by_sm = int(math.ceil(sm_count * factor))
    by_target_items = int(math.ceil(float(max(1, num_work_items)) / float(target_items)))
    by_min_items = int(math.ceil(float(max(1, num_work_items)) / float(min_items_per_worker)))
    target = max(1, min(by_sm, by_min_items))
    target = max(target, by_target_items)
    return max(1, min(target, max(1, num_work_items), max_workers))


def _resolve_bwd_dq_bq(head_dim: int, block_size: int) -> int:
    """
    Resolve BLOCK_SIZE_Q for dQ backward compute.
    Env: FSA_LOCAL_BWD_DQ_BQ in {32,64,128,256} or 'auto'.
    """
    raw = os.getenv("FSA_LOCAL_BWD_DQ_BQ", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            val = int(raw)
            if val in (32, 64, 128, 256):
                return val
        except Exception:
            pass
    policy_val = _policy_get("dq_bq", head_dim=head_dim, block_size=block_size, default=None)
    if policy_val in (32, 64, 128, 256):
        return int(policy_val)
    if block_size >= 512:
        return 32
    if block_size >= 256:
        return 64
    if head_dim >= 128:
        return 64
    return 128 if IS_HOPPER_GPU else 64


def _resolve_bwd_dq_num_q_blocks(block_size: int) -> int:
    """
    Resolve number of q-subloops in dQ compute kernel.
    Env: FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS (positive int) or 'auto'.
    """
    raw = os.getenv("FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            val = int(raw)
            if val > 0:
                return val
        except Exception:
            pass
    policy_val = _policy_get("dq_num_q_blocks", head_dim=64, block_size=block_size, default=None)
    if isinstance(policy_val, int) and policy_val > 0:
        return int(policy_val)
    if block_size >= 512:
        return 4 if IS_HOPPER_GPU else 2
    if block_size >= 256:
        return 8 if IS_HOPPER_GPU else 4
    return 8 if IS_HOPPER_GPU else 4


def _maybe_precompact_kv_for_seq(
    k_seq: torch.Tensor,          # [1, Tk, HK, D]
    v_seq: torch.Tensor,          # [1, Tk, HK, D]
    bi_seq: torch.Tensor,         # [1, Tq_active, HK, S]
    block_size: int,
):
    """
    P1.1 preprocess-compaction stage:
    compact selected KV chapters once and remap block ids before fused forward.
    """
    _FWD_PRECOMPACT_STATS["attempts"] = int(_FWD_PRECOMPACT_STATS.get("attempts", 0)) + 1
    mode = os.getenv("FSA_LOCAL_FWD_PRECOMPACT", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False

    tk = int(k_seq.shape[1])
    if tk <= 0:
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False
    # Current compaction path assumes chapter-aligned memory.
    # If TK is not divisible by block_size, last chapter is partial and token gather
    # would need padding-aware handling; skip for safety.
    if (tk % block_size) != 0:
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False
    m_blocks = int((tk + block_size - 1) // block_size)
    if m_blocks <= 1:
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False

    valid = (bi_seq >= 0) & (bi_seq < m_blocks)
    if not bool(torch.any(valid)):
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False

    selected = torch.unique(bi_seq[valid].to(torch.int64), sorted=True)
    num_selected = int(selected.numel())
    if num_selected <= 0 or num_selected >= m_blocks:
        _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
        return k_seq, v_seq, bi_seq, False

    min_blocks_raw = os.getenv("FSA_LOCAL_FWD_PRECOMPACT_MIN_BLOCKS", "64").strip().lower()
    min_savings_raw = os.getenv("FSA_LOCAL_FWD_PRECOMPACT_MIN_SAVINGS", "0.15").strip().lower()
    try:
        min_blocks = max(1, int(min_blocks_raw))
    except Exception:
        min_blocks = 64
    try:
        min_savings = float(min_savings_raw)
    except Exception:
        min_savings = 0.15
    savings = 1.0 - (float(num_selected) / float(max(1, m_blocks)))
    if mode in ("", "auto"):
        if m_blocks < min_blocks or savings < min_savings:
            _FWD_PRECOMPACT_STATS["skipped"] = int(_FWD_PRECOMPACT_STATS.get("skipped", 0)) + 1
            return k_seq, v_seq, bi_seq, False

    lut = torch.full((m_blocks,), -1, dtype=torch.int32, device=k_seq.device)
    lut[selected] = torch.arange(num_selected, device=k_seq.device, dtype=torch.int32)
    bi_new = torch.where(
        valid,
        lut[bi_seq.to(torch.int64)],
        torch.full_like(bi_seq, -1),
    ).to(torch.int32)

    offs = torch.arange(block_size, device=k_seq.device, dtype=torch.int64).view(1, block_size)
    tok = (selected.view(-1, 1) * block_size + offs).reshape(-1)
    k_new = k_seq.index_select(1, tok)
    v_new = v_seq.index_select(1, tok)

    _FWD_PRECOMPACT_STATS["applied"] = int(_FWD_PRECOMPACT_STATS.get("applied", 0)) + 1
    return k_new, v_new, bi_new, True


@triton.jit
def block_to_token_kernel(
    topk_idx_ptr,
    result_ptr,
    N_token,
    K,
    min_block_id,
    max_block_id,
    padding_value,
    ts_h,
    ts_b,
    ts_n,
    rs_h,
    rs_b,
    rs_n,
    num_q_loops: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # token index i
    pid_h = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)  # [0, 1, ..., K-1]

    offs_q = tl.arange(0, num_q_loops)

    pid_j = pid * num_q_loops + offs_q

    topk_idx_offset = pid_h * ts_h + pid_j[None, :] * K + offs[:, None]
    block_ids = tl.load(
        topk_idx_ptr + topk_idx_offset, mask=(pid_j < N_token)[None, :] & (offs < K)[:, None], other=padding_value
    )

    result_ptrs = result_ptr + pid_h * rs_h + block_ids * N_token + pid_j[None, :]

    mask = (block_ids >= 0) & (block_ids != padding_value) & (pid_j < N_token)[None, :]
    tl.store(result_ptrs, pid_j[None, :], mask=mask)


def build_block_to_token_triton(
    result: torch.Tensor, topk_idx: torch.Tensor, min_block_id: int, max_block_id: int, padding_value: int = -1
):
    """
    Args:
        topk_idx: [num_heads, N_token, TopK], block indices per token, padded with padding_value for invalid blocks
        num_blocks: int
        padding_value: int

    Returns:
        result: [num_blocks, N_token], token indices per block, padded by padding_value
    """
    assert topk_idx.ndim == 3
    assert padding_value == -1
    num_heads, N_token, TopK = topk_idx.shape

    # 每个 token，每个head 一个 program
    num_q_loops = 4
    grid = (triton.cdiv(N_token, num_q_loops), num_heads)
    BLOCK_K = triton.next_power_of_2(TopK)
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="index_map",
        head_dim=64,
        block_size=max(32, int(BLOCK_K)),
        default_warps=2,
        default_stages=3,
    )
    block_to_token_kernel[grid](
        topk_idx,
        result,
        N_token,
        TopK,
        min_block_id,
        max_block_id,
        padding_value,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        num_q_loops,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return result


@triton.jit
def reduce_kernel(
    lse_ptr,  # float32 [H, N]
    m_ij_ptr,  # float32 [H, B, N]
    l_ij_first_ptr,  # float32 [H, 1, N]
    l_ij_rest_ptr,  # float32 [H, B, N]
    m_ij_last_ptr,  # float32 [H, N]
    o_ptr,  # o: n x h x d
    o_tiles_first_ptr,  # o_tiles: n x h x 1 x d
    o_tiles_rest_ptr,  # o_tiles: n x h x b x d
    acc_o_scales_first_ptr,  # acc_o_scales: n x h x 1
    acc_o_scales_rest_ptr,  # acc_o_scales: n x h x b
    t_ptr,  # topk_idx: h x n x k
    token_index_mapping_ptr,
    start_head_id,
    num_qz_loop,
    pid_q_offset,
    query_start_idx,
    query_tokens_count,
    TOPK,
    total_len,
    HEAD_DIM,
    # stride
    stride_lse_h,
    stride_lse_n,
    stride_m_ij_h,
    stride_m_ij_b,
    stride_m_ij_n,
    stride_l_ij_fh,
    stride_l_ij_fb,
    stride_l_ij_fn,
    stride_l_ij_rh,
    stride_l_ij_rb,
    stride_l_ij_rn,
    stride_on,
    stride_oh,
    stride_od,
    stride_otfh,
    stride_otfb,
    stride_otfn,
    stride_otfd,
    stride_otrh,
    stride_otrb,
    stride_otrn,
    stride_otrd,
    stride_acc_fh,
    stride_acc_fb,
    stride_acc_fn,
    stride_acc_rh,
    stride_acc_rb,
    stride_acc_rn,
    stride_th,
    stride_tn,
    stride_tk,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_qy = tl.program_id(0)
    pid_q = tl.program_id(1) + pid_q_offset  # token

    pid_q_local = pid_q + pid_qy * num_qz_loop
    if pid_q_local >= query_tokens_count:
        return
    pid_q_j = query_start_idx + pid_q_local
    if pid_q_j >= total_len:
        return
    t_ptr_j = t_ptr + pid_q_j * stride_tn

    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = o_ptr + pid_q_j * stride_on + off_d
    last_acc_o = tl.load(o_ptrs, mask=off_d < HEAD_DIM, other=0.0)
    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    acc_o += last_acc_o

    lse_ptrs = lse_ptr + pid_q_j * stride_lse_n
    # Load lse
    lse = tl.load(lse_ptrs, mask=pid_q_j < total_len, other=float("-inf"))

    # the stride is 1 for m_ij_last
    m_ij_last = tl.load(m_ij_last_ptr + pid_q_j)

    for block_id in range(TOPK):
        t = tl.load(t_ptr_j + block_id * stride_tk, mask=block_id < TOPK, other=-1)
        if t != -1:
            if t == 0:
                real_block_pos = 0
                l_ij_ptr = l_ij_first_ptr
                o_tiles_ptr = o_tiles_first_ptr
                acc_o_scales_ptr = acc_o_scales_first_ptr
                stride_l_ij_b = stride_l_ij_fb
                stride_l_ij_n = stride_l_ij_fn
                stride_acc_b = stride_acc_fb
                stride_acc_n = stride_acc_fn
                stride_otb = stride_otfb
                stride_otn = stride_otfn
            else:
                real_block_pos = t - 1
                l_ij_ptr = l_ij_rest_ptr
                o_tiles_ptr = o_tiles_rest_ptr
                acc_o_scales_ptr = acc_o_scales_rest_ptr
                stride_l_ij_b = stride_l_ij_rb
                stride_l_ij_n = stride_l_ij_rn
                stride_acc_b = stride_acc_rb
                stride_acc_n = stride_acc_rn
                stride_otb = stride_otrb
                stride_otn = stride_otrn

            # init pointers
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + t.to(tl.int64) * stride_tim_b + (pid_q_j) * stride_tim_n
            )
            real_token_index = tl.load(token_index_mapping_ptrs)

            m_ij = tl.load(
                m_ij_ptr + t * stride_m_ij_b + pid_q_j * stride_m_ij_n, mask=pid_q_j < total_len, other=float("-inf")
            )
            l_ij = tl.load(
                l_ij_ptr + real_block_pos * stride_l_ij_b + real_token_index * stride_l_ij_n,
                mask=real_token_index < total_len,
                other=0.0,
            )
            delta = lse - m_ij

            log_delta = tl.exp2(delta) + l_ij

            # Update lse
            lse = m_ij + tl.log2(log_delta)

            o_tiles_ptrs = (
                o_tiles_ptr + real_block_pos.to(tl.int64) * stride_otb + (real_token_index) * stride_otn + off_d
            )
            acc_o_scales_ptrs = acc_o_scales_ptr + real_block_pos * stride_acc_b + (real_token_index) * stride_acc_n

            o_tiles = tl.load(o_tiles_ptrs, mask=off_d < HEAD_DIM, other=0.0)
            acc_o_scales_tiles = tl.load(acc_o_scales_ptrs)
            acc_o = o_tiles + acc_o * acc_o_scales_tiles

    # final scale
    acc_o = acc_o * tl.exp2(m_ij_last - lse)
    tl.store(o_ptrs, acc_o, mask=off_d < HEAD_DIM)

    # Store back
    tl.store(
        lse_ptrs,
        lse,
        mask=pid_q_j < total_len,
    )


@triton.jit
def qk_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    m_i_tiles_ptr,  # m_i: h x b x n
    selected_tokens_ptr,  # selected_tokens: sum(valid_lens),
    valid_lens_ptr,  # valid_lens: (h x b),
    valid_start_indices_ptr,  # valid_start_indices: (h x b),
    num_heads,
    num_blocks,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    HEAD_DIM,
    # sm_scale
    sm_scale,
    num_q_blocks,
    num_b_blocks,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_m_i_tiles_h,
    stride_m_i_tiles_b,
    stride_m_i_tiles_n,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_block_grid = tl.program_id(0) // num_heads  # block id
    head_id = tl.program_id(0) % num_heads
    pid_q = tl.program_id(1)  # token

    # get q k start and len after rmpad
    k_len = tl.load(cu_seqlens_k + 1)
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + head_id * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )

    for bb in range(num_b_blocks):
        pid_block = bb + pid_block_grid * num_b_blocks

        start_id = tl.load(valid_start_indices_ptr + head_id * num_blocks + pid_block)
        valid_tokens = tl.load(valid_lens_ptr + head_id * num_blocks + pid_block)
        if pid_q * BLOCK_SIZE_Q < valid_tokens:

            c = pid_block * BLOCK_SIZE_K

            # load k
            k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")

            off_k = tl.arange(0, BLOCK_SIZE_K)
            off_d = tl.arange(0, BLOCK_SIZE_D)
            for j in range(num_q_blocks):
                pid_q_j = pid_q * num_q_blocks + j
                # Enable early return
                if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
                    # one thread block for one KV block, a subset of selected tokens
                    st_offs = start_id + (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
                    # st should be in shape [BLOCK_SIZE_Q]
                    st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

                    st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
                    # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
                    q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd
                    q_ptrs = q_ptr + head_id * stride_qh + q_ptrs_off
                    # load q
                    q_mask = (st != -1)[:, None] & (off_d < HEAD_DIM)[None, :]
                    q = tl.load(q_ptrs, mask=q_mask, other=0)
                    # compute qk
                    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
                    qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))
                    # [BLOCK_SIZE_Q, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_Q, BLOCK_SIZE_K]
                    qk += tl.dot(q, k) * qk_scale

                    m_i = tl.max(qk, axis=1)

                    m_i_tiles_ptrs = (
                        m_i_tiles_ptr
                        + head_id * stride_m_i_tiles_h
                        + pid_block * stride_m_i_tiles_b
                        + st * stride_m_i_tiles_n
                    )
                    tl.store(m_i_tiles_ptrs, m_i, mask=(st != -1))


@triton.jit
def forward_kernel_opt(
    q_ptr,
    k_ptr,
    v_ptr,  # V: n x h x d
    o_tiles_ptr,  # O: n x h x b x d
    acc_o_scales_ptr,  # acc_o_scales: h x b x n
    m_ij_tiles_ptr,
    l_ij_ptr,  # h x b x n
    token_index_mapping_ptr,
    selected_tokens_ptr,  # selected_tokens: sum(valid_lens),
    valid_lens_ptr,  # valid_lens: (h x b),
    valid_start_indices_ptr,  # valid_start_indices: (h x b),
    min_block_id,
    cur_max_valid_tokens,
    num_heads,
    num_blocks,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    HEAD_DIM,
    # sm_scale
    sm_scale,
    num_q_blocks,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_oth,
    stride_otb,
    stride_otn,
    stride_otd,
    stride_acc_oh,
    stride_acc_ob,
    stride_acc_on,
    stride_m_ij_tiles_h,
    stride_m_ij_tiles_b,
    stride_m_ij_tiles_n,
    stride_l_ij_h,
    stride_l_ij_b,
    stride_l_ij_n,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    # get batch id and head id
    pid_block = tl.program_id(0) // num_heads  # block id
    head_id = tl.program_id(0) % num_heads
    pid_q = tl.program_id(1)  # token
    # seq packing is not supported yet
    q_start = 0
    k_start = 0

    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + head_id * num_blocks + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + head_id * num_blocks + pid_block)
    if num_q_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (min_block_id + pid_block) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + head_id * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    # load k
    k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")

    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + head_id * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # load v
    v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    for j in range(num_q_blocks):
        pid_q_j = pid_q * num_q_blocks + j
        if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
            # one thread block for one KV block, a subset of selected tokens
            st_offs = start_id + (q_start + pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            # st should be in shape [BLOCK_SIZE_Q]
            st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

            st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

            # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
            q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd

            # load m_i
            mask = st != -1

            m_ij_tiles_ptrs = (
                m_ij_tiles_ptr
                + head_id * stride_m_ij_tiles_h
                + (q_start + st) * stride_m_ij_tiles_n
                + (pid_block + min_block_id) * stride_m_ij_tiles_b
            )
            m_ij = tl.load(m_ij_tiles_ptrs, mask=mask, other=float("-inf"))

            m_ij_tiles_prev_ptrs = (
                m_ij_tiles_ptr
                + head_id * stride_m_ij_tiles_h
                + (q_start + st) * stride_m_ij_tiles_n
                + (pid_block + min_block_id - 1) * stride_m_ij_tiles_b
            )
            m_ij_prev = tl.load(m_ij_tiles_prev_ptrs, mask=mask & (pid_block + min_block_id > 0), other=float("-inf"))

            m_i_minus_m_ij = m_ij_prev - m_ij

            q_ptrs = q_ptr + q_start * stride_qn + head_id * stride_qh + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)

            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))

            # [BLOCK_SIZE_Q, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            qk_scale = sm_scale * 1.44269504
            qk += tl.dot(q, k) * qk_scale

            # init statistics
            acc_o_buffer = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)

            # load m_ij and compute l_ij
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # load token index mapping
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + (st) * stride_tim_n + (pid_block + min_block_id) * stride_tim_b
            )
            token_index_mapping = tl.load(token_index_mapping_ptrs, mask=mask, other=-1)

            l_ij_ptrs = (
                l_ij_ptr
                + head_id * stride_l_ij_h
                + (q_start + token_index_mapping) * stride_l_ij_n
                + (pid_block) * stride_l_ij_b
            )
            tl.store(l_ij_ptrs, l_ij, mask=mask)
            # scale acc_o
            if pid_block + min_block_id == 0:
                acc_o_scale = tl.full((BLOCK_SIZE_Q,), 1.0, dtype=tl.float32)
            else:
                acc_o_scale = tl.exp2(m_i_minus_m_ij)

            tl.store(
                acc_o_scales_ptr
                + head_id * stride_acc_oh
                + (pid_block) * stride_acc_ob
                + (q_start + token_index_mapping) * stride_acc_on,
                acc_o_scale,
                mask=(st != -1),
            )

            p = p.to(v.dtype)
            acc_o_buffer = tl.dot(p, v)

            o_ptrs_off = token_index_mapping[:, None] * stride_otn + off_d[None, :] * stride_otd
            o_ptrs = o_tiles_ptr + head_id * stride_oth + o_ptrs_off + (pid_block).to(tl.int64) * stride_otb
            tl.store(o_ptrs, acc_o_buffer.to(o_tiles_ptr.dtype.element_ty), mask=q_mask)


def _topk_sparse_attention_fwd_opt(
    q: torch.Tensor,  # [total_len, num_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal=True,
):
    """
        Sequence packing is handled at wrapper level for multi-sequence varlen inputs.
        Includes a single-sequence fast path and optional stream-parallel multi-sequence dispatch.
    """
    o = torch.empty_like(q)
    total_len, num_heads, _ = q.shape
    lse = torch.empty((num_heads, total_len), dtype=torch.float32, device=q.device)

    seq_meta, cu_q_local_all, cu_k_local_all = _build_seq_dispatch_meta(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        device=cu_seqlens_q.device,
    )
    # OPT-12: Multi-sequence wrapper loop elimination (universal packed timeline path).
    if len(seq_meta) > 1:
        packed = _pack_varlen_unified_timeline(
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_size=block_size,
        )
        if packed is not None:
            q_u = packed["q_u"]
            k_u = packed["k_u"]
            v_u = packed["v_u"]
            topk_u = packed["topk_u"]
            cu_u = packed["cu_u"]
            packed_meta = packed["packed_meta"]

            o_u, lse_u, _perm_flat = _topk_sparse_attention_fwd_opt_per_seq(
            q=q_u,
            k=k_u,
            v=v_u,
            topk_idx=topk_u,
            block_size=block_size,
            cu_seqlens_q=cu_u,
            cu_seqlens_k=cu_u,
            max_seqlen_q=int(q_u.shape[0]),
            max_seqlen_k=int(k_u.shape[0]),
            sm_scale=sm_scale,
            causal=causal,
            )
            _unpack_varlen_unified_q(
                packed_meta=packed_meta,
                src_u=o_u,
                dst=o,
                by_head_first=False,
            )
            _unpack_varlen_unified_q(
                packed_meta=packed_meta,
                src_u=lse_u,
                dst=lse,
                by_head_first=True,
            )
            # Flat multi-seq backward rebuilds metadata from packed timeline when needed.
            return o, lse, None
        raise RuntimeError("FSA multi-seq flat forward packing failed unexpectedly.")

    permute_results = []
    if len(seq_meta) == 1:
        q_start, q_end, k_start, k_end, q_len, k_len = seq_meta[0]
        cu_q_local = cu_q_local_all[0]
        cu_k_local = cu_k_local_all[0]
        o_seq, lse_seq, permute_results_seq = _topk_sparse_attention_fwd_opt_per_seq(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            block_size,
            cu_q_local,
            cu_k_local,
            q_len,
            k_len,
            sm_scale,
            causal,
        )
        o[q_start:q_end] = o_seq
        lse[:, q_start:q_end] = lse_seq
        permute_results.append(permute_results_seq)
        return o, lse, permute_results

    raise RuntimeError("Unexpected forward wrapper state: multi-seq path must run via flat packed timeline.")


@triton.jit
def index_mapping_kernel(
    token_index_mapping_ptr,
    selected_tokens_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    stride_im_h,
    stride_im_b,
    stride_im_n,
    num_heads,
    num_blocks,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_hb = tl.program_id(0)
    pid_h = pid_hb // num_blocks
    pid_b = pid_hb % num_blocks
    pid_n = tl.program_id(1)

    offs_q = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_K + offs_q

    start_id = tl.load(valid_start_indices_ptr + pid_h * num_blocks + pid_b)
    valid_tokens = tl.load(valid_lens_ptr + pid_h * num_blocks + pid_b)

    st_offs = start_id + offs_n
    # st should be in shape [BLOCK_SIZE_K]
    st_mask = offs_n < valid_tokens

    st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

    token_im_ptrs = token_index_mapping_ptr + pid_h * stride_im_h + pid_b * stride_im_b + st * stride_im_n

    tl.store(token_im_ptrs, offs_n, mask=st_mask)


def index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks):
    max_tokens = int(valid_lens.max().to(dtype=torch.int32).cpu().tolist()) if valid_lens.numel() > 0 else 0
    if max_tokens <= 0:
        return
    num_heads = int(valid_lens.shape[0]) if valid_lens.ndim == 2 else 1
    BLOCK_SIZE_K = 1024
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="index_map",
        head_dim=64,
        block_size=BLOCK_SIZE_K,
        default_warps=2,
        default_stages=3,
    )
    grid = (num_heads * num_blocks, triton.cdiv(max_tokens, BLOCK_SIZE_K))

    index_mapping_kernel[grid](
        token_index_mapping,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        num_heads,
        num_blocks,
        BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def online_softmax(
    q_tile,
    k_tile,
    m_i_cur_tiles,
    valid_topk_idx_permuted_tile,
    valid_lens,
    valid_start_indices,
    compute_min_block_id,
    cur_max_valid_tokens,
    block_size,
    num_blocks,
    head_tile,
    head_dim,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
):

    # launch kernel
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_q_blocks = 16 if IS_HOPPER_GPU else 8
    num_b_blocks = 1
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="fwd_qk",
        head_dim=head_dim,
        block_size=block_size,
        default_warps=8,
        default_stages=3,
    )
    grid_qk = lambda META: (
        triton.cdiv(num_blocks, num_b_blocks),
        triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_q_blocks),
    )
    qk_kernel[grid_qk](
        q_tile,
        k_tile,
        m_i_cur_tiles,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        head_tile,
        num_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        head_dim,
        sm_scale,
        num_q_blocks,
        num_b_blocks,
        q_tile.stride(0),
        q_tile.stride(1),
        q_tile.stride(2),
        k_tile.stride(0),
        k_tile.stride(1),
        k_tile.stride(2),
        m_i_cur_tiles.stride(0),
        m_i_cur_tiles.stride(1),
        m_i_cur_tiles.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    m_ij_tiles = m_i_cur_tiles.cummax(dim=1).values
    m_ij_last = m_ij_tiles[:, -1]

    return m_ij_tiles, m_ij_last


def qkv_kernel(
    q_tile,
    k_tile,
    v_tile,
    o_tiles,
    acc_o_scales,
    m_ij_tiles,
    l_ij,
    token_index_mapping,
    valid_topk_idx_permuted_tile,
    valid_lens,
    valid_start_indices,
    compute_min_block_id,
    cur_max_valid_tokens,
    head_tile,
    compute_tile_size,
    cu_seqlens_q,
    cu_seqlens_k,
    head_dim,
    sm_scale,
    block_size,
):
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    # Hopper can typically handle deeper query-loop unrolling better.
    num_q_blocks = 16 if IS_HOPPER_GPU else 8
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="fwd_qkv",
        head_dim=head_dim,
        block_size=block_size,
        default_warps=4,
        default_stages=3,
    )

    grid_fwd = lambda META: (
        compute_tile_size * head_tile,
        triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_q_blocks),
    )

    forward_kernel_opt[grid_fwd](
        q_tile,
        k_tile,
        v_tile,
        o_tiles,
        acc_o_scales,
        m_ij_tiles,
        l_ij,
        token_index_mapping,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        compute_min_block_id,
        cur_max_valid_tokens,
        head_tile,
        compute_tile_size,
        cu_seqlens_q,
        cu_seqlens_k,
        head_dim,
        sm_scale,
        num_q_blocks,
        q_tile.stride(0),
        q_tile.stride(1),
        q_tile.stride(2),
        k_tile.stride(0),
        k_tile.stride(1),
        k_tile.stride(2),
        v_tile.stride(0),
        v_tile.stride(1),
        v_tile.stride(2),
        o_tiles.stride(0),
        o_tiles.stride(1),
        o_tiles.stride(2),
        o_tiles.stride(3),
        acc_o_scales.stride(0),
        acc_o_scales.stride(1),
        acc_o_scales.stride(2),
        m_ij_tiles.stride(0),
        m_ij_tiles.stride(1),
        m_ij_tiles.stride(2),
        l_ij.stride(0),
        l_ij.stride(1),
        l_ij.stride(2),
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_stages=num_stages,
        num_warps=num_warps,
    )


def reduce_output(
    lse,
    o,
    o_tiles_first,
    o_tiles_rest,
    m_ij_tiles,
    l_ij_first,
    l_ij_rest,
    m_ij_last,
    acc_o_scales_first,
    acc_o_scales_rest,
    topk_idx_tile,
    token_index_mapping,
    head_start_idx,
    head_tile,
    total_len,
    TOPK,
    head_dim,
    query_start_idx=0,
    query_tokens_count=None,
):
    if query_tokens_count is None:
        query_tokens_count = total_len
    query_tokens_count = int(query_tokens_count)
    if query_tokens_count <= 0:
        return
    num_qy_loop = 4
    num_qz_loop = max(1, query_tokens_count // num_qy_loop)
    grid_x = num_qy_loop + (query_tokens_count % num_qy_loop != 0)
    max_grid_y = 65535
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="fwd_reduce",
        head_dim=head_dim,
        block_size=max(32, int(TOPK)),
        default_warps=1,
        default_stages=2,
    )

    for q_off in range(0, num_qz_loop, max_grid_y):
        grid_y = min(max_grid_y, num_qz_loop - q_off)
        grid_reduce = (grid_x, grid_y)
        reduce_kernel[grid_reduce](
            lse,
            m_ij_tiles,
            l_ij_first,
            l_ij_rest,
            m_ij_last,
            o,
            o_tiles_first,
            o_tiles_rest,
            acc_o_scales_first,
            acc_o_scales_rest,
            topk_idx_tile,
            token_index_mapping,
            head_start_idx,
            num_qz_loop,
            q_off,
            query_start_idx,
            query_tokens_count,
            TOPK,
            total_len,
            head_dim,
            lse.stride(0),
            lse.stride(1),
            m_ij_tiles.stride(0),
            m_ij_tiles.stride(1),
            m_ij_tiles.stride(2),
            l_ij_first.stride(0),
            l_ij_first.stride(1),
            l_ij_first.stride(2),
            l_ij_rest.stride(0),
            l_ij_rest.stride(1),
            l_ij_rest.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o_tiles_first.stride(0),
            o_tiles_first.stride(1),
            o_tiles_first.stride(2),
            o_tiles_first.stride(3),
            o_tiles_rest.stride(0),
            o_tiles_rest.stride(1),
            o_tiles_rest.stride(2),
            o_tiles_rest.stride(3),
            acc_o_scales_first.stride(0),
            acc_o_scales_first.stride(1),
            acc_o_scales_first.stride(2),
            acc_o_scales_rest.stride(0),
            acc_o_scales_rest.stride(1),
            acc_o_scales_rest.stride(2),
            topk_idx_tile.stride(0),
            topk_idx_tile.stride(1),
            topk_idx_tile.stride(2),
            token_index_mapping.stride(0),
            token_index_mapping.stride(1),
            token_index_mapping.stride(2),
            BLOCK_SIZE_T=triton.next_power_of_2(TOPK),
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _topk_sparse_attention_fwd_opt_per_seq_all_heads(
    q: torch.Tensor,  # [total_len_q, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len_k, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len_k, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len_q, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float,
):
    """
    Fully de-serialized forward across all query heads.

    This path builds routing metadata per KV head once, then runs the forward math
    over all query heads in a single set of kernel launches.

    Returns:
      (o, lse, permute_results) on success
      None when preconditions are not met and caller should fallback to legacy path.
    """
    _record_fwd_full_deser_stat("attempts")

    total_len_q, num_q_heads, head_dim = q.shape
    total_len_k, num_kv_heads, _ = k.shape

    # P1.1 preprocess-compaction stage for the full-deserialized forward path.
    # Compact selected chapters once per sequence and remap chapter ids before
    # routing metadata + fused math.
    k_seq_c, v_seq_c, bi_seq_c, _compacted = _maybe_precompact_kv_for_seq(
        k_seq=k.unsqueeze(0),
        v_seq=v.unsqueeze(0),
        bi_seq=topk_idx.permute(1, 0, 2).unsqueeze(0),
        block_size=block_size,
    )
    if _compacted:
        k = k_seq_c[0]
        v = v_seq_c[0]
        topk_idx = bi_seq_c[0].permute(1, 0, 2)
        total_len_k = int(k.shape[0])

    if num_q_heads % num_kv_heads != 0:
        _record_fwd_full_deser_stat("fallback_hq_hk")
        _maybe_print_fwd_full_deser_notice(
            "FSA full-deser forward fallback -> legacy: precondition failed (HQ % HK != 0)."
        )
        return None

    gqa_deg = num_q_heads // num_kv_heads
    topk = int(topk_idx.shape[-1])
    real_num_blocks = int(math.ceil(total_len_k / block_size))
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    permute_results = _build_permute_results_per_seq_for_bwd(
        topk_idx=topk_idx,
        total_len_k=total_len_k,
        block_size=block_size,
        head_dim=head_dim,
        ensure_atomic_metadata=True,
    )

    num_blocks = int(permute_results["num_blocks"])
    num_blocks_full = int(permute_results["num_blocks_full"])
    valid_lens_all_full = permute_results["valid_lens_all"]
    valid_lens_all = valid_lens_all_full[:, :num_blocks]
    reduce_tile_size = max(0, num_blocks - 1)

    active_starts, active_counts = _detect_active_token_ranges_per_kv_head(topk_idx)
    active_pairs = torch.stack((active_starts, active_counts), dim=1).to(torch.int32).tolist()
    active_ends = active_starts + active_counts
    routed_heads = active_counts > 0
    if bool(torch.any(routed_heads)):
        _q_bounds = torch.stack((active_starts[routed_heads].min(), active_ends[routed_heads].max())).to(torch.int32).tolist()
        query_start_idx = int(_q_bounds[0])
        query_end_idx = int(_q_bounds[1])
        query_tokens_count = max(0, query_end_idx - query_start_idx)
        if num_kv_heads > 1:
            ref_start = int(active_pairs[0][0])
            ref_count = int(active_pairs[0][1])
            same_ranges = bool(torch.all((active_starts == ref_start) & (active_counts == ref_count)))
            if not same_ranges:
                _record_fwd_full_deser_stat("expanded_active_range")
    else:
        query_start_idx = 0
        query_tokens_count = 0

    global_max_valid_tokens = (
        int(valid_lens_all[:, 1:].max().to(dtype=torch.int32).cpu().tolist())
        if num_blocks > 1
        else int(valid_lens_all.max().to(dtype=torch.int32).cpu().tolist())
    )

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_q_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)

    selected_tokens_all = permute_results.get("valid_topk_idx_concat")
    kh_offsets = permute_results.get("valid_topk_idx_offsets")
    valid_lens_stack = permute_results.get("valid_lens_stack")
    valid_start_stack = permute_results.get("valid_start_indices_stack")
    if selected_tokens_all is None or kh_offsets is None or valid_lens_stack is None or valid_start_stack is None:
        # Defensive fallback for stale metadata produced by older checkpoints.
        permute_results = _ensure_dq_atomic_metadata(
            permute_results=permute_results,
            num_kv_heads=num_kv_heads,
            num_blocks=num_blocks,
            device=q.device,
        )
        selected_tokens_all = permute_results["valid_topk_idx_concat"]
        kh_offsets = permute_results["valid_topk_idx_offsets"]
        valid_lens_stack = permute_results["valid_lens_stack"]
        valid_start_stack = permute_results["valid_start_indices_stack"]
    if int(selected_tokens_all.numel()) <= 0:
        _record_fwd_full_deser_stat("success")
        return o_full, lse_full, permute_results

    use_fwd_packed_gqa = _resolve_fwd_packed_gqa(
        num_share_q_heads=gqa_deg,
        head_dim=head_dim,
        block_size=block_size,
    )
    if use_fwd_packed_gqa and gqa_deg > 1:
        head_tile = _resolve_head_tile(gqa_deg)
        token_index_mapping = _workspace_empty(
            "fwd_fulldeser_token_index_mapping",
            (1, num_blocks, total_len_q),
            torch.int32,
            q.device,
        )
        o_tiles_first = _workspace_empty(
            "fwd_fulldeser_o_tiles_first",
            (head_tile, 1, total_len_q, head_dim),
            torch.bfloat16,
            q.device,
        )
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            o_tiles_rest = _workspace_empty(
                "fwd_fulldeser_o_tiles_rest",
                (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim),
                torch.bfloat16,
                q.device,
            )
        else:
            o_tiles_rest = _workspace_empty(
                "fwd_fulldeser_o_tiles_rest_dummy",
                (head_tile, 1, 1, head_dim),
                torch.bfloat16,
                q.device,
            )
        m_i_cur_tiles = _workspace_empty(
            "fwd_fulldeser_m_i_cur_tiles",
            (head_tile, num_blocks, total_len_q),
            torch.float32,
            q.device,
        )
        l_ij_first = _workspace_empty(
            "fwd_fulldeser_l_ij_first",
            (head_tile, 1, total_len_q),
            torch.float32,
            q.device,
        )
        acc_o_scales_first = _workspace_empty(
            "fwd_fulldeser_acc_o_scales_first",
            (head_tile, 1, total_len_q),
            torch.float32,
            q.device,
        )
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            l_ij_rest = _workspace_empty(
                "fwd_fulldeser_l_ij_rest",
                (head_tile, reduce_tile_size, global_max_valid_tokens),
                torch.float32,
                q.device,
            )
            acc_o_scales_rest = _workspace_empty(
                "fwd_fulldeser_acc_o_scales_rest",
                (head_tile, reduce_tile_size, global_max_valid_tokens),
                torch.float32,
                q.device,
            )
        else:
            l_ij_rest = _workspace_empty("fwd_fulldeser_l_ij_rest_dummy", (head_tile, 1, 1), torch.float32, q.device)
            acc_o_scales_rest = _workspace_empty(
                "fwd_fulldeser_acc_o_scales_rest_dummy",
                (head_tile, 1, 1),
                torch.float32,
                q.device,
            )

        for kh in range(num_kv_heads):
            valid_topk_idx_permuted_tile = permute_results["valid_topk_idx_permuted_tile"][kh]
            valid_lens = permute_results["valid_lens"][kh]
            valid_start_indices = permute_results["valid_start_indices"][kh]
            if int(valid_topk_idx_permuted_tile.numel()) <= 0:
                continue
            index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

            query_start_idx_kh = int(active_pairs[kh][0])
            query_tokens_count_kh = int(active_pairs[kh][1])
            if query_tokens_count_kh <= 0:
                continue

            topk_idx_tile_base = topk_idx[kh:kh + 1]
            valid_lens_host = [int(x) for x in valid_lens.to(dtype=torch.int32).tolist()]
            max_valid_first = valid_lens_host[0] if len(valid_lens_host) > 0 else 0
            max_valid_rest = max(valid_lens_host[1:], default=0)

            for sh0 in range(0, gqa_deg, head_tile):
                ht = min(head_tile, gqa_deg - sh0)
                qh_start = kh * gqa_deg + sh0
                qh_end = qh_start + ht

                q_tile = q[:, qh_start:qh_end]
                o = o_full[:, qh_start:qh_end]
                lse = lse_full[qh_start:qh_end]

                k_base = k[:, kh:kh + 1]
                v_base = v[:, kh:kh + 1]
                if ht == 1:
                    k_tile = k_base
                    v_tile = v_base
                    topk_idx_tile = topk_idx_tile_base
                    token_index_mapping_tile = token_index_mapping
                else:
                    k_tile = k_base.expand(-1, ht, -1)
                    v_tile = v_base.expand(-1, ht, -1)
                    topk_idx_tile = topk_idx_tile_base.expand(ht, -1, -1)
                    token_index_mapping_tile = token_index_mapping.expand(ht, -1, -1)

                valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1)
                valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1)

                m_i_cur_tiles_tile = m_i_cur_tiles[:ht]
                m_i_cur_tiles_tile.fill_(float("-inf"))
                l_ij_first_tile = l_ij_first[:ht]
                acc_o_scales_first_tile = acc_o_scales_first[:ht]
                l_ij_first_tile.fill_(0)
                acc_o_scales_first_tile.fill_(1)
                if reduce_tile_size > 0:
                    l_ij_rest_tile = l_ij_rest[:ht]
                    acc_o_scales_rest_tile = acc_o_scales_rest[:ht]
                    l_ij_rest_tile.fill_(0)
                    acc_o_scales_rest_tile.fill_(1)
                else:
                    l_ij_rest_tile = l_ij_rest[:ht]
                    acc_o_scales_rest_tile = acc_o_scales_rest[:ht]

                m_ij_tiles, m_ij_last = online_softmax(
                    q_tile,
                    k_tile,
                    m_i_cur_tiles_tile,
                    valid_topk_idx_permuted_tile,
                    valid_lens_tile,
                    valid_start_indices_tile,
                    0,
                    global_max_valid_tokens,
                    block_size,
                    num_blocks,
                    ht,
                    head_dim,
                    sm_scale,
                    cu_seqlens_q,
                    cu_seqlens_k,
                )

                for compute_min_block_id in range(min(2, num_blocks)):
                    if compute_min_block_id == 0:
                        compute_tile_size = 1
                        cur_max_valid_tokens = max_valid_first
                        cur_valid_lens = valid_lens_tile[:, 0]
                        cur_valid_start_indices = valid_start_indices_tile[:, 0]
                        o_tiles = o_tiles_first[:ht]
                        l_ij = l_ij_first_tile
                        acc_o_scales = acc_o_scales_first_tile
                    else:
                        compute_tile_size = num_blocks - 1
                        if compute_tile_size <= 0:
                            continue
                        cur_max_valid_tokens = max_valid_rest
                        if cur_max_valid_tokens <= 0:
                            continue
                        cur_valid_lens = valid_lens_tile[:, compute_min_block_id:]
                        cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:]
                        o_tiles = o_tiles_rest[:ht]
                        l_ij = l_ij_rest_tile
                        acc_o_scales = acc_o_scales_rest_tile

                    if cur_max_valid_tokens <= 0:
                        continue

                    qkv_kernel(
                        q_tile,
                        k_tile,
                        v_tile,
                        o_tiles,
                        acc_o_scales,
                        m_ij_tiles,
                        l_ij,
                        token_index_mapping_tile,
                        valid_topk_idx_permuted_tile,
                        cur_valid_lens,
                        cur_valid_start_indices,
                        compute_min_block_id,
                        cur_max_valid_tokens,
                        ht,
                        compute_tile_size,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        head_dim,
                        sm_scale,
                        block_size,
                    )

                reduce_output(
                    lse,
                    o,
                    o_tiles_first[:ht],
                    o_tiles_rest[:ht],
                    m_ij_tiles,
                    l_ij_first_tile,
                    l_ij_rest_tile,
                    m_ij_last,
                    acc_o_scales_first_tile,
                    acc_o_scales_rest_tile,
                    topk_idx_tile,
                    token_index_mapping_tile,
                    qh_start,
                    ht,
                    total_len_q,
                    topk,
                    head_dim,
                    query_start_idx=query_start_idx_kh,
                    query_tokens_count=query_tokens_count_kh,
                )

                o_full[:, qh_start:qh_end] = o
                lse_full[qh_start:qh_end] = lse
    else:
        qh_to_kh = torch.div(
            torch.arange(num_q_heads, device=q.device, dtype=torch.int64),
            gqa_deg,
            rounding_mode="floor",
        ).to(torch.long)

        valid_lens_qh = valid_lens_stack.index_select(0, qh_to_kh)  # [HQ, num_blocks]
        valid_start_qh = valid_start_stack.index_select(0, qh_to_kh) + kh_offsets.index_select(0, qh_to_kh).view(-1, 1)
        token_index_mapping_qh = _workspace_empty(
            "fwd_fulldeser_token_index_mapping_qh",
            (num_q_heads, num_blocks, total_len_q),
            torch.int32,
            q.device,
        )
        index_mapping(token_index_mapping_qh, selected_tokens_all, valid_lens_qh, valid_start_qh, num_blocks)

        topk_idx_qh = topk_idx.index_select(0, qh_to_kh)
        k_qh = k.index_select(1, qh_to_kh)
        v_qh = v.index_select(1, qh_to_kh)

        head_tile = num_q_heads
        o_tiles_first = _workspace_empty(
            "fwd_fulldeser_o_tiles_first_qh",
            (head_tile, 1, total_len_q, head_dim),
            torch.bfloat16,
            q.device,
        )
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            o_tiles_rest = _workspace_empty(
                "fwd_fulldeser_o_tiles_rest_qh",
                (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim),
                torch.bfloat16,
                q.device,
            )
        else:
            o_tiles_rest = _workspace_empty(
                "fwd_fulldeser_o_tiles_rest_qh_dummy",
                (head_tile, 1, 1, head_dim),
                torch.bfloat16,
                q.device,
            )
        m_i_cur_tiles = _workspace_empty(
            "fwd_fulldeser_m_i_cur_tiles_qh",
            (head_tile, num_blocks, total_len_q),
            torch.float32,
            q.device,
        )
        l_ij_first = _workspace_empty(
            "fwd_fulldeser_l_ij_first_qh",
            (head_tile, 1, total_len_q),
            torch.float32,
            q.device,
        )
        acc_o_scales_first = _workspace_empty(
            "fwd_fulldeser_acc_o_scales_first_qh",
            (head_tile, 1, total_len_q),
            torch.float32,
            q.device,
        )
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            l_ij_rest = _workspace_empty(
                "fwd_fulldeser_l_ij_rest_qh",
                (head_tile, reduce_tile_size, global_max_valid_tokens),
                torch.float32,
                q.device,
            )
            acc_o_scales_rest = _workspace_empty(
                "fwd_fulldeser_acc_o_scales_rest_qh",
                (head_tile, reduce_tile_size, global_max_valid_tokens),
                torch.float32,
                q.device,
            )
        else:
            l_ij_rest = _workspace_empty("fwd_fulldeser_l_ij_rest_qh_dummy", (head_tile, 1, 1), torch.float32, q.device)
            acc_o_scales_rest = _workspace_empty(
                "fwd_fulldeser_acc_o_scales_rest_qh_dummy",
                (head_tile, 1, 1),
                torch.float32,
                q.device,
            )

        m_i_cur_tiles.fill_(float("-inf"))
        l_ij_first.fill_(0)
        acc_o_scales_first.fill_(1)
        if reduce_tile_size > 0:
            l_ij_rest.fill_(0)
            acc_o_scales_rest.fill_(1)

        m_ij_tiles, m_ij_last = online_softmax(
            q,
            k_qh,
            m_i_cur_tiles,
            selected_tokens_all,
            valid_lens_qh,
            valid_start_qh,
            0,
            global_max_valid_tokens,
            block_size,
            num_blocks,
            head_tile,
            head_dim,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
        )

        if valid_lens_qh.numel() > 0:
            max_valid_first = int(valid_lens_qh[:, 0].max().to(dtype=torch.int32).cpu().tolist())
        else:
            max_valid_first = 0
        if num_blocks > 1 and valid_lens_qh.shape[1] > 1:
            max_valid_rest = int(valid_lens_qh[:, 1:].max().to(dtype=torch.int32).cpu().tolist())
        else:
            max_valid_rest = 0

        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                compute_tile_size = 1
                cur_max_valid_tokens = max_valid_first
                cur_valid_lens = valid_lens_qh[:, 0]
                cur_valid_start_indices = valid_start_qh[:, 0]
                o_tiles = o_tiles_first
                l_ij = l_ij_first
                acc_o_scales = acc_o_scales_first
            else:
                compute_tile_size = num_blocks - 1
                if compute_tile_size <= 0:
                    continue
                cur_max_valid_tokens = max_valid_rest
                if cur_max_valid_tokens <= 0:
                    continue
                cur_valid_lens = valid_lens_qh[:, compute_min_block_id:]
                cur_valid_start_indices = valid_start_qh[:, compute_min_block_id:]
                o_tiles = o_tiles_rest
                l_ij = l_ij_rest
                acc_o_scales = acc_o_scales_rest

            if cur_max_valid_tokens <= 0:
                continue

            qkv_kernel(
                q,
                k_qh,
                v_qh,
                o_tiles,
                acc_o_scales,
                m_ij_tiles,
                l_ij,
                token_index_mapping_qh,
                selected_tokens_all,
                cur_valid_lens,
                cur_valid_start_indices,
                compute_min_block_id,
                cur_max_valid_tokens,
                head_tile,
                compute_tile_size,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim,
                sm_scale,
                block_size,
            )

        reduce_output(
            lse_full,
            o_full,
            o_tiles_first,
            o_tiles_rest,
            m_ij_tiles,
            l_ij_first,
            l_ij_rest,
            m_ij_last,
            acc_o_scales_first,
            acc_o_scales_rest,
            topk_idx_qh,
            token_index_mapping_qh,
            0,
            head_tile,
            total_len_q,
            topk,
            head_dim,
            query_start_idx=query_start_idx,
            query_tokens_count=query_tokens_count,
        )

    _record_fwd_full_deser_stat("success")
    return o_full, lse_full, permute_results


def _topk_sparse_attention_fwd_opt_per_seq(
    q: torch.Tensor,  # [total_len, num_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal=True,
):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256, 512, 1024}

    total_len_q, num_heads, head_dim = q.shape
    total_len_k, num_kv_heads, _ = k.shape
    assert num_heads % num_kv_heads == 0
    gqa_deg = num_heads // num_kv_heads

    # P1.1 preprocess-compaction stage for legacy per-seq path.
    # Apply once per sequence before head-wise metadata/math loops.
    k_seq_c, v_seq_c, bi_seq_c, _compacted = _maybe_precompact_kv_for_seq(
        k_seq=k.unsqueeze(0),
        v_seq=v.unsqueeze(0),
        bi_seq=topk_idx.permute(1, 0, 2).unsqueeze(0),
        block_size=block_size,
    )
    if _compacted:
        k = k_seq_c[0]
        v = v_seq_c[0]
        topk_idx = bi_seq_c[0].permute(1, 0, 2)
        total_len_k = int(k.shape[0])

    # Optional fully de-serialized forward across all query heads.
    # Falls back to the legacy path if preconditions are not met.
    use_full_deser = os.getenv("FSA_LOCAL_FWD_FULL_DESERIALIZE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if use_full_deser:
        # Stability guard for no-prefix ("all query tokens routed") long-sequence mode.
        # This regime can stress very large metadata tensors and has shown kernel fragility
        # on some setups. Keep it opt-in via env when needed.
        starts_all, counts_all = _detect_active_token_ranges_per_kv_head(topk_idx)
        all_routed_no_prefix = bool(
            torch.all((starts_all == 0) & (counts_all == int(total_len_q)))
        ) if counts_all.numel() > 0 else False
        allow_no_prefix = os.getenv("FSA_LOCAL_FWD_FULL_DESER_ALLOW_NO_PREFIX", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )

        # Additional guard on very large dense mapping tensors.
        # token_index_mapping_qh is [HQ, num_blocks, Tq].
        # Default cap chosen for safety on very long sequences.
        max_map_elems_raw = os.getenv("FSA_LOCAL_FWD_FULL_DESER_MAX_MAP_ELEMS", "500000000").strip().lower()
        try:
            max_map_elems = int(max_map_elems_raw)
        except Exception:
            max_map_elems = 500000000
        est_num_blocks = max(math.ceil(total_len_k / block_size), int(topk_idx.shape[-1]))
        est_map_elems = int(num_heads) * int(est_num_blocks) * int(total_len_q)
        map_too_large = est_map_elems > max(1, max_map_elems)

        if (all_routed_no_prefix and not allow_no_prefix) or map_too_large:
            use_full_deser = False
            why = []
            if all_routed_no_prefix and not allow_no_prefix:
                why.append("no-prefix/all-routed regime")
            if map_too_large:
                why.append(f"estimated map too large ({est_map_elems} elems)")
            _maybe_print_fwd_full_deser_notice(
                "FSA full-deser forward disabled -> legacy path for stability: " + ", ".join(why) + ". "
                "Override via FSA_LOCAL_FWD_FULL_DESER_ALLOW_NO_PREFIX=1 and/or "
                "FSA_LOCAL_FWD_FULL_DESER_MAX_MAP_ELEMS."
            )
    if use_full_deser:
        fast = _topk_sparse_attention_fwd_opt_per_seq_all_heads(
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            sm_scale=sm_scale,
        )
        if fast is not None:
            _maybe_print_fwd_full_deser_notice(
                "FSA full-deser forward active: preconditions satisfied "
                "(HQ % HK == 0). Active query ranges are unioned across KV heads when needed."
            )
            return fast

    head_tile = _resolve_head_tile(gqa_deg)

    TOPK = int(topk_idx.shape[-1])
    real_num_blocks = int(math.ceil(total_len_k / block_size))
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    permute_results = _build_permute_results_per_seq_for_bwd(
        topk_idx=topk_idx,
        total_len_k=total_len_k,
        block_size=block_size,
        head_dim=head_dim,
        ensure_atomic_metadata=True,
    )
    num_blocks = int(permute_results["num_blocks"])
    num_blocks_full = int(permute_results["num_blocks_full"])
    valid_lens_all_full = permute_results["valid_lens_all"]
    reduce_tile_size = num_blocks - 1

    valid_lens_all = valid_lens_all_full[:, :num_blocks]

    active_starts, active_counts = _detect_active_token_ranges_per_kv_head(topk_idx)
    active_pairs = torch.stack((active_starts, active_counts), dim=1).to(torch.int32).tolist()

    global_max_valid_tokens = (
        int(valid_lens_all[:, 1:].max().to(dtype=torch.int32).cpu().tolist())
        if num_blocks > 1
        else int(valid_lens_all.max().to(dtype=torch.int32).cpu().tolist())
    )

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)

    token_index_mapping = _workspace_empty(
        "fwd_legacy_token_index_mapping",
        (1, num_blocks, total_len_q),
        torch.int32,
        q.device,
    )

    o_tiles_first = _workspace_empty(
        "fwd_legacy_o_tiles_first",
        (head_tile, 1, total_len_q, head_dim),
        torch.bfloat16,
        q.device,
    )
    if reduce_tile_size > 0 and global_max_valid_tokens > 0:
        o_tiles_rest = _workspace_empty(
            "fwd_legacy_o_tiles_rest",
            (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim),
            torch.bfloat16,
            q.device,
        )
    else:
        o_tiles_rest = _workspace_empty("fwd_legacy_o_tiles_rest_dummy", (head_tile, 1, 1, head_dim), torch.bfloat16, q.device)
    m_i_cur_tiles = _workspace_empty(
        "fwd_legacy_m_i_cur_tiles",
        (head_tile, num_blocks, total_len_q),
        torch.float32,
        q.device,
    )
    l_ij_first = _workspace_empty("fwd_legacy_l_ij_first", (head_tile, 1, total_len_q), torch.float32, q.device)
    acc_o_scales_first = _workspace_empty(
        "fwd_legacy_acc_o_scales_first",
        (head_tile, 1, total_len_q),
        torch.float32,
        q.device,
    )
    if reduce_tile_size > 0 and global_max_valid_tokens > 0:
        l_ij_rest = _workspace_empty(
            "fwd_legacy_l_ij_rest",
            (head_tile, reduce_tile_size, global_max_valid_tokens),
            torch.float32,
            q.device,
        )
        acc_o_scales_rest = _workspace_empty(
            "fwd_legacy_acc_o_scales_rest",
            (head_tile, reduce_tile_size, global_max_valid_tokens),
            torch.float32,
            q.device,
        )
    else:
        l_ij_rest = _workspace_empty("fwd_legacy_l_ij_rest_dummy", (head_tile, 1, 1), torch.float32, q.device)
        acc_o_scales_rest = _workspace_empty("fwd_legacy_acc_o_scales_rest_dummy", (head_tile, 1, 1), torch.float32, q.device)

    for kh in range(num_kv_heads):
        topk_idx_tile_base = topk_idx[kh:kh + 1]
        valid_topk_idx_permuted_tile = permute_results["valid_topk_idx_permuted_tile"][kh]
        valid_lens = permute_results["valid_lens"][kh]
        valid_start_indices = permute_results["valid_start_indices"][kh]
        index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

        query_start_idx = int(active_pairs[kh][0])
        query_tokens_count = int(active_pairs[kh][1])
        if query_tokens_count <= 0:
            continue
        valid_lens_host = [int(x) for x in valid_lens.to(dtype=torch.int32).tolist()]
        max_valid_first = valid_lens_host[0] if len(valid_lens_host) > 0 else 0
        max_valid_rest = max(valid_lens_host[1:], default=0)

        for sh0 in range(0, gqa_deg, head_tile):
            ht = min(head_tile, gqa_deg - sh0)
            qh_start = kh * gqa_deg + sh0
            qh_end = qh_start + ht

            q_tile = q[:, qh_start:qh_end]
            o = o_full[:, qh_start:qh_end]
            lse = lse_full[qh_start:qh_end]

            k_base = k[:, kh:kh + 1]
            v_base = v[:, kh:kh + 1]
            if ht == 1:
                k_tile = k_base
                v_tile = v_base
                topk_idx_tile = topk_idx_tile_base
                token_index_mapping_tile = token_index_mapping
            else:
                k_tile = k_base.expand(-1, ht, -1)
                v_tile = v_base.expand(-1, ht, -1)
                topk_idx_tile = topk_idx_tile_base.expand(ht, -1, -1)
                token_index_mapping_tile = token_index_mapping.expand(ht, -1, -1)

            valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1)
            valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1)

            m_i_cur_tiles_tile = m_i_cur_tiles[:ht]
            m_i_cur_tiles_tile.fill_(float("-inf"))
            l_ij_first_tile = l_ij_first[:ht]
            acc_o_scales_first_tile = acc_o_scales_first[:ht]
            l_ij_first_tile.fill_(0)
            acc_o_scales_first_tile.fill_(1)
            if reduce_tile_size > 0:
                l_ij_rest_tile = l_ij_rest[:ht]
                acc_o_scales_rest_tile = acc_o_scales_rest[:ht]
                l_ij_rest_tile.fill_(0)
                acc_o_scales_rest_tile.fill_(1)
            else:
                l_ij_rest_tile = l_ij_rest[:ht]
                acc_o_scales_rest_tile = acc_o_scales_rest[:ht]

            m_ij_tiles, m_ij_last = online_softmax(
                q_tile,
                k_tile,
                m_i_cur_tiles_tile,
                valid_topk_idx_permuted_tile,
                valid_lens_tile,
                valid_start_indices_tile,
                0,
                global_max_valid_tokens,
                block_size,
                num_blocks,
                ht,
                head_dim,
                sm_scale,
                cu_seqlens_q,
                cu_seqlens_k,
            )

            for compute_min_block_id in range(min(2, num_blocks)):
                if compute_min_block_id == 0:
                    compute_tile_size = 1
                    cur_max_valid_tokens = max_valid_first
                    cur_valid_lens = valid_lens_tile[:, 0]
                    cur_valid_start_indices = valid_start_indices_tile[:, 0]
                    o_tiles = o_tiles_first[:ht]
                    l_ij = l_ij_first_tile
                    acc_o_scales = acc_o_scales_first_tile
                else:
                    compute_tile_size = num_blocks - 1
                    if compute_tile_size <= 0:
                        continue
                    cur_valid_lens = valid_lens_tile[:, compute_min_block_id:]
                    if cur_valid_lens.numel() == 0:
                        continue
                    cur_max_valid_tokens = max_valid_rest
                    if cur_max_valid_tokens <= 0:
                        continue
                    cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:]
                    o_tiles = o_tiles_rest[:ht]
                    l_ij = l_ij_rest_tile
                    acc_o_scales = acc_o_scales_rest_tile

                qkv_kernel(
                    q_tile,
                    k_tile,
                    v_tile,
                    o_tiles,
                    acc_o_scales,
                    m_ij_tiles,
                    l_ij,
                    token_index_mapping_tile,
                    valid_topk_idx_permuted_tile,
                    cur_valid_lens,
                    cur_valid_start_indices,
                    compute_min_block_id,
                    cur_max_valid_tokens,
                    ht,
                    compute_tile_size,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    head_dim,
                    sm_scale,
                    block_size,
                )

            reduce_output(
                lse,
                o,
                o_tiles_first[:ht],
                o_tiles_rest[:ht],
                m_ij_tiles,
                l_ij_first_tile,
                l_ij_rest_tile,
                m_ij_last,
                acc_o_scales_first_tile,
                acc_o_scales_rest_tile,
                topk_idx_tile,
                token_index_mapping_tile,
                qh_start,
                ht,
                total_len_q,
                TOPK,
                head_dim,
                query_start_idx=query_start_idx,
                query_tokens_count=query_tokens_count,
            )

            o_full[:, qh_start:qh_end] = o
            lse_full[qh_start:qh_end] = lse

    return o_full, lse_full, permute_results


def _build_permute_results_per_seq_for_bwd(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len_q, topk]
    total_len_k: int,
    block_size: int,
    head_dim: int = 64,
    ensure_atomic_metadata: bool = True,
):
    """
    Build only the permutation metadata needed by backward, without running forward math kernels.
    """
    _, total_len_q, topk = topk_idx.shape
    num_kv_heads = topk_idx.shape[0]
    real_num_blocks = math.ceil(total_len_k / block_size)
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    num_blocks_full = max(real_num_blocks, topk)

    # OPT-3: Vectorized bincount across all KV heads using offset indices.
    # Replaces num_kv_heads individual bincounts with a single scatter operation.
    valid_mask = (topk_idx >= 0) & (topk_idx < real_num_blocks)
    kh_offset = torch.arange(num_kv_heads, device=topk_idx.device, dtype=torch.int64).view(-1, 1, 1) * num_blocks_full
    offset_idx = topk_idx.to(torch.int64) + kh_offset
    flat_valid = offset_idx[valid_mask]
    counts = torch.bincount(flat_valid, minlength=num_kv_heads * num_blocks_full)
    valid_lens_all_full = counts.view(num_kv_heads, num_blocks_full).to(torch.int32)

    num_blocks = _resolve_effective_num_blocks(
        valid_lens_all=valid_lens_all_full,
        num_blocks_full=num_blocks_full,
        real_num_blocks=real_num_blocks,
        head_dim=int(head_dim),
        block_size=block_size,
    )
    valid_lens_all = valid_lens_all_full[:, :num_blocks]

    global_max_valid_tokens = valid_lens_all[:, 1:].max() if num_blocks > 1 else valid_lens_all.max()

    # OPT-3: Single batched Triton kernel call for all heads (instead of num_kv_heads launches).
    topk_idx_permuted = torch.full((num_kv_heads, num_blocks, total_len_q), -1, dtype=torch.int32, device=topk_idx.device)
    build_block_to_token_triton(topk_idx_permuted, topk_idx, 0, num_blocks, padding_value=-1)

    # OPT-3: Vectorized cumsum for valid_start_indices across all heads at once.
    valid_start_all = torch.nn.functional.pad(
        valid_lens_all.cumsum(dim=1)[:, :-1], (1, 0), value=0
    )

    permute_results = {
        "global_max_valid_tokens": global_max_valid_tokens,
        "num_blocks": num_blocks,
        "num_blocks_full": num_blocks_full,
        "real_num_blocks": real_num_blocks,
        "valid_topk_idx_permuted_tile": [],
        "valid_lens_all": valid_lens_all_full,
        "valid_lens": [],
        "valid_start_indices": [],
    }

    # Extract per-head variable-length results (lightweight loop — only slicing + masking).
    for kh in range(num_kv_heads):
        perm_kh = topk_idx_permuted[kh]
        permute_results["valid_topk_idx_permuted_tile"].append(perm_kh[perm_kh != -1])
        permute_results["valid_lens"].append(valid_lens_all[kh])
        permute_results["valid_start_indices"].append(valid_start_all[kh])

    if ensure_atomic_metadata:
        # P0.3: precompute concatenated routed-token buffers and offsets for atomic dQ path.
        permute_results = _ensure_dq_atomic_metadata(
            permute_results=permute_results,
            num_kv_heads=num_kv_heads,
            num_blocks=num_blocks,
            device=topk_idx.device,
        )

    return permute_results


def _build_permute_results_for_bwd(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len_q, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_dim: int = 64,
):
    """
    Varlen wrapper for backward permutation metadata.
    """
    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) == 1 and len(k_ranges) == 1:
        (q_start, q_end) = q_ranges[0]
        (k_start, k_end) = k_ranges[0]
        return [
            _build_permute_results_per_seq_for_bwd(
                topk_idx=topk_idx[:, q_start:q_end],
                total_len_k=(k_end - k_start),
                block_size=block_size,
                head_dim=head_dim,
            )
        ]
    return [
        _build_permute_results_per_seq_for_bwd(
            topk_idx=topk_idx[:, q_start:q_end],
            total_len_k=(k_end - k_start),
            block_size=block_size,
            head_dim=head_dim,
        )
        for (q_start, q_end), (k_start, k_end) in zip(q_ranges, k_ranges)
    ]


def _permute_results_need_rebuild(permute_results: Any) -> bool:
    """
    Validate backward metadata container shape/content.
    Returns True when metadata is missing/invalid and must be rebuilt.
    """
    if permute_results is None:
        return True
    if isinstance(permute_results, dict):
        permute_results = [permute_results]
    if not isinstance(permute_results, (list, tuple)) or len(permute_results) == 0:
        return True

    required = (
        "valid_lens_all",
        "real_num_blocks",
        "num_blocks",
        "valid_topk_idx_permuted_tile",
        "valid_lens",
        "valid_start_indices",
    )
    for item in permute_results:
        if item is None or not isinstance(item, dict):
            return True
        for key in required:
            if key not in item or item[key] is None:
                return True
    return False


def _topk_sparse_attention_fwd_nsa_style(
    q: torch.Tensor,  # [total_len_q, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len_k, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len_k, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len_q, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float,
):
    """
    NSA-style fused selected-attention forward:
      - one kernel does indexed KV load + QK + online-softmax + PV
      - no forward-side permutation/reduction staging buffers
    """
    from memory_cross_attn import memory_cross_attn_forward

    total_len_q, num_q_heads, _ = q.shape
    # Prefix timeline has inactive tokens (no routed chapters). Initialize explicitly:
    # - output for inactive tokens is zero
    # - lse for inactive tokens is -inf
    o = torch.zeros_like(q)
    # FSA backward uses exp2-space lse, so convert from natural-log lse.
    lse = torch.full((num_q_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)
    log2e = 1.4426950408889634
    use_native_nsa_fwd = os.getenv("FSA_LOCAL_USE_NSA_NATIVE_FWD", "0").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    pad_small_g_to_16 = os.getenv("FSA_LOCAL_PAD_G_TO_16", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "fallback").strip().lower()
    # Modes:
    #  - fallback: default; for G<16 use legacy local FSA forward path in caller
    #  - pad:      pad G<16 to 16 and run fused memory_cross_attn forward
    #  - fma:      no pad, use small-G fallback math path in memory_cross_attn
    #  - torch:    no pad, use PyTorch matmul/softmax path for small-G forward experiment
    #  - fallback: handled in caller (use legacy FSA forward instead)
    if small_g_mode not in ("pad", "fma", "torch", "fallback"):
        small_g_mode = "pad"
    if small_g_mode == "fma":
        pad_small_g_to_16 = False
    native_nsa_fwd = _try_get_native_nsa_parallel_fwd() if use_native_nsa_fwd else None

    max_tokens_env = os.getenv("FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL", "auto").strip().lower()
    torch_chunk_env = os.getenv("FSA_LOCAL_TORCH_CHUNK_TOKENS", "512").strip().lower()

    def _torch_small_g_forward_chunk(
        q_chunk: torch.Tensor,     # [1, Tq, HQ, D]
        k_seq: torch.Tensor,       # [1, Tk, HK, D]
        v_seq: torch.Tensor,       # [1, Tk, HK, D]
        bi_chunk: torch.Tensor,    # [1, Tq, HK, S]
    ):
        """
        Experimental small-G forward implemented with PyTorch matmul/softmax.
        Returns:
          o_chunk: [1, Tq, HQ, D]
          lse_chunk_e: [1, Tq, HQ] in natural-log space.
        """
        _, tqa, hq_real, d = q_chunk.shape
        tk = int(k_seq.shape[1])
        hk = int(k_seq.shape[2])
        s = int(bi_chunk.shape[-1])
        gqa_deg_local = hq_real // hk
        ksel = s * block_size

        qg = q_chunk.view(1, tqa, hk, gqa_deg_local, d)[0].to(torch.float32)   # [Tq, HK, G, D]
        chapters = bi_chunk[0].to(torch.long)                                    # [Tq, HK, S]

        offs = torch.arange(block_size, device=q_chunk.device, dtype=torch.long).view(1, 1, 1, block_size)
        tok_idx = chapters.unsqueeze(-1) * block_size + offs                     # [Tq, HK, S, BS]
        valid_tok = (chapters.unsqueeze(-1) >= 0) & (tok_idx >= 0) & (tok_idx < tk)

        # clamp only used for safe gather; invalid rows are fully masked out before softmax
        tok_idx = tok_idx.clamp_(0, max(tk - 1, 0))
        tok_idx = tok_idx.view(tqa, hk, ksel)                                    # [Tq, HK, Ksel]
        valid_tok = valid_tok.view(tqa, hk, ksel)                                # [Tq, HK, Ksel]

        k_h = k_seq[0].permute(1, 0, 2).to(torch.float32)                        # [HK, Tk, D]
        v_h = v_seq[0].permute(1, 0, 2).to(torch.float32)                        # [HK, Tk, D]

        hk_idx = torch.arange(hk, device=q_chunk.device, dtype=torch.long).view(1, hk, 1)
        k_sel = k_h[hk_idx, tok_idx]                                             # [Tq, HK, Ksel, D]
        v_sel = v_h[hk_idx, tok_idx]                                             # [Tq, HK, Ksel, D]

        scores = torch.matmul(qg, k_sel.transpose(-1, -2)) * float(sm_scale)     # [Tq, HK, G, Ksel]
        score_mask = valid_tok.unsqueeze(2)                                       # [Tq, HK, 1, Ksel]
        scores = scores.masked_fill(~score_mask, float("-inf"))

        lse_chunk_e = torch.logsumexp(scores, dim=-1)                             # [Tq, HK, G]
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(score_mask, probs, torch.zeros_like(probs))
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.matmul(probs, v_sel)                                          # [Tq, HK, G, D]
        any_valid = valid_tok.any(dim=-1, keepdim=True)                           # [Tq, HK, 1]
        lse_chunk_e = torch.where(
            any_valid.expand(-1, -1, gqa_deg_local),
            lse_chunk_e,
            torch.full_like(lse_chunk_e, float("-inf")),
        )

        o_chunk = out.to(q_chunk.dtype).reshape(1, tqa, hq_real, d)
        lse_chunk_e = lse_chunk_e.reshape(1, tqa, hq_real).to(torch.float32)
        return o_chunk, lse_chunk_e

    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) > 1:
        packed = _pack_varlen_unified_timeline(
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_size=block_size,
        )
        if packed is None:
            raise RuntimeError("FSA NSA-style multi-seq flat forward packing failed unexpectedly.")
        o_u, lse_u = _topk_sparse_attention_fwd_nsa_style(
            q=packed["q_u"],
            k=packed["k_u"],
            v=packed["v_u"],
            topk_idx=packed["topk_u"],
            block_size=block_size,
            cu_seqlens_q=packed["cu_u"],
            cu_seqlens_k=packed["cu_u"],
            sm_scale=sm_scale,
        )
        _unpack_varlen_unified_q(
            packed_meta=packed["packed_meta"],
            src_u=o_u,
            dst=o,
            by_head_first=False,
        )
        _unpack_varlen_unified_q(
            packed_meta=packed["packed_meta"],
            src_u=lse_u,
            dst=lse,
            by_head_first=True,
        )
        return o, lse

    for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        q_len_seq = q_end - q_start
        k_len_seq = k_end - k_start
        real_num_blocks_seq = math.ceil(max(0, k_len_seq) / block_size)
        topk_idx_seq = topk_idx[:, q_start:q_end, :]   # [HK, Tq, S]
        topk_idx_seq, _ = _sanitize_topk_block_indices(
            topk_idx_seq,
            real_num_blocks=real_num_blocks_seq,
        )

        if native_nsa_fwd is not None:
            # Native NSA selected-fwd requires q/k/v to share the same timeline length.
            # Build a prefix-timeline view: [memory | query], matching benchmark Path-A layout.
            q_seq_full = q[q_start:q_end].unsqueeze(0)  # [1, Tfull, HQ, D]
            k_seq_mem = k[k_start:k_end].unsqueeze(0)   # [1, Tk, HK, D]
            v_seq_mem = v[k_start:k_end].unsqueeze(0)   # [1, Tk, HK, D]
            bi_seq_full = (
                topk_idx_seq
                .permute(1, 0, 2)
                .unsqueeze(0)
            )                                                         # [1, Tfull, HK, S]

            hq = int(q_seq_full.shape[2])
            hk = int(k_seq_mem.shape[2])
            g = (hq // hk) if (hk > 0 and hq % hk == 0) else -1
            nsa_shape_ok = (
                g >= 16
                and (g & (g - 1)) == 0
                and block_size >= 16
                and int(q_seq_full.shape[-1]) >= 16
            )

            if nsa_shape_ok:
                t_full = int(q_seq_full.shape[1])
                t_mem = int(k_seq_mem.shape[1])
                # Avoid zero-fill of inactive tail; selected path only indexes memory blocks.
                k_seq_full = torch.empty(
                    (1, t_full, hk, int(k_seq_mem.shape[-1])),
                    dtype=k_seq_mem.dtype,
                    device=k_seq_mem.device,
                )
                v_seq_full = torch.empty(
                    (1, t_full, hk, int(v_seq_mem.shape[-1])),
                    dtype=v_seq_mem.dtype,
                    device=v_seq_mem.device,
                )
                k_seq_full[:, :t_mem].copy_(k_seq_mem)
                v_seq_full[:, :t_mem].copy_(v_seq_mem)

                o_seq, lse_seq_e = native_nsa_fwd(
                    q=q_seq_full,
                    k=k_seq_full,
                    v=v_seq_full,
                    block_indices=bi_seq_full,
                    block_counts=int(bi_seq_full.shape[-1]),
                    block_size=block_size,
                    scale=sm_scale,
                    offsets=None,
                    token_indices=None,
                )
                o[q_start:q_end] = o_seq[0]
                lse[:, q_start:q_end] = lse_seq_e[0].transpose(0, 1) * log2e
                continue

        # Fast path for the NSA-style prefix layout used in the benchmark:
        #   Q timeline is [memory-prefix | real queries], and the prefix has no routed blocks (-1).
        # For true "varlen q/k" usage (no prefix), this check falls back to the generic scan.
        prefix_mode = q_len_seq > k_len_seq
        if prefix_mode:
            prefix_mode = bool(torch.all(topk_idx_seq[:, 0:1, :] < 0))
        if prefix_mode:
            query_start_idx = k_len_seq
            query_tokens_count = q_len_seq - k_len_seq
        else:
            query_start_idx, query_tokens_count = _detect_active_token_range(topk_idx_seq)
        if query_tokens_count == 0:
            continue

        q_sub_start = q_start + query_start_idx
        q_sub_end = q_sub_start + query_tokens_count

        q_seq = q[q_sub_start:q_sub_end].unsqueeze(0)  # [1, Tq_active, HQ, D]
        k_seq = k[k_start:k_end].unsqueeze(0)          # [1, Tk, HK, D]
        v_seq = v[k_start:k_end].unsqueeze(0)          # [1, Tk, HK, D]
        bi_seq = (
            topk_idx_seq[:, query_start_idx: query_start_idx + query_tokens_count, :]
            .permute(1, 0, 2)
            .unsqueeze(0)
        )                                                           # [1, Tq_active, HK, S]

        # P1.1: optional preprocess-compaction (compact selected chapters once per sequence).
        # This reduces forward-side random access in long sparse routes.
        k_seq, v_seq, bi_seq, _compacted = _maybe_precompact_kv_for_seq(
            k_seq=k_seq,
            v_seq=v_seq,
            bi_seq=bi_seq,
            block_size=block_size,
        )

        hq_real = int(q_seq.shape[2])
        hk = int(k_seq.shape[2])
        gqa_deg = (hq_real // hk) if (hk > 0 and hq_real % hk == 0) else -1
        use_g16_pad = pad_small_g_to_16 and (gqa_deg > 0 and gqa_deg < 16)
        use_torch_small_g = (small_g_mode == "torch") and (gqa_deg > 0 and gqa_deg < 16)
        use_nsa_packed_gqa = _resolve_nsa_packed_gqa(
            num_share_q_heads=max(1, gqa_deg),
            head_dim=int(q_seq.shape[-1]),
            block_size=block_size,
        )
        packed_scope = os.getenv("FSA_LOCAL_NSA_PACKED_GQA_SCOPE", "small_g").strip().lower()
        packed_scope_all = packed_scope in ("1", "true", "yes", "on", "all", "force")
        use_packed_nsa_chunk = (
            use_nsa_packed_gqa
            and gqa_deg > 1
            and (packed_scope_all or gqa_deg < 16)
        )
        # OPT-9: small-G specialized path (no pad-to-16). Process each KV head/group directly.
        # Keeps kernel arithmetic dense while avoiding pad overhead for G<16.
        force_small_g_specialized = os.getenv("FSA_LOCAL_SMALL_G_SPECIALIZED", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        if (
            force_small_g_specialized
            and gqa_deg > 1
            and gqa_deg < 16
            and small_g_mode != "torch"
        ):
            use_packed_nsa_chunk = True
            use_g16_pad = False

        d = int(q_seq.shape[3])
        qh_call = hk * 16 if use_g16_pad else hq_real

        # Guard against 32-bit offset overflow inside Triton pointer math:
        # q_tok * (HQ * D) should stay safely below int32 max.
        int32_max = (1 << 31) - 1
        safe_tokens_auto = max(1, (int32_max // max(1, qh_call * d)) - 1)
        if max_tokens_env not in ("", "auto"):
            try:
                env_cap = int(max_tokens_env)
            except Exception:
                env_cap = safe_tokens_auto
            if env_cap > 0:
                safe_tokens = min(safe_tokens_auto, env_cap)
            else:
                safe_tokens = safe_tokens_auto
        else:
            safe_tokens = safe_tokens_auto
        safe_tokens = max(1, safe_tokens)
        if use_torch_small_g:
            try:
                torch_chunk = int(torch_chunk_env)
            except Exception:
                torch_chunk = 512
            if torch_chunk <= 0:
                torch_chunk = 512
            safe_tokens = min(safe_tokens, torch_chunk)

        tqa_total = int(q_seq.shape[1])
        for t0 in range(0, tqa_total, safe_tokens):
            t1 = min(tqa_total, t0 + safe_tokens)
            q_chunk = q_seq[:, t0:t1]
            bi_chunk = bi_seq[:, t0:t1]
            used_legacy_pad_path = False

            if use_torch_small_g:
                o_chunk, lse_chunk_e = _torch_small_g_forward_chunk(
                    q_chunk=q_chunk,
                    k_seq=k_seq,
                    v_seq=v_seq,
                    bi_chunk=bi_chunk,
                )
            else:
                if use_packed_nsa_chunk:
                    tqa = int(q_chunk.shape[1])
                    o_chunk = torch.empty((1, tqa, hq_real, d), dtype=q_chunk.dtype, device=q_chunk.device)
                    lse_chunk_e = torch.full((1, tqa, hq_real), float("-inf"), dtype=torch.float32, device=q_chunk.device)
                    head_tile_packed = _resolve_head_tile(gqa_deg)
                    head_tile_packed = max(1, min(head_tile_packed, gqa_deg))
                    for kh_i in range(hk):
                        bi_base = bi_chunk[:, :, kh_i:kh_i + 1, :]
                        k_base = k_seq[:, :, kh_i:kh_i + 1, :]
                        v_base = v_seq[:, :, kh_i:kh_i + 1, :]
                        for sh0 in range(0, gqa_deg, head_tile_packed):
                            ht = min(head_tile_packed, gqa_deg - sh0)
                            qh_start = kh_i * gqa_deg + sh0
                            qh_end = qh_start + ht
                            q_tile = q_chunk[:, :, qh_start:qh_end, :]
                            if use_g16_pad and ht < 16:
                                g_target = 16
                                q_call = torch.zeros((1, tqa, g_target, d), dtype=q_tile.dtype, device=q_tile.device)
                                q_call[:, :, :ht, :] = q_tile
                            else:
                                q_call = q_tile
                            o_tile, lse_tile = memory_cross_attn_forward(
                                q=q_call,
                                k=k_base,
                                v=v_base,
                                block_indices=bi_base,
                                block_size=block_size,
                                scale=sm_scale,
                            )
                            if use_g16_pad and ht < 16:
                                o_tile = o_tile[:, :, :ht, :]
                                lse_tile = lse_tile[:, :, :ht]
                            o_chunk[:, :, qh_start:qh_end, :] = o_tile
                            lse_chunk_e[:, :, qh_start:qh_end] = lse_tile.to(torch.float32)
                else:
                    if use_g16_pad:
                        g_target = 16
                        tqa = int(q_chunk.shape[1])
                        q_grouped = q_chunk.view(1, tqa, hk, gqa_deg, d)
                        q_padded_grouped = torch.zeros(
                            (1, tqa, hk, g_target, d),
                            dtype=q_chunk.dtype,
                            device=q_chunk.device,
                        )
                        q_padded_grouped[:, :, :, :gqa_deg, :] = q_grouped
                        q_call = q_padded_grouped.view(1, tqa, hk * g_target, d)
                        used_legacy_pad_path = True
                    else:
                        q_call = q_chunk

                    o_chunk, lse_chunk_e = memory_cross_attn_forward(
                        q=q_call,
                        k=k_seq,
                        v=v_seq,
                        block_indices=bi_chunk,
                        block_size=block_size,
                        scale=sm_scale,
                    )

            if used_legacy_pad_path:
                g_target = 16
                tqa = int(o_chunk.shape[1])
                o_grouped = o_chunk.view(1, tqa, hk, g_target, d)
                o_chunk = o_grouped[:, :, :, :gqa_deg, :].reshape(1, tqa, hq_real, d)

                lse_grouped = lse_chunk_e.view(1, tqa, hk, g_target)
                lse_chunk_e = lse_grouped[:, :, :, :gqa_deg].reshape(1, tqa, hq_real)

            q_abs_start = q_sub_start + t0
            q_abs_end = q_sub_start + t1
            o[q_abs_start:q_abs_end] = o_chunk[0]
            lse[:, q_abs_start:q_abs_end] = lse_chunk_e[0].transpose(0, 1) * log2e

    return o, lse


@triton.jit
def dq_compute_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_tiles_ptr,
    token_index_mapping_ptr,
    selected_tokens_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    cur_max_valid_tokens,
    compute_min_block_id,
    head_tile,
    num_blocks,
    HEAD_DIM,
    TOTAL_LEN_Q,
    cu_seqlens_k,
    num_dq_blocks,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    stride_dqth,
    stride_dqtb,
    stride_dqtn,
    stride_dqtd,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    pid_block_h = tl.program_id(0)
    pid_h = pid_block_h % head_tile
    pid_block = pid_block_h // head_tile
    pid_q = tl.program_id(1)  # token
    # seq packing is not supported yet
    q_start = 0
    k_start = 0

    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + pid_h * num_blocks + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + pid_h * num_blocks + pid_block)
    if num_dq_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (pid_block + compute_min_block_id) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_h * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # load k
    k = tl.load(tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero")
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_h * stride_vh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_vd, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )

    # load v
    v = tl.load(tl.advance(v_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero")

    qk_scale = sm_scale * 1.44269504

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    for j in range(num_dq_blocks):
        pid_q_j = pid_q * num_dq_blocks + j
        if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
            # one thread block for one KV block, a subset of selected tokens
            st_offs = start_id + (q_start + pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            # st should be in shape [BLOCK_SIZE_Q]
            st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

            st_raw = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
            # Guard against malformed metadata indices.
            token_valid = (st_raw >= 0) & (st_raw < TOTAL_LEN_Q)
            st = tl.where(token_valid, st_raw, 0)
            # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
            q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd

            mask = token_valid

            q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + st[:, None] * stride_don + off_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            if USE_PRECOMPUTED_DELTA:
                delta_ptrs = delta_ptr + pid_h * stride_dh + st[:, None] * stride_dn
                d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            else:
                o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + st[:, None] * stride_on + off_d[None, :] * stride_od
                o_val = tl.load(o_ptrs, mask=q_mask, other=0)
                d = tl.sum(do.to(tl.float32) * o_val.to(tl.float32), axis=1)[:, None]
            lse_ptrs = lse_ptr + pid_h * stride_lh + st[:, None] * stride_ln
            lse = tl.load(lse_ptrs, mask=mask[:, None], other=0)

            dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k)) * qk_scale  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            if not DISABLE_CAUSAL_MASK:
                qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))
            p = tl.exp2(qk - lse)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            dp = tl.dot(do, v)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            ds = sm_scale * p * (dp - d)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            ds = ds.to(q.dtype)
            dq = tl.dot(ds, k)  # [BLOCK_SIZE_Q, BLOCK_SIZE_D]

            # load token index mapping
            token_index_mapping_ptrs = (
                token_index_mapping_ptr
                + pid_h * stride_tim_h
                + (st) * stride_tim_n
                + (pid_block + compute_min_block_id) * stride_tim_b
            )
            token_index_mapping = tl.load(token_index_mapping_ptrs, mask=mask, other=-1)
            # Safety guard: mapping can be invalid for some masked/irregular entries.
            # Prevent OOB writes into dq_tiles staging buffers.
            valid_map = (token_index_mapping >= 0) & (token_index_mapping < cur_max_valid_tokens)
            store_mask = q_mask & valid_map[:, None]
            token_index_mapping_safe = tl.where(valid_map, token_index_mapping, 0)
            dq_ptrs_off = token_index_mapping_safe[:, None] * stride_dqtn + off_d[None, :] * stride_dqtd
            dq_tiles_ptrs = (
                dq_tiles_ptr
                + dq_ptrs_off
                + (pid_block).to(tl.int64) * stride_dqtb
                + pid_h.to(tl.int64) * stride_dqth
            )
            tl.store(dq_tiles_ptrs, dq.to(dq_tiles_ptr.dtype.element_ty), mask=store_mask)


@triton.jit
def dq_compute_atomic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_accum_ptr,           # float32 [N, H, D]
    selected_tokens_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    cur_max_valid_tokens,
    compute_min_block_id,
    head_tile,
    num_blocks,
    HEAD_DIM,
    TOTAL_LEN_Q,
    cu_seqlens_k,
    num_dq_blocks,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    pid_block_h = tl.program_id(0)
    pid_h = pid_block_h % head_tile
    pid_block = pid_block_h // head_tile
    pid_q = tl.program_id(1)

    q_start = 0
    k_start = 0
    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + pid_h * num_blocks + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + pid_h * num_blocks + pid_block)
    if num_dq_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (pid_block + compute_min_block_id) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_h * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k = tl.load(tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero")

    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_h * stride_vh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_vd, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v = tl.load(tl.advance(v_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero")

    qk_scale = sm_scale * 1.44269504
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    for j in range(num_dq_blocks):
        pid_q_j = pid_q * num_dq_blocks + j
        if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
            st_offs = start_id + (q_start + pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens
            st_raw = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
            token_valid = (st_raw >= 0) & (st_raw < TOTAL_LEN_Q)
            st = tl.where(token_valid, st_raw, 0)
            mask = token_valid
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]

            q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + st[:, None] * stride_qn + off_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + st[:, None] * stride_don + off_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            if USE_PRECOMPUTED_DELTA:
                delta_ptrs = delta_ptr + pid_h * stride_dh + st[:, None] * stride_dn
                d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            else:
                o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + st[:, None] * stride_on + off_d[None, :] * stride_od
                o_val = tl.load(o_ptrs, mask=q_mask, other=0)
                d = tl.sum(do.to(tl.float32) * o_val.to(tl.float32), axis=1)[:, None]
            lse_ptrs = lse_ptr + pid_h * stride_lh + st[:, None] * stride_ln
            lse = tl.load(lse_ptrs, mask=mask[:, None], other=0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
            if not DISABLE_CAUSAL_MASK:
                qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))

            p = tl.exp2(qk - lse)
            dp = tl.dot(do, v)
            ds = sm_scale * p * (dp - d)
            ds = ds.to(q.dtype)
            dq_part = tl.dot(ds, k).to(tl.float32)

            dq_ptrs = dq_accum_ptr + st[:, None] * stride_dqn + pid_h * stride_dqh + off_d[None, :] * stride_dqd
            tl.atomic_add(dq_ptrs, dq_part, mask=q_mask)


@triton.jit
def dq_reduce_kernel(
    dq_buffer_first_ptr,  # [H, 1, N, D]
    dq_buffer_rest_ptr,  # [H, B, N, D]
    dq_ptr,  # o: n x h x d
    t_ptr,  # topk_idx: h x n x k
    token_index_mapping_ptr,
    num_qz_loop,
    pid_q_offset,
    query_start_idx,
    query_tokens_count,
    head_tile,
    TOPK,
    total_len,
    HEAD_DIM,
    # stride
    stride_dqtfh,
    stride_dqtfb,
    stride_dqtfn,
    stride_dqtfd,
    stride_dqtrh,
    stride_dqtrb,
    stride_dqtrn,
    stride_dqtrd,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_qy = tl.program_id(0)
    pid_q = tl.program_id(1) + pid_q_offset  # token
    pid_h = tl.program_id(2)

    if pid_h >= head_tile:
        return

    pid_q_local = pid_q + pid_qy * num_qz_loop
    if pid_q_local >= query_tokens_count:
        return
    pid_q_j = query_start_idx + pid_q_local
    if pid_q_j >= total_len:
        return
    t_ptr_j = t_ptr + pid_h * stride_th + pid_q_j * stride_tn

    off_d = tl.arange(0, BLOCK_SIZE_D)
    dq_ptrs = dq_ptr + pid_q_j * stride_dqn + pid_h * stride_dqh + off_d * stride_dqd
    acc_dq = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for block_id in range(TOPK):
        t = tl.load(t_ptr_j + block_id * stride_tk, mask=block_id < TOPK, other=-1)
        if t != -1:
            if t == 0:
                dq_buffer_ptr = dq_buffer_first_ptr + pid_h.to(tl.int64) * stride_dqtfh
                stride_dqtb = stride_dqtfb
                stride_dqtn = stride_dqtfn
                stride_dqtd = stride_dqtfd
                real_block_pos = 0
            else:
                dq_buffer_ptr = dq_buffer_rest_ptr + pid_h.to(tl.int64) * stride_dqtrh
                stride_dqtb = stride_dqtrb
                stride_dqtn = stride_dqtrn
                stride_dqtd = stride_dqtrd
                real_block_pos = t - 1

            # init pointers
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + pid_h.to(tl.int64) * stride_tim_h + t.to(tl.int64) * stride_tim_b + (pid_q_j) * stride_tim_n
            )
            real_token_index = tl.load(token_index_mapping_ptrs)
            if real_token_index >= 0:
                dq_buffer_ptrs = (
                    dq_buffer_ptr
                    + real_block_pos.to(tl.int64) * stride_dqtb
                    + (real_token_index) * stride_dqtn
                    + off_d * stride_dqtd
                )

                dq_buffers = tl.load(dq_buffer_ptrs, mask=off_d < HEAD_DIM, other=0.0)
                acc_dq = dq_buffers + acc_dq

    tl.store(dq_ptrs, acc_dq, mask=off_d < HEAD_DIM)


def backward_dq_opt(
    o,  # [total_len, num_heads, head_dim]
    q,  # [total_len, num_heads, head_dim]
    k,  # [total_len, num_k_heads, head_dim]
    v,  # [total_len, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    lse,  # [num_heads, total_len]
    delta,  # [num_heads, total_len]
    do,  # [total_len, num_heads, head_dim]
    dq,  # [total_len, num_heads, head_dim]
    cu_seqlens_q,
    cu_seqlens_k,
    num_k_heads,
    num_share_q_heads,
    head_dim,
    topk,
    sm_scale,
    block_size,
    permute_results,
    dq_block_size_q: Optional[int] = None,
    dq_num_q_blocks: Optional[int] = None,
    disable_causal_mask=False,
):
    """
        Sequence packing is handled at wrapper level for multi-sequence varlen inputs.
        Includes a single-sequence fast path and optional stream-parallel multi-sequence dispatch.
    """
    expected_seqs = int(cu_seqlens_q.numel() - 1)
    if _permute_results_need_rebuild(permute_results) or (
        not isinstance(permute_results, (list, tuple))
    ) or len(permute_results) != expected_seqs:
        permute_results = _build_permute_results_for_bwd(
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            head_dim=int(q.shape[-1]),
        )

    seq_meta, cu_q_local_all, cu_k_local_all = _build_seq_dispatch_meta(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        device=cu_seqlens_q.device,
    )
    if len(permute_results) != len(seq_meta):
        raise RuntimeError(
            f"Mismatched permute_results: len(permute_results)={len(permute_results)} vs sequences={len(seq_meta)}."
        )

    # OPT-12: Multi-sequence wrapper loop elimination for dQ wrapper.
    if len(seq_meta) > 1:
        packed = _pack_varlen_unified_timeline(
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_size=block_size,
            o=o,
            do=do,
            lse=lse,
            delta=delta,
        )
        if packed is not None:
            q_u = packed["q_u"]
            k_u = packed["k_u"]
            v_u = packed["v_u"]
            topk_u = packed["topk_u"]
            o_u = packed["o_u"]
            do_u = packed["do_u"]
            lse_u = packed["lse_u"]
            delta_u = packed["delta_u"]
            cu_u = packed["cu_u"]
            packed_meta = packed["packed_meta"]

            dq_u = torch.zeros_like(q_u)
            perm_flat = _build_permute_results_per_seq_for_bwd(
                topk_idx=topk_u,
                total_len_k=int(k_u.shape[0]),
                block_size=block_size,
                head_dim=int(head_dim),
                ensure_atomic_metadata=True,
            )
            backward_dq_opt_per_seq(
                o_u,
                q_u,
                k_u,
                v_u,
                topk_u,
                lse_u,
                delta_u,
                do_u,
                dq_u,
                cu_u,
                cu_u,
                num_k_heads,
                num_share_q_heads,
                head_dim,
                topk,
                sm_scale,
                block_size,
                perm_flat,
                dq_block_size_q=dq_block_size_q,
                dq_num_q_blocks=dq_num_q_blocks,
                disable_causal_mask=disable_causal_mask,
            )
            _unpack_varlen_unified_q(
                packed_meta=packed_meta,
                src_u=dq_u,
                dst=dq,
                by_head_first=False,
            )
            return dq
        raise RuntimeError("FSA multi-seq flat dQ packing failed unexpectedly.")

    if len(seq_meta) == 1:
        q_start, q_end, k_start, k_end, q_len, k_len = seq_meta[0]
        cu_q_local = cu_q_local_all[0]
        cu_k_local = cu_k_local_all[0]
        backward_dq_opt_per_seq(
            o[q_start:q_end],
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            lse[:, q_start:q_end],
            delta[:, q_start:q_end],
            do[q_start:q_end],
            dq[q_start:q_end],
            cu_q_local,
            cu_k_local,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            sm_scale,
            block_size,
            permute_results[0],
            dq_block_size_q=dq_block_size_q,
            dq_num_q_blocks=dq_num_q_blocks,
            disable_causal_mask=disable_causal_mask,
        )
        return dq

    raise RuntimeError("Unexpected dQ wrapper state: multi-seq path must run via flat packed timeline.")


def backward_dq_opt_per_seq(
    o,  # [total_len, num_heads, head_dim]
    q,  # [total_len, num_heads, head_dim]
    k,  # [total_len_k, num_k_heads, head_dim]
    v,  # [total_len_k, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    lse,  # [num_heads, total_len]
    delta,  # [num_heads, total_len]
    do,  # [total_len, num_heads, head_dim]
    dq,  # [total_len, num_heads, head_dim]
    cu_seqlens_q,
    cu_seqlens_k,
    num_k_heads,
    num_share_q_heads,
    head_dim,
    topk,
    sm_scale,
    block_size,
    permute_results,
    dq_block_size_q: Optional[int] = None,
    dq_num_q_blocks: Optional[int] = None,
    disable_causal_mask=False,
):
    """
    dQ backward (per-sequence).

    Key optimizations vs the original local copy:
      1) Build token_index_mapping **once per KV head** (shared across its G query heads),
         instead of re-running index_mapping for every query head.
      2) Avoid the expensive GPU scan + CPU sync per head in `_detect_active_token_range`
         by reusing the per-KV-head result (and using a fast prefix-mode heuristic when applicable).
      3) Allocate dQ staging buffers based on the **max per-block valid count**, not `total_len`,
         and use `torch.empty` (no memset) since kernels overwrite all read elements.
      4) Optional staging dtype control via `FSA_LOCAL_DQ_BUFFER_DTYPE`:
         - auto (default): fp32 (more stable dQ accumulation)
         - fp16 / bf16 / fp32
    """
    num_q_heads = int(q.shape[1])
    head_tile = _resolve_head_tile(num_share_q_heads)
    total_len = int(topk_idx.shape[1])
    total_len_k = int(k.shape[0])
    dq_mode = os.getenv("FSA_LOCAL_DQ_ACCUM_MODE", "atomic").strip().lower()
    dq_mode_explicit = "FSA_LOCAL_DQ_ACCUM_MODE" in os.environ
    use_atomic_dq = dq_mode in ("", "auto", "atomic", "1", "true", "yes", "on")

    force_atomic_raw = os.environ.get("FSA_LOCAL_DQ_FORCE_ATOMIC", None)
    force_atomic_explicit = force_atomic_raw is not None
    if force_atomic_raw is None:
        force_atomic_dq = True
    else:
        force_atomic_dq = force_atomic_raw.strip().lower() not in ("0", "false", "no", "off", "")

    # Full dQ de-serialization mode: force all-query-head atomic path.
    dq_full_deser_raw = os.environ.get("FSA_LOCAL_DQ_FULL_DESERIALIZE", None)
    dq_full_deser_explicit = dq_full_deser_raw is not None
    if dq_full_deser_raw is None:
        dq_full_deser = True
    else:
        dq_full_deser = dq_full_deser_raw.strip().lower() not in ("0", "false", "no", "off", "")
    if dq_full_deser:
        use_atomic_dq = True
    if force_atomic_dq:
        use_atomic_dq = True
    use_precomputed_delta = os.getenv("FSA_LOCAL_DQ_USE_PRECOMPUTED_DELTA", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )

    # Safety guard for known-fragile long-sequence atomic dQ regime on some Triton/CUDA stacks:
    # when user did not explicitly force atomic/full-deser, auto-fallback to buffered dQ.
    dq_atomic_guard = os.getenv("FSA_LOCAL_DQ_ATOMIC_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    risky_atomic_regime = (
        num_k_heads == 1
        and num_share_q_heads >= 16
        and block_size >= 128
        and total_len >= 131072
    )
    if (
        dq_atomic_guard
        and use_atomic_dq
        and risky_atomic_regime
        and (not force_atomic_explicit)
        and (not dq_full_deser_explicit)
        and (not dq_mode_explicit)
    ):
        use_atomic_dq = False
        dq_full_deser = False

    num_blocks = int(permute_results["num_blocks"])
    reduce_tile_size = max(0, num_blocks - 1)
    valid_lens_all = permute_results["valid_lens_all"]
    max_tokens_any_block = int(valid_lens_all.max().to(dtype=torch.int32).cpu().tolist()) if valid_lens_all.numel() > 0 else 0
    global_max_valid_tokens = (
        int(permute_results["global_max_valid_tokens"])
        if num_blocks > 1
        else max_tokens_any_block
    )

    # No routed queries at all -> dQ is zero.
    if max_tokens_any_block <= 0:
        return dq

    # OPT-5: Use cached workspace buffers to avoid re-allocating every backward call.
    _use_workspace_cache = os.getenv("FSA_LOCAL_WORKSPACE_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if use_atomic_dq:
        _dq_shape = (dq.shape[0], dq.shape[1], dq.shape[2])
        if _use_workspace_cache:
            dq_accum = _get_cached_zeros("dq_accum", _dq_shape, torch.float32, dq.device)
        else:
            dq_accum = torch.zeros(_dq_shape, dtype=torch.float32, device=dq.device)
        dq_buffer_first = None
        dq_buffer_rest = None
        token_index_mapping = None
    else:
        # Choose staging dtype (defaults to fp32 for stable dQ accumulation).
        dq_buf_mode = os.getenv("FSA_LOCAL_DQ_BUFFER_DTYPE", "auto").strip().lower()
        if dq_buf_mode in ("fp32", "f32", "float32"):
            dq_buf_dtype = torch.float32
        elif dq_buf_mode in ("bf16", "bfloat16"):
            dq_buf_dtype = torch.bfloat16
        elif dq_buf_mode in ("fp16", "f16", "float16"):
            dq_buf_dtype = torch.float16
        else:  # auto / unknown
            dq_buf_dtype = torch.float32

        # Staging buffers:
        # - block 0 uses its own buffer so we can index [t==0] without subtracting 1.
        _bf_shape = (head_tile, 1, max_tokens_any_block, head_dim)
        if _use_workspace_cache:
            dq_buffer_first = _get_cached_empty("dq_buf_first", _bf_shape, dq_buf_dtype, dq.device)
        else:
            dq_buffer_first = torch.empty(_bf_shape, dtype=dq_buf_dtype, device=dq.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            _br_shape = (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim)
            if _use_workspace_cache:
                dq_buffer_rest = _get_cached_empty("dq_buf_rest", _br_shape, dq_buf_dtype, dq.device)
            else:
                dq_buffer_rest = torch.empty(_br_shape, dtype=dq_buf_dtype, device=dq.device)
        else:
            # Minimal dummy tensor (won't be read if num_blocks==1)
            dq_buffer_rest = torch.empty((head_tile, 1, 1, head_dim), dtype=dq_buf_dtype, device=dq.device)

        # Dense mapping: token -> position inside each block's compacted list.
        # Use empty() since index_mapping overwrites all entries that will ever be read.
        _im_shape = (1, num_blocks, total_len)
        dq_safe_token_index = os.getenv("FSA_LOCAL_DQ_SAFE_TOKEN_INDEX", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        if _use_workspace_cache:
            token_index_mapping = _get_cached_empty("token_idx_map", _im_shape, torch.int32, q.device)
        else:
            token_index_mapping = torch.empty(_im_shape, dtype=torch.int32, device=q.device)
        if dq_safe_token_index:
            # Safety-first default: avoid stale/uninitialized indices in reduce path.
            token_index_mapping.fill_(-1)

    # Precompute static kernel META once.
    BLOCK_SIZE_Q = int(dq_block_size_q) if dq_block_size_q is not None else _resolve_bwd_dq_bq(
        head_dim=head_dim, block_size=block_size
    )
    num_dq_blocks = int(dq_num_q_blocks) if dq_num_q_blocks is not None else _resolve_bwd_dq_num_q_blocks(
        block_size=block_size
    )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    _base_warps, _base_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="bwd_dq",
        head_dim=head_dim,
        block_size=block_size,
        default_warps=_base_warps,
        default_stages=_base_stages,
    )
    dq_reduce_warps, dq_reduce_stages = _resolve_launch_warps_stages(
        op="bwd_dq",
        head_dim=head_dim,
        block_size=max(32, int(topk)),
        default_warps=1,
        default_stages=2,
    )

    # Process per KV head once (mapping + active range), then loop its share-Q heads.
    if not use_atomic_dq:
        if total_len > total_len_k:
            prefix_mask_per_kh = (topk_idx[:, 0, :] < 0).all(dim=-1).to(dtype=torch.bool)
        else:
            prefix_mask_per_kh = torch.zeros((num_k_heads,), dtype=torch.bool, device=topk_idx.device)
        active_starts, active_counts = _detect_active_token_ranges_per_kv_head(topk_idx)
        prefix_mask_list = [bool(x) for x in prefix_mask_per_kh.tolist()]
        active_pairs = torch.stack((active_starts, active_counts), dim=1).to(torch.int32).tolist()

    # Atomic dQ fast path: de-serialize across all query heads at once.
    # This removes the Python per-KV-head loop in the hot backward path.
    if use_atomic_dq:
        permute_results = _ensure_dq_atomic_metadata(
            permute_results=permute_results,
            num_kv_heads=num_k_heads,
            num_blocks=num_blocks,
            device=q.device,
        )
        qh_to_kh = torch.div(
            torch.arange(num_q_heads, device=q.device, dtype=torch.int64),
            num_share_q_heads,
            rounding_mode="floor",
        )
        # P0.3: consume prebuilt metadata to avoid host-side per-head orchestration.
        selected_tokens_all = permute_results["valid_topk_idx_concat"]
        kh_offsets = permute_results["valid_topk_idx_offsets"]
        valid_lens_stack = permute_results["valid_lens_stack"]
        valid_start_stack = permute_results["valid_start_indices_stack"]
        if int(selected_tokens_all.numel()) <= 0:
            return dq

        use_packed_gqa = _resolve_dq_packed_gqa(
            num_share_q_heads=num_share_q_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        if num_share_q_heads <= 1:
            use_packed_gqa = False

        if not use_packed_gqa:
            valid_lens_qh = valid_lens_stack.index_select(0, qh_to_kh)  # [HQ, num_blocks]
            valid_start_qh = (
                valid_start_stack.index_select(0, qh_to_kh)
                + kh_offsets.index_select(0, qh_to_kh).view(-1, 1)
            )
        else:
            valid_lens_qh = None
            valid_start_qh = None

        # Use full head de-serialization by default in atomic mode.
        # Fallback to tiled mode only when explicitly disabled.
        if use_packed_gqa and num_share_q_heads > 1:
            # Keep tiles aligned to KV-head groups so K/V pointers can be broadcast
            # with stride_h=0 instead of materialized index_select copies.
            head_tile_qh = num_share_q_heads
        elif dq_full_deser:
            head_tile_qh = num_q_heads
        else:
            head_tile_qh = _resolve_head_tile(num_q_heads)
        head_tile_qh = max(1, min(head_tile_qh, num_q_heads))
        BLOCK_SIZE_Q = _resolve_bwd_dq_bq(head_dim=head_dim, block_size=block_size)
        num_dq_blocks = _resolve_bwd_dq_num_q_blocks(block_size=block_size)
        BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
        BLOCK_SIZE_K = triton.next_power_of_2(block_size)
        _base_warps, _base_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
        num_warps, num_stages = _resolve_launch_warps_stages(
            op="bwd_dq",
            head_dim=head_dim,
            block_size=block_size,
            default_warps=_base_warps,
            default_stages=_base_stages,
        )

        # OPT-1: Pre-compute CPU-side head mapping and max valid tokens to eliminate
        # per-tile GPU->CPU .item() syncs in the hot loop.
        _qh_to_kh_cpu = [h // num_share_q_heads for h in range(num_q_heads)]
        _max_valid_per_kh = valid_lens_stack.max(dim=1).values  # [HK]
        if valid_lens_qh is not None:
            _max_valid_per_qh = valid_lens_qh.max(dim=1).values  # [HQ]
            _mvt_batch = torch.cat([_max_valid_per_kh, _max_valid_per_qh]).tolist()
            _mvt_kh = [int(x) for x in _mvt_batch[:num_k_heads]]
            _mvt_qh = [int(x) for x in _mvt_batch[num_k_heads:]]
        else:
            _mvt_kh = [int(x) for x in _max_valid_per_kh.tolist()]
            _mvt_qh = None

        for h_start in range(0, num_q_heads, head_tile_qh):
            h_end = min(num_q_heads, h_start + head_tile_qh)
            ht = h_end - h_start
            kh_idx = qh_to_kh[h_start:h_end]

            q_tile = q[:, h_start:h_end]
            o_tile = o[:, h_start:h_end]
            do_tile = do[:, h_start:h_end]
            lse_tile = lse[h_start:h_end]
            delta_tile = delta[h_start:h_end]
            dq_accum_tile = dq_accum[:, h_start:h_end]

            packed_tile = bool(use_packed_gqa) and all(
                _qh_to_kh_cpu[h] == _qh_to_kh_cpu[h_start] for h in range(h_start, h_end)
            )
            if packed_tile:
                kh0 = _qh_to_kh_cpu[h_start]
                k_base = k[:, kh0:kh0 + 1]
                v_base = v[:, kh0:kh0 + 1]
                if ht == 1:
                    k_tile = k_base
                    v_tile = v_base
                else:
                    k_tile = k_base.expand(-1, ht, -1)
                    v_tile = v_base.expand(-1, ht, -1)

                base_lens = valid_lens_stack[kh0]
                base_starts = valid_start_stack[kh0] + kh_offsets[kh0]
                valid_lens_tile = base_lens.view(1, -1).expand(ht, -1)
                valid_start_indices_tile = base_starts.view(1, -1).expand(ht, -1)
            else:
                k_tile = k.index_select(1, kh_idx)
                v_tile = v.index_select(1, kh_idx)
                valid_lens_tile = valid_lens_qh[h_start:h_end]
                valid_start_indices_tile = valid_start_qh[h_start:h_end]

            # Single-pass over all blocks per head-tile (compute_min_block_id=0, tile_size=num_blocks).
            # OPT-1: Use pre-computed CPU max instead of per-tile .item() sync.
            if packed_tile:
                cur_max_valid_tokens = _mvt_kh[kh0] if kh0 < len(_mvt_kh) else 0
            elif _mvt_qh is not None:
                cur_max_valid_tokens = int(max(_mvt_qh[h_start:h_end])) if h_end > h_start else 0
            else:
                cur_max_valid_tokens = int(valid_lens_tile.max().to(dtype=torch.int32).cpu().tolist()) if valid_lens_tile.numel() > 0 else 0
            if cur_max_valid_tokens <= 0:
                continue
            grid_dq = lambda META: (
                num_blocks * ht,
                triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_dq_blocks),
            )
            dq_compute_atomic_kernel[grid_dq](
                q_tile,
                k_tile,
                v_tile,
                o_tile,
                lse_tile,
                delta_tile,
                do_tile,
                dq_accum_tile,
                selected_tokens_all,
                valid_lens_tile,
                valid_start_indices_tile,
                cur_max_valid_tokens,
                0,
                ht,
                num_blocks,
                head_dim,
                total_len,
                cu_seqlens_k,
                num_dq_blocks,
                sm_scale,
                q_tile.stride(0),
                q_tile.stride(1),
                q_tile.stride(2),
                k_tile.stride(0),
                k_tile.stride(1),
                k_tile.stride(2),
                v_tile.stride(0),
                v_tile.stride(1),
                v_tile.stride(2),
                do_tile.stride(0),
                do_tile.stride(1),
                do_tile.stride(2),
                o_tile.stride(0),
                o_tile.stride(1),
                o_tile.stride(2),
                lse_tile.stride(0),
                lse_tile.stride(1),
                delta_tile.stride(0),
                delta_tile.stride(1),
                dq_accum_tile.stride(0),
                dq_accum_tile.stride(1),
                dq_accum_tile.stride(2),
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                DISABLE_CAUSAL_MASK=disable_causal_mask,
                USE_PRECOMPUTED_DELTA=use_precomputed_delta,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        dq.copy_(dq_accum.to(dq.dtype))
        return dq

    for kh in range(num_k_heads):
        valid_topk_idx_permuted_tile = permute_results["valid_topk_idx_permuted_tile"][kh]
        valid_lens = permute_results["valid_lens"][kh]
        valid_start_indices = permute_results["valid_start_indices"][kh]
        # Build token->rank mapping for this KV head (shared across G query heads).
        if not use_atomic_dq:
            index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

        topk_idx_tile_base = topk_idx[kh:kh + 1]

        if not use_atomic_dq:
            # Active token range for this KV head.
            # - Prefix-mode fast path: [memory-prefix | query] with no routes on the prefix.
            prefix_mode = prefix_mask_list[kh]
            if prefix_mode:
                query_start_idx = total_len_k
                query_tokens_count = total_len - total_len_k
            else:
                query_start_idx = int(active_pairs[kh][0])
                query_tokens_count = int(active_pairs[kh][1])

            if query_tokens_count <= 0:
                continue
        valid_lens_host = [int(x) for x in valid_lens.to(dtype=torch.int32).tolist()]
        max_valid_first = valid_lens_host[0] if len(valid_lens_host) > 0 else 0
        max_valid_rest = max(valid_lens_host[1:], default=0)

        # Iterate over the G query heads that share this KV head in head tiles.
        for sh0 in range(0, num_share_q_heads, head_tile):
            ht = min(head_tile, num_share_q_heads - sh0)
            h_start = kh * num_share_q_heads + sh0
            h_end = h_start + ht

            q_tile = q[:, h_start:h_end]
            o_tile = o[:, h_start:h_end]
            do_tile = do[:, h_start:h_end]
            lse_tile = lse[h_start:h_end]
            delta_tile = delta[h_start:h_end]
            dq_tile = dq[:, h_start:h_end]

            # Shared KV head for this group; stride_h may be 0 for ht>1 (intentional broadcast).
            k_base = k[:, kh:kh + 1]
            v_base = v[:, kh:kh + 1]
            if ht == 1:
                k_tile = k_base
                v_tile = v_base
                topk_idx_tile = topk_idx_tile_base
                token_index_mapping_tile = token_index_mapping if token_index_mapping is not None else None
            else:
                k_tile = k_base.expand(-1, ht, -1)
                v_tile = v_base.expand(-1, ht, -1)
                topk_idx_tile = topk_idx_tile_base.expand(ht, -1, -1)
                if token_index_mapping is not None:
                    token_index_mapping_tile = token_index_mapping.expand(ht, -1, -1)
                else:
                    token_index_mapping_tile = None

            valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1)
            valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1)

            # Two-stage: (block 0) + (blocks 1..num_blocks-1) to keep the "t==0" and "t>0" buffers compact.
            for compute_min_block_id in range(min(2, num_blocks)):
                if compute_min_block_id == 0:
                    compute_tile_size = 1
                    cur_max_valid_tokens = max_valid_first
                    cur_valid_lens = valid_lens_tile[:, 0]
                    cur_valid_start_indices = valid_start_indices_tile[:, 0]
                    if not use_atomic_dq:
                        dq_buffer = dq_buffer_first[:ht]
                    else:
                        dq_buffer = None
                else:
                    compute_tile_size = num_blocks - 1
                    # If there are no blocks > 0, skip.
                    if compute_tile_size <= 0:
                        continue
                    cur_valid_lens = valid_lens_tile[:, compute_min_block_id:]
                    if cur_valid_lens.numel() == 0:
                        continue
                    cur_max_valid_tokens = max_valid_rest
                    if cur_max_valid_tokens <= 0:
                        continue
                    cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:]
                    if not use_atomic_dq:
                        dq_buffer = dq_buffer_rest[:ht]
                    else:
                        dq_buffer = None

                if cur_max_valid_tokens <= 0:
                    continue

                grid_dq = lambda META: (
                    compute_tile_size * ht,
                    triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_dq_blocks),
                )

                if use_atomic_dq:
                    dq_accum_tile = dq_accum[:, h_start:h_end]
                    dq_compute_atomic_kernel[grid_dq](
                        q_tile,
                        k_tile,
                        v_tile,
                        o_tile,
                        lse_tile,
                        delta_tile,
                        do_tile,
                        dq_accum_tile,
                        valid_topk_idx_permuted_tile,
                        cur_valid_lens,
                        cur_valid_start_indices,
                        cur_max_valid_tokens,
                        compute_min_block_id,
                        ht,
                        compute_tile_size,
                        head_dim,
                        total_len,
                        cu_seqlens_k,
                        num_dq_blocks,
                        sm_scale,
                        q_tile.stride(0),
                        q_tile.stride(1),
                        q_tile.stride(2),
                        k_tile.stride(0),
                        k_tile.stride(1),
                        k_tile.stride(2),
                        v_tile.stride(0),
                        v_tile.stride(1),
                        v_tile.stride(2),
                        do_tile.stride(0),
                        do_tile.stride(1),
                        do_tile.stride(2),
                        o_tile.stride(0),
                        o_tile.stride(1),
                        o_tile.stride(2),
                        lse_tile.stride(0),
                        lse_tile.stride(1),
                        delta_tile.stride(0),
                        delta_tile.stride(1),
                        dq_accum_tile.stride(0),
                        dq_accum_tile.stride(1),
                        dq_accum_tile.stride(2),
                        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        BLOCK_SIZE_D=BLOCK_SIZE_D,
                        DISABLE_CAUSAL_MASK=disable_causal_mask,
                        USE_PRECOMPUTED_DELTA=use_precomputed_delta,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                else:
                    dq_compute_kernel[grid_dq](
                        q_tile,
                        k_tile,
                        v_tile,
                        o_tile,
                        lse_tile,
                        delta_tile,
                        do_tile,
                        dq_buffer,
                        token_index_mapping_tile,
                        valid_topk_idx_permuted_tile,
                        cur_valid_lens,
                        cur_valid_start_indices,
                        cur_max_valid_tokens,
                        compute_min_block_id,
                        ht,
                        compute_tile_size,
                        head_dim,
                        total_len,
                        cu_seqlens_k,
                        num_dq_blocks,
                        sm_scale,
                        q_tile.stride(0),
                        q_tile.stride(1),
                        q_tile.stride(2),
                        k_tile.stride(0),
                        k_tile.stride(1),
                        k_tile.stride(2),
                        v_tile.stride(0),
                        v_tile.stride(1),
                        v_tile.stride(2),
                        do_tile.stride(0),
                        do_tile.stride(1),
                        do_tile.stride(2),
                        o_tile.stride(0),
                        o_tile.stride(1),
                        o_tile.stride(2),
                        lse_tile.stride(0),
                        lse_tile.stride(1),
                        delta_tile.stride(0),
                        delta_tile.stride(1),
                        token_index_mapping_tile.stride(0),
                        token_index_mapping_tile.stride(1),
                        token_index_mapping_tile.stride(2),
                        dq_buffer.stride(0),
                        dq_buffer.stride(1),
                        dq_buffer.stride(2),
                        dq_buffer.stride(3),
                        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        BLOCK_SIZE_D=BLOCK_SIZE_D,
                        DISABLE_CAUSAL_MASK=disable_causal_mask,
                        USE_PRECOMPUTED_DELTA=use_precomputed_delta,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )

            if not use_atomic_dq:
                # Reduce per-query across its top-k blocks.
                num_qy_loop = 4
                num_qz_loop = max(1, query_tokens_count // num_qy_loop)
                grid_x = num_qy_loop + (query_tokens_count % num_qy_loop != 0)
                max_grid_y = 65535

                for q_off in range(0, num_qz_loop, max_grid_y):
                    grid_y = min(max_grid_y, num_qz_loop - q_off)
                    grid_reduce = (grid_x, grid_y, ht)

                    dq_reduce_kernel[grid_reduce](
                        dq_buffer_first[:ht],
                        dq_buffer_rest[:ht],
                        dq_tile,
                        topk_idx_tile,
                        token_index_mapping_tile,
                        num_qz_loop,
                        q_off,
                        query_start_idx,
                        query_tokens_count,
                        ht,
                        topk,
                        total_len,
                        head_dim,
                        dq_buffer_first.stride(0),
                        dq_buffer_first.stride(1),
                        dq_buffer_first.stride(2),
                        dq_buffer_first.stride(3),
                        dq_buffer_rest.stride(0),
                        dq_buffer_rest.stride(1),
                        dq_buffer_rest.stride(2),
                        dq_buffer_rest.stride(3),
                        dq_tile.stride(0),
                        dq_tile.stride(1),
                        dq_tile.stride(2),
                        topk_idx_tile.stride(0),
                        topk_idx_tile.stride(1),
                        topk_idx_tile.stride(2),
                        token_index_mapping_tile.stride(0),
                        token_index_mapping_tile.stride(1),
                        token_index_mapping_tile.stride(2),
                        BLOCK_SIZE_T=triton.next_power_of_2(topk),
                        BLOCK_SIZE_D=BLOCK_SIZE_D,
                        num_warps=dq_reduce_warps,
                        num_stages=dq_reduce_stages,
                    )

                dq[:, h_start:h_end] = dq_tile

    if use_atomic_dq:
        dq.copy_(dq_accum.to(dq.dtype))

    return dq


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    o_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DK: sh x n x kh x d
    # seqlens
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    cu_topk_q_count,  # [kh, total_blocks]
    active_block_idx,  # [kh, batch, max_active_blocks], local block ids per (kh,b)
    active_block_count,  # [kh, batch]
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tqh,
    stride_tqn,
    stride_ctqh,
    stride_ctqn,
    stride_abh,
    stride_abb,
    stride_abk,
    stride_ac_h,
    stride_ac_b,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    PIPELINE_CHUNKS: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    USE_ACTIVE_BLOCK_MAP: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if USE_ACTIVE_BLOCK_MAP:
        num_active = tl.load(active_block_count + pid_kh * stride_ac_h + pid_b * stride_ac_b).to(tl.int32)
        if pid_k >= num_active:
            return
        real_pid_k = tl.load(
            active_block_idx + pid_kh * stride_abh + pid_b * stride_abb + pid_k * stride_abk
        ).to(tl.int32)
        if real_pid_k < 0:
            return
    else:
        real_pid_k = pid_k
    if BLOCK_SIZE_K * real_pid_k >= k_len:
        return
    # get topk_q_idx
    b_start = tl.load(cu_seqblocks + pid_b)  # how many blocks before current sequence
    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k) * stride_ctqn)
    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k + 1) * stride_ctqn)
    act_q_len = act_q_end - act_q_start
    if act_q_len <= 0:
        return
    tq_ptr = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + real_pid_k * BLOCK_SIZE_K
    off_d = tl.arange(0, BLOCK_SIZE_D)
    # load k v and keep in SRAM
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dk dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    # init ptrs
    q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + off_d[None, :] * stride_qd
    do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod
    o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od
    d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
    lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh
    # loop for q blocks
    step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
    for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
        for u in tl.static_range(0, PIPELINE_CHUNKS):
            i = ib + u * BLOCK_SIZE_Q
            if i < act_q_len:
                # load
                idx_q = tl.load(tq_ptr + i + off_q, mask=off_q < act_q_len - i, other=0).to(tl.int32)
                q = tl.load(
                    q_ptrs + idx_q[:, None] * stride_qn,
                    mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                    other=0,
                )
                do = tl.load(
                    do_ptrs + idx_q[:, None] * stride_don,
                    mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                    other=0,
                )
                lse = tl.load(
                    lse_ptrs + idx_q[:, None] * stride_ln,
                    mask=(off_q < act_q_len - i)[:, None],
                    other=0,
                )
                if USE_PRECOMPUTED_DELTA:
                    d = tl.load(
                        d_ptrs + idx_q[:, None] * stride_dn,
                        mask=(off_q < act_q_len - i)[:, None],
                        other=0,
                    )
                else:
                    o = tl.load(
                        o_ptrs + idx_q[:, None] * stride_on,
                        mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    d = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)[:, None]
                # compute qk
                qk = tl.dot(q, k.T) * qk_scale
                if not DISABLE_CAUSAL_MASK:
                    qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                # compute p, ds
                p = tl.exp2(qk - lse)
                dp = tl.dot(do, v.T)
                ds = sm_scale * p * (dp - d)
                # cast dtype
                p = p.to(do.dtype)
                ds = ds.to(q.dtype)
                # update dk and dv
                dk += tl.dot(ds.T, q)
                dv += tl.dot(p.T, do)
    # save dk dv
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dkdv_gqa_fused(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    o_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    # seqlens
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    cu_topk_q_count,  # [kh, total_blocks + 1]
    active_block_idx,  # [kh, batch, max_active_blocks], local block ids per (kh,b)
    active_block_count,  # [kh, batch]
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tqh,
    stride_tqn,
    stride_ctqh,
    stride_ctqn,
    stride_abh,
    stride_abb,
    stride_abk,
    stride_ac_h,
    stride_ac_b,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    PIPELINE_CHUNKS: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    USE_ACTIVE_BLOCK_MAP: tl.constexpr,
    COMPUTE_DK: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_k = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    if USE_ACTIVE_BLOCK_MAP:
        num_active = tl.load(active_block_count + pid_kh * stride_ac_h + pid_b * stride_ac_b).to(tl.int32)
        if pid_k >= num_active:
            return
        real_pid_k = tl.load(
            active_block_idx + pid_kh * stride_abh + pid_b * stride_abb + pid_k * stride_abk
        ).to(tl.int32)
        if real_pid_k < 0:
            return
    else:
        real_pid_k = pid_k

    if BLOCK_SIZE_K * real_pid_k >= k_len:
        return

    b_start = tl.load(cu_seqblocks + pid_b)
    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k) * stride_ctqn)
    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k + 1) * stride_ctqn)
    act_q_len = act_q_end - act_q_start
    if act_q_len <= 0:
        return

    tq_ptr = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn

    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + real_pid_k * BLOCK_SIZE_K
    off_d = tl.arange(0, BLOCK_SIZE_D)

    k_blk = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v_blk = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

    for pid_sh in range(NUM_SHARE_Q_HEADS):
        pid_h = pid_kh * NUM_SHARE_Q_HEADS + pid_sh
        q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + off_d[None, :] * stride_qd
        do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod
        o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od
        d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
        lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

        step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
        for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
            for u in tl.static_range(0, PIPELINE_CHUNKS):
                i = ib + u * BLOCK_SIZE_Q
                if i < act_q_len:
                    idx_q = tl.load(tq_ptr + i + off_q, mask=off_q < act_q_len - i, other=0).to(tl.int32)
                    q = tl.load(
                        q_ptrs + idx_q[:, None] * stride_qn,
                        mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    do = tl.load(
                        do_ptrs + idx_q[:, None] * stride_don,
                        mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    lse = tl.load(
                        lse_ptrs + idx_q[:, None] * stride_ln,
                        mask=(off_q < act_q_len - i)[:, None],
                        other=0,
                    )
                    if USE_PRECOMPUTED_DELTA:
                        d = tl.load(
                            d_ptrs + idx_q[:, None] * stride_dn,
                            mask=(off_q < act_q_len - i)[:, None],
                            other=0,
                        )
                    else:
                        o = tl.load(
                            o_ptrs + idx_q[:, None] * stride_on,
                            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                            other=0,
                        )
                        d = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)[:, None]
                    qk = tl.dot(q, k_blk.T) * qk_scale
                    if not DISABLE_CAUSAL_MASK:
                        qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                    p = tl.exp2(qk - lse)
                    if COMPUTE_DV:
                        p_cast = p.to(do.dtype)
                        dv += tl.dot(p_cast.T, do)
                    if COMPUTE_DK:
                        dp = tl.dot(do, v_blk.T)
                        ds = sm_scale * p * (dp - d)
                        ds = ds.to(q.dtype)
                        dk += tl.dot(ds.T, q)

    if COMPUTE_DK:
        tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    if COMPUTE_DV:
        tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dkdv_gqa_fused_worklist(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    o_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    # seqlens
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    cu_topk_q_count,  # [kh, total_blocks + 1]
    worklist_ptr,  # [N, 3] => (batch, kv_head, local_k_block)
    num_work_items,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tqh,
    stride_tqn,
    stride_ctqh,
    stride_ctqn,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_wln,
    stride_wlc,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    PIPELINE_CHUNKS: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    COMPUTE_DK: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_work_items:
        return

    qk_scale = sm_scale * 1.44269504
    pid_b = tl.load(worklist_ptr + pid * stride_wln + 0 * stride_wlc).to(tl.int32)
    pid_kh = tl.load(worklist_ptr + pid * stride_wln + 1 * stride_wlc).to(tl.int32)
    real_pid_k = tl.load(worklist_ptr + pid * stride_wln + 2 * stride_wlc).to(tl.int32)

    q_start = tl.load(cu_seqlens_q + pid_b)
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    if BLOCK_SIZE_K * real_pid_k >= k_len:
        return

    b_start = tl.load(cu_seqblocks + pid_b)
    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k) * stride_ctqn)
    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k + 1) * stride_ctqn)
    act_q_len = act_q_end - act_q_start
    if act_q_len <= 0:
        return

    tq_ptr = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn

    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(real_pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + real_pid_k * BLOCK_SIZE_K
    off_d = tl.arange(0, BLOCK_SIZE_D)

    k_blk = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v_blk = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

    for pid_sh in range(NUM_SHARE_Q_HEADS):
        pid_h = pid_kh * NUM_SHARE_Q_HEADS + pid_sh
        q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + off_d[None, :] * stride_qd
        do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod
        o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od
        d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
        lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

        step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
        for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
            for u in tl.static_range(0, PIPELINE_CHUNKS):
                i = ib + u * BLOCK_SIZE_Q
                if i < act_q_len:
                    idx_q = tl.load(tq_ptr + i + off_q, mask=off_q < act_q_len - i, other=0).to(tl.int32)
                    q = tl.load(
                        q_ptrs + idx_q[:, None] * stride_qn,
                        mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    do = tl.load(
                        do_ptrs + idx_q[:, None] * stride_don,
                        mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    lse = tl.load(
                        lse_ptrs + idx_q[:, None] * stride_ln,
                        mask=(off_q < act_q_len - i)[:, None],
                        other=0,
                    )
                    if USE_PRECOMPUTED_DELTA:
                        d = tl.load(
                            d_ptrs + idx_q[:, None] * stride_dn,
                            mask=(off_q < act_q_len - i)[:, None],
                            other=0,
                        )
                    else:
                        o = tl.load(
                            o_ptrs + idx_q[:, None] * stride_on,
                            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                            other=0,
                        )
                        d = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)[:, None]
                    qk = tl.dot(q, k_blk.T) * qk_scale
                    if not DISABLE_CAUSAL_MASK:
                        qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                    p = tl.exp2(qk - lse)
                    if COMPUTE_DV:
                        p_cast = p.to(do.dtype)
                        dv += tl.dot(p_cast.T, do)
                    if COMPUTE_DK:
                        dp = tl.dot(do, v_blk.T)
                        ds = sm_scale * p * (dp - d)
                        ds = ds.to(q.dtype)
                        dk += tl.dot(ds.T, q)

    if COMPUTE_DK:
        tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    if COMPUTE_DV:
        tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dkdv_gqa_fused_persistent_queue(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    o_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    # seqlens
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    cu_topk_q_count,  # [kh, total_blocks + 1]
    worklist_ptr,  # [N, 3] => (batch, kv_head, local_k_block)
    queue_ptr,  # int32 scalar cursor
    num_work_items,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tqh,
    stride_tqn,
    stride_ctqh,
    stride_ctqn,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_on,
    stride_oh,
    stride_od,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_wln,
    stride_wlc,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    PIPELINE_CHUNKS: tl.constexpr,
    WORK_STEAL_CHUNK: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    COMPUTE_DK: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
    USE_PRECOMPUTED_DELTA: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    wid = tl.atomic_add(queue_ptr, WORK_STEAL_CHUNK)
    while wid < num_work_items:
        for chunk_i in tl.static_range(0, WORK_STEAL_CHUNK):
            pid = wid + chunk_i
            if pid < num_work_items:
                pid_b = tl.load(worklist_ptr + pid * stride_wln + 0 * stride_wlc).to(tl.int32)
                pid_kh = tl.load(worklist_ptr + pid * stride_wln + 1 * stride_wlc).to(tl.int32)
                real_pid_k = tl.load(worklist_ptr + pid * stride_wln + 2 * stride_wlc).to(tl.int32)

                q_start = tl.load(cu_seqlens_q + pid_b)
                k_start = tl.load(cu_seqlens_k + pid_b)
                k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

                if BLOCK_SIZE_K * real_pid_k < k_len:
                    b_start = tl.load(cu_seqblocks + pid_b)
                    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k) * stride_ctqn)
                    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + real_pid_k + 1) * stride_ctqn)
                    act_q_len = act_q_end - act_q_start
                    if act_q_len > 0:
                        tq_ptr_w = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn

                        k_ptrs = tl.make_block_ptr(
                            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
                            shape=(k_len, HEAD_DIM),
                            strides=(stride_kn, stride_kd),
                            offsets=(real_pid_k * BLOCK_SIZE_K, 0),
                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                            order=(1, 0),
                        )
                        v_ptrs = tl.make_block_ptr(
                            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
                            shape=(k_len, HEAD_DIM),
                            strides=(stride_vn, stride_vd),
                            offsets=(real_pid_k * BLOCK_SIZE_K, 0),
                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                            order=(1, 0),
                        )
                        dk_ptrs = tl.make_block_ptr(
                            base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh,
                            shape=(k_len, HEAD_DIM),
                            strides=(stride_dkn, stride_dkd),
                            offsets=(real_pid_k * BLOCK_SIZE_K, 0),
                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                            order=(1, 0),
                        )
                        dv_ptrs = tl.make_block_ptr(
                            base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh,
                            shape=(k_len, HEAD_DIM),
                            strides=(stride_dvn, stride_dvd),
                            offsets=(real_pid_k * BLOCK_SIZE_K, 0),
                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                            order=(1, 0),
                        )

                        off_q = tl.arange(0, BLOCK_SIZE_Q)
                        off_k = tl.arange(0, BLOCK_SIZE_K) + real_pid_k * BLOCK_SIZE_K
                        off_d = tl.arange(0, BLOCK_SIZE_D)

                        k_blk = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
                        v_blk = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

                        dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
                        dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

                        for pid_sh in range(NUM_SHARE_Q_HEADS):
                            pid_h = pid_kh * NUM_SHARE_Q_HEADS + pid_sh
                            q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + off_d[None, :] * stride_qd
                            do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod
                            o_ptrs = o_ptr + q_start * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od
                            d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
                            lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

                            step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
                            for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
                                for u in tl.static_range(0, PIPELINE_CHUNKS):
                                    i = ib + u * BLOCK_SIZE_Q
                                    if i < act_q_len:
                                        idx_q = tl.load(tq_ptr_w + i + off_q, mask=off_q < act_q_len - i, other=0).to(tl.int32)
                                        qv = tl.load(
                                            q_ptrs + idx_q[:, None] * stride_qn,
                                            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                                            other=0,
                                        )
                                        do_v = tl.load(
                                            do_ptrs + idx_q[:, None] * stride_don,
                                            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                                            other=0,
                                        )
                                        lse_v = tl.load(
                                            lse_ptrs + idx_q[:, None] * stride_ln,
                                            mask=(off_q < act_q_len - i)[:, None],
                                            other=0,
                                        )
                                        if USE_PRECOMPUTED_DELTA:
                                            d_v = tl.load(
                                                d_ptrs + idx_q[:, None] * stride_dn,
                                                mask=(off_q < act_q_len - i)[:, None],
                                                other=0,
                                            )
                                        else:
                                            o_v = tl.load(
                                                o_ptrs + idx_q[:, None] * stride_on,
                                                mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
                                                other=0,
                                            )
                                            d_v = tl.sum(do_v.to(tl.float32) * o_v.to(tl.float32), axis=1)[:, None]
                                        qk = tl.dot(qv, k_blk.T) * qk_scale
                                        if not DISABLE_CAUSAL_MASK:
                                            qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                                        p = tl.exp2(qk - lse_v)
                                        if COMPUTE_DV:
                                            p_cast = p.to(do_v.dtype)
                                            dv += tl.dot(p_cast.T, do_v)
                                        if COMPUTE_DK:
                                            dp = tl.dot(do_v, v_blk.T)
                                            ds = sm_scale * p * (dp - d_v)
                                            ds = ds.to(qv.dtype)
                                            dk += tl.dot(ds.T, qv)

                        if COMPUTE_DK:
                            tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
                        if COMPUTE_DV:
                            tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))

        wid = tl.atomic_add(queue_ptr, WORK_STEAL_CHUNK)


def _topk_sparse_attention_bwd_opt_core(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    permute_results,
    disable_causal_mask: bool = False,
    dq_out: Optional[torch.Tensor] = None,
    dk_out: Optional[torch.Tensor] = None,
    dv_out: Optional[torch.Tensor] = None,
):

    assert block_size in {32, 64, 128, 256, 512, 1024}
    if isinstance(permute_results, dict):
        permute_results = [permute_results]
    expected_seqs = int(cu_seqlens_q.numel() - 1)
    if _permute_results_need_rebuild(permute_results) or (not isinstance(permute_results, (list, tuple))) or len(permute_results) != expected_seqs:
        permute_results = _build_permute_results_for_bwd(
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            head_dim=int(q.shape[-1]),
        )

    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    topk = topk_idx.shape[-1]
    use_precomputed_delta_dkdv = os.getenv("FSA_LOCAL_DKDV_USE_PRECOMPUTED_DELTA", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    use_precomputed_delta_dq = os.getenv("FSA_LOCAL_DQ_USE_PRECOMPUTED_DELTA", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    need_precomputed_delta = use_precomputed_delta_dkdv or use_precomputed_delta_dq
    # compute D
    if need_precomputed_delta:
        delta = _workspace_zeros(
            "bwd_delta",
            (num_o_heads, o_len),
            torch.float32,
            o.device,
        )
    else:
        # Delta values are computed on-the-fly inside dQ/dK/dV kernels; keep only shaped workspace.
        delta = _workspace_empty(
        "bwd_delta",
        (num_o_heads, o_len),
        torch.float32,
        o.device,
    )
    BLOCK_SIZE_O = 256
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    _base_warps, _base_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O, IS_HOPPER_GPU)
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="bwd_delta",
        head_dim=head_dim,
        block_size=block_size,
        default_warps=_base_warps,
        default_stages=_base_stages,
    )
    grid = (triton.cdiv(o_len, BLOCK_SIZE_O), num_o_heads)
    if need_precomputed_delta:
        backward_sum_o_do[grid](
            o,
            do,
            delta,
            o_len,
            head_dim,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            delta.stride(0),
            delta.stride(1),
            BLOCK_SIZE_O=BLOCK_SIZE_O,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    # count active querys for each key block, shape: (num_k_heads, total_k_blocks)
    seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seqblocks = torch.ceil(seqlens / block_size).to(torch.int32)
    cu_seqblocks = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(seqblocks, dim=0),
        ]
    ).to(torch.int32)

    # Defensive metadata validation: avoid None/invalid slots propagating into kernel launches.
    # If any sequence metadata is malformed, rebuild once from topk_idx + cu_seqlens.
    malformed = False
    if len(permute_results) != expected_seqs:
        malformed = True
    else:
        for i in range(expected_seqs):
            item = permute_results[i]
            if item is None or not isinstance(item, dict):
                malformed = True
                break
            if ("valid_lens_all" not in item) or ("real_num_blocks" not in item):
                malformed = True
                break
            if item["valid_lens_all"] is None:
                malformed = True
                break
    if malformed:
        permute_results = _build_permute_results_for_bwd(
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            head_dim=head_dim,
        )
        if len(permute_results) != expected_seqs:
            raise RuntimeError(
                f"Invalid permute_results length after rebuild: got {len(permute_results)}, expected {expected_seqs}."
            )

    topk_idx_for_reorder = topk_idx
    sanitize_enabled = os.getenv("FSA_LOCAL_BLOCK_PRUNING_SANITIZE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if sanitize_enabled:
        q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
        if len(q_ranges) == expected_seqs:
            topk_idx_for_reorder = topk_idx.clone()

    topk_q_count_list = []
    for i in range(expected_seqs):
        item = permute_results[i]
        if item is None or not isinstance(item, dict):
            raise RuntimeError(f"permute_results[{i}] is invalid: {type(item)}")
        valid_lens_all_i = item.get("valid_lens_all", None)
        real_num_blocks_i = item.get("real_num_blocks", None)
        if valid_lens_all_i is None or real_num_blocks_i is None:
            raise RuntimeError(f"permute_results[{i}] missing required fields.")
        if sanitize_enabled and len(q_ranges) == expected_seqs:
            q_start, q_end = q_ranges[i]
            topk_idx_for_reorder[:, q_start:q_end], _ = _sanitize_topk_block_indices(
                topk_idx_for_reorder[:, q_start:q_end],
                real_num_blocks=int(real_num_blocks_i),
            )
        topk_q_count_list.append(valid_lens_all_i[:, :int(real_num_blocks_i)])
    topk_q_count = torch.cat(topk_q_count_list, dim=1)

    cu_topk_q_count = torch.cat(
        [
            torch.zeros(topk_q_count.shape[0], 1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(topk_q_count, dim=-1),
        ],
        dim=-1,
    ).to(torch.int32)
    total_active_q = int(cu_topk_q_count[:, -1].to(dtype=torch.int64).sum().cpu().tolist())
    if total_active_q == 0:
        # No routed queries for any KV block -> all attention grads are zero.
        if dq_out is not None and tuple(dq_out.shape) == tuple(q.shape):
            dq_out.zero_()
            dq_ret = dq_out
        else:
            dq_ret = torch.zeros_like(q)
        if dk_out is not None and tuple(dk_out.shape) == tuple(k.shape):
            dk_out.zero_()
            dk_ret = dk_out
        else:
            dk_ret = torch.zeros_like(k)
        if dv_out is not None and tuple(dv_out.shape) == tuple(v.shape):
            dv_out.zero_()
            dv_ret = dv_out
        else:
            dv_ret = torch.zeros_like(v)
        return dq_ret, dk_ret, dv_ret

    batch_size = cu_seqlens_q.shape[0] - 1
    # OPT-4: Unified prep stage for reordered indices + segmented sort + active-map metadata.
    (
        topk_q_idx,
        active_idx,
        active_count,
        use_active_map,
        max_active_blocks,
        active_ratio,
        active_work_items_est,
    ) = _prepare_bwd_reorder_sort_and_active(
        topk_idx_for_reorder=topk_idx_for_reorder,
        cu_topk_q_count=cu_topk_q_count,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqblocks=cu_seqblocks,
        block_size=block_size,
        num_k_heads=num_k_heads,
        batch_size=batch_size,
        head_dim=head_dim,
        total_active_q=total_active_q,
    )

    # compute dk dv
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_Q = _resolve_bwd_dkdv_bq(head_dim=head_dim, block_size=block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    _base_warps, _base_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    num_warps, num_stages = _resolve_launch_warps_stages(
        op="bwd_dkdv",
        head_dim=head_dim,
        block_size=block_size,
        default_warps=_base_warps,
        default_stages=_base_stages,
    )
    loop_stages = max(1, min(int(num_stages), 4))
    pipeline_chunks = _resolve_hopper_pipeline_chunks(head_dim=head_dim, block_size=block_size)
    if pipeline_chunks > 1:
        loop_stages = max(loop_stages, 3)

    dkdv_mode = _resolve_dkdv_mode(num_share_q_heads, head_dim=head_dim, block_size=block_size)
    use_gqa_fused_dkdv = dkdv_mode == "gqa_fused"
    dkdv_two_pass = _resolve_dkdv_two_pass(
        num_share_q_heads=num_share_q_heads, head_dim=head_dim, block_size=block_size
    ) and use_gqa_fused_dkdv
    # OPT-6: Skip two-pass for small workloads where double kernel launch overhead
    # exceeds the register pressure savings from splitting dK/dV.
    _two_pass_min_raw = os.getenv("FSA_LOCAL_DKDV_TWO_PASS_MIN_ACTIVE_Q", "32768").strip().lower()
    try:
        _two_pass_min = int(_two_pass_min_raw)
    except Exception:
        _two_pass_min = 32768
    if dkdv_two_pass and total_active_q < _two_pass_min:
        dkdv_two_pass = False
    # OPT-5: Collapse historical two-pass dK/dV dispatch into one launch by default.
    # Keeps two-pass policy signal, but avoids double launch overhead in orchestration.
    fuse_two_pass = dkdv_two_pass and os.getenv("FSA_LOCAL_DKDV_TWO_PASS_FUSED", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    allow_legacy_two_pass = os.getenv("FSA_LOCAL_DKDV_TWO_PASS_LEGACY", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if fuse_two_pass or (dkdv_two_pass and not allow_legacy_two_pass):
        dkdv_two_pass = False
    dkdv_sched_mode = _resolve_dkdv_schedule_mode(
        use_active_map=use_active_map,
        total_active_q=total_active_q,
        active_ratio=active_ratio,
        active_work_items=active_work_items_est,
        batch_size=batch_size,
        num_share_q_heads=num_share_q_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    if use_gqa_fused_dkdv:
        if dk_out is not None and tuple(dk_out.shape) == (k_len, num_k_heads, head_dim):
            dk = dk_out
            dk.zero_()
        else:
            dk = torch.zeros(k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
        if dv_out is not None and tuple(dv_out.shape) == (k_len, num_k_heads, head_dim):
            dv = dv_out
            dv.zero_()
        else:
            dv = torch.zeros(k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
    else:
        dk = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
        dv = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)

    if use_active_map:
        grid = (batch_size, num_k_heads if use_gqa_fused_dkdv else num_q_heads, max_active_blocks)
    else:
        grid = (batch_size, num_k_heads if use_gqa_fused_dkdv else num_q_heads, triton.cdiv(max_seqlen_k, BLOCK_SIZE_K))

    if use_gqa_fused_dkdv:
        use_worklist = dkdv_sched_mode in ("worklist", "persistent") and use_active_map
        use_persistent = dkdv_sched_mode == "persistent" and use_active_map
        if use_worklist:
            worklist = _build_active_kv_worklist(active_idx, active_count)
            num_work_items = int(worklist.shape[0])
            if num_work_items > 0:
                if use_persistent:
                    avg_active_q_per_item = float(total_active_q) / float(max(1, num_work_items))
                    num_workers = _resolve_dkdv_persistent_workers(
                        q_device=q.device,
                        num_work_items=num_work_items,
                        avg_active_q_per_item=avg_active_q_per_item,
                        head_dim=head_dim,
                        block_size=block_size,
                    )
                    work_steal_chunk = _resolve_dkdv_persistent_chunk(
                        head_dim=head_dim,
                        block_size=block_size,
                        active_ratio=active_ratio,
                        avg_active_q_per_item=avg_active_q_per_item,
                        num_work_items=num_work_items,
                    )
                    grid_wl = (num_workers,)
                    if dkdv_two_pass:
                        # OPT-6: Use separate queue tensors for each pass to eliminate
                        # the queue.zero_() pipeline bubble between passes.
                        queue_dv = torch.zeros((1,), dtype=torch.int32, device=q.device)
                        queue_dk = torch.zeros((1,), dtype=torch.int32, device=q.device)
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue_dv, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=False, COMPUTE_DV=True,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue_dk, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=False,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                    else:
                        queue = torch.zeros((1,), dtype=torch.int32, device=q.device)
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=True,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                else:
                    grid_wl = (num_work_items,)
                    if dkdv_two_pass:
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=False, COMPUTE_DV=True,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=False,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                    else:
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=True,
                            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                            num_warps=num_warps, num_stages=num_stages,
                        )
        else:
            if dkdv_two_pass:
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=False, COMPUTE_DV=True,
                    USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                    num_warps=num_warps, num_stages=num_stages,
                )
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=True, COMPUTE_DV=False,
                    USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                    num_warps=num_warps, num_stages=num_stages,
                )
            else:
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, lse, delta, do, o, dk, dv,
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=True, COMPUTE_DV=True,
                    USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
                    num_warps=num_warps, num_stages=num_stages,
                )
    else:
        backward_dkdv[grid](
            q,
            k,
            v,
            topk_q_idx,
            lse,
            delta,
            do,
            o,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqblocks,
            cu_topk_q_count,
            active_idx,
            active_count,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            topk_q_idx.stride(0),
            topk_q_idx.stride(1),
            cu_topk_q_count.stride(0),
            cu_topk_q_count.stride(1),
            active_idx.stride(0),
            active_idx.stride(1),
            active_idx.stride(2),
            active_count.stride(0),
            active_count.stride(1),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            LOOP_STAGES=loop_stages,
            PIPELINE_CHUNKS=pipeline_chunks,
            DISABLE_CAUSAL_MASK=disable_causal_mask,
            USE_ACTIVE_BLOCK_MAP=use_active_map,
            USE_PRECOMPUTED_DELTA=use_precomputed_delta_dkdv,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dk = dk.sum(0)
        dv = dv.sum(0)
    # P1.2: shape-family policy for dQ scheduling knobs.
    dq_block_size_q = _resolve_bwd_dq_bq(head_dim=head_dim, block_size=block_size)
    dq_num_q_blocks = _resolve_bwd_dq_num_q_blocks(block_size=block_size)
    if total_active_q >= 262144:
        if block_size <= 128:
            dq_num_q_blocks = max(dq_num_q_blocks, 8 if IS_HOPPER_GPU else 4)
        elif block_size >= 512:
            dq_num_q_blocks = max(1, min(dq_num_q_blocks, 4 if IS_HOPPER_GPU else 2))

    # compute dq
    if dq_out is not None and tuple(dq_out.shape) == tuple(q.shape):
        dq = dq_out
        dq.zero_()
    else:
        dq = torch.zeros_like(q)
    num_q_loop = max_seqlen_q // 32768 + 1  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, triton.cdiv(max_seqlen_q, num_q_loop))

    backward_dq_opt(
        o,
        q,
        k,
        v,
        topk_idx,
        lse,
        delta,
        do,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        sm_scale,
        block_size,
        permute_results,
        dq_block_size_q=dq_block_size_q,
        dq_num_q_blocks=dq_num_q_blocks,
        disable_causal_mask=disable_causal_mask,
    )

    return dq, dk, dv


def _topk_sparse_attention_bwd_opt_seq_parallel(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    permute_results,
    disable_causal_mask: bool = False,
):
    """
    Dedicated sequence-parallel backward path.
    Splits by (q_seq, k_seq), runs core backward per sequence, and writes into global outputs.
    """
    seq_meta, cu_q_local_all, cu_k_local_all = _build_seq_dispatch_meta(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        device=q.device,
    )
    nseq = len(seq_meta)
    if nseq > 1:
        packed = _pack_varlen_unified_timeline(
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_size=block_size,
            o=o,
            do=do,
            lse=lse,
        )
        if packed is not None:
            q_u = packed["q_u"]
            k_u = packed["k_u"]
            v_u = packed["v_u"]
            topk_u = packed["topk_u"]
            o_u = packed["o_u"]
            do_u = packed["do_u"]
            lse_u = packed["lse_u"]
            cu_u = packed["cu_u"]
            packed_meta = packed["packed_meta"]

            dq_u, dk_u, dv_u = _topk_sparse_attention_bwd_opt_core(
                o=o_u,
                do=do_u,
                lse=lse_u,
                q=q_u,
                k=k_u,
                v=v_u,
                topk_idx=topk_u,
                block_size=block_size,
                cu_seqlens_q=cu_u,
                cu_seqlens_k=cu_u,
                max_seqlen_q=int(q_u.shape[0]),
                max_seqlen_k=int(k_u.shape[0]),
                sm_scale=sm_scale,
                permute_results=None,
                disable_causal_mask=disable_causal_mask,
            )
            dq_out = torch.zeros_like(q)
            dk_out = torch.zeros_like(k)
            dv_out = torch.zeros_like(v)
            _unpack_varlen_unified_q(
                packed_meta=packed_meta,
                src_u=dq_u,
                dst=dq_out,
                by_head_first=False,
            )
            _unpack_varlen_unified_kv(
                packed_meta=packed_meta,
                dk_u=dk_u,
                dv_u=dv_u,
                dk_dst=dk_out,
                dv_dst=dv_out,
            )
            return dq_out, dk_out, dv_out
        raise RuntimeError("FSA multi-seq flat backward packing failed unexpectedly.")

    if nseq <= 1:
        return _topk_sparse_attention_bwd_opt_core(
            o=o,
            do=do,
            lse=lse,
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            permute_results=permute_results,
            disable_causal_mask=disable_causal_mask,
        )

    raise RuntimeError("Unexpected backward wrapper state: multi-seq path must run via flat packed timeline.")


def _topk_sparse_attention_bwd_opt(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    permute_results,
    disable_causal_mask: bool = False,
):
    expected_seqs = int(cu_seqlens_q.numel() - 1)
    if expected_seqs > 1:
        # OPT-12: force unified multi-seq packed backward path.
        return _topk_sparse_attention_bwd_opt_seq_parallel(
            o=o,
            do=do,
            lse=lse,
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            permute_results=permute_results,
            disable_causal_mask=disable_causal_mask,
        )
    use_seq_parallel = _resolve_bwd_sequence_parallel(
        expected_seqs=expected_seqs,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        head_dim=int(q.shape[-1]),
        block_size=block_size,
    )
    if use_seq_parallel:
        return _topk_sparse_attention_bwd_opt_seq_parallel(
            o=o,
            do=do,
            lse=lse,
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            permute_results=permute_results,
            disable_causal_mask=disable_causal_mask,
        )
    return _topk_sparse_attention_bwd_opt_core(
        o=o,
        do=do,
        lse=lse,
        q=q,
        k=k,
        v=v,
        topk_idx=topk_idx,
        block_size=block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
        permute_results=permute_results,
        disable_causal_mask=disable_causal_mask,
    )


class FSATopkSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
        k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
        v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
        topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
        block_size: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        sm_scale=None,
        disable_causal_mask: bool = False,
    ):
        # dtype check
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert topk_idx.dtype == torch.int32
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])

        hq = int(q.shape[1])
        hk = int(k.shape[1])
        gqa_deg = (hq // hk) if (hk > 0 and hq % hk == 0) else -1
        use_nsa_style_fwd = os.getenv("FSA_LOCAL_USE_NSA_STYLE_FWD", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        force_nsa_style_small_g = os.getenv("FSA_LOCAL_FORCE_NSA_STYLE_FWD_SMALL_G", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        # Default small-G mode is "fallback": for G<16, use legacy local FSA forward path.
        small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "fallback").strip().lower()
        if small_g_mode not in ("pad", "fma", "torch", "fallback"):
            small_g_mode = "pad"
        fwd_packed_gqa = _resolve_fwd_packed_gqa(
            num_share_q_heads=max(1, gqa_deg),
            head_dim=int(q.shape[-1]),
            block_size=block_size,
        )
        if (
            gqa_deg > 0
            and gqa_deg < 16
            and small_g_mode == "fallback"
            and (not force_nsa_style_small_g)
            and (not fwd_packed_gqa)
        ):
            use_nsa_style_fwd = False
        permute_results = None

        if use_nsa_style_fwd:
            o, lse = _topk_sparse_attention_fwd_nsa_style(
                q=q,
                k=k,
                v=v,
                topk_idx=topk_idx,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                sm_scale=sm_scale,
            )
            # OPT-2: Eagerly build backward permutation metadata in forward to eliminate
            # the full rebuild in backward. Cost is small relative to attention kernel and
            # avoids the expensive lazy rebuild during the backward pass.
            _eager_bwd_meta = os.getenv("FSA_LOCAL_EAGER_BWD_META", "1").strip().lower() not in (
                "0", "false", "no", "off", ""
            )
            expected_seqs = int(cu_seqlens_q.numel() - 1)
            if _eager_bwd_meta and expected_seqs <= 1:
                permute_results = _build_permute_results_for_bwd(
                    topk_idx=topk_idx,
                    block_size=block_size,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    head_dim=int(q.shape[-1]),
                )
            else:
                permute_results = None
        else:
            o, lse, permute_results = _topk_sparse_attention_fwd_opt(
                q,
                k,
                v,
                topk_idx,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )

        if o is None or lse is None:
            raise RuntimeError("FSA forward produced None output/lse.")

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx)
        ctx.permute_results = permute_results
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.block_size = block_size
        ctx.disable_causal_mask = disable_causal_mask
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx = ctx.saved_tensors
        permute_results = ctx.permute_results

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        block_size = ctx.block_size
        disable_causal_mask = ctx.disable_causal_mask
        assert block_size in {32, 64, 128, 256, 512, 1024}
        expected_seqs = int(cu_seqlens_q.numel() - 1)
        need_rebuild = _permute_results_need_rebuild(permute_results)
        if need_rebuild and expected_seqs <= 1:
            permute_results = _build_permute_results_for_bwd(
                topk_idx=topk_idx,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                head_dim=int(q.shape[-1]),
            )

        dq, dk, dv = _topk_sparse_attention_bwd_opt(
                o,
                do,
                lse,
                q,
                k,
                v,
                topk_idx,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
                permute_results,
                disable_causal_mask=disable_causal_mask,
            )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def FSA_topk_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    disable_causal_mask: bool = False,
) -> torch.Tensor:
    """Topk sparse attention varlen version implemented in triton.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.int32).max().cpu().tolist())
    return FSATopkSparseAttention.apply(
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale,
        disable_causal_mask,
    )


def _resolve_internal_block_size(block_size: int) -> int:
    """
    Choose a kernel-safe internal KV block size.

    Large external chapter sizes (e.g. 512/1024) can exceed shared-memory limits in
    current Triton kernels. We keep semantics by splitting each chapter into multiple
    internal sub-blocks and expanding top-k indices accordingly.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    cap = int(os.getenv("FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE", "128"))
    if cap not in (32, 64, 128, 256):
        cap = 128
    internal = min(block_size, cap)
    # Internal kernel assumes power-of-two block sizes from this set.
    if internal not in (32, 64, 128, 256):
        raise ValueError(
            f"Unsupported internal block_size={internal}. "
            f"Use FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE in {{32,64,128,256}}."
        )
    if block_size % internal != 0:
        raise ValueError(
            f"block_size={block_size} must be divisible by internal block size {internal}. "
            "Adjust FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE."
        )
    return internal


def _expand_topk_for_internal_blocks(
    topk_idx: torch.Tensor,
    block_size: int,
    internal_block_size: int,
) -> torch.Tensor:
    """
    Expand chapter-level top-k ids into internal sub-block ids.

    Example:
      block_size=1024, internal=256, ratio=4
      chapter c -> sub-block ids [4c, 4c+1, 4c+2, 4c+3]
    """
    ratio = block_size // internal_block_size
    if ratio == 1:
        return topk_idx
    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)

    # topk_idx: [H, N, S] -> expanded: [H, N, S*ratio]
    offsets = torch.arange(ratio, device=topk_idx.device, dtype=topk_idx.dtype).view(1, 1, 1, ratio)
    base = topk_idx.unsqueeze(-1)
    expanded = base * ratio + offsets
    expanded = torch.where(base >= 0, expanded, torch.full_like(expanded, -1))
    return expanded.reshape(topk_idx.shape[0], topk_idx.shape[1], topk_idx.shape[2] * ratio)


def _maybe_sort_reordered_topk_q_idx(
    topk_q_idx: torch.Tensor,
    cu_topk_q_count: torch.Tensor,
) -> torch.Tensor:
    """
    Optional segmented sort for reordered query indices (per KV-head, per KV-block segment).

    Helps locality in dK/dV kernel when reordered indices are highly scattered.
    Controlled by env:
      FSA_LOCAL_SORT_TOPK_Q_IDX=1     -> force enable
      FSA_LOCAL_SORT_TOPK_Q_IDX=0     -> force disable
      FSA_LOCAL_SORT_TOPK_Q_IDX=auto  -> enable only for large dense-enough routed sets (default)
    """
    mode = os.getenv("FSA_LOCAL_SORT_TOPK_Q_IDX", "auto").strip().lower()
    if mode in ("", "0", "false", "no", "off"):
        enabled = False
    elif mode in ("1", "true", "yes", "on"):
        enabled = True
    else:
        enabled = True  # auto, decided below
    if not enabled:
        return topk_q_idx
    if topk_q_idx.ndim != 2 or cu_topk_q_count.ndim != 2:
        return topk_q_idx

    num_kh, max_active_slots = int(topk_q_idx.shape[0]), int(topk_q_idx.shape[1])
    if num_kh <= 0 or max_active_slots <= 1:
        return topk_q_idx

    num_segments = max(1, int(cu_topk_q_count.shape[1]) - 1)
    n_active = cu_topk_q_count[:, -1].to(dtype=torch.int64)
    n_active = torch.clamp(n_active, min=0, max=max_active_slots)
    total_active = int(n_active.sum().to(dtype=torch.int64).cpu().tolist()) if n_active.numel() > 0 else 0

    if mode not in ("1", "true", "yes", "on"):
        avg_seg = (float(total_active) / float(max(1, num_kh * num_segments)))
        # Auto-enable only when routing is large enough for locality to matter.
        if total_active < 65536 or avg_seg < 2.0:
            return topk_q_idx

    max_active = int(n_active.max().to(dtype=torch.int32).cpu().tolist()) if n_active.numel() > 0 else 0
    if max_active <= 1 or total_active <= 1:
        return topk_q_idx

    out = topk_q_idx.clone()
    pos = torch.arange(max_active_slots, device=out.device, dtype=torch.int64).view(1, max_active_slots)
    pos = pos.expand(num_kh, max_active_slots)
    valid = pos < n_active.view(-1, 1)
    if not bool(torch.any(valid)):
        return out

    # Build row-major segment group ids for active entries only.
    seg_counts = (cu_topk_q_count[:, 1:] - cu_topk_q_count[:, :-1]).to(dtype=torch.int64)
    seg_counts = torch.clamp(seg_counts, min=0)
    seg_counts_flat = seg_counts.reshape(-1)
    group_basis = torch.arange(num_kh * num_segments, device=out.device, dtype=torch.int64)
    flat_group = torch.repeat_interleave(group_basis, seg_counts_flat)

    flat_vals = out[valid]
    if int(flat_vals.numel()) <= 1:
        return out
    if int(flat_group.numel()) != int(flat_vals.numel()):
        # Defensive fallback for malformed metadata; keep original ordering.
        return out

    # Single-pass segmented stable sort via lexicographic key:
    # key = group_id * value_span + value.
    # Stable argsort preserves relative order for ties.
    vals_i64 = flat_vals.to(torch.int64)
    v_min, v_max = torch.aminmax(vals_i64)
    v_min_i, v_max_i = [int(x) for x in torch.stack((v_min, v_max)).cpu().tolist()]
    span = (v_max_i - v_min_i) + 1
    if span <= 0:
        return out

    key = flat_group * span + (vals_i64 - v_min_i)
    ord_key = torch.argsort(key, stable=True)
    out[valid] = flat_vals.index_select(0, ord_key)
    return out


def _build_active_kv_block_map(
    cu_topk_q_count: torch.Tensor,  # [HK, total_blocks + 1]
    cu_seqblocks: torch.Tensor,     # [B + 1]
):
    """
    Build compact local KV-block id lists per (kv_head, batch) for dK/dV launch compaction.

    Returns:
      active_block_idx:   [HK, B, max_active] int32 local block ids (0..seq_blocks_b-1), padded with -1
      active_block_count: [HK, B] int32 number of active blocks
      max_active: int
      active_ratio: float in [0,1], fraction of active blocks across all (kh,b)
    """
    if cu_topk_q_count.ndim != 2 or cu_seqblocks.ndim != 1:
        raise ValueError("Invalid shapes for active block map build.")

    device = cu_topk_q_count.device
    hk = int(cu_topk_q_count.shape[0])
    batch = int(cu_seqblocks.numel() - 1)
    if hk <= 0 or batch <= 0:
        active_block_idx = torch.full((max(hk, 1), max(batch, 1), 1), -1, dtype=torch.int32, device=device)
        active_block_count = torch.zeros((max(hk, 1), max(batch, 1)), dtype=torch.int32, device=device)
        return active_block_idx[:hk, :batch], active_block_count[:hk, :batch], 0, 0.0

    block_counts = cu_topk_q_count[:, 1:] - cu_topk_q_count[:, :-1]  # [HK, total_blocks]
    return _build_active_kv_block_map_from_counts(
        block_counts=block_counts,
        cu_seqblocks=cu_seqblocks,
    )


def _build_active_kv_block_map_from_counts(
    block_counts: torch.Tensor,     # [HK, total_blocks]
    cu_seqblocks: torch.Tensor,     # [B + 1]
):
    """
    Build compact active local-block map from precomputed per-block counts.
    Reused by fused prep path to avoid recomputing block-count deltas.
    """
    if block_counts.ndim != 2 or cu_seqblocks.ndim != 1:
        raise ValueError("Invalid shapes for active block map build.")
    device = block_counts.device
    hk = int(block_counts.shape[0])
    batch = int(cu_seqblocks.numel() - 1)
    if hk <= 0 or batch <= 0:
        active_block_idx = torch.full((max(hk, 1), max(batch, 1), 1), -1, dtype=torch.int32, device=device)
        active_block_count = torch.zeros((max(hk, 1), max(batch, 1)), dtype=torch.int32, device=device)
        return active_block_idx[:hk, :batch], active_block_count[:hk, :batch], 0, 0.0

    active_mask = block_counts > 0
    nz = torch.nonzero(active_mask, as_tuple=False)  # [N, 2] -> (kh, global_block)
    total_active = int(nz.shape[0])

    _seqblocks_last = int(cu_seqblocks[-1].to(dtype=torch.int32).cpu().tolist()) if cu_seqblocks.numel() > 0 else 0
    total_blocks_all = hk * _seqblocks_last
    if total_active <= 0:
        active_block_idx = torch.full((hk, batch, 1), -1, dtype=torch.int32, device=device)
        active_block_count = torch.zeros((hk, batch), dtype=torch.int32, device=device)
        return active_block_idx, active_block_count, 0, 0.0

    kh_idx = nz[:, 0].to(torch.int64)  # [N]
    global_blk = nz[:, 1].to(torch.int64)  # [N]

    # Map global block -> (batch, local_block) using cumulative sequence blocks.
    seq_end = cu_seqblocks[1:].to(torch.int64)  # [B]
    b_idx = torch.bucketize(global_blk, seq_end, right=True)  # [N], 0..B-1
    b_idx = torch.clamp(b_idx, min=0, max=max(batch - 1, 0))
    seq_start = cu_seqblocks.to(torch.int64).index_select(0, b_idx)
    local_blk = (global_blk - seq_start).to(torch.int32)  # [N], local block id inside sequence

    pair = kh_idx * batch + b_idx  # unique id for (kh,b)
    pair_i64 = pair.to(torch.int64)
    active_block_count = torch.bincount(pair_i64, minlength=hk * batch).to(torch.int32).view(hk, batch)
    max_active = int(active_block_count.max().to(dtype=torch.int32).cpu().tolist())
    if max_active <= 0:
        active_block_idx = torch.full((hk, batch, 1), -1, dtype=torch.int32, device=device)
        return active_block_idx, active_block_count, 0, 0.0

    # Deterministic order: sort by pair then by local block id.
    sort_key = pair.to(torch.int64) * (_seqblocks_last + 1) + local_blk.to(torch.int64)
    order = torch.argsort(sort_key, stable=True)
    pair_s = pair.index_select(0, order)
    local_s = local_blk.index_select(0, order).to(torch.int32)

    n = int(pair_s.numel())
    idx = torch.arange(n, device=device, dtype=torch.int64)
    is_new = torch.ones((n,), device=device, dtype=torch.bool)
    if n > 1:
        is_new[1:] = pair_s[1:] != pair_s[:-1]
    start_marker = torch.where(is_new, idx, torch.full_like(idx, -1))
    first_idx, _ = torch.cummax(start_marker, dim=0)
    rank = (idx - first_idx).to(torch.int64)  # position inside (kh,b) list

    hk_s = (pair_s // batch).to(torch.int64)
    b_s = (pair_s % batch).to(torch.int64)
    active_block_idx = torch.full((hk, batch, max_active), -1, dtype=torch.int32, device=device)
    active_block_idx[hk_s, b_s, rank] = local_s

    active_ratio = float(total_active) / float(max(1, total_blocks_all))
    return active_block_idx, active_block_count, max_active, active_ratio


def _build_active_kv_worklist(
    active_block_idx: torch.Tensor,   # [HK, B, max_active]
    active_block_count: torch.Tensor, # [HK, B]
) -> torch.Tensor:
    """
    Build compact active worklist of tuples (batch, kv_head, local_k_block).
    Returns int32 tensor [N, 3] on device.
    """
    if active_block_idx.ndim != 3 or active_block_count.ndim != 2:
        raise ValueError("Invalid active map shapes for worklist build.")
    hk, batch, max_active = active_block_idx.shape
    if max_active <= 0:
        return torch.empty((0, 3), dtype=torch.int32, device=active_block_idx.device)

    offs = torch.arange(max_active, device=active_block_idx.device, dtype=torch.int32).view(1, 1, max_active)
    valid = offs < active_block_count.unsqueeze(-1)
    valid = valid & (active_block_idx >= 0)
    if not torch.any(valid):
        return torch.empty((0, 3), dtype=torch.int32, device=active_block_idx.device)

    kh_grid = torch.arange(hk, device=active_block_idx.device, dtype=torch.int32).view(hk, 1, 1).expand(hk, batch, max_active)
    b_grid = torch.arange(batch, device=active_block_idx.device, dtype=torch.int32).view(1, batch, 1).expand(hk, batch, max_active)

    return torch.stack(
        (
            b_grid[valid],
            kh_grid[valid],
            active_block_idx[valid].to(torch.int32),
        ),
        dim=-1,
    )


def _prepare_bwd_reorder_sort_and_active(
    topk_idx_for_reorder: torch.Tensor,
    cu_topk_q_count: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqblocks: torch.Tensor,
    block_size: int,
    num_k_heads: int,
    batch_size: int,
    head_dim: int,
    total_active_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int, float, int]:
    """
    Build reordered query indices and active-map metadata in one preparation stage.
    OPT-4 fused prep:
      - single fused GPU orchestration path for reorder + segmented sort + active-map build
      - legacy reorder/map builders are retained only as explicit strictness fallback
    """
    device = topk_idx_for_reorder.device
    active_mode = os.getenv("FSA_LOCAL_COMPACT_ACTIVE_BLOCKS", "auto").strip().lower()
    active_ratio_threshold = _resolve_active_map_ratio_threshold(head_dim=head_dim, block_size=block_size)
    block_counts = (cu_topk_q_count[:, 1:] - cu_topk_q_count[:, :-1]).to(torch.int32)
    active_idx = _workspace_empty(
        "bwd_active_idx",
        (num_k_heads, batch_size, 1),
        torch.int32,
        device,
    )
    active_idx.fill_(-1)
    active_count = _workspace_zeros(
        "bwd_active_count",
        (num_k_heads, batch_size),
        torch.int32,
        device,
    )
    use_active_map = False
    max_active_blocks = 0
    active_ratio = 1.0
    active_work_items_est = 0

    def _build_reorder_sorted_and_active_fused() -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, int]]:
        """
        Single fused prep path:
          - reorder + per-segment query sorting
          - active-map derivation from the same routed-pair stream
        """
        if topk_idx_for_reorder.ndim != 3 or cu_topk_q_count.ndim != 2:
            return None
        hk = int(topk_idx_for_reorder.shape[0])
        total_q = int(topk_idx_for_reorder.shape[1])
        total_blocks = int(cu_topk_q_count.shape[1] - 1)
        batch = int(cu_seqblocks.numel() - 1)
        if hk <= 0 or total_q <= 0 or total_blocks <= 0 or batch <= 0:
            topk_q_idx_empty = _workspace_empty("bwd_topk_q_idx_fused_empty", (max(hk, 1), 0), torch.int32, device)[:hk, :0]
            active_idx_empty = _workspace_empty("bwd_active_idx_fused_empty", (max(hk, 1), max(batch, 1), 1), torch.int32, device)[:hk, :batch]
            active_idx_empty.fill_(-1)
            active_count_empty = _workspace_zeros("bwd_active_count_fused_empty", (max(hk, 1), max(batch, 1)), torch.int32, device)[:hk, :batch]
            return topk_q_idx_empty, active_idx_empty, active_count_empty, 0, 0.0, 0

        total_active = int(cu_topk_q_count[:, -1].to(dtype=torch.int64).sum().cpu().tolist()) if cu_topk_q_count.numel() > 0 else 0
        if total_active <= 0:
            topk_q_idx_empty = _workspace_empty("bwd_topk_q_idx_fused_empty2", (hk, 0), torch.int32, device)
            active_idx_empty = _workspace_empty("bwd_active_idx_fused_empty2", (hk, batch, 1), torch.int32, device)
            active_idx_empty.fill_(-1)
            active_count_empty = _workspace_zeros("bwd_active_count_fused_empty2", (hk, batch), torch.int32, device)
            return topk_q_idx_empty, active_idx_empty, active_count_empty, 0, 0.0, 0

        q_tok = torch.arange(total_q, device=device, dtype=torch.int64)
        q_seq = torch.bucketize(q_tok, cu_seqlens_q[1:].to(dtype=torch.int64), right=True)
        blk_off = cu_seqblocks.index_select(0, q_seq).to(dtype=topk_idx_for_reorder.dtype).view(1, total_q, 1)
        gblk = torch.where(
            topk_idx_for_reorder >= 0,
            topk_idx_for_reorder + blk_off,
            torch.full_like(topk_idx_for_reorder, -1),
        )
        valid = (gblk >= 0) & (gblk < total_blocks)
        if not bool(torch.any(valid)):
            topk_q_idx_empty = _workspace_empty("bwd_topk_q_idx_fused_empty3", (hk, 0), torch.int32, device)
            active_idx_empty = _workspace_empty("bwd_active_idx_fused_empty3", (hk, batch, 1), torch.int32, device)
            active_idx_empty.fill_(-1)
            active_count_empty = _workspace_zeros("bwd_active_count_fused_empty3", (hk, batch), torch.int32, device)
            return topk_q_idx_empty, active_idx_empty, active_count_empty, 0, 0.0, 0

        kh_grid = torch.arange(hk, device=device, dtype=torch.int64).view(hk, 1, 1).expand_as(gblk)
        q_grid = q_tok.view(1, total_q, 1).expand_as(gblk)
        pair = kh_grid[valid] * total_blocks + gblk[valid].to(torch.int64)
        q_vals = q_grid[valid].to(torch.int32)
        n_valid = int(pair.numel())
        if n_valid <= 0:
            topk_q_idx_empty = _workspace_empty("bwd_topk_q_idx_fused_empty4", (hk, 0), torch.int32, device)
            active_idx_empty = _workspace_empty("bwd_active_idx_fused_empty4", (hk, batch, 1), torch.int32, device)
            active_idx_empty.fill_(-1)
            active_count_empty = _workspace_zeros("bwd_active_count_fused_empty4", (hk, batch), torch.int32, device)
            return topk_q_idx_empty, active_idx_empty, active_count_empty, 0, 0.0, 0

        sort_key = pair * (total_q + 1) + q_vals.to(torch.int64)
        order = torch.argsort(sort_key, stable=True)
        pair_s = pair.index_select(0, order)
        q_s = q_vals.index_select(0, order)

        idx = torch.arange(n_valid, device=device, dtype=torch.int64)
        is_new = torch.ones((n_valid,), device=device, dtype=torch.bool)
        if n_valid > 1:
            is_new[1:] = pair_s[1:] != pair_s[:-1]
        start_marker = torch.where(is_new, idx, torch.full_like(idx, -1))
        first_idx, _ = torch.cummax(start_marker, dim=0)
        rank = idx - first_idx

        # Defensive consistency check: fallback only when metadata and routed pairs disagree.
        if int(n_valid) != int(total_active):
            return None

        starts_flat = cu_topk_q_count[:, :-1].to(dtype=torch.int64).reshape(-1)
        dst = starts_flat.index_select(0, pair_s) + rank
        topk_q_idx = _workspace_empty("bwd_topk_q_idx_fused", (hk, total_active), torch.int32, device)
        topk_q_idx.zero_()
        kh_s = torch.div(pair_s, total_blocks, rounding_mode="floor").to(torch.int64)
        topk_q_idx[kh_s, dst.to(torch.int64)] = q_s

        # Active-map from unique (kv_head, global_block) pairs.
        pair_u = pair_s[is_new]
        if int(pair_u.numel()) <= 0:
            active_idx_empty = _workspace_empty("bwd_active_idx_fused_empty5", (hk, batch, 1), torch.int32, device)
            active_idx_empty.fill_(-1)
            active_count_empty = _workspace_zeros("bwd_active_count_fused_empty5", (hk, batch), torch.int32, device)
            return topk_q_idx, active_idx_empty, active_count_empty, 0, 0.0, 0

        kh_u = torch.div(pair_u, total_blocks, rounding_mode="floor").to(torch.int64)
        gblk_u = torch.remainder(pair_u, total_blocks).to(torch.int64)
        seq_end = cu_seqblocks[1:].to(torch.int64)
        b_u = torch.bucketize(gblk_u, seq_end, right=True)
        b_u = torch.clamp(b_u, min=0, max=max(batch - 1, 0))
        seq_start = cu_seqblocks.to(torch.int64).index_select(0, b_u)
        local_u = (gblk_u - seq_start).to(torch.int32)

        pair_b = kh_u * batch + b_u
        active_count_built = torch.bincount(pair_b, minlength=hk * batch).to(torch.int32).view(hk, batch)
        max_active = int(active_count_built.max().to(dtype=torch.int32).cpu().tolist())
        if max_active <= 0:
            active_idx_built = _workspace_empty("bwd_active_idx_fused_empty6", (hk, batch, 1), torch.int32, device)
            active_idx_built.fill_(-1)
            return topk_q_idx, active_idx_built, active_count_built, 0, 0.0, 0

        # Deterministic order in active_idx: (kh,batch,local_block).
        sort_key2 = pair_b * (total_blocks + 1) + local_u.to(torch.int64)
        order2 = torch.argsort(sort_key2, stable=True)
        pair_b_s = pair_b.index_select(0, order2)
        local_s = local_u.index_select(0, order2).to(torch.int32)
        n2 = int(pair_b_s.numel())
        idx2 = torch.arange(n2, device=device, dtype=torch.int64)
        is_new2 = torch.ones((n2,), device=device, dtype=torch.bool)
        if n2 > 1:
            is_new2[1:] = pair_b_s[1:] != pair_b_s[:-1]
        start_marker2 = torch.where(is_new2, idx2, torch.full_like(idx2, -1))
        first_idx2, _ = torch.cummax(start_marker2, dim=0)
        rank2 = idx2 - first_idx2

        kh_b = torch.div(pair_b_s, batch, rounding_mode="floor").to(torch.int64)
        b_b = torch.remainder(pair_b_s, batch).to(torch.int64)
        active_idx_built = _workspace_empty("bwd_active_idx_fused", (hk, batch, max_active), torch.int32, device)
        active_idx_built.fill_(-1)
        active_idx_built[kh_b, b_b, rank2.to(torch.int64)] = local_s

        total_blocks_all = hk * int(cu_seqblocks[-1].to(dtype=torch.int32).cpu().tolist())
        active_ratio_built = float(int(pair_u.numel())) / float(max(1, total_blocks_all))
        active_work_items_built = int(active_count_built.to(dtype=torch.int64).sum().cpu().tolist())
        return topk_q_idx, active_idx_built, active_count_built, max_active, active_ratio_built, active_work_items_built

    prep_mode = os.getenv("FSA_LOCAL_BWD_PREP_MODE", "auto").strip().lower()
    force_legacy_prep = prep_mode in ("legacy", "fallback", "unfused", "0", "off", "false", "no")
    force_fused_prep = prep_mode in ("fused", "1", "on", "true", "yes")

    fused_all = None if force_legacy_prep else _build_reorder_sorted_and_active_fused()
    if force_fused_prep and fused_all is None:
        raise RuntimeError("Fused reorder/sort/active prep forced but unavailable for this input.")

    if fused_all is None:
        strict_fused = os.getenv("FSA_LOCAL_FUSED_PREP_STRICT", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        if strict_fused and (not force_legacy_prep):
            raise RuntimeError("Fused reorder/sort/active prep failed consistency checks.")
        topk_q_idx = reorder_topk_idx(topk_idx_for_reorder, cu_topk_q_count, cu_seqlens_q, cu_seqblocks, block_size)
        topk_q_idx = _maybe_sort_reordered_topk_q_idx(topk_q_idx, cu_topk_q_count)
        active_idx_built, active_count_built, max_active_blocks, active_ratio = _build_active_kv_block_map_from_counts(
            block_counts=block_counts,
            cu_seqblocks=cu_seqblocks,
        )
        active_work_items_est = int(active_count_built.to(dtype=torch.int64).sum().cpu().tolist())
    else:
        (
            topk_q_idx,
            active_idx_built,
            active_count_built,
            max_active_blocks,
            active_ratio,
            active_work_items_est,
        ) = fused_all

    # Decide whether to enable active-map launch compaction from fused metadata.
    if active_mode not in ("0", "false", "no", "off"):
        max_auto_active_q_raw = os.getenv("FSA_LOCAL_MAX_AUTO_ACTIVE_Q_FOR_MAP", "8000000").strip().lower()
        try:
            max_auto_active_q_for_map = int(max_auto_active_q_raw)
        except Exception:
            max_auto_active_q_for_map = 8000000
        auto_map_allowed = not (
            active_mode in ("", "auto") and total_active_q > max(1, max_auto_active_q_for_map)
        )
        if auto_map_allowed or active_mode in ("1", "true", "yes", "on"):
            if active_mode in ("1", "true", "yes", "on"):
                use_active_map = True
            else:
                use_active_map = max_active_blocks > 0 and active_ratio < active_ratio_threshold
            if use_active_map:
                active_idx = active_idx_built
                active_count = active_count_built
            else:
                active_work_items_est = 0
        else:
            active_work_items_est = 0
    else:
        active_work_items_est = 0

    return (
        topk_q_idx,
        active_idx,
        active_count,
        use_active_map,
        max_active_blocks,
        active_ratio,
        active_work_items_est,
    )


def FSA_topk_sparse_attention_varlen_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    disable_causal_mask: bool = False,
) -> torch.Tensor:
    """
    Local API extension (minimal change): expose separate q/k cu_seqlens.
    Note: core kernels still assume same per-sequence total length behavior.
    """
    internal_block_size = _resolve_internal_block_size(block_size)
    topk_idx_internal = _expand_topk_for_internal_blocks(topk_idx, block_size, internal_block_size)

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(dtype=torch.int32).max().cpu().tolist())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(dtype=torch.int32).max().cpu().tolist())
    return FSATopkSparseAttention.apply(
        q,
        k,
        v,
        topk_idx_internal,
        internal_block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        disable_causal_mask,
    )


def FSA_topk_sparse_attention_bthd(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    block_indices_bths: Optional[torch.Tensor],
    block_size: int,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    topk_idx_hns: Optional[torch.Tensor] = None,
    assume_sorted_topk: bool = False,
    disable_causal_mask: bool = False,
) -> torch.Tensor:
    """
    Benchmark-facing wrapper for this project layout:
      q: [B, Tq, HQ, D]
      k/v: [B, Tk, HK, D]
      block_indices: [B, Tq, HK, topk] (preferred) or [B, Tq, HQ, topk]
    Returns:
      out: [B, Tq, HQ, D]
    """
    if q_bthd.ndim != 4 or k_bthd.ndim != 4 or v_bthd.ndim != 4:
        raise ValueError("q/k/v must be rank-4 [B,T,H,D].")
    if k_bthd.shape != v_bthd.shape:
        raise ValueError("k and v must have identical [B,Tk,H,D] shape.")
    if q_bthd.shape[0] != k_bthd.shape[0]:
        raise ValueError("q and k/v batch size must match.")
    if q_bthd.shape[3] != k_bthd.shape[3]:
        raise ValueError("q and k/v head dim must match.")
    if q_bthd.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"FSA kernels require fp16/bf16; got {q_bthd.dtype}.")
    if k_bthd.dtype != q_bthd.dtype or v_bthd.dtype != q_bthd.dtype:
        raise ValueError("q/k/v dtype must match.")
    if block_size not in {32, 64, 128, 256, 512, 1024}:
        raise ValueError(f"FSA wrappers support block_size in {{32,64,128,256,512,1024}}; got {block_size}.")

    B, Tq, HQ, D = q_bthd.shape
    _, Tk, HK, _ = k_bthd.shape
    if HQ % HK != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HK ({HK}) for GQA.")
    gqa_deg = HQ // HK
    device = q_bthd.device

    q = q_bthd.reshape(B * Tq, HQ, D).contiguous()
    k = k_bthd.reshape(B * Tk, HK, D).contiguous()
    v = v_bthd.reshape(B * Tk, HK, D).contiguous()

    if topk_idx_hns is not None:
        if topk_idx_hns.ndim != 3:
            raise ValueError("topk_idx_hns must be rank-3 [HK or HQ, B*Tq, topk].")
        if topk_idx_hns.shape[1] != (B * Tq):
            raise ValueError(
                f"topk_idx_hns shape mismatch, expected second dim B*Tq={B*Tq}, got {tuple(topk_idx_hns.shape)}."
            )
        if topk_idx_hns.shape[0] == HK:
            topk_idx = topk_idx_hns
        elif topk_idx_hns.shape[0] == HQ:
            # Convert per-query-head routes to per-kv-head routes by taking the first query-head
            # in each GQA group. In typical GQA use these are shared across the group.
            topk_idx = _collapse_hq_routes_to_hk(topk_idx_hns, hk=HK, gqa_deg=gqa_deg)
        else:
            raise ValueError(
                f"topk_idx_hns first dim must be HK={HK} or HQ={HQ}, got {topk_idx_hns.shape[0]}."
            )
    else:
        if block_indices_bths is None:
            raise ValueError("Either block_indices_bths [B,Tq,H,topk] or topk_idx_hns [H,B*Tq,topk] must be provided.")
        if block_indices_bths.ndim != 4:
            raise ValueError("block_indices_bths must be rank-4 [B,Tq,H,topk].")
        if block_indices_bths.shape[0] != B or block_indices_bths.shape[1] != Tq:
            raise ValueError("block_indices_bths must have prefix [B,Tq,...] matching q.")
        if block_indices_bths.shape[2] == HK:
            topk_idx = block_indices_bths.permute(0, 2, 1, 3).reshape(HK, B * Tq, -1)
        elif block_indices_bths.shape[2] == HQ:
            topk_idx_q = block_indices_bths.permute(0, 2, 1, 3).reshape(HQ, B * Tq, -1)
            topk_idx = _collapse_hq_routes_to_hk(topk_idx_q, hk=HK, gqa_deg=gqa_deg)
        else:
            raise ValueError(
                f"block_indices_bths third dim must be HK={HK} or HQ={HQ}, got {block_indices_bths.shape[2]}."
            )

    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    if not assume_sorted_topk:
        # FSA kernels expect per-query top-k entries to be ordered; unsorted entries
        # can mis-handle causal-valid counts vs traversal order.
        topk_idx = topk_idx.sort(dim=-1).values
    topk_idx = topk_idx.contiguous()

    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(B + 1, device=device, dtype=torch.int32) * Tq
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.arange(B + 1, device=device, dtype=torch.int32) * Tk

    out = FSA_topk_sparse_attention_varlen_qk(
        q=q,
        k=k,
        v=v,
        topk_idx=topk_idx,
        block_size=block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        disable_causal_mask=disable_causal_mask,
    )
    return out.reshape(B, Tq, HQ, D)


def _collapse_hq_routes_to_hk(
    topk_idx_q: torch.Tensor,
    hk: int,
    gqa_deg: int,
) -> torch.Tensor:
    """
    Collapse per-query-head routing [HQ, N, S] to per-kv-head routing [HK, N, S].

    Modes via `FSA_LOCAL_GQA_ROUTE_COLLAPSE`:
      - strict/validate: require identical routes inside each GQA group
      - first: take first query-head route in each GQA group
      - auto (default): validate, then fallback to first with a warning
    """
    mode = os.getenv("FSA_LOCAL_GQA_ROUTE_COLLAPSE", "auto").strip().lower()
    grouped = topk_idx_q.reshape(hk, gqa_deg, topk_idx_q.shape[1], topk_idx_q.shape[2])
    if mode in ("first", "head0"):
        return grouped[:, 0, :, :]

    ref = grouped[:, 0:1, :, :]
    shared = bool(torch.equal(grouped, ref.expand_as(grouped)))
    if shared:
        return grouped[:, 0, :, :]

    if mode in ("strict", "validate", "error"):
        raise ValueError(
            "Packed-GQA requires shared routing inside each GQA group, but HQ routes differ. "
            "Set FSA_LOCAL_GQA_ROUTE_COLLAPSE=first to force best-effort collapse."
        )

    # auto and unknown modes: best-effort fallback to head0.
    global _GQA_ROUTE_COLLAPSE_WARNED
    if not _GQA_ROUTE_COLLAPSE_WARNED:
        print(
            "FSA local warning: HQ routes differ within GQA groups; "
            "falling back to head0 collapse (set FSA_LOCAL_GQA_ROUTE_COLLAPSE=strict to enforce)."
        )
        _GQA_ROUTE_COLLAPSE_WARNED = True
    return grouped[:, 0, :, :]
