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
      - valid_topk_w_concat (only when weights are present)
      - valid_topk_idx_offsets
      - valid_lens_stack
      - valid_start_indices_stack
    """
    base_ok = (
        permute_results.get("valid_topk_idx_concat", None) is not None
        and permute_results.get("valid_topk_idx_offsets", None) is not None
        and permute_results.get("valid_lens_stack", None) is not None
        and permute_results.get("valid_start_indices_stack", None) is not None
    )
    if base_ok:
        has_w_list = permute_results.get("valid_topk_w_permuted_tile", None) is not None
        if not has_w_list:
            return permute_results
        if permute_results.get("valid_topk_w_concat", None) is not None:
            return permute_results

    valid_lens = permute_results.get("valid_lens", [])
    valid_start_indices = permute_results.get("valid_start_indices", [])
    valid_topk_idx_permuted_tile = permute_results.get("valid_topk_idx_permuted_tile", [])
    valid_topk_w_permuted_tile = permute_results.get("valid_topk_w_permuted_tile", None)

    if len(valid_lens) == 0 or len(valid_start_indices) == 0 or len(valid_topk_idx_permuted_tile) == 0:
        permute_results["valid_topk_idx_concat"] = torch.empty((0,), dtype=torch.int32, device=device)
        if valid_topk_w_permuted_tile is not None:
            permute_results["valid_topk_w_concat"] = torch.empty((0,), dtype=torch.float32, device=device)
        permute_results["valid_topk_idx_offsets"] = torch.zeros((num_kv_heads,), dtype=torch.int32, device=device)
        permute_results["valid_lens_stack"] = torch.zeros((num_kv_heads, num_blocks), dtype=torch.int32, device=device)
        permute_results["valid_start_indices_stack"] = torch.zeros((num_kv_heads, num_blocks), dtype=torch.int32, device=device)
        return permute_results

    valid_lens_stack = torch.stack(valid_lens, dim=0).contiguous()
    valid_start_indices_stack = torch.stack(valid_start_indices, dim=0).contiguous()
    per_kh_counts = valid_lens_stack.sum(dim=1).to(torch.int32).contiguous()
    offsets = torch.zeros((num_kv_heads,), dtype=torch.int32, device=device)
    if num_kv_heads > 1:
        offsets[1:] = torch.cumsum(per_kh_counts, dim=0)[:-1]
    total_sel = int(per_kh_counts.sum().item())
    if total_sel > 0:
        concat = torch.cat(valid_topk_idx_permuted_tile, dim=0).contiguous()
        if valid_topk_w_permuted_tile is not None and all(x is not None for x in valid_topk_w_permuted_tile):
            w_concat = torch.cat(valid_topk_w_permuted_tile, dim=0).to(dtype=torch.float32).contiguous()
        else:
            w_concat = None
    else:
        concat = torch.empty((0,), dtype=torch.int32, device=device)
        w_concat = torch.empty((0,), dtype=torch.float32, device=device) if valid_topk_w_permuted_tile is not None else None

    permute_results["valid_topk_idx_concat"] = concat
    if w_concat is not None:
        permute_results["valid_topk_w_concat"] = w_concat
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
    first = int(nz[0].item())
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
    return starts.contiguous(), full_counts.contiguous()


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
        dropped = int(invalid.to(dtype=torch.int64).sum().item())
        if dropped <= 0:
            return topk_idx, 0
        out = torch.full_like(topk_idx, -1).contiguous()
        _record_block_prune_stat("sanitize_applied")
        _record_block_prune_stat("sanitize_dropped", dropped)
        return out, dropped

    valid = (topk_idx >= 0) & (topk_idx < real_num_blocks)
    invalid = (topk_idx >= 0) & (~valid)
    dropped = int(invalid.to(dtype=torch.int64).sum().item())
    if dropped <= 0:
        return topk_idx, 0

    out = torch.where(valid, topk_idx, torch.full_like(topk_idx, -1))
    # Keep deterministic sorted order after sanitization.
    out = out.sort(dim=-1).values.contiguous()
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
        eff = int(nz[-1].item()) + 1
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
    ).to(torch.int32).contiguous()

    offs = torch.arange(block_size, device=k_seq.device, dtype=torch.int64).view(1, block_size)
    tok = (selected.view(-1, 1) * block_size + offs).reshape(-1)
    k_new = k_seq.index_select(1, tok).contiguous()
    v_new = v_seq.index_select(1, tok).contiguous()

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


@triton.jit
def block_to_token_kernel_with_weights(
    topk_idx_ptr,
    topk_w_ptr,
    result_idx_ptr,
    result_w_ptr,
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
    rws_h,
    rws_b,
    rws_n,
    num_q_loops: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = 0
    offs = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, num_q_loops)
    pid_j = pid * num_q_loops + offs_q

    topk_idx_offset = pid_h * ts_h + pid_j[None, :] * K + offs[:, None]
    block_ids = tl.load(
        topk_idx_ptr + topk_idx_offset,
        mask=(pid_j < N_token)[None, :] & (offs < K)[:, None],
        other=padding_value,
    )
    w = tl.load(
        topk_w_ptr + topk_idx_offset,
        mask=(pid_j < N_token)[None, :] & (offs < K)[:, None],
        other=0.0,
    ).to(tl.float32)

    result_idx_ptrs = result_idx_ptr + pid_h * rs_h + block_ids * N_token + pid_j[None, :]
    result_w_ptrs = result_w_ptr + pid_h * rws_h + block_ids * N_token + pid_j[None, :]
    mask = (block_ids >= 0) & (block_ids != padding_value) & (pid_j < N_token)[None, :]
    tl.store(result_idx_ptrs, pid_j[None, :], mask=mask)
    tl.store(result_w_ptrs, w, mask=mask)


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


def build_block_to_token_with_weights_triton(
    result_idx: torch.Tensor,
    result_w: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    min_block_id: int,
    max_block_id: int,
    padding_value: int = -1,
):
    """
    Weighted variant of build_block_to_token_triton.

    Writes:
      result_idx: [num_heads, num_blocks, N_token] with token indices (or -1)
      result_w:   [num_heads, num_blocks, N_token] with per-(block,token) weights (0 for padding)
    """
    assert topk_idx.ndim == 3
    assert topk_w.ndim == 3
    assert topk_w.shape == topk_idx.shape
    assert padding_value == -1
    num_heads, N_token, TopK = topk_idx.shape

    num_q_loops = 4
    grid = (triton.cdiv(N_token, num_q_loops),)
    BLOCK_K = triton.next_power_of_2(TopK)
    block_to_token_kernel_with_weights[grid](
        topk_idx,
        topk_w,
        result_idx,
        result_w,
        N_token,
        TopK,
        min_block_id,
        max_block_id,
        padding_value,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        result_idx.stride(0),
        result_idx.stride(1),
        result_idx.stride(2),
        result_w.stride(0),
        result_w.stride(1),
        result_w.stride(2),
        num_q_loops,
        BLOCK_K=BLOCK_K,
        num_warps=2,
        num_stages=3,
    )
    return result_idx, result_w

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
    selected_weights_ptr,  # selected_weights: sum(valid_lens),
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
    HAS_WEIGHTS: tl.constexpr,
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
            if HAS_WEIGHTS:
                w = tl.load(selected_weights_ptr + st_offs, mask=st_mask, other=0.0).to(tl.float32)
                w = tl.maximum(w, 0.0)
                p = p * w[:, None]
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
    topk_w: Optional[torch.Tensor],  # [num_heads, total_len, topk] or None
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal=True,
):
    """
        Sequence packing is still done at wrapper level for multi-sequence varlen inputs.
        Fast path avoids loop overhead when there is only one sequence.
    """
    o = torch.empty_like(q)
    total_len, num_heads, _ = q.shape
    lse = torch.empty((num_heads, total_len), dtype=torch.float32, device=q.device)

    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) != len(k_ranges):
        raise RuntimeError(
            f"Mismatched sequence partitions: len(q_ranges)={len(q_ranges)} vs len(k_ranges)={len(k_ranges)}."
        )

    permute_results = []
    if len(q_ranges) == 1:
        (q_start, q_end) = q_ranges[0]
        (k_start, k_end) = k_ranges[0]
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        cu_q_local = torch.tensor([0, q_len], dtype=torch.int32, device=cu_seqlens_q.device)
        cu_k_local = torch.tensor([0, k_len], dtype=torch.int32, device=cu_seqlens_k.device)
        o_seq, lse_seq, permute_results_seq = _topk_sparse_attention_fwd_opt_per_seq(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            topk_w[:, q_start:q_end] if topk_w is not None else None,
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

    for (q_start, q_end), (k_start, k_end) in zip(q_ranges, k_ranges):
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        cu_q_local = torch.tensor([0, q_len], dtype=torch.int32, device=cu_seqlens_q.device)
        cu_k_local = torch.tensor([0, k_len], dtype=torch.int32, device=cu_seqlens_k.device)
        o_seq, lse_seq, permute_results_seq = _topk_sparse_attention_fwd_opt_per_seq(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            topk_w[:, q_start:q_end] if topk_w is not None else None,
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
    max_tokens = int(valid_lens.max().item()) if valid_lens.numel() > 0 else 0
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
    valid_topk_w_permuted_tile,
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
        valid_topk_w_permuted_tile if valid_topk_w_permuted_tile is not None else valid_topk_idx_permuted_tile,
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
        HAS_WEIGHTS=(valid_topk_w_permuted_tile is not None),
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
    topk_w: Optional[torch.Tensor],  # [num_kv_heads, total_len_q, topk] or None
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
        bi_seq=topk_idx.permute(1, 0, 2).contiguous().unsqueeze(0),
        block_size=block_size,
    )
    if _compacted:
        k = k_seq_c[0].contiguous()
        v = v_seq_c[0].contiguous()
        topk_idx = bi_seq_c[0].permute(1, 0, 2).contiguous()
        total_len_k = int(k.shape[0])

    if num_q_heads % num_kv_heads != 0:
        _record_fwd_full_deser_stat("fallback_hq_hk")
        _maybe_print_fwd_full_deser_notice(
            "FSA full-deser forward fallback -> legacy: precondition failed (HQ % HK != 0)."
        )
        return None

    gqa_deg = num_q_heads // num_kv_heads
    topk = topk_idx.shape[-1]
    real_num_blocks = math.ceil(total_len_k / block_size)
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    if topk_w is not None:
        if topk_w.shape != topk_idx.shape:
            raise ValueError("topk_w must match topk_idx shape [H, N, S].")
        topk_w = topk_w.to(dtype=torch.float32)
        topk_w = torch.where(topk_idx >= 0, topk_w, torch.zeros_like(topk_w))
    num_blocks_full = max(real_num_blocks, topk)

    valid_lens_all_full = torch.zeros((num_kv_heads, num_blocks_full), dtype=torch.int32, device=q.device)
    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        topk_idx_nonneg = topk_idx_tile[(topk_idx_tile >= 0) & (topk_idx_tile < real_num_blocks)]
        valid_lens_all_full[kh:kh + 1] = torch.bincount(topk_idx_nonneg.reshape(-1), minlength=num_blocks_full)

    num_blocks = _resolve_effective_num_blocks(
        valid_lens_all=valid_lens_all_full,
        num_blocks_full=num_blocks_full,
        real_num_blocks=real_num_blocks,
        head_dim=head_dim,
        block_size=block_size,
    )
    valid_lens_all = valid_lens_all_full[:, :num_blocks].contiguous()
    reduce_tile_size = max(0, num_blocks - 1)

    active_starts, active_counts = _detect_active_token_ranges_per_kv_head(topk_idx)
    active_ends = active_starts + active_counts
    routed_heads = active_counts > 0
    if bool(torch.any(routed_heads)):
        query_start_idx = int(active_starts[routed_heads].min().item())
        query_end_idx = int(active_ends[routed_heads].max().item())
        query_tokens_count = max(0, query_end_idx - query_start_idx)
        if num_kv_heads > 1:
            ref_start = int(active_starts[0].item())
            ref_count = int(active_counts[0].item())
            same_ranges = bool(torch.all((active_starts == ref_start) & (active_counts == ref_count)))
            if not same_ranges:
                _record_fwd_full_deser_stat("expanded_active_range")
    else:
        query_start_idx = 0
        query_tokens_count = 0

    global_max_valid_tokens = int(valid_lens_all[:, 1:].max().item()) if num_blocks > 1 else int(valid_lens_all.max().item())

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_q_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)

    topk_idx_permuted_tile = torch.full((1, num_blocks, total_len_q), -1, dtype=torch.int32, device=q.device)
    topk_w_permuted_tile = None
    if topk_w is not None:
        topk_w_permuted_tile = torch.zeros((1, num_blocks, total_len_q), dtype=torch.float32, device=q.device)
    permute_results = {
        "global_max_valid_tokens": global_max_valid_tokens,
        "num_blocks": num_blocks,
        "num_blocks_full": num_blocks_full,
        "real_num_blocks": real_num_blocks,
        "valid_topk_idx_permuted_tile": [],
        "valid_topk_w_permuted_tile": [],
        "valid_lens_all": valid_lens_all_full,
        "valid_lens": [],
        "valid_start_indices": [],
    }
    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        if topk_w is None:
            build_block_to_token_triton(topk_idx_permuted_tile, topk_idx_tile, 0, num_blocks, padding_value=-1)
        else:
            assert topk_w_permuted_tile is not None
            topk_w_tile = topk_w[kh:kh + 1].to(torch.float32).contiguous()
            build_block_to_token_with_weights_triton(
                topk_idx_permuted_tile,
                topk_w_permuted_tile,
                topk_idx_tile,
                topk_w_tile,
                0,
                num_blocks,
                padding_value=-1,
            )
        mask_valid = topk_idx_permuted_tile != -1
        valid_topk_idx_permuted_tile = topk_idx_permuted_tile[mask_valid]
        valid_topk_w_permuted_tile = None
        if topk_w is not None:
            assert topk_w_permuted_tile is not None
            valid_topk_w_permuted_tile = topk_w_permuted_tile[mask_valid].to(torch.float32)
        valid_lens = valid_lens_all[kh]
        valid_start_indices = torch.nn.functional.pad(valid_lens.cumsum(0)[:-1], (1, 0), value=0)
        permute_results["valid_topk_idx_permuted_tile"].append(valid_topk_idx_permuted_tile)
        permute_results["valid_topk_w_permuted_tile"].append(valid_topk_w_permuted_tile)
        permute_results["valid_lens"].append(valid_lens)
        permute_results["valid_start_indices"].append(valid_start_indices)
        topk_idx_permuted_tile.fill_(-1)
        if topk_w_permuted_tile is not None:
            topk_w_permuted_tile.zero_()

    # Build one concatenated selected-token list; each query head uses offsets into this list.
    kh_offsets = torch.zeros((num_kv_heads,), dtype=torch.int32, device=q.device)
    selected_count = 0
    for kh in range(num_kv_heads):
        kh_offsets[kh] = selected_count
        selected_count += int(permute_results["valid_topk_idx_permuted_tile"][kh].numel())
    if selected_count <= 0:
        _record_fwd_full_deser_stat("success")
        return o_full, lse_full, permute_results

    selected_tokens_all = torch.cat(permute_results["valid_topk_idx_permuted_tile"], dim=0).contiguous()
    selected_weights_all = None
    if topk_w is not None:
        selected_weights_all = torch.cat(permute_results["valid_topk_w_permuted_tile"], dim=0).contiguous()
    valid_lens_stack = torch.stack(permute_results["valid_lens"], dim=0).contiguous()  # [HK, num_blocks]
    valid_start_stack = torch.stack(permute_results["valid_start_indices"], dim=0).contiguous()  # [HK, num_blocks]

    use_fwd_packed_gqa = _resolve_fwd_packed_gqa(
        num_share_q_heads=gqa_deg,
        head_dim=head_dim,
        block_size=block_size,
    )
    if use_fwd_packed_gqa and gqa_deg > 1:
        head_tile = _resolve_head_tile(gqa_deg)
        token_index_mapping = torch.empty((1, num_blocks, total_len_q), dtype=torch.int32, device=q.device)
        o_tiles_first = torch.empty((head_tile, 1, total_len_q, head_dim), dtype=torch.bfloat16, device=q.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            o_tiles_rest = torch.empty(
                (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim),
                dtype=torch.bfloat16,
                device=q.device,
            )
        else:
            o_tiles_rest = torch.empty((head_tile, 1, 1, head_dim), dtype=torch.bfloat16, device=q.device)
        m_i_cur_tiles = torch.empty((head_tile, num_blocks, total_len_q), dtype=torch.float32, device=q.device)
        l_ij_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
        acc_o_scales_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            l_ij_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
            acc_o_scales_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
        else:
            l_ij_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)
            acc_o_scales_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)

        for kh in range(num_kv_heads):
            valid_topk_idx_permuted_tile = permute_results["valid_topk_idx_permuted_tile"][kh]
            valid_topk_w_permuted_tile = None
            if "valid_topk_w_permuted_tile" in permute_results:
                valid_topk_w_permuted_tile = permute_results["valid_topk_w_permuted_tile"][kh]
            valid_lens = permute_results["valid_lens"][kh]
            valid_start_indices = permute_results["valid_start_indices"][kh]
            if int(valid_topk_idx_permuted_tile.numel()) <= 0:
                continue
            index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

            query_start_idx_kh = int(active_starts[kh].item())
            query_tokens_count_kh = int(active_counts[kh].item())
            if query_tokens_count_kh <= 0:
                continue

            topk_idx_tile_base = topk_idx[kh:kh + 1]
            max_valid_first = int(valid_lens[0].item()) if valid_lens.numel() > 0 else 0
            max_valid_rest = (
                int(valid_lens[1:].max().item())
                if (num_blocks > 1 and valid_lens.shape[0] > 1)
                else 0
            )

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

                valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1).contiguous()
                valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1).contiguous()

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
                        cur_valid_lens = valid_lens_tile[:, 0].contiguous()
                        cur_valid_start_indices = valid_start_indices_tile[:, 0].contiguous()
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
                        cur_valid_lens = valid_lens_tile[:, compute_min_block_id:].contiguous()
                        cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:].contiguous()
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
                        valid_topk_w_permuted_tile,
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

        valid_lens_qh = valid_lens_stack.index_select(0, qh_to_kh).contiguous()  # [HQ, num_blocks]
        valid_start_qh = valid_start_stack.index_select(0, qh_to_kh).contiguous() + kh_offsets.index_select(0, qh_to_kh).view(-1, 1)
        token_index_mapping_qh = torch.empty((num_q_heads, num_blocks, total_len_q), dtype=torch.int32, device=q.device)
        index_mapping(token_index_mapping_qh, selected_tokens_all, valid_lens_qh, valid_start_qh, num_blocks)

        topk_idx_qh = topk_idx.index_select(0, qh_to_kh).contiguous()
        k_qh = k.index_select(1, qh_to_kh).contiguous()
        v_qh = v.index_select(1, qh_to_kh).contiguous()

        head_tile = num_q_heads
        o_tiles_first = torch.empty((head_tile, 1, total_len_q, head_dim), dtype=torch.bfloat16, device=q.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            o_tiles_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=q.device)
        else:
            o_tiles_rest = torch.empty((head_tile, 1, 1, head_dim), dtype=torch.bfloat16, device=q.device)
        m_i_cur_tiles = torch.empty((head_tile, num_blocks, total_len_q), dtype=torch.float32, device=q.device)
        l_ij_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
        acc_o_scales_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            l_ij_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
            acc_o_scales_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
        else:
            l_ij_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)
            acc_o_scales_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)

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

        max_valid_first = int(valid_lens_qh[:, 0].max().item()) if valid_lens_qh.numel() > 0 else 0
        max_valid_rest = (
            int(valid_lens_qh[:, 1:].max().item())
            if (num_blocks > 1 and valid_lens_qh.shape[1] > 1)
            else 0
        )

        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                compute_tile_size = 1
                cur_max_valid_tokens = max_valid_first
                cur_valid_lens = valid_lens_qh[:, 0].contiguous()
                cur_valid_start_indices = valid_start_qh[:, 0].contiguous()
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
                cur_valid_lens = valid_lens_qh[:, compute_min_block_id:].contiguous()
                cur_valid_start_indices = valid_start_qh[:, compute_min_block_id:].contiguous()
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
                selected_weights_all,
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

    permute_results = _ensure_dq_atomic_metadata(
        permute_results=permute_results,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        device=q.device,
    )
    _record_fwd_full_deser_stat("success")
    return o_full, lse_full, permute_results


def _topk_sparse_attention_fwd_opt_per_seq(
    q: torch.Tensor,  # [total_len, num_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_heads, total_len, topk]
    topk_w: Optional[torch.Tensor],  # [num_kv_heads, total_len, topk] or None
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
        bi_seq=topk_idx.permute(1, 0, 2).contiguous().unsqueeze(0),
        block_size=block_size,
    )
    if _compacted:
        k = k_seq_c[0].contiguous()
        v = v_seq_c[0].contiguous()
        topk_idx = bi_seq_c[0].permute(1, 0, 2).contiguous()
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
            topk_w=topk_w,
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

    TOPK = topk_idx.shape[-1]
    real_num_blocks = math.ceil(total_len_k / block_size)
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    if topk_w is not None:
        if topk_w.shape != topk_idx.shape:
            raise ValueError("topk_w must match topk_idx shape [H, N, S].")
        topk_w = topk_w.to(dtype=torch.float32)
        topk_w = torch.where(topk_idx >= 0, topk_w, torch.zeros_like(topk_w))
    num_blocks_full = max(real_num_blocks, TOPK)
    valid_lens_all_full = torch.zeros((num_kv_heads, num_blocks_full), dtype=torch.int32, device=q.device)
    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        topk_idx_nonneg = topk_idx_tile[(topk_idx_tile >= 0) & (topk_idx_tile < real_num_blocks)]
        valid_lens_all_full[kh:kh + 1] = torch.bincount(topk_idx_nonneg.reshape(-1), minlength=num_blocks_full)
    num_blocks = _resolve_effective_num_blocks(
        valid_lens_all=valid_lens_all_full,
        num_blocks_full=num_blocks_full,
        real_num_blocks=real_num_blocks,
        head_dim=head_dim,
        block_size=block_size,
    )
    reduce_tile_size = num_blocks - 1

    valid_lens_all = valid_lens_all_full[:, :num_blocks].contiguous()

    active_starts, active_counts = _detect_active_token_ranges_per_kv_head(topk_idx)

    global_max_valid_tokens = (
        int(valid_lens_all[:, 1:].max().item())
        if num_blocks > 1
        else int(valid_lens_all.max().item())
    )

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)

    topk_idx_permuted_tile = torch.full((1, num_blocks, total_len_q), -1, dtype=torch.int32, device=q.device)
    topk_w_permuted_tile = None
    if topk_w is not None:
        topk_w_permuted_tile = torch.zeros((1, num_blocks, total_len_q), dtype=torch.float32, device=q.device)
    token_index_mapping = torch.empty((1, num_blocks, total_len_q), dtype=torch.int32, device=q.device)

    o_tiles_first = torch.empty((head_tile, 1, total_len_q, head_dim), dtype=torch.bfloat16, device=q.device)
    if reduce_tile_size > 0 and global_max_valid_tokens > 0:
        o_tiles_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=q.device)
    else:
        o_tiles_rest = torch.empty((head_tile, 1, 1, head_dim), dtype=torch.bfloat16, device=q.device)
    m_i_cur_tiles = torch.empty((head_tile, num_blocks, total_len_q), dtype=torch.float32, device=q.device)
    l_ij_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
    acc_o_scales_first = torch.empty((head_tile, 1, total_len_q), dtype=torch.float32, device=q.device)
    if reduce_tile_size > 0 and global_max_valid_tokens > 0:
        l_ij_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
        acc_o_scales_rest = torch.empty((head_tile, reduce_tile_size, global_max_valid_tokens), dtype=torch.float32, device=q.device)
    else:
        l_ij_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)
        acc_o_scales_rest = torch.empty((head_tile, 1, 1), dtype=torch.float32, device=q.device)

    permute_results = {
        "global_max_valid_tokens": global_max_valid_tokens,
        "num_blocks": num_blocks,
        "num_blocks_full": num_blocks_full,
        "real_num_blocks": real_num_blocks,
        "valid_topk_idx_permuted_tile": [],
        "valid_topk_w_permuted_tile": [],
        "valid_lens_all": valid_lens_all_full,
        "valid_lens": [],
        "valid_start_indices": [],
    }

    for kh in range(num_kv_heads):
        topk_idx_tile_base = topk_idx[kh:kh + 1]
        if topk_w is None:
            build_block_to_token_triton(topk_idx_permuted_tile, topk_idx_tile_base, 0, num_blocks, padding_value=-1)
        else:
            assert topk_w_permuted_tile is not None
            topk_w_tile = topk_w[kh:kh + 1].to(torch.float32).contiguous()
            build_block_to_token_with_weights_triton(
                topk_idx_permuted_tile,
                topk_w_permuted_tile,
                topk_idx_tile_base,
                topk_w_tile,
                0,
                num_blocks,
                padding_value=-1,
            )
        mask_valid = topk_idx_permuted_tile != -1
        valid_topk_idx_permuted_tile = topk_idx_permuted_tile[mask_valid]
        valid_topk_w_permuted_tile = None
        if topk_w is not None:
            assert topk_w_permuted_tile is not None
            valid_topk_w_permuted_tile = topk_w_permuted_tile[mask_valid].to(torch.float32)
        valid_lens = valid_lens_all[kh]
        valid_start_indices = torch.nn.functional.pad(valid_lens.cumsum(0)[:-1], (1, 0), value=0)
        index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

        permute_results["valid_topk_idx_permuted_tile"].append(valid_topk_idx_permuted_tile)
        permute_results["valid_topk_w_permuted_tile"].append(valid_topk_w_permuted_tile)
        permute_results["valid_lens"].append(valid_lens)
        permute_results["valid_start_indices"].append(valid_start_indices)

        query_start_idx = int(active_starts[kh].item())
        query_tokens_count = int(active_counts[kh].item())
        if query_tokens_count <= 0:
            fused_fill(topk_idx_permuted_tile, m_i_cur_tiles[:1])
            if topk_w_permuted_tile is not None:
                topk_w_permuted_tile.zero_()
            continue

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

            valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1).contiguous()
            valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1).contiguous()

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
                    cur_max_valid_tokens = int(valid_lens[0].item()) if valid_lens.numel() > 0 else 0
                    cur_valid_lens = valid_lens_tile[:, 0].contiguous()
                    cur_valid_start_indices = valid_start_indices_tile[:, 0].contiguous()
                    o_tiles = o_tiles_first[:ht]
                    l_ij = l_ij_first_tile
                    acc_o_scales = acc_o_scales_first_tile
                else:
                    compute_tile_size = num_blocks - 1
                    if compute_tile_size <= 0:
                        continue
                    cur_valid_lens = valid_lens_tile[:, compute_min_block_id:].contiguous()
                    if cur_valid_lens.numel() == 0:
                        continue
                    cur_max_valid_tokens = int(cur_valid_lens.max().item())
                    if cur_max_valid_tokens <= 0:
                        continue
                    cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:].contiguous()
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
                    valid_topk_w_permuted_tile,
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

        fused_fill(topk_idx_permuted_tile, m_i_cur_tiles[:1])
        if topk_w_permuted_tile is not None:
            topk_w_permuted_tile.zero_()

    return o_full, lse_full, permute_results


def _build_permute_results_per_seq_for_bwd(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len_q, topk]
    topk_w: Optional[torch.Tensor],  # [num_kv_heads, total_len_q, topk] or None
    total_len_k: int,
    block_size: int,
):
    """
    Build only the permutation metadata needed by backward, without running forward math kernels.
    """
    _, total_len_q, topk = topk_idx.shape
    num_kv_heads = topk_idx.shape[0]
    real_num_blocks = math.ceil(total_len_k / block_size)
    topk_idx, _ = _sanitize_topk_block_indices(topk_idx, real_num_blocks=real_num_blocks)
    if topk_w is not None:
        if topk_w.shape != topk_idx.shape:
            raise ValueError("topk_w must match topk_idx shape [H, N, S].")
        topk_w = topk_w.to(dtype=torch.float32)
        topk_w = torch.where(topk_idx >= 0, topk_w, torch.zeros_like(topk_w))
    num_blocks_full = max(real_num_blocks, topk)

    valid_lens_all_full = torch.zeros((num_kv_heads, num_blocks_full), dtype=torch.int32, device=topk_idx.device)
    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        topk_nonneg = topk_idx_tile[(topk_idx_tile >= 0) & (topk_idx_tile < real_num_blocks)]
        valid_lens = torch.bincount(topk_nonneg.reshape(-1), minlength=num_blocks_full)
        valid_lens_all_full[kh:kh + 1] = valid_lens

    num_blocks = _resolve_effective_num_blocks(
        valid_lens_all=valid_lens_all_full,
        num_blocks_full=num_blocks_full,
        real_num_blocks=real_num_blocks,
        head_dim=64,
        block_size=block_size,
    )
    valid_lens_all = valid_lens_all_full[:, :num_blocks].contiguous()

    global_max_valid_tokens = valid_lens_all[:, 1:].max() if num_blocks > 1 else valid_lens_all.max()
    topk_idx_permuted_tile = torch.full((1, num_blocks, total_len_q), -1, dtype=torch.int32, device=topk_idx.device)
    topk_w_permuted_tile = None
    if topk_w is not None:
        topk_w_permuted_tile = torch.zeros((1, num_blocks, total_len_q), dtype=torch.float32, device=topk_idx.device)

    permute_results = {
        "global_max_valid_tokens": global_max_valid_tokens,
        "num_blocks": num_blocks,
        "num_blocks_full": num_blocks_full,
        "real_num_blocks": real_num_blocks,
        "valid_topk_idx_permuted_tile": [],
        "valid_topk_w_permuted_tile": [],
        "valid_lens_all": valid_lens_all_full,
        "valid_lens": [],
        "valid_start_indices": [],
    }

    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        if topk_w is None:
            build_block_to_token_triton(topk_idx_permuted_tile, topk_idx_tile, 0, num_blocks, padding_value=-1)
        else:
            assert topk_w_permuted_tile is not None
            topk_w_tile = topk_w[kh:kh + 1].to(torch.float32).contiguous()
            build_block_to_token_with_weights_triton(
                topk_idx_permuted_tile,
                topk_w_permuted_tile,
                topk_idx_tile,
                topk_w_tile,
                0,
                num_blocks,
                padding_value=-1,
            )
        mask_valid = topk_idx_permuted_tile != -1
        valid_topk_idx_permuted_tile = topk_idx_permuted_tile[mask_valid]
        valid_topk_w_permuted_tile = None
        if topk_w is not None:
            assert topk_w_permuted_tile is not None
            valid_topk_w_permuted_tile = topk_w_permuted_tile[mask_valid].to(torch.float32)
        valid_lens = valid_lens_all[kh]
        valid_start_indices = torch.nn.functional.pad(valid_lens.cumsum(0)[:-1], (1, 0), value=0)

        permute_results["valid_topk_idx_permuted_tile"].append(valid_topk_idx_permuted_tile)
        permute_results["valid_topk_w_permuted_tile"].append(valid_topk_w_permuted_tile)
        permute_results["valid_lens"].append(valid_lens)
        permute_results["valid_start_indices"].append(valid_start_indices)

        topk_idx_permuted_tile.fill_(-1)
        if topk_w_permuted_tile is not None:
            topk_w_permuted_tile.zero_()

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
    topk_w: Optional[torch.Tensor],
):
    """
    Varlen wrapper for backward permutation metadata.
    """
    results = []
    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        topk_idx_seq = topk_idx[:, q_start:q_end].contiguous()
        topk_w_seq = topk_w[:, q_start:q_end].contiguous() if topk_w is not None else None
        results.append(
            _build_permute_results_per_seq_for_bwd(
                topk_idx=topk_idx_seq,
                topk_w=topk_w_seq,
                total_len_k=(k_end - k_start),
                block_size=block_size,
            )
        )
    return results


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
    topk_w: Optional[torch.Tensor],  # [num_kv_heads, total_len_q, topk] or None
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
    # Native NSA selected-fwd currently does not support per-block weights. When weights are present,
    # force the memory_cross_attn path for correctness.
    native_nsa_fwd = _try_get_native_nsa_parallel_fwd() if (use_native_nsa_fwd and topk_w is None) else None

    max_tokens_env = os.getenv("FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL", "auto").strip().lower()
    torch_chunk_env = os.getenv("FSA_LOCAL_TORCH_CHUNK_TOKENS", "512").strip().lower()

    def _torch_small_g_forward_chunk(
        q_chunk: torch.Tensor,     # [1, Tq, HQ, D]
        k_seq: torch.Tensor,       # [1, Tk, HK, D]
        v_seq: torch.Tensor,       # [1, Tk, HK, D]
        bi_chunk: torch.Tensor,    # [1, Tq, HK, S]
        bw_chunk: Optional[torch.Tensor],  # [1, Tq, HK, S] or None
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

        k_h = k_seq[0].permute(1, 0, 2).contiguous().to(torch.float32)           # [HK, Tk, D]
        v_h = v_seq[0].permute(1, 0, 2).contiguous().to(torch.float32)           # [HK, Tk, D]

        hk_idx = torch.arange(hk, device=q_chunk.device, dtype=torch.long).view(1, hk, 1)
        k_sel = k_h[hk_idx, tok_idx]                                             # [Tq, HK, Ksel, D]
        v_sel = v_h[hk_idx, tok_idx]                                             # [Tq, HK, Ksel, D]

        scores = torch.matmul(qg, k_sel.transpose(-1, -2)) * float(sm_scale)     # [Tq, HK, G, Ksel]
        score_mask = valid_tok.unsqueeze(2)                                       # [Tq, HK, 1, Ksel]
        scores = scores.masked_fill(~score_mask, float("-inf"))
        if bw_chunk is not None:
            # Apply per-chapter weights uniformly across that chapter's tokens (equivalent to +log(w) on logits).
            w = bw_chunk[0].to(dtype=torch.float32)  # [Tq, HK, S]
            w = w.unsqueeze(-1).expand(tqa, hk, s, block_size).reshape(tqa, hk, ksel)  # [Tq, HK, Ksel]
            logw = torch.where(
                w > 0,
                torch.log(w),
                torch.full_like(w, float("-inf")),
            )
            scores = scores.to(dtype=torch.float32) + logw.unsqueeze(2)  # [Tq, HK, G, Ksel]
            scores = scores.to(dtype=q_chunk.dtype)

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

        o_chunk = out.to(q_chunk.dtype).reshape(1, tqa, hq_real, d).contiguous()
        lse_chunk_e = lse_chunk_e.reshape(1, tqa, hq_real).to(torch.float32).contiguous()
        return o_chunk, lse_chunk_e

    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        q_len_seq = q_end - q_start
        k_len_seq = k_end - k_start
        real_num_blocks_seq = math.ceil(max(0, k_len_seq) / block_size)
        topk_idx_seq = topk_idx[:, q_start:q_end, :].contiguous()   # [HK, Tq, S]
        topk_w_seq = topk_w[:, q_start:q_end, :].contiguous() if topk_w is not None else None
        topk_idx_seq, _ = _sanitize_topk_block_indices(
            topk_idx_seq,
            real_num_blocks=real_num_blocks_seq,
        )
        if topk_w_seq is not None:
            if topk_w_seq.dtype != torch.float32:
                topk_w_seq = topk_w_seq.to(dtype=torch.float32)
            topk_w_seq = torch.where(topk_idx_seq >= 0, topk_w_seq, torch.zeros_like(topk_w_seq))

        if native_nsa_fwd is not None:
            # Native NSA selected-fwd requires q/k/v to share the same timeline length.
            # Build a prefix-timeline view: [memory | query], matching benchmark Path-A layout.
            q_seq_full = q[q_start:q_end].contiguous().unsqueeze(0)  # [1, Tfull, HQ, D]
            k_seq_mem = k[k_start:k_end].contiguous().unsqueeze(0)   # [1, Tk, HK, D]
            v_seq_mem = v[k_start:k_end].contiguous().unsqueeze(0)   # [1, Tk, HK, D]
            bi_seq_full = (
                topk_idx_seq
                .permute(1, 0, 2)
                .contiguous()
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

        q_seq = q[q_sub_start:q_sub_end].contiguous().unsqueeze(0)  # [1, Tq_active, HQ, D]
        k_seq = k[k_start:k_end].contiguous().unsqueeze(0)          # [1, Tk, HK, D]
        v_seq = v[k_start:k_end].contiguous().unsqueeze(0)          # [1, Tk, HK, D]
        bi_seq = (
            topk_idx_seq[:, query_start_idx: query_start_idx + query_tokens_count, :]
            .permute(1, 0, 2)
            .contiguous()
            .unsqueeze(0)
        )                                                           # [1, Tq_active, HK, S]
        bw_seq = None
        if topk_w_seq is not None:
            bw_seq = (
                topk_w_seq[:, query_start_idx: query_start_idx + query_tokens_count, :]
                .permute(1, 0, 2)
                .contiguous()
                .unsqueeze(0)
            )                                                       # [1, Tq_active, HK, S]

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
            q_chunk = q_seq[:, t0:t1].contiguous()
            bi_chunk = bi_seq[:, t0:t1].contiguous()
            bw_chunk = bw_seq[:, t0:t1].contiguous() if bw_seq is not None else None
            used_legacy_pad_path = False

            if use_torch_small_g:
                o_chunk, lse_chunk_e = _torch_small_g_forward_chunk(
                    q_chunk=q_chunk,
                    k_seq=k_seq,
                    v_seq=v_seq,
                    bi_chunk=bi_chunk,
                    bw_chunk=bw_chunk,
                )
            else:
                if use_packed_nsa_chunk:
                    tqa = int(q_chunk.shape[1])
                    o_chunk = torch.empty((1, tqa, hq_real, d), dtype=q_chunk.dtype, device=q_chunk.device)
                    lse_chunk_e = torch.full((1, tqa, hq_real), float("-inf"), dtype=torch.float32, device=q_chunk.device)
                    head_tile_packed = _resolve_head_tile(gqa_deg)
                    head_tile_packed = max(1, min(head_tile_packed, gqa_deg))
                    for kh_i in range(hk):
                        bi_base = bi_chunk[:, :, kh_i:kh_i + 1, :].contiguous()
                        bw_base = bw_chunk[:, :, kh_i:kh_i + 1, :].contiguous() if bw_chunk is not None else None
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
                                block_weights=bw_base,
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
                        q_call = q_padded_grouped.view(1, tqa, hk * g_target, d).contiguous()
                        used_legacy_pad_path = True
                    else:
                        q_call = q_chunk

                    o_chunk, lse_chunk_e = memory_cross_attn_forward(
                        q=q_call,
                        k=k_seq,
                        v=v_seq,
                        block_indices=bi_chunk,
                        block_weights=bw_chunk,
                        block_size=block_size,
                        scale=sm_scale,
                    )

            if used_legacy_pad_path:
                g_target = 16
                tqa = int(o_chunk.shape[1])
                o_grouped = o_chunk.view(1, tqa, hk, g_target, d)
                o_chunk = o_grouped[:, :, :, :gqa_deg, :].reshape(1, tqa, hq_real, d).contiguous()

                lse_grouped = lse_chunk_e.view(1, tqa, hk, g_target)
                lse_chunk_e = lse_grouped[:, :, :, :gqa_deg].reshape(1, tqa, hq_real).contiguous()

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
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_tiles_ptr,
    token_index_mapping_ptr,
    selected_tokens_ptr,
    selected_weights_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    cur_max_valid_tokens,
    compute_min_block_id,
    head_tile,
    num_blocks,
    HEAD_DIM,
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
    HAS_WEIGHTS: tl.constexpr,
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

            st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
            # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
            q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd

            mask = st != -1

            q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + st[:, None] * stride_don + off_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            delta_ptrs = delta_ptr + pid_h * stride_dh + st[:, None] * stride_dn
            d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            lse_ptrs = lse_ptr + pid_h * stride_lh + st[:, None] * stride_ln
            lse = tl.load(lse_ptrs, mask=mask[:, None], other=0)

            dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k)) * qk_scale  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            if not DISABLE_CAUSAL_MASK:
                qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))
            p = tl.exp2(qk - lse)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            if HAS_WEIGHTS:
                w = tl.load(selected_weights_ptr + st_offs, mask=st_mask, other=0.0).to(tl.float32)
                w = tl.maximum(w, 0.0)
                p = p * w[:, None]
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

            dq_ptrs_off = token_index_mapping[:, None] * stride_dqtn + off_d[None, :] * stride_dqtd
            dq_tiles_ptrs = dq_tiles_ptr + dq_ptrs_off + (pid_block).to(tl.int64) * stride_dqtb + pid_h.to(tl.int64) * stride_dqth
            tl.store(dq_tiles_ptrs, dq.to(dq_tiles_ptr.dtype.element_ty), mask=q_mask)


@triton.jit
def dq_compute_atomic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_accum_ptr,           # float32 [N, H, D]
    selected_tokens_ptr,
    selected_weights_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    cur_max_valid_tokens,
    compute_min_block_id,
    head_tile,
    num_blocks,
    HEAD_DIM,
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
    HAS_WEIGHTS: tl.constexpr,
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
            st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

            mask = st != -1
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]

            q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + st[:, None] * stride_qn + off_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + st[:, None] * stride_don + off_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            delta_ptrs = delta_ptr + pid_h * stride_dh + st[:, None] * stride_dn
            d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            lse_ptrs = lse_ptr + pid_h * stride_lh + st[:, None] * stride_ln
            lse = tl.load(lse_ptrs, mask=mask[:, None], other=0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
            if not DISABLE_CAUSAL_MASK:
                qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))

            p = tl.exp2(qk - lse)
            if HAS_WEIGHTS:
                w = tl.load(selected_weights_ptr + st_offs, mask=st_mask, other=0.0).to(tl.float32)
                w = tl.maximum(w, 0.0)
                p = p * w[:, None]
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
    q,  # [total_len, num_heads, head_dim]
    k,  # [total_len, num_k_heads, head_dim]
    v,  # [total_len, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    topk_w,  # [num_k_heads, total_len, topk] or None
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
        Sequence packing is still done at wrapper level for multi-sequence varlen inputs.
        Fast path avoids loop overhead when there is only one sequence.
    """
    expected_seqs = int(cu_seqlens_q.numel() - 1)
    if topk_w is not None:
        for item in (permute_results or []):
            if not isinstance(item, dict) or "valid_topk_w_permuted_tile" not in item:
                permute_results = None
                break
    if _permute_results_need_rebuild(permute_results) or (
        not isinstance(permute_results, (list, tuple))
    ) or len(permute_results) != expected_seqs:
        permute_results = _build_permute_results_for_bwd(
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            topk_w=topk_w,
        )

    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) != len(k_ranges):
        raise RuntimeError(
            f"Mismatched sequence partitions: len(q_ranges)={len(q_ranges)} vs len(k_ranges)={len(k_ranges)}."
        )
    if len(permute_results) != len(q_ranges):
        raise RuntimeError(
            f"Mismatched permute_results: len(permute_results)={len(permute_results)} vs sequences={len(q_ranges)}."
        )

    if len(q_ranges) == 1:
        (q_start, q_end) = q_ranges[0]
        (k_start, k_end) = k_ranges[0]
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        cu_q_local = torch.tensor([0, q_len], dtype=torch.int32, device=cu_seqlens_q.device)
        cu_k_local = torch.tensor([0, k_len], dtype=torch.int32, device=cu_seqlens_k.device)
        backward_dq_opt_per_seq(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            topk_w[:, q_start:q_end] if topk_w is not None else None,
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

    for seq_idx, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        cu_q_local = torch.tensor([0, q_len], dtype=torch.int32, device=cu_seqlens_q.device)
        cu_k_local = torch.tensor([0, k_len], dtype=torch.int32, device=cu_seqlens_k.device)

        backward_dq_opt_per_seq(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            topk_idx[:, q_start:q_end],
            topk_w[:, q_start:q_end] if topk_w is not None else None,
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
            permute_results[seq_idx],
            dq_block_size_q=dq_block_size_q,
            dq_num_q_blocks=dq_num_q_blocks,
            disable_causal_mask=disable_causal_mask,
        )

    return dq


def backward_dq_opt_per_seq(
    q,  # [total_len, num_heads, head_dim]
    k,  # [total_len_k, num_k_heads, head_dim]
    v,  # [total_len_k, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    topk_w,  # [num_k_heads, total_len, topk] or None
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
    use_atomic_dq = dq_mode in ("", "auto", "atomic", "1", "true", "yes", "on")
    force_atomic_dq = os.getenv("FSA_LOCAL_DQ_FORCE_ATOMIC", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    # Full dQ de-serialization mode: force all-query-head atomic path.
    dq_full_deser = os.getenv("FSA_LOCAL_DQ_FULL_DESERIALIZE", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if dq_full_deser:
        use_atomic_dq = True
    if force_atomic_dq:
        use_atomic_dq = True

    num_blocks = int(permute_results["num_blocks"])
    reduce_tile_size = max(0, num_blocks - 1)
    valid_lens_all = permute_results["valid_lens_all"]
    max_tokens_any_block = int(valid_lens_all.max().item()) if valid_lens_all.numel() > 0 else 0
    global_max_valid_tokens = (
        int(permute_results["global_max_valid_tokens"])
        if num_blocks > 1
        else max_tokens_any_block
    )

    # No routed queries at all -> dQ is zero.
    if max_tokens_any_block <= 0:
        return dq

    if use_atomic_dq:
        dq_accum = torch.zeros_like(dq, dtype=torch.float32)
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
        dq_buffer_first = torch.empty((head_tile, 1, max_tokens_any_block, head_dim), dtype=dq_buf_dtype, device=dq.device)
        if reduce_tile_size > 0 and global_max_valid_tokens > 0:
            dq_buffer_rest = torch.empty(
                (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=dq_buf_dtype, device=dq.device
            )
        else:
            # Minimal dummy tensor (won't be read if num_blocks==1)
            dq_buffer_rest = torch.empty((head_tile, 1, 1, head_dim), dtype=dq_buf_dtype, device=dq.device)

        # Dense mapping: token -> position inside each block's compacted list.
        # Use empty() since index_mapping overwrites all entries that will ever be read.
        token_index_mapping = torch.empty((1, num_blocks, total_len), dtype=torch.int32, device=q.device)

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
        selected_weights_all = permute_results.get("valid_topk_w_concat", None)
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
            valid_lens_qh = valid_lens_stack.index_select(0, qh_to_kh).contiguous()  # [HQ, num_blocks]
            valid_start_qh = (
                valid_start_stack.index_select(0, qh_to_kh)
                + kh_offsets.index_select(0, qh_to_kh).view(-1, 1)
            ).contiguous()
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

        for h_start in range(0, num_q_heads, head_tile_qh):
            h_end = min(num_q_heads, h_start + head_tile_qh)
            ht = h_end - h_start
            kh_idx = qh_to_kh[h_start:h_end]

            q_tile = q[:, h_start:h_end]
            do_tile = do[:, h_start:h_end]
            lse_tile = lse[h_start:h_end]
            delta_tile = delta[h_start:h_end]
            dq_accum_tile = dq_accum[:, h_start:h_end]

            packed_tile = bool(use_packed_gqa) and bool(torch.all(kh_idx == kh_idx[0]))
            if packed_tile:
                kh0 = int(kh_idx[0].item())
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
                valid_lens_tile = base_lens.view(1, -1).expand(ht, -1).contiguous()
                valid_start_indices_tile = base_starts.view(1, -1).expand(ht, -1).contiguous()
            else:
                k_tile = k.index_select(1, kh_idx).contiguous()
                v_tile = v.index_select(1, kh_idx).contiguous()
                valid_lens_tile = valid_lens_qh[h_start:h_end].contiguous()
                valid_start_indices_tile = valid_start_qh[h_start:h_end].contiguous()

            # Single-pass over all blocks per head-tile (compute_min_block_id=0, tile_size=num_blocks).
            cur_max_valid_tokens = int(valid_lens_tile.max().item()) if valid_lens_tile.numel() > 0 else 0
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
                lse_tile,
                delta_tile,
                do_tile,
                dq_accum_tile,
                selected_tokens_all,
                selected_weights_all if selected_weights_all is not None else selected_tokens_all,
                valid_lens_tile,
                valid_start_indices_tile,
                cur_max_valid_tokens,
                0,
                ht,
                num_blocks,
                head_dim,
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
                HAS_WEIGHTS=(selected_weights_all is not None),
                num_warps=num_warps,
                num_stages=num_stages,
            )

        dq.copy_(dq_accum.to(dq.dtype))
        return dq

    for kh in range(num_k_heads):
        valid_topk_idx_permuted_tile = permute_results["valid_topk_idx_permuted_tile"][kh]
        valid_topk_w_permuted_tile = None
        if "valid_topk_w_permuted_tile" in permute_results:
            valid_topk_w_permuted_tile = permute_results["valid_topk_w_permuted_tile"][kh]
        valid_lens = permute_results["valid_lens"][kh]
        valid_start_indices = permute_results["valid_start_indices"][kh]
        # Build token->rank mapping for this KV head (shared across G query heads).
        if not use_atomic_dq:
            index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)

        topk_idx_tile_base = topk_idx[kh:kh + 1]

        if not use_atomic_dq:
            # Active token range for this KV head.
            # - Prefix-mode fast path: [memory-prefix | query] with no routes on the prefix.
            prefix_mode = bool(prefix_mask_per_kh[kh].item())
            if prefix_mode:
                query_start_idx = total_len_k
                query_tokens_count = total_len - total_len_k
            else:
                query_start_idx = int(active_starts[kh].item())
                query_tokens_count = int(active_counts[kh].item())

            if query_tokens_count <= 0:
                continue

        # Iterate over the G query heads that share this KV head in head tiles.
        for sh0 in range(0, num_share_q_heads, head_tile):
            ht = min(head_tile, num_share_q_heads - sh0)
            h_start = kh * num_share_q_heads + sh0
            h_end = h_start + ht

            q_tile = q[:, h_start:h_end]
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

            valid_lens_tile = valid_lens.view(1, -1).expand(ht, -1).contiguous()
            valid_start_indices_tile = valid_start_indices.view(1, -1).expand(ht, -1).contiguous()

            # Two-stage: (block 0) + (blocks 1..num_blocks-1) to keep the "t==0" and "t>0" buffers compact.
            for compute_min_block_id in range(min(2, num_blocks)):
                if compute_min_block_id == 0:
                    compute_tile_size = 1
                    cur_max_valid_tokens = int(valid_lens[0].item()) if valid_lens.numel() > 0 else 0
                    cur_valid_lens = valid_lens_tile[:, 0].contiguous()
                    cur_valid_start_indices = valid_start_indices_tile[:, 0].contiguous()
                    if not use_atomic_dq:
                        dq_buffer = dq_buffer_first[:ht]
                    else:
                        dq_buffer = None
                else:
                    compute_tile_size = num_blocks - 1
                    # If there are no blocks > 0, skip.
                    if compute_tile_size <= 0:
                        continue
                    cur_valid_lens = valid_lens_tile[:, compute_min_block_id:].contiguous()
                    if cur_valid_lens.numel() == 0:
                        continue
                    cur_max_valid_tokens = int(cur_valid_lens.max().item())
                    if cur_max_valid_tokens <= 0:
                        continue
                    cur_valid_start_indices = valid_start_indices_tile[:, compute_min_block_id:].contiguous()
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
                        lse_tile,
                        delta_tile,
                        do_tile,
                        dq_accum_tile,
                        valid_topk_idx_permuted_tile,
                        valid_topk_w_permuted_tile if valid_topk_w_permuted_tile is not None else valid_topk_idx_permuted_tile,
                        cur_valid_lens,
                        cur_valid_start_indices,
                        cur_max_valid_tokens,
                        compute_min_block_id,
                        ht,
                        compute_tile_size,
                        head_dim,
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
                        HAS_WEIGHTS=(valid_topk_w_permuted_tile is not None),
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                else:
                    dq_compute_kernel[grid_dq](
                        q_tile,
                        k_tile,
                        v_tile,
                        lse_tile,
                        delta_tile,
                        do_tile,
                        dq_buffer,
                        token_index_mapping_tile,
                        valid_topk_idx_permuted_tile,
                        valid_topk_w_permuted_tile if valid_topk_w_permuted_tile is not None else valid_topk_idx_permuted_tile,
                        cur_valid_lens,
                        cur_valid_start_indices,
                        cur_max_valid_tokens,
                        compute_min_block_id,
                        ht,
                        compute_tile_size,
                        head_dim,
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
                        HAS_WEIGHTS=(valid_topk_w_permuted_tile is not None),
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
    tq_slot_ptr,  # topk_q_slot: kh x N
    tq_w_ptr,  # topk_q_w: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DK: sh x n x kh x d
    dw_ptr,  # DW: kh x n x TOPK (float32) or dummy when HAS_WEIGHTS=0
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
    stride_tqsh,
    stride_tqsn,
    stride_tqwh,
    stride_tqwn,
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
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_dwh,
    stride_dwn,
    stride_dwk,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    PIPELINE_CHUNKS: tl.constexpr,
    DISABLE_CAUSAL_MASK: tl.constexpr,
    USE_ACTIVE_BLOCK_MAP: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
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
    if HAS_WEIGHTS:
        tq_slot_ptr = tq_slot_ptr + pid_kh * stride_tqsh + act_q_start * stride_tqsn
        tq_w_ptr = tq_w_ptr + pid_kh * stride_tqwh + act_q_start * stride_tqwn
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
    d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
    lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh
    # loop for q blocks
    step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
    for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
        for u in tl.static_range(0, PIPELINE_CHUNKS):
            i = ib + u * BLOCK_SIZE_Q
            if i < act_q_len:
                # load
                q_mask = (off_q < act_q_len - i)
                idx_q = tl.load(tq_ptr + i + off_q, mask=q_mask, other=0).to(tl.int32)
                if HAS_WEIGHTS:
                    slot = tl.load(tq_slot_ptr + i + off_q, mask=q_mask, other=-1).to(tl.int32)
                    w = tl.load(tq_w_ptr + i + off_q, mask=q_mask, other=0.0).to(tl.float32)
                    w = tl.maximum(w, 0.0)
                q = tl.load(
                    q_ptrs + idx_q[:, None] * stride_qn,
                    mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                    other=0,
                )
                do = tl.load(
                    do_ptrs + idx_q[:, None] * stride_don,
                    mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                    other=0,
                )
                lse = tl.load(
                    lse_ptrs + idx_q[:, None] * stride_ln,
                    mask=q_mask[:, None],
                    other=0,
                )
                d = tl.load(
                    d_ptrs + idx_q[:, None] * stride_dn,
                    mask=q_mask[:, None],
                    other=0,
                )
                # compute qk
                qk = tl.dot(q, k.T) * qk_scale
                if not DISABLE_CAUSAL_MASK:
                    qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                # compute p, ds
                p = tl.exp2(qk - lse)
                if HAS_WEIGHTS:
                    p = p * w[:, None]
                dp = tl.dot(do, v.T)
                ds_fp32 = sm_scale * p * (dp - d)
                # cast dtype
                p = p.to(do.dtype)
                ds = ds_fp32.to(q.dtype)
                # update dk and dv
                dk += tl.dot(ds.T, q)
                dv += tl.dot(p.T, do)
                if HAS_WEIGHTS:
                    dlogw = tl.sum(ds_fp32, axis=1) / sm_scale
                    dw = tl.where(w > 0, dlogw / w, 0.0).to(tl.float32)
                    q_abs = q_start + idx_q
                    m_dw = q_mask & (slot >= 0) & (slot < TOPK) & (w > 0)
                    dw_ptrs = dw_ptr + pid_kh * stride_dwh + q_abs * stride_dwn + slot * stride_dwk
                    tl.atomic_add(dw_ptrs, dw, mask=m_dw)
    # save dk dv
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dkdv_gqa_fused(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    tq_slot_ptr,  # topk_q_slot: kh x N
    tq_w_ptr,  # topk_q_w: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    dw_ptr,  # DW: kh x n x TOPK (float32) or dummy when HAS_WEIGHTS=0
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
    stride_tqsh,
    stride_tqsn,
    stride_tqwh,
    stride_tqwn,
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
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_dwh,
    stride_dwn,
    stride_dwk,
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
    HAS_WEIGHTS: tl.constexpr,
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
    if HAS_WEIGHTS:
        tq_slot_ptr = tq_slot_ptr + pid_kh * stride_tqsh + act_q_start * stride_tqsn
        tq_w_ptr = tq_w_ptr + pid_kh * stride_tqwh + act_q_start * stride_tqwn

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
        d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
        lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

        step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
        for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
            for u in tl.static_range(0, PIPELINE_CHUNKS):
                i = ib + u * BLOCK_SIZE_Q
                if i < act_q_len:
                    q_mask = (off_q < act_q_len - i)
                    idx_q = tl.load(tq_ptr + i + off_q, mask=q_mask, other=0).to(tl.int32)
                    if HAS_WEIGHTS:
                        slot = tl.load(tq_slot_ptr + i + off_q, mask=q_mask, other=-1).to(tl.int32)
                        w = tl.load(tq_w_ptr + i + off_q, mask=q_mask, other=0.0).to(tl.float32)
                        w = tl.maximum(w, 0.0)
                    q = tl.load(
                        q_ptrs + idx_q[:, None] * stride_qn,
                        mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    do = tl.load(
                        do_ptrs + idx_q[:, None] * stride_don,
                        mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    lse = tl.load(
                        lse_ptrs + idx_q[:, None] * stride_ln,
                        mask=q_mask[:, None],
                        other=0,
                    )
                    d = tl.load(
                        d_ptrs + idx_q[:, None] * stride_dn,
                        mask=q_mask[:, None],
                        other=0,
                    )
                    qk = tl.dot(q, k_blk.T) * qk_scale
                    if not DISABLE_CAUSAL_MASK:
                        qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                    p = tl.exp2(qk - lse)
                    if HAS_WEIGHTS:
                        p = p * w[:, None]
                    if COMPUTE_DV:
                        p_cast = p.to(do.dtype)
                        dv += tl.dot(p_cast.T, do)
                    if COMPUTE_DK:
                        dp = tl.dot(do, v_blk.T)
                        ds_fp32 = sm_scale * p * (dp - d)
                        ds = ds_fp32.to(q.dtype)
                        dk += tl.dot(ds.T, q)
                        if HAS_WEIGHTS:
                            dlogw = tl.sum(ds_fp32, axis=1) / sm_scale
                            dw = tl.where(w > 0, dlogw / w, 0.0).to(tl.float32)
                            q_abs = q_start + idx_q
                            m_dw = q_mask & (slot >= 0) & (slot < TOPK) & (w > 0)
                            dw_ptrs = dw_ptr + pid_kh * stride_dwh + q_abs * stride_dwn + slot * stride_dwk
                            tl.atomic_add(dw_ptrs, dw, mask=m_dw)

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
    tq_slot_ptr,  # topk_q_slot: kh x N
    tq_w_ptr,  # topk_q_w: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    dw_ptr,  # DW: kh x n x TOPK (float32) or dummy when HAS_WEIGHTS=0
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
    stride_tqsh,
    stride_tqsn,
    stride_tqwh,
    stride_tqwn,
    stride_ctqh,
    stride_ctqn,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_dwh,
    stride_dwn,
    stride_dwk,
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
    HAS_WEIGHTS: tl.constexpr,
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
    if HAS_WEIGHTS:
        tq_slot_ptr = tq_slot_ptr + pid_kh * stride_tqsh + act_q_start * stride_tqsn
        tq_w_ptr = tq_w_ptr + pid_kh * stride_tqwh + act_q_start * stride_tqwn

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
        d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
        lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

        step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
        for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
            for u in tl.static_range(0, PIPELINE_CHUNKS):
                i = ib + u * BLOCK_SIZE_Q
                if i < act_q_len:
                    q_mask = (off_q < act_q_len - i)
                    idx_q = tl.load(tq_ptr + i + off_q, mask=q_mask, other=0).to(tl.int32)
                    if HAS_WEIGHTS:
                        slot = tl.load(tq_slot_ptr + i + off_q, mask=q_mask, other=-1).to(tl.int32)
                        w = tl.load(tq_w_ptr + i + off_q, mask=q_mask, other=0.0).to(tl.float32)
                        w = tl.maximum(w, 0.0)
                    q = tl.load(
                        q_ptrs + idx_q[:, None] * stride_qn,
                        mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    do = tl.load(
                        do_ptrs + idx_q[:, None] * stride_don,
                        mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                        other=0,
                    )
                    lse = tl.load(
                        lse_ptrs + idx_q[:, None] * stride_ln,
                        mask=q_mask[:, None],
                        other=0,
                    )
                    d = tl.load(
                        d_ptrs + idx_q[:, None] * stride_dn,
                        mask=q_mask[:, None],
                        other=0,
                    )
                    qk = tl.dot(q, k_blk.T) * qk_scale
                    if not DISABLE_CAUSAL_MASK:
                        qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                    p = tl.exp2(qk - lse)
                    if HAS_WEIGHTS:
                        p = p * w[:, None]
                    if COMPUTE_DV:
                        p_cast = p.to(do.dtype)
                        dv += tl.dot(p_cast.T, do)
                    if COMPUTE_DK:
                        dp = tl.dot(do, v_blk.T)
                        ds_fp32 = sm_scale * p * (dp - d)
                        ds = ds_fp32.to(q.dtype)
                        dk += tl.dot(ds.T, q)
                        if HAS_WEIGHTS:
                            dlogw = tl.sum(ds_fp32, axis=1) / sm_scale
                            dw = tl.where(w > 0, dlogw / w, 0.0).to(tl.float32)
                            q_abs = q_start + idx_q
                            m_dw = q_mask & (slot >= 0) & (slot < TOPK) & (w > 0)
                            dw_ptrs = dw_ptr + pid_kh * stride_dwh + q_abs * stride_dwn + slot * stride_dwk
                            tl.atomic_add(dw_ptrs, dw, mask=m_dw)

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
    tq_slot_ptr,  # topk_q_slot: kh x N
    tq_w_ptr,  # topk_q_w: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: n x kh x d
    dv_ptr,  # DV: n x kh x d
    dw_ptr,  # DW: kh x n x TOPK (float32) or dummy when HAS_WEIGHTS=0
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
    stride_tqsh,
    stride_tqsn,
    stride_tqwh,
    stride_tqwn,
    stride_ctqh,
    stride_ctqn,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_dwh,
    stride_dwn,
    stride_dwk,
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
    HAS_WEIGHTS: tl.constexpr,
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
                        if HAS_WEIGHTS:
                            tq_slot_ptr_w = tq_slot_ptr + pid_kh * stride_tqsh + act_q_start * stride_tqsn
                            tq_w_ptr_w = tq_w_ptr + pid_kh * stride_tqwh + act_q_start * stride_tqwn

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
                            d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
                            lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh

                            step_q = BLOCK_SIZE_Q * PIPELINE_CHUNKS
                            for ib in tl.range(0, act_q_len, step_q, num_stages=LOOP_STAGES):
                                for u in tl.static_range(0, PIPELINE_CHUNKS):
                                    i = ib + u * BLOCK_SIZE_Q
                                    if i < act_q_len:
                                        q_mask = (off_q < act_q_len - i)
                                        idx_q = tl.load(tq_ptr_w + i + off_q, mask=q_mask, other=0).to(tl.int32)
                                        if HAS_WEIGHTS:
                                            slot = tl.load(tq_slot_ptr_w + i + off_q, mask=q_mask, other=-1).to(tl.int32)
                                            w = tl.load(tq_w_ptr_w + i + off_q, mask=q_mask, other=0.0).to(tl.float32)
                                            w = tl.maximum(w, 0.0)
                                        qv = tl.load(
                                            q_ptrs + idx_q[:, None] * stride_qn,
                                            mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                                            other=0,
                                        )
                                        do_v = tl.load(
                                            do_ptrs + idx_q[:, None] * stride_don,
                                            mask=q_mask[:, None] & (off_d < HEAD_DIM)[None, :],
                                            other=0,
                                        )
                                        lse_v = tl.load(
                                            lse_ptrs + idx_q[:, None] * stride_ln,
                                            mask=q_mask[:, None],
                                            other=0,
                                        )
                                        d_v = tl.load(
                                            d_ptrs + idx_q[:, None] * stride_dn,
                                            mask=q_mask[:, None],
                                            other=0,
                                        )
                                        qk = tl.dot(qv, k_blk.T) * qk_scale
                                        if not DISABLE_CAUSAL_MASK:
                                            qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
                                        p = tl.exp2(qk - lse_v)
                                        if HAS_WEIGHTS:
                                            p = p * w[:, None]
                                        if COMPUTE_DV:
                                            p_cast = p.to(do_v.dtype)
                                            dv += tl.dot(p_cast.T, do_v)
                                        if COMPUTE_DK:
                                            dp = tl.dot(do_v, v_blk.T)
                                            ds_fp32 = sm_scale * p * (dp - d_v)
                                            ds = ds_fp32.to(qv.dtype)
                                            dk += tl.dot(ds.T, qv)
                                            if HAS_WEIGHTS:
                                                dlogw = tl.sum(ds_fp32, axis=1) / sm_scale
                                                dw = tl.where(w > 0, dlogw / w, 0.0).to(tl.float32)
                                                q_abs = q_start + idx_q
                                                m_dw = q_mask & (slot >= 0) & (slot < TOPK) & (w > 0)
                                                dw_ptrs = dw_ptr + pid_kh * stride_dwh + q_abs * stride_dwn + slot * stride_dwk
                                                tl.atomic_add(dw_ptrs, dw, mask=m_dw)

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
    topk_w: Optional[torch.Tensor],
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    permute_results,
    disable_causal_mask: bool = False,
):

    assert block_size in {32, 64, 128, 256, 512, 1024}
    if isinstance(permute_results, dict):
        permute_results = [permute_results]
    expected_seqs = int(cu_seqlens_q.numel() - 1)
    if topk_w is not None:
        for item in (permute_results or []):
            if not isinstance(item, dict) or "valid_topk_w_permuted_tile" not in item:
                permute_results = None
                break
    if _permute_results_need_rebuild(permute_results) or (not isinstance(permute_results, (list, tuple))) or len(permute_results) != expected_seqs:
        permute_results = _build_permute_results_for_bwd(
            topk_idx=topk_idx,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            topk_w=topk_w,
        )

    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    topk = topk_idx.shape[-1]
    # compute D
    delta = torch.zeros([num_o_heads, o_len], device=o.device, dtype=torch.float32)
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
            topk_w=topk_w,
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
    total_active_q = int(cu_topk_q_count[:, -1].to(dtype=torch.int64).sum().item())
    if total_active_q == 0:
        # No routed queries for any KV block -> all attention grads are zero.
        if topk_w is None:
            return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), None
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(topk_w, dtype=torch.float32)

    # active query idx for each key block
    # how to get active query idx for sequence b, head h, kv block i?
    has_weights = topk_w is not None
    if has_weights:
        if topk_w.dtype != torch.float32:
            topk_w = topk_w.to(dtype=torch.float32)
        topk_q_idx, topk_q_slot, topk_q_w = reorder_topk_idx(
            topk_idx_for_reorder,
            cu_topk_q_count,
            cu_seqlens_q,
            cu_seqblocks,
            block_size,
            topk_w=topk_w,
            return_slot=True,
            return_weights=True,
        )
        topk_q_idx, topk_q_slot, topk_q_w = _maybe_sort_reordered_topk_q_meta(
            topk_q_idx, cu_topk_q_count, topk_q_slot, topk_q_w
        )
        dw = torch.zeros_like(topk_w, dtype=torch.float32, device=topk_w.device)
    else:
        topk_q_idx = reorder_topk_idx(topk_idx_for_reorder, cu_topk_q_count, cu_seqlens_q, cu_seqblocks, block_size)
        topk_q_idx = _maybe_sort_reordered_topk_q_idx(topk_q_idx, cu_topk_q_count)
        topk_q_slot = topk_q_idx  # dummy - never read when HAS_WEIGHTS=0
        topk_q_w = delta[:1, :1]  # dummy - never read when HAS_WEIGHTS=0
        dw = None
    # compute dk dv
    batch_size = cu_seqlens_q.shape[0] - 1
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

    active_mode = os.getenv("FSA_LOCAL_COMPACT_ACTIVE_BLOCKS", "auto").strip().lower()
    active_idx = torch.full((num_k_heads, batch_size, 1), -1, dtype=torch.int32, device=k.device)
    active_count = torch.zeros((num_k_heads, batch_size), dtype=torch.int32, device=k.device)
    use_active_map = False
    max_active_blocks = 0
    active_ratio = 1.0
    active_work_items_est = 0
    active_ratio_threshold = _resolve_active_map_ratio_threshold(head_dim=head_dim, block_size=block_size)
    if active_mode not in ("0", "false", "no", "off"):
        # Stability guard: in extremely large routed workloads, the active-map path can
        # become fragile on some driver/runtime combinations. Keep force-on behavior intact.
        max_auto_active_q_raw = os.getenv("FSA_LOCAL_MAX_AUTO_ACTIVE_Q_FOR_MAP", "8000000").strip().lower()
        try:
            max_auto_active_q_for_map = int(max_auto_active_q_raw)
        except Exception:
            max_auto_active_q_for_map = 8000000
        auto_map_allowed = not (
            active_mode in ("", "auto") and total_active_q > max(1, max_auto_active_q_for_map)
        )

        if auto_map_allowed or active_mode in ("1", "true", "yes", "on"):
            active_idx_built, active_count_built, max_active_blocks, active_ratio = _build_active_kv_block_map(
                cu_topk_q_count=cu_topk_q_count,
                cu_seqblocks=cu_seqblocks,
            )
            if active_mode in ("1", "true", "yes", "on"):
                use_active_map = True
            else:
                # auto: enable compaction when a non-trivial fraction of blocks are empty.
                use_active_map = max_active_blocks > 0 and active_ratio < active_ratio_threshold
            if use_active_map:
                active_idx = active_idx_built
                active_count = active_count_built
                active_work_items_est = int(active_count.to(dtype=torch.int64).sum().item())

    dkdv_mode = _resolve_dkdv_mode(num_share_q_heads, head_dim=head_dim, block_size=block_size)
    use_gqa_fused_dkdv = dkdv_mode == "gqa_fused"
    dkdv_two_pass = _resolve_dkdv_two_pass(
        num_share_q_heads=num_share_q_heads, head_dim=head_dim, block_size=block_size
    ) and use_gqa_fused_dkdv
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
        dk = torch.zeros(k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
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
                    queue = torch.zeros((1,), dtype=torch.int32, device=q.device)
                    if dkdv_two_pass:
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=False, COMPUTE_DV=True,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        queue.zero_()
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=False,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                    else:
                        backward_dkdv_gqa_fused_persistent_queue[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, queue, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            WORK_STEAL_CHUNK=work_steal_chunk,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=True,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                else:
                    grid_wl = (num_work_items,)
                    if dkdv_two_pass:
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=False, COMPUTE_DV=True,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=False,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                    else:
                        backward_dkdv_gqa_fused_worklist[grid_wl](
                            q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                            cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count,
                            worklist, num_work_items,
                            num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            topk_q_idx.stride(0), topk_q_idx.stride(1),
                            topk_q_slot.stride(0), topk_q_slot.stride(1),
                            topk_q_w.stride(0), topk_q_w.stride(1),
                            cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                            lse.stride(0), lse.stride(1),
                            delta.stride(0), delta.stride(1),
                            do.stride(0), do.stride(1), do.stride(2),
                            dk.stride(0), dk.stride(1), dk.stride(2),
                            dv.stride(0), dv.stride(1), dv.stride(2),
                            (dw.stride(0) if dw is not None else 0),
                            (dw.stride(1) if dw is not None else 0),
                            (dw.stride(2) if dw is not None else 0),
                            worklist.stride(0), worklist.stride(1),
                            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                            PIPELINE_CHUNKS=pipeline_chunks,
                            DISABLE_CAUSAL_MASK=disable_causal_mask, COMPUTE_DK=True, COMPUTE_DV=True,
                            HAS_WEIGHTS=has_weights,
                            num_warps=num_warps, num_stages=num_stages,
                        )
        else:
            if dkdv_two_pass:
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    topk_q_slot.stride(0), topk_q_slot.stride(1),
                    topk_q_w.stride(0), topk_q_w.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    (dw.stride(0) if dw is not None else 0),
                    (dw.stride(1) if dw is not None else 0),
                    (dw.stride(2) if dw is not None else 0),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=False, COMPUTE_DV=True,
                    HAS_WEIGHTS=has_weights,
                    num_warps=num_warps, num_stages=num_stages,
                )
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    topk_q_slot.stride(0), topk_q_slot.stride(1),
                    topk_q_w.stride(0), topk_q_w.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    (dw.stride(0) if dw is not None else 0),
                    (dw.stride(1) if dw is not None else 0),
                    (dw.stride(2) if dw is not None else 0),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=True, COMPUTE_DV=False,
                    HAS_WEIGHTS=has_weights,
                    num_warps=num_warps, num_stages=num_stages,
                )
            else:
                backward_dkdv_gqa_fused[grid](
                    q, k, v, topk_q_idx, topk_q_slot, topk_q_w, lse, delta, do, dk, dv, dw if dw is not None else delta[:1, :1],
                    cu_seqlens_q, cu_seqlens_k, cu_seqblocks, cu_topk_q_count, active_idx, active_count,
                    num_k_heads, num_share_q_heads, head_dim, topk, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    topk_q_idx.stride(0), topk_q_idx.stride(1),
                    topk_q_slot.stride(0), topk_q_slot.stride(1),
                    topk_q_w.stride(0), topk_q_w.stride(1),
                    cu_topk_q_count.stride(0), cu_topk_q_count.stride(1),
                    active_idx.stride(0), active_idx.stride(1), active_idx.stride(2),
                    active_count.stride(0), active_count.stride(1),
                    lse.stride(0), lse.stride(1), delta.stride(0), delta.stride(1),
                    do.stride(0), do.stride(1), do.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    dv.stride(0), dv.stride(1), dv.stride(2),
                    (dw.stride(0) if dw is not None else 0),
                    (dw.stride(1) if dw is not None else 0),
                    (dw.stride(2) if dw is not None else 0),
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, LOOP_STAGES=loop_stages,
                    PIPELINE_CHUNKS=pipeline_chunks,
                    DISABLE_CAUSAL_MASK=disable_causal_mask, USE_ACTIVE_BLOCK_MAP=use_active_map,
                    COMPUTE_DK=True, COMPUTE_DV=True,
                    HAS_WEIGHTS=has_weights,
                    num_warps=num_warps, num_stages=num_stages,
                )
    else:
        backward_dkdv[grid](
            q,
            k,
            v,
            topk_q_idx,
            topk_q_slot,
            topk_q_w,
            lse,
            delta,
            do,
            dk,
            dv,
            dw if dw is not None else delta[:1, :1],
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
            topk_q_slot.stride(0),
            topk_q_slot.stride(1),
            topk_q_w.stride(0),
            topk_q_w.stride(1),
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
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            (dw.stride(0) if dw is not None else 0),
            (dw.stride(1) if dw is not None else 0),
            (dw.stride(2) if dw is not None else 0),
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            LOOP_STAGES=loop_stages,
            PIPELINE_CHUNKS=pipeline_chunks,
            DISABLE_CAUSAL_MASK=disable_causal_mask,
            USE_ACTIVE_BLOCK_MAP=use_active_map,
            HAS_WEIGHTS=has_weights,
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
    dq = torch.zeros_like(q)
    num_q_loop = max_seqlen_q // 32768 + 1  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, triton.cdiv(max_seqlen_q, num_q_loop))

    backward_dq_opt(
        q,
        k,
        v,
        topk_idx,
        topk_w,
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

    return dq, dk, dv, dw


def _topk_sparse_attention_bwd_opt_seq_parallel(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
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
    q_ranges = _cu_seqlens_to_ranges(cu_seqlens_q)
    k_ranges = _cu_seqlens_to_ranges(cu_seqlens_k)
    if len(q_ranges) != len(k_ranges):
        raise RuntimeError(
            f"Mismatched sequence partitions: len(q_ranges)={len(q_ranges)} vs len(k_ranges)={len(k_ranges)}."
        )
    nseq = len(q_ranges)
    if nseq <= 1:
        return _topk_sparse_attention_bwd_opt_core(
            o=o,
            do=do,
            lse=lse,
            q=q,
            k=k,
            v=v,
            topk_idx=topk_idx,
            topk_w=topk_w,
            block_size=block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            permute_results=permute_results,
            disable_causal_mask=disable_causal_mask,
        )

    if isinstance(permute_results, dict):
        permute_list = [permute_results]
    elif isinstance(permute_results, (list, tuple)):
        permute_list = list(permute_results)
    else:
        permute_list = []

    dq_out = torch.zeros_like(q)
    dk_out = torch.zeros_like(k)
    dv_out = torch.zeros_like(v)
    dw_out = torch.zeros_like(topk_w, dtype=torch.float32) if topk_w is not None else None

    use_cuda_streams = bool(q.is_cuda and torch.cuda.is_available())
    num_streams = _resolve_bwd_sequence_parallel_streams(
        q_device=q.device,
        expected_seqs=nseq,
        head_dim=int(q.shape[-1]),
        block_size=block_size,
    ) if use_cuda_streams else 1
    num_streams = max(1, min(num_streams, nseq))
    streams = [torch.cuda.Stream(device=q.device) for _ in range(num_streams)] if use_cuda_streams and num_streams > 1 else [None]

    for seq_idx, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        q_len = int(q_end - q_start)
        k_len = int(k_end - k_start)
        if q_len <= 0 or k_len <= 0:
            continue
        cu_q_local = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
        cu_k_local = torch.tensor([0, k_len], dtype=torch.int32, device=q.device)
        perm_i = permute_list[seq_idx] if seq_idx < len(permute_list) else None

        stream = streams[seq_idx % len(streams)]
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            dq_i, dk_i, dv_i, dw_i = _topk_sparse_attention_bwd_opt_core(
                o=o[q_start:q_end].contiguous(),
                do=do[q_start:q_end].contiguous(),
                lse=lse[:, q_start:q_end].contiguous(),
                q=q[q_start:q_end].contiguous(),
                k=k[k_start:k_end].contiguous(),
                v=v[k_start:k_end].contiguous(),
                topk_idx=topk_idx[:, q_start:q_end].contiguous(),
                topk_w=topk_w[:, q_start:q_end].contiguous() if topk_w is not None else None,
                block_size=block_size,
                cu_seqlens_q=cu_q_local,
                cu_seqlens_k=cu_k_local,
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                sm_scale=sm_scale,
                permute_results=perm_i,
                disable_causal_mask=disable_causal_mask,
            )
            dq_out[q_start:q_end].copy_(dq_i)
            dk_out[k_start:k_end].copy_(dk_i)
            dv_out[k_start:k_end].copy_(dv_i)
            if dw_out is not None and dw_i is not None:
                dw_out[:, q_start:q_end].copy_(dw_i)

    if use_cuda_streams and num_streams > 1:
        for s in streams:
            s.synchronize()

    return dq_out, dk_out, dv_out, dw_out


def _topk_sparse_attention_bwd_opt(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
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
            topk_w=topk_w,
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
        topk_w=topk_w,
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
                topk_w=None,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                sm_scale=sm_scale,
            )
            # Keep forward close to pure fused attention timing. Build backward-side
            # permutation metadata lazily in backward when gradients are requested.
            permute_results = None
        else:
            o, lse, permute_results = _topk_sparse_attention_fwd_opt(
                q,
                k,
                v,
                topk_idx,
                None,
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
        if _permute_results_need_rebuild(permute_results):
            permute_results = _build_permute_results_for_bwd(
                topk_idx=topk_idx,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                topk_w=None,
            )

        dq, dk, dv, _dw = _topk_sparse_attention_bwd_opt(
                o,
                do,
                lse,
                q,
                k,
                v,
                topk_idx,
                None,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
                permute_results,
                disable_causal_mask=disable_causal_mask,
            )
        return dq, dk, dv, None, None, None, None, None, None, None, None


class FSATopkSparseAttentionWeighted(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_w: torch.Tensor,
        block_size: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        sm_scale=None,
        disable_causal_mask: bool = False,
    ):
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert topk_idx.dtype == torch.int32
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        assert topk_w.shape == topk_idx.shape
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

        if use_nsa_style_fwd:
            o, lse = _topk_sparse_attention_fwd_nsa_style(
                q=q,
                k=k,
                v=v,
                topk_idx=topk_idx,
                topk_w=topk_w,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                sm_scale=sm_scale,
            )
            permute_results = None
        else:
            o, lse, permute_results = _topk_sparse_attention_fwd_opt(
                q,
                k,
                v,
                topk_idx,
                topk_w,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx, topk_w)
        ctx.permute_results = permute_results
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.block_size = block_size
        ctx.disable_causal_mask = disable_causal_mask
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx, topk_w = ctx.saved_tensors
        permute_results = ctx.permute_results

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        block_size = ctx.block_size
        disable_causal_mask = ctx.disable_causal_mask
        assert block_size in {32, 64, 128, 256, 512, 1024}
        if _permute_results_need_rebuild(permute_results):
            permute_results = _build_permute_results_for_bwd(
                topk_idx=topk_idx,
                block_size=block_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                topk_w=topk_w,
            )

        dq, dk, dv, dw = _topk_sparse_attention_bwd_opt(
            o,
            do,
            lse,
            q,
            k,
            v,
            topk_idx,
            topk_w,
            block_size,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            permute_results,
            disable_causal_mask=disable_causal_mask,
        )
        return dq, dk, dv, None, dw, None, None, None, None, None, None, None


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

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.int32).max().item())
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
    return expanded.reshape(topk_idx.shape[0], topk_idx.shape[1], topk_idx.shape[2] * ratio).contiguous()


def _expand_topk_w_for_internal_blocks(
    topk_w: torch.Tensor,
    block_size: int,
    internal_block_size: int,
) -> torch.Tensor:
    """
    Match _expand_topk_for_internal_blocks expansion for per-topk weights.

    For each original chapter weight w, replicate across sub-block ids created by the split.
    """
    ratio = block_size // internal_block_size
    if ratio == 1:
        return topk_w
    w = topk_w.to(torch.float32)
    expanded = w.unsqueeze(-1).expand(w.shape[0], w.shape[1], w.shape[2], ratio)
    return expanded.reshape(w.shape[0], w.shape[1], w.shape[2] * ratio).contiguous()


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

    if mode not in ("1", "true", "yes", "on"):
        total_active = int(cu_topk_q_count[:, -1].to(dtype=torch.int64).sum().item()) if cu_topk_q_count.numel() > 0 else 0
        num_segments = int(cu_topk_q_count.shape[0]) * max(1, int(cu_topk_q_count.shape[1]) - 1)
        avg_seg = (total_active / max(1, num_segments))
        # Auto-enable only when routing is large enough for locality to matter.
        if total_active < 65536 or avg_seg < 2.0:
            return topk_q_idx

    out = topk_q_idx.clone()
    num_kh, max_active_slots = int(out.shape[0]), int(out.shape[1])
    if num_kh <= 0 or max_active_slots <= 1:
        return out

    n_active = cu_topk_q_count[:, -1].to(dtype=torch.int64)
    n_active = torch.clamp(n_active, min=0, max=max_active_slots)
    if int(n_active.max().item()) <= 1:
        return out

    pos = torch.arange(max_active_slots, device=out.device, dtype=torch.int64).view(1, max_active_slots)
    pos = pos.expand(num_kh, max_active_slots).contiguous()
    valid = pos < n_active.view(-1, 1)
    if not bool(torch.any(valid)):
        return out

    num_segments = max(1, int(cu_topk_q_count.shape[1]) - 1)
    head_id = torch.arange(num_kh, device=out.device, dtype=torch.int64).view(-1, 1)
    try:
        # Row-wise segment id in [0, num_segments) for each active position.
        seg = torch.searchsorted(cu_topk_q_count[:, 1:].to(dtype=torch.int64), pos, right=True)
        group_id = head_id * num_segments + seg.to(dtype=torch.int64)

        flat_vals = out[valid]
        flat_group = group_id[valid]
        if int(flat_vals.numel()) <= 1:
            return out

        # Stable sort by value inside each (head, segment):
        # two-pass stable sort over flattened active entries.
        ord_val = torch.argsort(flat_vals, stable=True)
        vals_by_val = flat_vals.index_select(0, ord_val)
        grp_by_val = flat_group.index_select(0, ord_val)
        ord_group = torch.argsort(grp_by_val, stable=True)
        out[valid] = vals_by_val.index_select(0, ord_group)
        return out
    except Exception:
        # Compatibility fallback for environments without N-D searchsorted support.
        for kh in range(num_kh):
            offsets = cu_topk_q_count[kh].to(dtype=torch.int64)
            n_kh = int(min(max_active_slots, int(offsets[-1].item())))
            if n_kh <= 1:
                continue
            vals = out[kh, :n_kh]
            pos_kh = torch.arange(n_kh, device=out.device, dtype=torch.int64)
            seg_kh = torch.bucketize(pos_kh, offsets[1:], right=True)
            ord_val = torch.argsort(vals, stable=True)
            seg_by_val = seg_kh.index_select(0, ord_val)
            ord_seg = torch.argsort(seg_by_val, stable=True)
            out[kh, :n_kh] = vals.index_select(0, ord_val.index_select(0, ord_seg))
        return out


def _maybe_sort_reordered_topk_q_meta(
    topk_q_idx: torch.Tensor,
    cu_topk_q_count: torch.Tensor,
    topk_q_slot: Optional[torch.Tensor] = None,
    topk_q_w: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Like _maybe_sort_reordered_topk_q_idx, but applies the exact same segmented-stable
    permutation to optional metadata arrays (slot / weight).
    """
    mode = os.getenv("FSA_LOCAL_SORT_TOPK_Q_IDX", "auto").strip().lower()
    if mode in ("", "0", "false", "no", "off"):
        enabled = False
    elif mode in ("1", "true", "yes", "on"):
        enabled = True
    else:
        enabled = True
    if not enabled or topk_q_idx.ndim != 2 or cu_topk_q_count.ndim != 2:
        return topk_q_idx, topk_q_slot, topk_q_w

    if mode not in ("1", "true", "yes", "on"):
        total_active = int(cu_topk_q_count[:, -1].to(dtype=torch.int64).sum().item()) if cu_topk_q_count.numel() > 0 else 0
        num_segments = int(cu_topk_q_count.shape[0]) * max(1, int(cu_topk_q_count.shape[1]) - 1)
        avg_seg = (total_active / max(1, num_segments))
        if total_active < 65536 or avg_seg < 2.0:
            return topk_q_idx, topk_q_slot, topk_q_w

    out_idx = topk_q_idx.clone()
    out_slot = topk_q_slot.clone() if topk_q_slot is not None else None
    out_w = topk_q_w.clone() if topk_q_w is not None else None

    num_kh, max_active_slots = int(out_idx.shape[0]), int(out_idx.shape[1])
    if num_kh <= 0 or max_active_slots <= 1:
        return out_idx, out_slot, out_w

    n_active = cu_topk_q_count[:, -1].to(dtype=torch.int64)
    n_active = torch.clamp(n_active, min=0, max=max_active_slots)
    if int(n_active.max().item()) <= 1:
        return out_idx, out_slot, out_w

    pos = torch.arange(max_active_slots, device=out_idx.device, dtype=torch.int64).view(1, max_active_slots)
    pos = pos.expand(num_kh, max_active_slots).contiguous()
    valid = pos < n_active.view(-1, 1)
    if not bool(torch.any(valid)):
        return out_idx, out_slot, out_w

    num_segments = max(1, int(cu_topk_q_count.shape[1]) - 1)
    head_id = torch.arange(num_kh, device=out_idx.device, dtype=torch.int64).view(-1, 1)
    try:
        seg = torch.searchsorted(cu_topk_q_count[:, 1:].to(dtype=torch.int64), pos, right=True)
        group_id = head_id * num_segments + seg.to(dtype=torch.int64)

        flat_vals = out_idx[valid]
        flat_group = group_id[valid]
        if int(flat_vals.numel()) <= 1:
            return out_idx, out_slot, out_w

        ord_val = torch.argsort(flat_vals, stable=True)
        vals_by_val = flat_vals.index_select(0, ord_val)
        grp_by_val = flat_group.index_select(0, ord_val)
        ord_group = torch.argsort(grp_by_val, stable=True)
        perm = ord_val.index_select(0, ord_group)

        out_idx[valid] = flat_vals.index_select(0, perm)
        if out_slot is not None:
            flat_slot = out_slot[valid]
            out_slot[valid] = flat_slot.index_select(0, perm)
        if out_w is not None:
            flat_w = out_w[valid]
            out_w[valid] = flat_w.index_select(0, perm)
        return out_idx, out_slot, out_w
    except Exception:
        for kh in range(num_kh):
            offsets = cu_topk_q_count[kh].to(dtype=torch.int64)
            n_kh = int(min(max_active_slots, int(offsets[-1].item())))
            if n_kh <= 1:
                continue
            vals = out_idx[kh, :n_kh]
            pos_kh = torch.arange(n_kh, device=out_idx.device, dtype=torch.int64)
            seg_kh = torch.bucketize(pos_kh, offsets[1:], right=True)
            ord_val = torch.argsort(vals, stable=True)
            seg_by_val = seg_kh.index_select(0, ord_val)
            ord_seg = torch.argsort(seg_by_val, stable=True)
            perm = ord_val.index_select(0, ord_seg)
            out_idx[kh, :n_kh] = vals.index_select(0, perm)
            if out_slot is not None:
                out_slot[kh, :n_kh] = out_slot[kh, :n_kh].index_select(0, perm)
            if out_w is not None:
                out_w[kh, :n_kh] = out_w[kh, :n_kh].index_select(0, perm)
        return out_idx, out_slot, out_w


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
    active_mask = block_counts > 0
    nz = torch.nonzero(active_mask, as_tuple=False)  # [N, 2] -> (kh, global_block)
    total_active = int(nz.shape[0])

    total_blocks_all = hk * int(cu_seqblocks[-1].item()) if cu_seqblocks.numel() > 0 else 0
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
    max_active = int(active_block_count.max().item())
    if max_active <= 0:
        active_block_idx = torch.full((hk, batch, 1), -1, dtype=torch.int32, device=device)
        return active_block_idx, active_block_count, 0, 0.0

    # Deterministic order: sort by pair then by local block id.
    sort_key = pair.to(torch.int64) * (int(cu_seqblocks[-1].item()) + 1) + local_blk.to(torch.int64)
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
    ).contiguous()


def FSA_topk_sparse_attention_varlen_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk_w: Optional[torch.Tensor] = None,
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
    topk_w_internal = (
        _expand_topk_w_for_internal_blocks(topk_w, block_size, internal_block_size)
        if topk_w is not None
        else None
    )

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(dtype=torch.int32).max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(dtype=torch.int32).max().item())
    if topk_w_internal is None:
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
    return FSATopkSparseAttentionWeighted.apply(
        q,
        k,
        v,
        topk_idx_internal,
        topk_w_internal,
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
    block_weights_bths: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    topk_idx_hns: Optional[torch.Tensor] = None,
    topk_w_hns: Optional[torch.Tensor] = None,
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

    topk_w: Optional[torch.Tensor] = None
    if topk_idx_hns is not None:
        if topk_idx_hns.ndim != 3:
            raise ValueError("topk_idx_hns must be rank-3 [HK or HQ, B*Tq, topk].")
        if topk_idx_hns.shape[1] != (B * Tq):
            raise ValueError(
                f"topk_idx_hns shape mismatch, expected second dim B*Tq={B*Tq}, got {tuple(topk_idx_hns.shape)}."
            )
        if topk_idx_hns.shape[0] == HK:
            topk_idx = topk_idx_hns
            if topk_w_hns is not None:
                if topk_w_hns.shape != topk_idx_hns.shape:
                    raise ValueError("topk_w_hns must match topk_idx_hns shape.")
                topk_w = topk_w_hns
        elif topk_idx_hns.shape[0] == HQ:
            # Convert per-query-head routes to per-kv-head routes by taking the first query-head
            # in each GQA group. In typical GQA use these are shared across the group.
            topk_idx_hns = topk_idx_hns.contiguous()
            topk_idx = _collapse_hq_routes_to_hk(topk_idx_hns, hk=HK, gqa_deg=gqa_deg)
            if topk_w_hns is not None:
                if topk_w_hns.shape != topk_idx_hns.shape:
                    raise ValueError("topk_w_hns must match topk_idx_hns shape.")
                topk_w_hns = topk_w_hns.contiguous()
                topk_w = _collapse_hq_routes_to_hk(topk_w_hns, hk=HK, gqa_deg=gqa_deg)
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
            if block_weights_bths is not None:
                if block_weights_bths.shape != block_indices_bths.shape:
                    raise ValueError("block_weights_bths must match block_indices_bths shape.")
                topk_w = block_weights_bths.permute(0, 2, 1, 3).reshape(HK, B * Tq, -1)
        elif block_indices_bths.shape[2] == HQ:
            topk_idx_q = block_indices_bths.permute(0, 2, 1, 3).reshape(HQ, B * Tq, -1).contiguous()
            topk_idx = _collapse_hq_routes_to_hk(topk_idx_q, hk=HK, gqa_deg=gqa_deg)
            if block_weights_bths is not None:
                if block_weights_bths.shape != block_indices_bths.shape:
                    raise ValueError("block_weights_bths must match block_indices_bths shape.")
                topk_w_q = block_weights_bths.permute(0, 2, 1, 3).reshape(HQ, B * Tq, -1).contiguous()
                topk_w = _collapse_hq_routes_to_hk(topk_w_q, hk=HK, gqa_deg=gqa_deg)
        else:
            raise ValueError(
                f"block_indices_bths third dim must be HK={HK} or HQ={HQ}, got {block_indices_bths.shape[2]}."
            )

    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    if not assume_sorted_topk:
        # FSA kernels expect per-query top-k entries to be ordered; unsorted entries
        # can mis-handle causal-valid counts vs traversal order.
        if topk_w is None:
            topk_idx = topk_idx.sort(dim=-1).values
        else:
            topk_idx, order = topk_idx.sort(dim=-1)
            topk_w = topk_w.to(torch.float32).gather(dim=-1, index=order)
    topk_idx = topk_idx.contiguous()
    if topk_w is not None:
        topk_w = topk_w.to(torch.float32).contiguous()

    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(B + 1, device=device, dtype=torch.int32) * Tq
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.arange(B + 1, device=device, dtype=torch.int32) * Tk

    out = FSA_topk_sparse_attention_varlen_qk(
        q=q,
        k=k,
        v=v,
        topk_idx=topk_idx,
        topk_w=topk_w,
        block_size=block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        disable_causal_mask=disable_causal_mask,
    )
    return out.reshape(B, Tq, HQ, D).contiguous()


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
    grouped = topk_idx_q.view(hk, gqa_deg, topk_idx_q.shape[1], topk_idx_q.shape[2]).contiguous()
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



