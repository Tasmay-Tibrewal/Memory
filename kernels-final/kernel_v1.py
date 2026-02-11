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

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    fused_fill_kernel[grid](
        tile_flat,
        m_i_cur_tiles_flat,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=3,
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
    pid_h = 0
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
    grid = (triton.cdiv(N_token, num_q_loops),)
    BLOCK_K = triton.next_power_of_2(TopK)
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
        num_warps=2,
        num_stages=3,
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
        TODO: Currently sequence packing is explicitly done in for loop, will merge in kernels.
    """
    o = torch.empty_like(q)
    total_len, num_heads, _ = q.shape
    lse = torch.empty((num_heads, total_len), dtype=torch.float32, device=q.device)

    permute_results = []
    for i in range(len(cu_seqlens_q) - 1):
        cu_seqlens_q_ = cu_seqlens_q[i: i + 2] - cu_seqlens_q[i]
        cu_seqlens_k_ = cu_seqlens_k[i: i + 2] - cu_seqlens_k[i]
        max_seqlen_q_ = cu_seqlens_q_[1] - cu_seqlens_q_[0]
        max_seqlen_k_ = cu_seqlens_k_[1] - cu_seqlens_k_[0]

        q_ = q[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        k_ = k[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        v_ = v[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        topk_idx_ = topk_idx[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        topk_w_ = topk_w[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]] if topk_w is not None else None
        o_seq, lse_seq, permute_results_seq = _topk_sparse_attention_fwd_opt_per_seq(
            q_,
            k_,
            v_,
            topk_idx_,
            topk_w_,
            block_size,
            cu_seqlens_q_,
            cu_seqlens_k_,
            max_seqlen_q_,
            max_seqlen_k_,
            sm_scale,
            causal,
        )
        o[cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = o_seq

        lse[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = lse_seq
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
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_q = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_K + offs_q

    start_id = tl.load(valid_start_indices_ptr + pid_b)
    valid_tokens = tl.load(valid_lens_ptr + pid_b)

    st_offs = start_id + offs_n
    # st should be in shape [BLOCK_SIZE_K]
    st_mask = offs_n < valid_tokens

    st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

    token_im_ptrs = token_index_mapping_ptr + pid_b * stride_im_b + st * stride_im_n

    tl.store(token_im_ptrs, offs_n, mask=st_mask)


def index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks):
    max_tokens = valid_lens.max()
    BLOCK_SIZE_K = 1024
    grid = (num_blocks, triton.cdiv(max_tokens, BLOCK_SIZE_K))

    index_mapping_kernel[grid](
        token_index_mapping,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        BLOCK_SIZE_K,
        num_warps=2,
        num_stages=3,
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
        num_warps=8,
        num_stages=3,
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
        num_stages=3,
        num_warps=4,
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
    h,
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
            h * head_tile,
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
            num_warps=1,
            num_stages=2,
        )


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
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256, 512, 1024}
    # shape

    total_len_q, num_heads, head_dim = q.shape
    total_len_k, num_kv_heads, head_dim = k.shape

    assert num_heads % num_kv_heads == 0
    gqa_deg = num_heads // num_kv_heads

    TOPK = topk_idx.shape[-1]

    real_num_blocks = math.ceil(total_len_k / block_size)
    num_blocks = max(real_num_blocks, TOPK)

    head_tile = 1
    reduce_tile_size = num_blocks - 1

    valid_lens_all = torch.zeros(
        (
            num_kv_heads,
            num_blocks,
        ),
        dtype=torch.int32,
        device=q.device,
    )
    for h in range(num_kv_heads):
        topk_idx_tile = topk_idx[h * head_tile: (h + 1) * head_tile]
        topk_idx_nonneg = topk_idx_tile[topk_idx_tile >= 0]
        valid_lens = torch.bincount(topk_idx_nonneg.view(-1), minlength=num_blocks)
        valid_lens_all[h * head_tile: (h + 1) * head_tile] = valid_lens

    global_max_valid_tokens = valid_lens_all[:, 1:].max() if num_blocks > 1 else valid_lens_all.max()

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_heads, total_len_q), float("-inf"), dtype=torch.float32, device=q.device)

    # New introduced buffers
    topk_idx_permuted_tile = torch.full((head_tile, num_blocks, total_len_q), -1, dtype=torch.int32, device=q.device)
    topk_w_permuted_tile = None
    if topk_w is not None:
        topk_w_permuted_tile = torch.zeros((head_tile, num_blocks, total_len_q), dtype=torch.float32, device=q.device)

    token_index_mapping = torch.full((head_tile, num_blocks, total_len_q), 0, dtype=torch.int32, device=q.device)
    # first KV block is computed seaprately
    o_tiles_first = torch.zeros((head_tile, 1, total_len_q, head_dim), dtype=torch.bfloat16, device=q.device)
    o_tiles_rest = torch.zeros(
        (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=q.device
    )

    # Statistics buffers
    # m_i_tiles: 历史最大, m_diff_tiles: 历史最大和当前最大的差值
    # m_i_cur_tiles: 当前最大, # m_ij_tiles: 考虑当前和历史后的最大
    m_i_cur_tiles: torch.Tensor = torch.full(
        (head_tile, num_blocks, total_len_q), float("-inf"), dtype=torch.float32, device=q.device
    )

    # first KV block is reduced separately
    l_ij_first = torch.full((head_tile, 1, total_len_q), 0, dtype=torch.float32, device=q.device)
    acc_o_scales_first = torch.full((head_tile, 1, total_len_q), 1, dtype=torch.float32, device=q.device)

    l_ij_rest = torch.full(
        (head_tile, reduce_tile_size, global_max_valid_tokens), 0, dtype=torch.float32, device=q.device
    )
    acc_o_scales_rest = torch.full(
        (head_tile, reduce_tile_size, global_max_valid_tokens), 1, dtype=torch.float32, device=q.device
    )

    permute_results = {}
    permute_results['global_max_valid_tokens'] = global_max_valid_tokens
    permute_results['num_blocks'] = num_blocks
    permute_results['real_num_blocks'] = real_num_blocks
    permute_results['valid_topk_idx_permuted_tile'] = []
    permute_results['valid_topk_w_permuted_tile'] = []
    permute_results['valid_lens_all'] = valid_lens_all
    permute_results['valid_lens'] = []
    permute_results['valid_start_indices'] = []

    for h in range(num_heads // head_tile):
        q_tile = q[:, h * head_tile: (h + 1) * head_tile]
        k_tile = k[:, (h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]
        v_tile = v[:, (h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]
        o = o_full[:, h * head_tile: (h + 1) * head_tile]
        lse = lse_full[h * head_tile: (h + 1) * head_tile]

        permute_min_block_id = 0
        permute_max_block_id = min(permute_min_block_id + num_blocks, num_blocks)

        topk_idx_tile = topk_idx[(h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]

        if h % gqa_deg == 0:
            if topk_w is None:
                topk_idx_permuted_tile = build_block_to_token_triton(
                    topk_idx_permuted_tile, topk_idx_tile, permute_min_block_id, permute_max_block_id, padding_value=-1
                )
            else:
                assert topk_w_permuted_tile is not None
                topk_w_tile = topk_w[(h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile].to(torch.float32)
                topk_idx_permuted_tile, topk_w_permuted_tile = build_block_to_token_with_weights_triton(
                    topk_idx_permuted_tile,
                    topk_w_permuted_tile,
                    topk_idx_tile,
                    topk_w_tile,
                    permute_min_block_id,
                    permute_max_block_id,
                    padding_value=-1,
                )

            mask_valid = topk_idx_permuted_tile != -1
            valid_topk_idx_permuted_tile = topk_idx_permuted_tile[mask_valid]
            valid_topk_w_permuted_tile = None
            if topk_w is not None:
                assert topk_w_permuted_tile is not None
                valid_topk_w_permuted_tile = topk_w_permuted_tile[mask_valid].to(torch.float32)
            valid_lens = valid_lens_all[(h // gqa_deg) * head_tile, :]
            valid_start_indices = torch.nn.functional.pad(valid_lens.cumsum(0)[:-1], (1, 0), value=0)

            index_mapping(
                token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks
            )

            permute_results['valid_topk_idx_permuted_tile'].append(valid_topk_idx_permuted_tile)
            permute_results['valid_topk_w_permuted_tile'].append(valid_topk_w_permuted_tile)
            permute_results['valid_lens'].append(valid_lens)
            permute_results['valid_start_indices'].append(valid_start_indices)

            m_ij_tiles, m_ij_last = online_softmax(
                q_tile,
                k_tile,
                m_i_cur_tiles,
                valid_topk_idx_permuted_tile,
                valid_lens,
                valid_start_indices,
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

            # Keep per-token online-softmax statistics as computed.
            # The token-0 broadcast used upstream can produce invalid stats in
            # prefix-mode layouts (dummy tokens at the front), leading to NaNs.
        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                cur_max_valid_tokens = valid_lens[0]
                cur_valid_lens = valid_lens[0]
                cur_valid_start_indices = valid_start_indices[0]
                o_tiles = o_tiles_first
                l_ij = l_ij_first
                acc_o_scales = acc_o_scales_first
                compute_tile_size = 1
            else:
                cur_max_valid_tokens = valid_lens[compute_min_block_id:].max()
                cur_valid_lens = valid_lens[compute_min_block_id:]
                cur_valid_start_indices = valid_start_indices[compute_min_block_id:]
                o_tiles = o_tiles_rest
                l_ij = l_ij_rest
                acc_o_scales = acc_o_scales_rest
                compute_tile_size = num_blocks - 1

            # launch kernel
            qkv_kernel(
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

        query_start_idx, query_tokens_count = _detect_active_token_range(topk_idx_tile)
        reduce_output(
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
            h,
            head_tile,
            total_len_q,
            TOPK,
            head_dim,
            query_start_idx=query_start_idx,
            query_tokens_count=query_tokens_count,
        )

        o_full[:, h * head_tile: (h + 1) * head_tile] = o
        lse_full[h * head_tile: (h + 1) * head_tile] = lse

        if h % gqa_deg == 0:
            fused_fill(topk_idx_permuted_tile, m_i_cur_tiles)
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
    num_blocks = max(real_num_blocks, topk)

    valid_lens_all = torch.zeros((num_kv_heads, num_blocks), dtype=torch.int32, device=topk_idx.device)
    for kh in range(num_kv_heads):
        topk_idx_tile = topk_idx[kh:kh + 1]
        topk_nonneg = topk_idx_tile[topk_idx_tile >= 0]
        valid_lens = torch.bincount(topk_nonneg.reshape(-1), minlength=num_blocks)
        valid_lens_all[kh:kh + 1] = valid_lens

    global_max_valid_tokens = valid_lens_all[:, 1:].max() if num_blocks > 1 else valid_lens_all.max()
    topk_idx_permuted_tile = torch.full((1, num_blocks, total_len_q), -1, dtype=torch.int32, device=topk_idx.device)
    topk_w_permuted_tile = None
    if topk_w is not None:
        topk_w_permuted_tile = torch.zeros((1, num_blocks, total_len_q), dtype=torch.float32, device=topk_idx.device)

    permute_results = {
        "global_max_valid_tokens": global_max_valid_tokens,
        "num_blocks": num_blocks,
        "real_num_blocks": real_num_blocks,
        "valid_topk_idx_permuted_tile": [],
        "valid_topk_w_permuted_tile": [],
        "valid_lens_all": valid_lens_all,
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
            build_block_to_token_kernel = build_block_to_token_with_weights_triton
            build_block_to_token_kernel(topk_idx_permuted_tile, topk_w_permuted_tile, topk_idx_tile, topk_w_tile, 0, num_blocks, padding_value=-1)
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
    for i in range(len(cu_seqlens_q) - 1):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
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
    small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "pad").strip().lower()
    # Modes:
    #  - pad:      pad G<16 to 16 (default)
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
            w_blk = bw_chunk[0].to(torch.float32).clamp_min_(0.0)                # [Tq, HK, S]
            w_tok = w_blk.unsqueeze(-1).expand(-1, -1, -1, block_size).reshape(tqa, hk, ksel)
            logw = torch.where(w_tok > 0, torch.log(w_tok), torch.full_like(w_tok, float("-inf")))
            scores = scores + logw.unsqueeze(2)

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

    for i in range(len(cu_seqlens_q) - 1):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())

        if native_nsa_fwd is not None:
            # Native NSA selected-fwd requires q/k/v to share the same timeline length.
            # Build a prefix-timeline view: [memory | query], matching benchmark Path-A layout.
            q_seq_full = q[q_start:q_end].contiguous().unsqueeze(0)  # [1, Tfull, HQ, D]
            k_seq_mem = k[k_start:k_end].contiguous().unsqueeze(0)   # [1, Tk, HK, D]
            v_seq_mem = v[k_start:k_end].contiguous().unsqueeze(0)   # [1, Tk, HK, D]
            bi_seq_full = (
                topk_idx[:, q_start:q_end, :]
                .permute(1, 0, 2)
                .contiguous()
                .unsqueeze(0)
            )                                                         # [1, Tfull, HK, S]
            bw_seq_full = None
            if topk_w is not None:
                bw_seq_full = (
                    topk_w[:, q_start:q_end, :]
                    .permute(1, 0, 2)
                    .contiguous()
                    .unsqueeze(0)
                    .to(dtype=torch.float32)
                )

            hq = int(q_seq_full.shape[2])
            hk = int(k_seq_mem.shape[2])
            g = (hq // hk) if (hk > 0 and hq % hk == 0) else -1
            nsa_shape_ok = (
                g >= 16
                and (g & (g - 1)) == 0
                and block_size >= 16
                and int(q_seq_full.shape[-1]) >= 16
            )

            if nsa_shape_ok and (topk_w is None):
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

        topk_idx_seq = topk_idx[:, q_start:q_end, :].contiguous()   # [HK, Tq, S]
        topk_w_seq = topk_w[:, q_start:q_end, :].contiguous() if topk_w is not None else None
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
                .to(dtype=torch.float32)
            )

        hq_real = int(q_seq.shape[2])
        hk = int(k_seq.shape[2])
        gqa_deg = (hq_real // hk) if (hk > 0 and hq_real % hk == 0) else -1
        use_g16_pad = pad_small_g_to_16 and (gqa_deg > 0 and gqa_deg < 16)
        use_torch_small_g = (small_g_mode == "torch") and (gqa_deg > 0 and gqa_deg < 16)

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

            if use_torch_small_g:
                o_chunk, lse_chunk_e = _torch_small_g_forward_chunk(
                    q_chunk=q_chunk,
                    k_seq=k_seq,
                    v_seq=v_seq,
                    bi_chunk=bi_chunk,
                    bw_chunk=bw_chunk,
                )
            elif use_g16_pad:
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
            else:
                q_call = q_chunk

            if not use_torch_small_g:
                o_chunk, lse_chunk_e = memory_cross_attn_forward(
                    q=q_call,
                    k=k_seq,
                    v=v_seq,
                    block_indices=bi_chunk,
                    block_weights=bw_chunk,
                    block_size=block_size,
                    scale=sm_scale,
                )

            if use_g16_pad and (not use_torch_small_g):
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

    pid_block = tl.program_id(0)
    pid_q = tl.program_id(1)  # token
    # seq packing is not supported yet
    q_start = 0
    k_start = 0

    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + pid_block)
    if num_dq_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (pid_block + compute_min_block_id) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # load k
    k = tl.load(tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero")
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn,
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

            q_ptrs = q_ptr + q_start * stride_qn + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_qn + q_ptrs_off
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            delta_ptrs = delta_ptr + st[:, None]
            d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            lse_ptrs = lse_ptr + st[:, None]
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
                token_index_mapping_ptr + (st) * stride_tim_n + (pid_block + compute_min_block_id) * stride_tim_b
            )
            token_index_mapping = tl.load(token_index_mapping_ptrs, mask=mask, other=-1)

            dq_ptrs_off = token_index_mapping[:, None] * stride_dqtn + off_d[None, :] * stride_dqtd
            dq_tiles_ptrs = dq_tiles_ptr + dq_ptrs_off + (pid_block).to(tl.int64) * stride_dqtb
            tl.store(dq_tiles_ptrs, dq.to(dq_tiles_ptr.dtype.element_ty), mask=q_mask)


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

    pid_q_local = pid_q + pid_qy * num_qz_loop
    if pid_q_local >= query_tokens_count:
        return
    pid_q_j = query_start_idx + pid_q_local
    if pid_q_j >= total_len:
        return
    t_ptr_j = t_ptr + pid_q_j * stride_tn

    off_d = tl.arange(0, BLOCK_SIZE_D)
    dq_ptrs = dq_ptr + pid_q_j * stride_dqn + off_d
    acc_dq = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for block_id in range(TOPK):
        t = tl.load(t_ptr_j + block_id * stride_tk, mask=block_id < TOPK, other=-1)
        if t != -1:
            if t == 0:
                dq_buffer_ptr = dq_buffer_first_ptr
                stride_dqtb = stride_dqtfb
                stride_dqtn = stride_dqtfn
                real_block_pos = 0
            else:
                dq_buffer_ptr = dq_buffer_rest_ptr
                stride_dqtb = stride_dqtrb
                stride_dqtn = stride_dqtrn
                real_block_pos = t - 1

            # init pointers
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + t.to(tl.int64) * stride_tim_b + (pid_q_j) * stride_tim_n
            )
            real_token_index = tl.load(token_index_mapping_ptrs)

            dq_buffer_ptrs = (
                dq_buffer_ptr + real_block_pos.to(tl.int64) * stride_dqtb + (real_token_index) * stride_dqtn + off_d
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
    disable_causal_mask=False,
):
    """
        TODO: Currently sequence packing is explicitly done in for loop, will merge in kernels.
    """
    for i in range(len(cu_seqlens_q) - 1):
        cu_seqlens_q_ = cu_seqlens_q[i: i + 2] - cu_seqlens_q[i]
        cu_seqlens_k_ = cu_seqlens_k[i: i + 2] - cu_seqlens_k[i]

        permute_results_ = permute_results[i]

        q_ = q[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        k_ = k[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        v_ = v[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        topk_idx_ = topk_idx[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        topk_w_ = topk_w[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]] if topk_w is not None else None
        lse_ = lse[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        delta_ = delta[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        do_ = do[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        dq_ = dq[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]

        backward_dq_opt_per_seq(
            q_,
            k_,
            v_,
            topk_idx_,
            topk_w_,
            lse_,
            delta_,
            do_,
            dq_,
            cu_seqlens_q_,
            cu_seqlens_k_,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            sm_scale,
            block_size,
            permute_results_,
            disable_causal_mask=disable_causal_mask,
        )

        dq[cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = dq_

    return dq


def backward_dq_opt_per_seq(
    q,  # [total_len, num_k_heads, head_dim]
    k,  # [total_len, num_k_heads, head_dim]
    v,  # [total_len, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    topk_w,  # [num_k_heads, total_len, topk] or None
    lse,  # [num_k_heads, total_len]
    delta,  # [num_k_heads, total_len]
    do,  # [total_len, num_k_heads, head_dim]
    dq,  # [total_len, num_k_heads, head_dim]
    cu_seqlens_q,
    cu_seqlens_k,
    num_k_heads,
    num_share_q_heads,
    head_dim,
    topk,
    sm_scale,
    block_size,
    permute_results,
    disable_causal_mask=False,
):
    head_tile = 1
    total_len = topk_idx.shape[1]
    global_max_valid_tokens = permute_results['global_max_valid_tokens']
    num_blocks = permute_results['num_blocks']
    reduce_tile_size = num_blocks - 1
    dq_buffer_first = torch.zeros((head_tile, 1, total_len, head_dim), dtype=torch.bfloat16, device=dq.device)
    dq_buffer_rest = torch.zeros(
        (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=dq.device
    )

    num_heads = num_share_q_heads * num_k_heads

    token_index_mapping = torch.full((head_tile, num_blocks, total_len), 0, dtype=torch.int32, device=q.device)
    for h in range(num_heads // head_tile):
        valid_topk_idx_permuted_tile = permute_results['valid_topk_idx_permuted_tile'][h // num_share_q_heads]
        valid_topk_w_permuted_tile = None
        if "valid_topk_w_permuted_tile" in permute_results:
            valid_topk_w_permuted_tile = permute_results["valid_topk_w_permuted_tile"][h // num_share_q_heads]

        valid_lens = permute_results['valid_lens'][h // num_share_q_heads]
        valid_start_indices = permute_results['valid_start_indices'][h // num_share_q_heads]

        index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)
        q_tile = q[:, h * head_tile: (h + 1) * head_tile]
        k_tile = k[:, (h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        v_tile = v[:, (h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        do_tile = do[:, h * head_tile: (h + 1) * head_tile]
        lse_tile = lse[h * head_tile: (h + 1) * head_tile]
        topk_idx_tile = topk_idx[(h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        delta_tile = delta[h * head_tile: (h + 1) * head_tile]
        dq_tile = dq[:, h * head_tile: (h + 1) * head_tile]

        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                compute_tile_size = 1
                cur_max_valid_tokens = valid_lens[0]
                cur_valid_lens = valid_lens[0]
                cur_valid_start_indices = valid_start_indices[0]
                dq_buffer = dq_buffer_first
            else:
                compute_tile_size = num_blocks - 1
                cur_max_valid_tokens = valid_lens[compute_min_block_id:].max()
                cur_valid_lens = valid_lens[compute_min_block_id:]
                cur_valid_start_indices = valid_start_indices[compute_min_block_id:]
                dq_buffer = dq_buffer_rest

            default_bq = 128
            default_loops = 16 if IS_HOPPER_GPU else 8
            BLOCK_SIZE_Q = int(os.getenv("FSA_LOCAL_BWD_DQ_BQ", str(default_bq)))
            if BLOCK_SIZE_Q not in (32, 64, 128, 256):
                BLOCK_SIZE_Q = default_bq
            num_dq_blocks = int(os.getenv("FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS", str(default_loops)))
            if num_dq_blocks <= 0:
                num_dq_blocks = default_loops
            grid_dq = lambda META: (
                compute_tile_size,
                triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_dq_blocks),
            )

            num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
            BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
            BLOCK_SIZE_K = triton.next_power_of_2(block_size)
            dq_compute_kernel[grid_dq](
                q_tile,
                k_tile,
                v_tile,
                lse_tile,
                delta_tile,
                do_tile,
                dq_buffer,
                token_index_mapping,
                valid_topk_idx_permuted_tile,
                valid_topk_w_permuted_tile if valid_topk_w_permuted_tile is not None else valid_topk_idx_permuted_tile,
                cur_valid_lens,
                cur_valid_start_indices,
                cur_max_valid_tokens,
                compute_min_block_id,
                head_tile,
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
                token_index_mapping.stride(0),
                token_index_mapping.stride(1),
                token_index_mapping.stride(2),
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

        query_start_idx, query_tokens_count = _detect_active_token_range(topk_idx_tile)
        if query_tokens_count <= 0:
            dq[:, h * head_tile: (h + 1) * head_tile] = dq_tile
            continue
        num_qy_loop = 4
        num_qz_loop = max(1, query_tokens_count // num_qy_loop)
        grid_x = num_qy_loop + (query_tokens_count % num_qy_loop != 0)
        max_grid_y = 65535

        for q_off in range(0, num_qz_loop, max_grid_y):
            grid_y = min(max_grid_y, num_qz_loop - q_off)
            grid_reduce = (grid_x, grid_y)
            dq_reduce_kernel[grid_reduce](
                dq_buffer_first,
                dq_buffer_rest,
                dq_tile,
                topk_idx_tile,
                token_index_mapping,
                num_qz_loop,
                q_off,
                query_start_idx,
                query_tokens_count,
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
                token_index_mapping.stride(0),
                token_index_mapping.stride(1),
                token_index_mapping.stride(2),
                BLOCK_SIZE_T=triton.next_power_of_2(topk),
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=1,
                num_stages=2,
            )

        dq[:, h * head_tile: (h + 1) * head_tile] = dq_tile

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
    DISABLE_CAUSAL_MASK: tl.constexpr,
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
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    # get topk_q_idx
    b_start = tl.load(cu_seqblocks + pid_b)  # how many blocks before current sequence
    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + pid_k) * stride_ctqn)
    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + pid_k + 1) * stride_ctqn)
    act_q_len = act_q_end - act_q_start
    tq_ptr = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn
    if HAS_WEIGHTS:
        tq_slot_ptr = tq_slot_ptr + pid_kh * stride_tqsh + act_q_start * stride_tqsn
        tq_w_ptr = tq_w_ptr + pid_kh * stride_tqwh + act_q_start * stride_tqwn
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
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
    for i in range(0, act_q_len, BLOCK_SIZE_Q):
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

    assert block_size in {32, 64, 128, 256, 512, 1024}
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
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O, IS_HOPPER_GPU)
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

    topk_q_count = torch.cat(
        [
            permute_results[i]['valid_lens_all'][:, : permute_results[i]['real_num_blocks']]
            for i in range(len(permute_results))
        ],
        dim=1,
    )

    cu_topk_q_count = torch.cat(
        [
            torch.zeros(topk_q_count.shape[0], 1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(topk_q_count, dim=-1),
        ],
        dim=-1,
    ).to(torch.int32)
    # active query idx for each key block
    # how to get active query idx for sequence b, head h, kv block i?
    has_weights = topk_w is not None
    if has_weights:
        topk_q_idx, topk_q_slot, topk_q_w = reorder_topk_idx(
            topk_idx,
            cu_topk_q_count,
            cu_seqlens_q,
            cu_seqblocks,
            block_size,
            topk_w=topk_w,
            return_slot=True,
            return_weights=True,
        )
        dw = torch.zeros_like(topk_w, dtype=torch.float32, device=topk_w.device)
    else:
        topk_q_idx = reorder_topk_idx(topk_idx, cu_topk_q_count, cu_seqlens_q, cu_seqblocks, block_size)
        topk_q_slot = topk_q_idx  # dummy - never read when HAS_WEIGHTS=0
        topk_q_w = delta[:1, :1]  # dummy - never read when HAS_WEIGHTS=0
        dw = None
    # compute dk dv
    dk = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
    dv = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
    batch_size = cu_seqlens_q.shape[0] - 1
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    default_bq_dkdv = 128 if IS_HOPPER_GPU else 64
    BLOCK_SIZE_Q = int(os.getenv("FSA_LOCAL_BWD_DKDV_BQ", str(default_bq_dkdv)))
    if BLOCK_SIZE_Q not in (32, 64, 128, 256):
        BLOCK_SIZE_Q = default_bq_dkdv
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    grid = (batch_size, num_q_heads, triton.cdiv(max_seqlen_k, BLOCK_SIZE_K))
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
        DISABLE_CAUSAL_MASK=disable_causal_mask,
        HAS_WEIGHTS=has_weights,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dk = dk.sum(0)
    dv = dv.sum(0)
    # compute dq
    dq = torch.zeros_like(q)
    num_q_loop = max_seqlen_q // 32768 + 1  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, triton.cdiv(max_seqlen_q, num_q_loop))
    BLOCK_SIZE_K = block_size
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)

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
        disable_causal_mask=disable_causal_mask,
    )

    return dq, dk, dv, dw


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
        small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "pad").strip().lower()
        if small_g_mode not in ("pad", "fma", "torch", "fallback"):
            small_g_mode = "pad"
        if gqa_deg > 0 and gqa_deg < 16 and small_g_mode == "fallback":
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
        if permute_results is None:
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
        small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "pad").strip().lower()
        if small_g_mode not in ("pad", "fma", "torch", "fallback"):
            small_g_mode = "pad"
        if gqa_deg > 0 and gqa_deg < 16 and small_g_mode == "fallback":
            use_nsa_style_fwd = False
        permute_results = None

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
        if permute_results is None:
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

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
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
    cap = int(os.getenv("FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE", "256"))
    if cap not in (32, 64, 128, 256):
        cap = 256
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
    topk_idx: torch.Tensor,
    block_size: int,
    internal_block_size: int,
) -> torch.Tensor:
    """
    Expand per-chapter top-k weights to match _expand_topk_for_internal_blocks.

    Uses repeat-interleave so gradients to the original weights sum across the
    duplicated internal sub-blocks.
    """
    ratio = block_size // internal_block_size
    if ratio == 1:
        return topk_w
    if topk_w.ndim != 3 or topk_idx.ndim != 3 or topk_w.shape != topk_idx.shape:
        raise ValueError("topk_w must match topk_idx shape [H, N, S].")
    w = topk_w.to(torch.float32)
    w = w.repeat_interleave(ratio, dim=-1)
    # Keep invalid padded entries at 0 weight.
    base = topk_idx.unsqueeze(-1)
    w = w.view(topk_idx.shape[0], topk_idx.shape[1], topk_idx.shape[2], ratio)
    w = torch.where(base >= 0, w, torch.zeros_like(w))
    return w.reshape(topk_idx.shape[0], topk_idx.shape[1], topk_idx.shape[2] * ratio).contiguous()


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
    topk_w_internal = None
    if topk_w is not None:
        topk_w_internal = _expand_topk_w_for_internal_blocks(topk_w, topk_idx, block_size, internal_block_size)

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
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
        topk_w_internal.to(dtype=torch.float32).contiguous(),
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

    if topk_idx_hns is not None:
        if topk_idx_hns.ndim != 3:
            raise ValueError("topk_idx_hns must be rank-3 [HK or HQ, B*Tq, topk].")
        if topk_idx_hns.shape[1] != (B * Tq):
            raise ValueError(
                f"topk_idx_hns shape mismatch, expected second dim B*Tq={B*Tq}, got {tuple(topk_idx_hns.shape)}."
            )
        if topk_w_hns is not None:
            if topk_w_hns.shape != topk_idx_hns.shape:
                raise ValueError("topk_w_hns must match topk_idx_hns shape [H, B*Tq, topk].")
        if topk_idx_hns.shape[0] == HK:
            topk_idx = topk_idx_hns
            topk_w = topk_w_hns
        elif topk_idx_hns.shape[0] == HQ:
            # Convert per-query-head routes to per-kv-head routes by taking the first query-head
            # in each GQA group. In typical GQA use these are shared across the group.
            topk_idx_hns = topk_idx_hns.contiguous()
            topk_idx = topk_idx_hns.view(HK, gqa_deg, B * Tq, topk_idx_hns.shape[-1])[:, 0, :, :]
            topk_w = None
            if topk_w_hns is not None:
                topk_w_hns = topk_w_hns.contiguous()
                topk_w = topk_w_hns.view(HK, gqa_deg, B * Tq, topk_w_hns.shape[-1])[:, 0, :, :]
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
        if block_weights_bths is not None and block_weights_bths.shape != block_indices_bths.shape:
            raise ValueError("block_weights_bths must match block_indices_bths shape [B,Tq,H,topk].")
        if block_indices_bths.shape[2] == HK:
            topk_idx = block_indices_bths.permute(0, 2, 1, 3).reshape(HK, B * Tq, -1)
            topk_w = (
                block_weights_bths.permute(0, 2, 1, 3).reshape(HK, B * Tq, -1)
                if block_weights_bths is not None
                else None
            )
        elif block_indices_bths.shape[2] == HQ:
            topk_idx_q = block_indices_bths.permute(0, 2, 1, 3).reshape(HQ, B * Tq, -1).contiguous()
            topk_idx = topk_idx_q.view(HK, gqa_deg, B * Tq, topk_idx_q.shape[-1])[:, 0, :, :]
            topk_w = None
            if block_weights_bths is not None:
                topk_w_q = block_weights_bths.permute(0, 2, 1, 3).reshape(HQ, B * Tq, -1).contiguous()
                topk_w = topk_w_q.view(HK, gqa_deg, B * Tq, topk_w_q.shape[-1])[:, 0, :, :]
        else:
            raise ValueError(
                f"block_indices_bths third dim must be HK={HK} or HQ={HQ}, got {block_indices_bths.shape[2]}."
            )

    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    if not assume_sorted_topk:
        # FSA kernels expect per-query top-k entries to be ordered; unsorted entries
        # can mis-handle causal-valid counts vs traversal order.
        topk_idx, sort_idx = topk_idx.sort(dim=-1)
        if topk_w is not None:
            topk_w = topk_w.gather(dim=-1, index=sort_idx)
    topk_idx = topk_idx.contiguous()
    if topk_w is not None:
        topk_w = topk_w.to(dtype=torch.float32).contiguous()

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
