# -*- coding: utf-8 -*-
"""
Memory Cross-Attention (FSA-inspired backend).

This module adds a new standalone implementation focused on a high-throughput
backward dK/dV path:
  - Forward: reuses the validated forward kernel from `memory_cross_attn.py`
  - Backward dQ: reuses the validated dQ kernel from `memory_cross_attn.py`
  - Backward dK/dV: uses a KV-block-outer schedule with a GPU-built
    inverted index (group = batch/head/chapter) inspired by NSA/FSA ideas.

Primary design goals:
  1) Avoid scan-all-query behavior in dK/dV.
  2) Avoid CPU-side index construction and large global sorts.
  3) Keep API compatible with existing benchmark scaffolding.

Expected tensor layouts:
  q: [B, TQ, HQ, Kdim]
  k: [B, TK, H,  Kdim]
  v: [B, TK, H,  Vdim]
  block_indices: [B, TQ, H, S] int32 chapter indices in [0, M), M = TK // block_size
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

import memory_cross_attn as base


# ---------------------------
# Autotune policy (cold-start control)
# ---------------------------

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _parse_num_warps(raw: Optional[str], default: Tuple[int, ...]) -> Tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return default
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            w = int(part)
        except ValueError:
            continue
        if w > 0:
            values.append(w)
    if not values:
        return default
    return tuple(dict.fromkeys(values))


_FAST_START = _env_flag("MEM_XATTN_FAST_START", True)
_DEFAULT_WARPS = (4,) if _FAST_START else (1, 2, 4, 8)
_NUM_WARPS = _parse_num_warps(os.getenv("MEM_XATTN_NUM_WARPS"), _DEFAULT_WARPS)
_AUTOTUNE_CONFIGS = [triton.Config({}, num_warps=w) for w in _NUM_WARPS]


# ---------------------------
# Triton helper
# ---------------------------

@triton.jit
def mm_or_fallback(a, b, USE_DOT: tl.constexpr):
    if USE_DOT:
        return tl.dot(a, b).to(tl.float32)
    else:
        a32 = a.to(tl.float32)
        b32 = b.to(tl.float32)
        return tl.sum(a32[:, :, None] * b32[None, :, :], axis=1)


# ---------------------------
# Inverted-index kernels (GPU)
# ---------------------------

@triton.jit
def mem_xattn_count_groups_kernel(
    block_indices,
    group_counts,
    TQ: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
):
    """
    Grid: (B*TQ*H,)
    One program handles one (b, t, h), loops over top-k chapter slots.
    """
    pid = tl.program_id(0)
    th = TQ * H
    i_b = pid // th
    rem = pid - i_b * th
    i_t = rem // H
    i_h = rem - i_t * H

    bh = i_b * H + i_h
    base = (i_b * TQ + i_t) * H * S + i_h * S

    for s in range(S):
        chap = tl.load(block_indices + base + s).to(tl.int32)
        if chap >= 0 and chap < M:
            group = bh * M + chap
            tl.atomic_add(group_counts + group, 1)


@triton.jit
def mem_xattn_scatter_groups_kernel(
    block_indices,
    group_offsets,
    group_cursors,
    sorted_q,
    TQ: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
):
    """
    Grid: (B*TQ*H,)
    Writes query indices into group buckets:
      group bucket = [group_offsets[g], group_offsets[g+1]).
    """
    pid = tl.program_id(0)
    th = TQ * H
    i_b = pid // th
    rem = pid - i_b * th
    i_t = rem // H
    i_h = rem - i_t * H

    bh = i_b * H + i_h
    base = (i_b * TQ + i_t) * H * S + i_h * S

    for s in range(S):
        chap = tl.load(block_indices + base + s).to(tl.int32)
        if chap >= 0 and chap < M:
            group = bh * M + chap
            # Position in group-local bucket.
            local_pos = tl.atomic_add(group_cursors + group, 1)
            group_start = tl.load(group_offsets + group)
            out_idx = group_start + local_pos
            tl.store(sorted_q + out_idx, i_t.to(tl.int32))


@dataclass
class InvertedIndexGPU:
    sorted_q: torch.Tensor      # [N_edges] int32
    group_offsets: torch.Tensor # [G_total+1] int32
    B: int
    TQ: int
    H: int
    M: int
    S: int


def build_inverted_index_gpu(block_indices: torch.Tensor, num_blocks: int) -> InvertedIndexGPU:
    """
    Build (group -> list of query indices) entirely on GPU using:
      pass 1: atomic counts
      pass 2: atomic scatter
    """
    if not block_indices.is_cuda:
        raise ValueError("block_indices must be CUDA tensor.")
    if block_indices.dtype != torch.int32:
        raise ValueError(f"block_indices must be int32, got {block_indices.dtype}.")
    if not block_indices.is_contiguous():
        raise ValueError("block_indices must be contiguous.")

    B, TQ, H, S = block_indices.shape
    M = int(num_blocks)
    G_total = B * H * M
    num_rows = B * TQ * H
    device = block_indices.device

    group_counts = torch.zeros((G_total,), device=device, dtype=torch.int32)
    mem_xattn_count_groups_kernel[(num_rows,)](
        block_indices=block_indices,
        group_counts=group_counts,
        TQ=TQ,
        H=H,
        S=S,
        M=M,
    )

    group_offsets = torch.empty((G_total + 1,), device=device, dtype=torch.int32)
    group_offsets[0] = 0
    torch.cumsum(group_counts, dim=0, out=group_offsets[1:])

    total_edges = int(group_offsets[-1].item())
    sorted_q = torch.empty((total_edges,), device=device, dtype=torch.int32)
    group_cursors = torch.zeros((G_total,), device=device, dtype=torch.int32)

    if total_edges > 0:
        mem_xattn_scatter_groups_kernel[(num_rows,)](
            block_indices=block_indices,
            group_offsets=group_offsets,
            group_cursors=group_cursors,
            sorted_q=sorted_q,
            TQ=TQ,
            H=H,
            S=S,
            M=M,
        )

    return InvertedIndexGPU(
        sorted_q=sorted_q,
        group_offsets=group_offsets,
        B=B,
        TQ=TQ,
        H=H,
        M=M,
        S=S,
    )


# ---------------------------
# dK/dV kernel (KV-block outer)
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dkv_fsa_kernel(
    q, k, v,
    lse, delta, do,
    dk, dv,
    sorted_q,         # int32 [N_edges]
    group_offsets,    # int32 [G_total+1]
    scale,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """
    Grid: (G_total, NV), where group=(b*h*m), NV=ceil(Vdim/BV).
    Each program owns one (group, value-chunk), no atomics required for dk/dv.
    """
    group = tl.program_id(0).to(tl.int32)
    i_v = tl.program_id(1)

    bh = group // M
    i_blk = group - bh * M
    i_b = bh // H
    i_h = bh - i_b * H

    tok0 = i_blk * BS

    start = tl.load(group_offsets + group)
    end = tl.load(group_offsets + group + 1)

    # Base pointers for this (b,h).
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

    # Load K/V once for this chapter block.
    p_k = tl.make_block_ptr(
        base=k + k_head_base + tok0 * H * Kdim,
        shape=(BS, Kdim),
        strides=(H * Kdim, 1),
        offsets=(0, 0),
        block_shape=(BS_PAD, BK),
        order=(1, 0),
    )
    p_v = tl.make_block_ptr(
        base=v + v_head_base + tok0 * H * Vdim,
        shape=(BS, Vdim),
        strides=(H * Vdim, 1),
        offsets=(0, i_v * BV),
        block_shape=(BS_PAD, BV),
        order=(1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_dk = tl.zeros([BS_PAD, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS_PAD, BV], dtype=tl.float32)

    for idx in range(start, end):
        i_t = tl.load(sorted_q + idx).to(tl.int32)
        q_tok = i_b * TQ + i_t

        p_q = tl.make_block_ptr(
            base=q + q_tok * HQ * Kdim,
            shape=(HQ, Kdim),
            strides=(Kdim, 1),
            offsets=(i_h * G, 0),
            block_shape=(G, BK),
            order=(1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_q = b_q * scale

        p_do = tl.make_block_ptr(
            base=do + q_tok * HQ * Vdim,
            shape=(HQ, Vdim),
            strides=(Vdim, 1),
            offsets=(i_h * G, i_v * BV),
            block_shape=(G, BV),
            order=(1, 0),
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))

        b_lse = tl.load(lse + q_tok * HQ + i_h * G + tl.arange(0, G))
        b_delta = tl.load(delta + q_tok * HQ + i_h * G + tl.arange(0, G))

        b_s = mm_or_fallback(b_k, tl.trans(b_q), USE_DOT=USE_DOT)  # [BS_PAD, G]
        b_p = tl.exp(b_s - b_lse[None, :])
        row_valid = (tl.arange(0, BS_PAD) < BS)[:, None]
        b_p = tl.where(row_valid, b_p, 0.0)

        if G == 1:
            b_dv += b_p.to(tl.float32) * b_do.to(tl.float32)
        else:
            b_dv += mm_or_fallback(b_p.to(b_do.dtype), b_do, USE_DOT=USE_DOT)

        b_dp = mm_or_fallback(b_v, tl.trans(b_do), USE_DOT=USE_DOT)
        b_ds = b_p * (b_dp - b_delta[None, :])

        if G == 1:
            b_dk += b_ds.to(tl.float32) * b_q.to(tl.float32)
        else:
            b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q, USE_DOT=USE_DOT)

    # Store group-local chapter gradients.
    p_dk = tl.make_block_ptr(
        base=dk + (i_v * B * TK * H + i_b * TK * H + i_h) * Kdim + tok0 * H * Kdim,
        shape=(BS, Kdim),
        strides=(H * Kdim, 1),
        offsets=(0, 0),
        block_shape=(BS_PAD, BK),
        order=(1, 0),
    )
    p_dv = tl.make_block_ptr(
        base=dv + (i_b * TK * H + i_h) * Vdim + tok0 * H * Vdim,
        shape=(BS, Vdim),
        strides=(H * Vdim, 1),
        offsets=(0, i_v * BV),
        block_shape=(BS_PAD, BV),
        order=(1, 0),
    )
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------
# Backward (dQ + FSA dKV)
# ---------------------------

def memory_cross_attn_backward_fsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (dq, dk, dv) using:
      - base dQ kernel
      - FSA-inspired dK/dV kernel backed by GPU-built inverted index
    """
    base._check_cuda(q, "q")
    base._check_cuda(k, "k")
    base._check_cuda(v, "v")
    base._check_cuda(o, "o")
    base._check_cuda(lse, "lse")
    base._check_cuda(do, "do")
    base._check_cuda(block_indices, "block_indices")

    B, TQ, HQ, Kdim = q.shape
    _, TK, H, _ = k.shape
    Vdim = v.shape[-1]
    S = block_indices.shape[-1]
    assert TK % block_size == 0
    M = TK // block_size

    if scale is None:
        scale = Kdim ** -0.5

    BK = min(256, base._next_pow2(Kdim))
    BV = min(256, base._next_pow2(Vdim))
    BS_PAD = base._next_pow2(block_size)
    NV = triton.cdiv(Vdim, BV)
    G = HQ // H
    assert HQ % H == 0, "HQ must be multiple of H."

    USE_DOT = (G >= 16) and (BK >= 16) and (BV >= 16) and (block_size >= 16)

    # ---- preprocess delta ----
    delta = base.memory_cross_attn_preprocess_delta(o, do)

    # ---- dQ (reuse validated base kernel) ----
    if NV == 1:
        dq = torch.empty_like(q)
        grid = (TQ, NV, B * H)
        base.mem_xattn_bwd_dq_kernel[grid](
            q=q, k=k, v=v,
            lse=lse, delta=delta, do=do,
            dq=dq,
            scale=scale,
            block_indices=block_indices,
            TQ=TQ, TK=TK,
            B=B, H=H, HQ=HQ, G=G,
            Kdim=Kdim, Vdim=Vdim,
            S=S, BS=block_size, BS_PAD=BS_PAD,
            BK=BK, BV=BV,
            USE_DOT=USE_DOT,
            DQ_HAS_NV=False,
            DQ_STRIDE_NV=0,
        )
    else:
        dq_part = torch.empty((NV, *q.shape), device=q.device, dtype=torch.float32)
        dq_stride_nv = dq_part.stride(0)
        grid = (TQ, NV, B * H)
        base.mem_xattn_bwd_dq_kernel[grid](
            q=q, k=k, v=v,
            lse=lse, delta=delta, do=do,
            dq=dq_part,
            scale=scale,
            block_indices=block_indices,
            TQ=TQ, TK=TK,
            B=B, H=H, HQ=HQ, G=G,
            Kdim=Kdim, Vdim=Vdim,
            S=S, BS=block_size, BS_PAD=BS_PAD,
            BK=BK, BV=BV,
            USE_DOT=USE_DOT,
            DQ_HAS_NV=True,
            DQ_STRIDE_NV=dq_stride_nv,
        )
        dq = dq_part.sum(0).to(q.dtype)

    # ---- dK/dV with GPU-built inverted index ----
    inv = build_inverted_index_gpu(block_indices, num_blocks=M)
    dk_part = torch.empty((NV, *k.shape), device=q.device, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    G_total = B * H * M
    grid = (G_total, NV)
    mem_xattn_bwd_dkv_fsa_kernel[grid](
        q=q, k=k, v=v,
        lse=lse, delta=delta, do=do,
        dk=dk_part, dv=dv,
        sorted_q=inv.sorted_q,
        group_offsets=inv.group_offsets,
        scale=scale,
        TQ=TQ, TK=TK,
        B=B, H=H, HQ=HQ, G=G,
        Kdim=Kdim, Vdim=Vdim,
        M=M, BS=block_size, BS_PAD=BS_PAD,
        BK=BK, BV=BV,
        USE_DOT=USE_DOT,
    )
    dk = dk_part.sum(0)

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


# ---------------------------
# Autograd wrapper / public API
# ---------------------------

class MemoryCrossAttnFSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_size: int, scale: Optional[float]):
        o, lse = base.memory_cross_attn_forward(q, k, v, block_indices, block_size, scale)
        ctx.save_for_backward(q, k, v, o, lse, block_indices)
        ctx.block_size = block_size
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices = ctx.saved_tensors
        dq, dk, dv = memory_cross_attn_backward_fsa(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=block_indices,
            block_size=ctx.block_size,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None


def memory_cross_attn_fsa_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Drop-in forward entrypoint:
      out = memory_cross_attn_fsa_opt(q, k, v, block_indices, block_size, scale)
    """
    return MemoryCrossAttnFSAFunction.apply(q, k, v, block_indices, block_size, scale)

