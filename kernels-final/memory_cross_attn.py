# -*- coding: utf-8 -*-
"""
Memory Cross-Attention (chapter-routed) Triton kernels.

Implements:
  Forward: online softmax over (topk * BS) tokens without KV duplication
  Backward: dQ + dK/dV accumulation to shared memory bank

dKV backends:
  A: KV-block parallel + block_mask + scan all queries (NSA-style)
  B: Query-chunk x KV-block parallel + atomic_add to dK/dV
  C: KV-block parallel + inverted index (chapter -> query list)
  D: Chunked inverted index + partial buffers + reduction pass

Assumptions / Notes:
- Layout is [B, T, H, D] contiguous (strides: (T*H*D, H*D, D, 1)).
- block_indices is [B, TQ, H, S] where each entry is a chapter index in [0, M).
- M = TK // BS (num chapters).
- For correctness, Approach A's boolean block_mask assumes each (query, head) selects each chapter at most once.
  Approaches C/D are correct even if duplicates exist (duplicates appear as multiple entries in inverted index).
  Approach B (scan) handles duplicates by counting occurrences and scaling contributions.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import triton
import triton.language as tl


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
    # Stable dedup while preserving order
    return tuple(dict.fromkeys(values))


_FAST_START = _env_flag("MEM_XATTN_FAST_START", True)
_DEFAULT_WARPS = (4,) if _FAST_START else (1, 2, 4, 8)
_NUM_WARPS = _parse_num_warps(os.getenv("MEM_XATTN_NUM_WARPS"), _DEFAULT_WARPS)
_AUTOTUNE_CONFIGS = [triton.Config({}, num_warps=w) for w in _NUM_WARPS]


# ---------------------------
# Triton helpers
# ---------------------------

@triton.jit
def mm_or_fallback(a, b, USE_DOT: tl.constexpr):
    """
    Safe matmul helper for small dimensions.

    Triton's `tl.dot` has minimum tile constraints on some backends; in particular,
    calling `tl.dot` with any of (M, N, K) < 16 can fail to compile. In our use-case,
    G (= HQ//H) can be 1, so we provide a compile-time fallback path that uses
    elementwise multiply + reduction (always legal).
    """
    if USE_DOT:
        return tl.dot(a, b).to(tl.float32)
    else:
        a32 = a.to(tl.float32)
        b32 = b.to(tl.float32)
        return tl.sum(a32[:, :, None] * b32[None, :, :], axis=1)


# ---------------------------
# Utilities / Shape helpers
# ---------------------------

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** (int(x - 1).bit_length())


def _check_cuda(x: torch.Tensor, name: str):
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")


def _check_contiguous_lastdim(x: torch.Tensor, name: str):
    # We rely on the standard contiguous layout for [B,T,H,D]
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous (expected [B,T,H,D] contiguous). Got strides={x.stride()}.")


# ---------------------------
# Forward kernel
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_fwd_kernel(
    q, k, v,
    o, lse,
    scale,
    block_indices,
    TQ: tl.constexpr,          # query length (per batch)
    TK: tl.constexpr,          # kv length (per batch)
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    S: tl.constexpr,           # topk
    BS: tl.constexpr,          # block size (chapter size)
    BS_PAD: tl.constexpr,      # padded power-of-two tile size
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """
    Grid: (TQ, NV, B*H)
      pid0 = query token index
      pid1 = value chunk
      pid2 = batch-head index
    """
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    # Pointers:
    # q: [B, TQ, HQ, Kdim]
    # k: [B, TK, H,  Kdim]
    # v: [B, TK, H,  Vdim]
    # block_indices: [B, TQ, H, S]

    # base offsets (token-major)
    q_tok = (i_b * TQ + i_t)
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim
    bi_base = (i_b * TQ + i_t) * H * S + i_h * S

    # Load Q block for grouped heads: [G, BK]
    p_q = tl.make_block_ptr(
        base=q + q_tok * HQ * Kdim,
        shape=(HQ, Kdim),
        strides=(Kdim, 1),
        offsets=(i_h * G, 0),
        block_shape=(G, BK),
        order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # Output accumulator:
    # b_o: [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)

    # iterate over selected chapters
    for s in range(S):
        chap = tl.load(block_indices + bi_base + s).to(tl.int32)
        # token start
        tok0 = chap * BS

        # skip invalid chap
        if tok0 >= 0 and tok0 < TK:
            # K block: treat K as (Kdim, TK) with stride along tokens = H*Kdim
            p_k = tl.make_block_ptr(
                base=k + k_head_base + tok0 * H * Kdim,
                shape=(Kdim, BS),
                strides=(1, H * Kdim),
                offsets=(0, 0),
                block_shape=(BK, BS_PAD),
                order=(0, 1),
            )
            # V block: treat V as (TK, Vdim) with stride along tokens = H*Vdim
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

            # scores: [G, BS]
            b_s = mm_or_fallback(b_q, b_k, USE_DOT=USE_DOT)  # [G, BS]

            # optional bound mask (usually TK == M*BS so this is always valid)
            offs_local = tl.arange(0, BS_PAD)
            offs_global = tok0 + offs_local
            valid = (offs_local < BS) & (offs_global < TK)
            b_s = tl.where(valid[None, :], b_s, float("-inf"))

            # online softmax update
            b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
            b_r = tl.exp(b_m - b_m_new)
            b_p = tl.exp(b_s - b_m_new[:, None])

            b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
            b_o = b_o * b_r[:, None] + mm_or_fallback(b_p.to(b_q.dtype), b_v, USE_DOT=USE_DOT)  # [G, BV]

            b_m = b_m_new

    # finalize
    # handle degenerate case
    b_o = tl.where(b_acc[:, None] > 0, b_o / b_acc[:, None], 0.0)
    b_lse = tl.where(b_acc > 0, b_m + tl.log(b_acc), 0.0)

    # store output
    p_o = tl.make_block_ptr(
        base=o + q_tok * HQ * Vdim,
        shape=(HQ, Vdim),
        strides=(Vdim, 1),
        offsets=(i_h * G, i_v * BV),
        block_shape=(G, BV),
        order=(1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # store lse once per (query, head-group)
    if i_v == 0:
        tl.store(
            lse + q_tok * HQ + i_h * G + tl.arange(0, G),
            b_lse.to(lse.dtype.element_ty),
        )


# ---------------------------
# Backward preprocess: delta = sum(o * do)
# ---------------------------

@triton.jit
def mem_xattn_bwd_preprocess_kernel(
    o, do, delta,
    BLOCK: tl.constexpr,
    Vdim: tl.constexpr,
):
    """
    1D grid over (B*TQ*HQ) rows, each row has Vdim elements.
    delta[row] = sum_j o[row, j] * do[row, j]
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < Vdim

    b_o = tl.load(o + row * Vdim + offs, mask=mask, other=0.0).to(tl.float32)
    b_do = tl.load(do + row * Vdim + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(delta + row, tl.sum(b_o * b_do).to(delta.dtype.element_ty))


# ---------------------------
# Backward dQ kernel
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dq_kernel(
    q, k, v,
    lse, delta, do,
    dq,
    scale,
    block_indices,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DOT: tl.constexpr,
    DQ_HAS_NV: tl.constexpr,
    DQ_STRIDE_NV: tl.constexpr,
):
    """
    Grid: (TQ, NV, B*H)
    Each program computes partial dQ for one value-chunk (i_v), then caller sums across NV.
    """
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_tok = i_b * TQ + i_t
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim
    bi_base = (i_b * TQ + i_t) * H * S + i_h * S

    # pointers row-local
    p_q = tl.make_block_ptr(
        base=q + q_tok * HQ * Kdim,
        shape=(HQ, Kdim),
        strides=(Kdim, 1),
        offsets=(i_h * G, 0),
        block_shape=(G, BK),
        order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

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

    b_dq = tl.zeros([G, BK], dtype=tl.float32)

    for s in range(S):
        chap = tl.load(block_indices + bi_base + s).to(tl.int32)
        tok0 = chap * BS
        if tok0 >= 0 and tok0 < TK:
            # K block [BK, BS]
            p_k = tl.make_block_ptr(
                base=k + k_head_base + tok0 * H * Kdim,
                shape=(Kdim, BS),
                strides=(1, H * Kdim),
                offsets=(0, 0),
                block_shape=(BK, BS_PAD),
                order=(0, 1),
            )
            # V block transposed for dp: [BV, BS]
            p_vt = tl.make_block_ptr(
                base=v + v_head_base + tok0 * H * Vdim,
                shape=(Vdim, BS),
                strides=(1, H * Vdim),
                offsets=(i_v * BV, 0),
                block_shape=(BV, BS_PAD),
                order=(0, 1),
            )

            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_vt = tl.load(p_vt, boundary_check=(0, 1))

            # scores [G, BS]
            b_s = mm_or_fallback(b_q, b_k, USE_DOT=USE_DOT)  # [G, BS]
            offs_local = tl.arange(0, BS_PAD)
            offs_global = tok0 + offs_local
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(((offs_local < BS) & (offs_global < TK))[None, :], b_p, 0.0)

            # dP = do @ V^T -> [G, BS]
            b_dp = mm_or_fallback(b_do, b_vt, USE_DOT=USE_DOT)  # [G, BS]
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])

            # dQ_scaled += dS @ K
            b_dq += mm_or_fallback(b_ds.to(b_k.dtype), tl.trans(b_k), USE_DOT=USE_DOT)  # [G, BK]

    # dQ = scale * dQ_scaled
    b_dq *= scale

    # store partial dQ for this value chunk
    dq_base = dq
    if DQ_HAS_NV:
        dq_base = dq + i_v.to(tl.int64) * tl.full([], DQ_STRIDE_NV, tl.int64)
    p_dq = tl.make_block_ptr(
        base=dq_base + q_tok * HQ * Kdim,
        shape=(HQ, Kdim),
        strides=(Kdim, 1),
        offsets=(i_h * G, 0),
        block_shape=(G, BK),
        order=(1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------
# Approach A: block_mask build + KV-block-parallel dKV (NSA-style)
# ---------------------------

@triton.jit
def mem_xattn_block_mask_kernel(
    block_indices,
    block_mask,
    TQ: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,   # num kv blocks (chapters)
):
    """
    block_indices: [B, TQ, H, S] (flattened pointer)
    block_mask:    [B, TQ, H, M] int32 counts
    Grid: (TQ, B, H*S)
    Accumulates per-(b,t,h,chap) selection counts (robust even if duplicates occur).
    """
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)
    i_hs = tl.program_id(2)

    i_h = i_hs // S
    i_s = i_hs % S

    chap = tl.load(block_indices + (i_b * TQ + i_t) * H * S + i_h * S + i_s).to(tl.int32)

    if chap >= 0 and chap < M:
        tl.atomic_add(
            block_mask + (i_b * TQ + i_t) * H * M + i_h * M + chap,
            tl.full([], 1, tl.int32),
        )


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dkv_a_kernel(
    q, k, v,
    lse, delta, do,
    dk, dv,
    block_mask,
    scale,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    M: tl.constexpr,    # num kv blocks (chapters)
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """
    Approach A dKV:
      Grid: (NV, M, B*H)
      For each (value-chunk, chapter-block, head), scan ALL queries and accumulate
      if block_mask says query attends to this chapter.
    """
    i_v = tl.program_id(0)
    i_blk = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    tok0 = i_blk * BS

    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

    # load K/V chapter block
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

    # scan all queries
    for i_t in range(0, TQ):
        m = tl.load(block_mask + (i_b * TQ + i_t) * H * M + i_h * M + i_blk).to(tl.int32)
        if m > 0:
            q_tok = i_b * TQ + i_t
            # Q_scaled
            p_q = tl.make_block_ptr(
                base=q + q_tok * HQ * Kdim,
                shape=(HQ, Kdim),
                strides=(Kdim, 1),
                offsets=(i_h * G, 0),
                block_shape=(G, BK),
                order=(1, 0),
            )
            # Keep scaled Q in fp32 for dK/dV accumulation fidelity.
            # Naive reference computes q.float() * scale.
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

            # scores [BS_PAD, G]
            b_s = mm_or_fallback(b_k, tl.trans(b_q), USE_DOT=USE_DOT)  # [BS, G]
            b_p = tl.exp(b_s - b_lse[None, :])
            row_valid = (tl.arange(0, BS_PAD) < BS)[:, None]
            b_p = tl.where(row_valid, b_p, 0.0)
            # Preserve exact semantics if duplicate chapters appear in block_indices.
            b_p = b_p * m.to(tl.float32)

            # dv += P @ dO
            if G == 1:
                # Outer product for G=1 without tl.dot.
                b_dv += b_p.to(tl.float32) * b_do.to(tl.float32)
            else:
                b_dv += mm_or_fallback(b_p.to(b_do.dtype), b_do, USE_DOT=USE_DOT)  # [BS, BV]

            # dP = V @ dO^T -> [BS, G]
            b_dp = mm_or_fallback(b_v, tl.trans(b_do), USE_DOT=USE_DOT)  # [BS, G] fp32
            b_ds = b_p * (b_dp - b_delta[None, :])

            # dk += dS @ Q_scaled
            if G == 1:
                # Outer product for G=1 without tl.dot.
                b_dk += b_ds.to(tl.float32) * b_q.to(tl.float32)
            else:
                b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q, USE_DOT=USE_DOT)  # [BS, BK]

    # store results (chapter-local rows)
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
# Approach B: Query-chunk x KV-block + atomic_add
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "CHUNK_Q", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dkv_b_atomic_kernel(
    q, k, v,
    lse, delta, do,
    dk, dv,
    block_indices,
    scale,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_Q: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """
    Approach B dKV:
      Grid: (NV, M, (num_q_chunks * B*H))
        pid0 = i_v
        pid1 = i_blk
        pid2 = packed (q_chunk, b, h)

      Each program:
        - loads K/V for one chapter block and head
        - iterates over queries in its chunk
        - checks membership by scanning block_indices[b, q, h, :]
        - accumulates local b_dk/b_dv
        - atomic_add into global dk/dv (float32)

    This avoids building block_mask. Duplicates are handled via occurrence count.
    """
    i_v = tl.program_id(0)
    i_blk = tl.program_id(1)
    pid2 = tl.program_id(2)

    # decode pid2
    # pid2 range: [0, num_q_chunks * B * H)
    # let bh = pid2 % (B*H), q_chunk = pid2 // (B*H)
    BH = B * H
    q_chunk = pid2 // BH
    i_bh = pid2 - q_chunk * BH
    i_b = i_bh // H
    i_h = i_bh % H

    q_start = q_chunk * CHUNK_Q
    q_end = tl.minimum(TQ, q_start + CHUNK_Q)

    tok0 = i_blk * BS

    # base pointers for head
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

    # load K/V chapter block
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

    # iterate over queries in chunk (static loop + runtime bound guard)
    for off in range(CHUNK_Q):
        i_t = q_start + off
        if i_t < q_end:
            # membership test: count occurrences of i_blk in block_indices[b,i_t,h,:]
            bi_base = (i_b * TQ + i_t) * H * S + i_h * S
            offs_s = tl.arange(0, S)
            chaps = tl.load(block_indices + bi_base + offs_s).to(tl.int32)
            occ = tl.sum((chaps == i_blk).to(tl.int32), axis=0)

            if occ > 0:
                q_tok = i_b * TQ + i_t

                # Q_scaled
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

                # scores [BS_PAD, G]
                b_s = mm_or_fallback(b_k, tl.trans(b_q), USE_DOT=USE_DOT)
                b_p = tl.exp(b_s - b_lse[None, :]).to(tl.float32)
                row_valid = (tl.arange(0, BS_PAD) < BS)[:, None]
                b_p = tl.where(row_valid, b_p, 0.0)

                # handle duplicates by scaling
                b_p = b_p * occ.to(tl.float32)

                # dv += P @ dO
                if G == 1:
                    b_dv += b_p.to(tl.float32) * b_do.to(tl.float32)
                else:
                    b_dv += mm_or_fallback(b_p.to(b_do.dtype), b_do, USE_DOT=USE_DOT)  # [BS, BV]

                # dP = V @ dO^T
                b_dp = mm_or_fallback(b_v, tl.trans(b_do), USE_DOT=USE_DOT)  # [BS, G] fp32
                b_ds = b_p * (b_dp - b_delta[None, :])

                # dk += dS @ Q_scaled
                if G == 1:
                    b_dk += b_ds.to(tl.float32) * b_q.to(tl.float32)
                else:
                    b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q, USE_DOT=USE_DOT)  # [BS, BK]

    # atomic add into dk/dv
    # dk layout: [NV, B, TK, H, Kdim] float32
    dk_base = dk + (i_v * B * TK * H + i_b * TK * H + i_h) * Kdim
    dv_base = dv + (i_b * TK * H + i_h) * Vdim

    offs_t_local = tl.arange(0, BS_PAD)
    offs_t = tok0 + offs_t_local
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    m_row = offs_t_local < BS
    m_t = (offs_t < TK) & m_row
    m_k = offs_k < Kdim
    m_v = (i_v * BV + offs_v) < Vdim

    # pointers for dk: [BS_PAD, BK]
    dk_ptrs = dk_base + (offs_t[:, None] * (H * Kdim) + offs_k[None, :])
    tl.atomic_add(dk_ptrs, b_dk, mask=(m_t[:, None] & m_k[None, :]))

    # pointers for dv: [BS_PAD, BV]
    dv_ptrs = dv_base + (offs_t[:, None] * (H * Vdim) + (i_v * BV + offs_v)[None, :])
    tl.atomic_add(dv_ptrs, b_dv, mask=(m_t[:, None] & m_v[None, :]))


# ---------------------------
# Approach C: Inverted index dKV (no wasted iterations)
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dkv_c_invidx_kernel(
    q, k, v,
    lse, delta, do,
    dk, dv,
    sorted_q,         # int32 [N_edges] query indices within batch (0..TQ-1)
    group_offsets,    # int32 [G_total+1] offsets into sorted_q for each group
    scale,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    M: tl.constexpr,      # num chapters
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """
    Approach C dKV:
      Grid: (NV, M, B*H)
      Each program handles one (i_v, chapter, batch-head)
      Iterates only over queries that routed to this chapter (via inverted index).
    """
    i_v = tl.program_id(0)
    i_blk = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    tok0 = i_blk * BS

    # group id (batch-head-chapter)
    group = (i_bh * M + i_blk).to(tl.int32)

    start = tl.load(group_offsets + group)
    end = tl.load(group_offsets + group + 1)

    # base pointers for head
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

    # load K/V chapter block
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

    # iterate only relevant queries
    for idx in range(start, end):
        i_t = tl.load(sorted_q + idx).to(tl.int32)   # within-batch query idx
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
            b_dv += mm_or_fallback(b_p.to(b_do.dtype), b_do, USE_DOT=USE_DOT)  # [BS, BV]

        b_dp = mm_or_fallback(b_v, tl.trans(b_do), USE_DOT=USE_DOT)  # [BS, G] fp32
        b_ds = b_p * (b_dp - b_delta[None, :])

        if G == 1:
            b_dk += b_ds.to(tl.float32) * b_q.to(tl.float32)
        else:
            b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q, USE_DOT=USE_DOT)  # [BS, BK]

    # store
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
# Approach D: Chunked inverted index -> partial buffers -> reduction
# ---------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV", "CHUNK_Q", "G", "USE_DOT"],
)
@triton.jit
def mem_xattn_bwd_dkv_d_partial_kernel(
    q, k, v,
    lse, delta, do,
    dk_part, dv_part,
    sorted_q,
    chunk_group,     # int32 [num_chunks] group id = (bh*M + blk)
    chunk_start,     # int32 [num_chunks] start idx into sorted_q
    chunk_end,       # int32 [num_chunks] end idx into sorted_q
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
    CHUNK_Q: tl.constexpr,   # max queries per chunk (used only for autotune key)
    USE_DOT: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """
    Grid: (num_chunks, NV)
      pid0 = chunk_id
      pid1 = i_v

    Each program processes exactly one chunk of queries for one (batch-head, chapter) group.
    Writes partial dk/dv into dk_part/dv_part.

    dk_part: [NV, num_chunks, BS, BK] float32
    dv_part: [NV, num_chunks, BS, BV] float32
    """
    c_id = tl.program_id(0)
    i_v = tl.program_id(1)

    group = tl.load(chunk_group + c_id).to(tl.int32)
    start = tl.load(chunk_start + c_id).to(tl.int32)
    end = tl.load(chunk_end + c_id).to(tl.int32)

    bh = group // M
    i_blk = group - bh * M

    i_b = bh // H
    i_h = bh - i_b * H

    tok0 = i_blk * BS

    # base pointers for head
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

    # load K/V chapter block
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
            b_dv += mm_or_fallback(b_p.to(b_do.dtype), b_do, USE_DOT=USE_DOT)  # [BS, BV]

        b_dp = mm_or_fallback(b_v, tl.trans(b_do), USE_DOT=USE_DOT)  # [BS, G] fp32
        b_ds = b_p * (b_dp - b_delta[None, :])

        if G == 1:
            b_dk += b_ds.to(tl.float32) * b_q.to(tl.float32)
        else:
            b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q, USE_DOT=USE_DOT)  # [BS, BK]

    # store partials
    # dk_part index: ((i_v * num_chunks + c_id) * BS * BK)
    # dk_part/dv_part are laid out as [NV, NUM_CHUNKS, BS, BK/BV] contiguous.
    dk_base = dk_part + (i_v * NUM_CHUNKS + c_id) * (BS * BK)
    dv_base = dv_part + (i_v * NUM_CHUNKS + c_id) * (BS * BV)

    offs_t = tl.arange(0, BS_PAD)[:, None]
    offs_k = tl.arange(0, BK)[None, :]
    offs_v = tl.arange(0, BV)[None, :]

    row_mask = offs_t < BS
    tl.store(dk_base + offs_t * BK + offs_k, b_dk, mask=row_mask)
    tl.store(dv_base + offs_t * BV + offs_v, b_dv, mask=row_mask)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BS", "BK", "BV"],
)
@triton.jit
def mem_xattn_bwd_dkv_d_reduce_kernel(
    dk_part, dv_part,
    dk, dv,
    chunk_offsets,   # int32 [G_total+1] offsets into chunk arrays for each group
    scale_dummy,     # unused, just for signature uniformity
    num_chunks: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    Kdim: tl.constexpr,
    Vdim: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BS_PAD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Reduction pass for Approach D.

    Grid: (G_total, NV)
      pid0 = group (bh*M + blk)
      pid1 = i_v

    Sums dk_part/dv_part across chunk ids belonging to group.

    dk_part: [NV, num_chunks, BS, BK]
    dv_part: [NV, num_chunks, BS, BV]

    Outputs:
      dk: [NV, B, TK, H, Kdim]
      dv: [B, TK, H, Vdim]
    """
    group = tl.program_id(0).to(tl.int32)
    i_v = tl.program_id(1)

    start = tl.load(chunk_offsets + group)
    end = tl.load(chunk_offsets + group + 1)

    bh = group // M
    i_blk = group - bh * M
    i_b = bh // H
    i_h = bh - i_b * H

    tok0 = i_blk * BS

    # accumulators
    b_dk = tl.zeros([BS_PAD, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS_PAD, BV], dtype=tl.float32)

    # sum over chunks
    for c_id in range(start, end):
        # base pointers into part buffers
        dk_base = dk_part + (i_v * num_chunks + c_id) * (BS * BK)
        dv_base = dv_part + (i_v * num_chunks + c_id) * (BS * BV)

        offs_t = tl.arange(0, BS_PAD)[:, None]
        offs_k = tl.arange(0, BK)[None, :]
        offs_v = tl.arange(0, BV)[None, :]

        row_mask = offs_t < BS
        b_dk += tl.load(dk_base + offs_t * BK + offs_k, mask=row_mask, other=0.0).to(tl.float32)
        b_dv += tl.load(dv_base + offs_t * BV + offs_v, mask=row_mask, other=0.0).to(tl.float32)

    # store to final dk/dv
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
# Python: inverted index builders (C/D)
# ---------------------------

@dataclass
class InvertedIndex:
    # sorted query indices within batch
    sorted_q: torch.Tensor          # [N_edges] int32
    # offsets per group: group = (bh*M + blk), bh=(b*H + h)
    group_offsets: torch.Tensor     # [G_total+1] int32
    # metadata
    B: int
    TQ: int
    H: int
    M: int
    S: int


def build_inverted_index(
    block_indices: torch.Tensor,
    num_blocks: int,
) -> InvertedIndex:
    """
    Build inverted index for approaches C/D.

    Input:
      block_indices: [B, TQ, H, S] (chapter indices in [0, num_blocks))
    Output:
      sorted_q: [N_edges] (query indices within batch, 0..TQ-1)
      group_offsets: [B*H*num_blocks + 1] offsets into sorted_q
    """
    _check_cuda(block_indices, "block_indices")
    B, TQ, H, S = block_indices.shape
    device = block_indices.device

    # Flatten chapters
    chapters = block_indices.reshape(-1).to(torch.int32)  # [B*TQ*H*S]

    # Build (b, t, h, s) -> group id = (bh * M + chap), bh=(b*H + h), chap in [0,M)
    # We store query index within-batch (t) only.
    # Build t ids:
    t_ids = torch.arange(TQ, device=device, dtype=torch.int32)  # [TQ]
    # For each batch, repeat TQ
    t_ids = t_ids.repeat(B)  # [B*TQ]
    # For each (t), repeat H*S (matches flatten ordering b,t,h,s)
    t_ids = t_ids.repeat_interleave(H * S)  # [B*TQ*H*S]

    # Head ids: for one token, flatten heads then slots -> [H*S]
    h_ids = torch.arange(H, device=device, dtype=torch.int32).repeat_interleave(S)  # [H*S]
    # Repeat for all (B*TQ) query tokens
    h_ids = h_ids.repeat(B * TQ)  # [B*TQ*H*S]

    # Batch ids: for each batch, there are TQ*H*S edges
    b_ids = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(TQ * H * S)

    # valid chapters
    valid = (chapters >= 0) & (chapters < num_blocks)

    chapters = chapters[valid]
    t_ids = t_ids[valid]
    h_ids = h_ids[valid]
    b_ids = b_ids[valid]

    bh = b_ids * H + h_ids  # [N_edges]
    group = bh * num_blocks + chapters  # [N_edges]

    # Sort by (group, t) so accumulation order is deterministic and closer
    # to naive reference loops (increasing query index within each group).
    key_scale = int(TQ) + 1
    sort_key = group.to(torch.int64) * key_scale + t_ids.to(torch.int64)
    sort_idx = torch.argsort(sort_key)
    group_sorted = group[sort_idx]
    sorted_q = t_ids[sort_idx]

    G_total = B * H * num_blocks
    # group_offsets[g] = first index where group_sorted >= g
    # Use searchsorted on sorted groups.
    # NOTE: torch.searchsorted expects ascending group_sorted.
    group_sorted64 = group_sorted.to(torch.int64)
    needles = torch.arange(G_total + 1, device=device, dtype=torch.int64)
    group_offsets = torch.searchsorted(group_sorted64, needles)

    return InvertedIndex(
        sorted_q=sorted_q,
        group_offsets=group_offsets.to(torch.int32),
        B=B, TQ=TQ, H=H, M=num_blocks, S=S,
    )


@dataclass
class ChunkedIndex:
    inv: InvertedIndex
    chunk_group: torch.Tensor     # [num_chunks] int32 group id
    chunk_start: torch.Tensor     # [num_chunks] int32 start idx into inv.sorted_q
    chunk_end: torch.Tensor       # [num_chunks] int32 end idx into inv.sorted_q
    chunk_offsets: torch.Tensor   # [G_total+1] int32 offsets into chunk arrays
    num_chunks: int


def build_chunked_index(inv: InvertedIndex, chunk_size: int) -> ChunkedIndex:
    """
    Build chunk metadata for approach D.
    """
    device = inv.sorted_q.device
    G_total = inv.B * inv.H * inv.M

    # boundaries are small; do it on CPU for simplicity
    offsets_cpu = inv.group_offsets.cpu().tolist()

    chunk_group = []
    chunk_start = []
    chunk_end = []
    chunk_offsets = [0]

    for g in range(G_total):
        s = offsets_cpu[g]
        e = offsets_cpu[g + 1]
        n = e - s
        n_chunks = (n + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            cs = s + c * chunk_size
            ce = min(cs + chunk_size, e)
            chunk_group.append(g)
            chunk_start.append(cs)
            chunk_end.append(ce)
        chunk_offsets.append(len(chunk_group))

    # move to GPU
    chunk_group_t = torch.tensor(chunk_group, device=device, dtype=torch.int32)
    chunk_start_t = torch.tensor(chunk_start, device=device, dtype=torch.int32)
    chunk_end_t = torch.tensor(chunk_end, device=device, dtype=torch.int32)
    chunk_offsets_t = torch.tensor(chunk_offsets, device=device, dtype=torch.int32)

    return ChunkedIndex(
        inv=inv,
        chunk_group=chunk_group_t,
        chunk_start=chunk_start_t,
        chunk_end=chunk_end_t,
        chunk_offsets=chunk_offsets_t,
        num_chunks=len(chunk_group),
    )


# ---------------------------
# Python: high-level API
# ---------------------------

def memory_cross_attn_forward(
    q: torch.Tensor,                # [B, TQ, HQ, K]
    k: torch.Tensor,                # [B, TK, H,  K]
    v: torch.Tensor,                # [B, TK, H,  V]
    block_indices: torch.Tensor,    # [B, TQ, H, S]
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_cuda(q, "q"); _check_cuda(k, "k"); _check_cuda(v, "v"); _check_cuda(block_indices, "block_indices")
    _check_contiguous_lastdim(q, "q"); _check_contiguous_lastdim(k, "k"); _check_contiguous_lastdim(v, "v")
    _check_contiguous_lastdim(block_indices, "block_indices")

    B, TQ, HQ, Kdim = q.shape
    Bk, TK, H, Kk = k.shape
    Bv, TKv, Hv, Vdim = v.shape
    assert B == Bk == Bv, "Batch mismatch"
    assert TK == TKv, "KV length mismatch"
    assert H == Hv, "KV heads mismatch"
    assert Kk == Kdim, "Key dim mismatch"
    assert block_indices.shape[:3] == (B, TQ, H), "block_indices shape must be [B,TQ,H,S]"
    S = block_indices.shape[-1]
    assert TK % block_size == 0, "TK must be divisible by block_size (chapter size) for clean chapter blocks."
    M = TK // block_size

    if scale is None:
        scale = Kdim ** -0.5

    BK = min(256, _next_pow2(Kdim))
    BV = min(256, _next_pow2(Vdim))
    BS_PAD = _next_pow2(block_size)
    assert triton.cdiv(Kdim, BK) == 1, "Kdim > 256 not supported in this kernel template."

    G = HQ // H
    assert HQ % H == 0, "HQ must be multiple of H (GQA grouping)."

    NV = triton.cdiv(Vdim, BV)

    # Avoid compiling any code path that calls tl.dot with dims < 16 (notably when G=1).
    USE_DOT = (G >= 16) and (BK >= 16) and (BV >= 16) and (block_size >= 16)

    o = torch.empty((B, TQ, HQ, Vdim), device=q.device, dtype=v.dtype)
    lse = torch.empty((B, TQ, HQ), device=q.device, dtype=torch.float32)

    grid = (TQ, NV, B * H)
    mem_xattn_fwd_kernel[grid](
        q=q, k=k, v=v,
        o=o, lse=lse,
        scale=scale,
        block_indices=block_indices,
        TQ=TQ, TK=TK,
        H=H, HQ=HQ, G=G,
        Kdim=Kdim, Vdim=Vdim,
        S=S, BS=block_size, BS_PAD=BS_PAD,
        BK=BK, BV=BV,
        USE_DOT=USE_DOT,
    )
    return o, lse


def memory_cross_attn_preprocess_delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    _check_cuda(o, "o"); _check_cuda(do, "do")
    assert o.shape == do.shape
    B, TQ, HQ, Vdim = o.shape
    delta = torch.empty((B, TQ, HQ), device=o.device, dtype=torch.float32)

    BLOCK = _next_pow2(Vdim)
    # flatten to rows
    mem_xattn_bwd_preprocess_kernel[(B * TQ * HQ,)](
        o=o,
        do=do,
        delta=delta,
        BLOCK=BLOCK,
        Vdim=Vdim,
    )
    return delta


def memory_cross_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
    dkv_strategy: Literal["a", "b", "c", "d"] = "b",
    # B strategy params
    q_chunk_size: int = 1024,
    # D strategy params
    d_chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: dq, dk, dv
    """
    _check_cuda(q, "q"); _check_cuda(k, "k"); _check_cuda(v, "v")
    _check_cuda(o, "o"); _check_cuda(lse, "lse"); _check_cuda(do, "do"); _check_cuda(block_indices, "block_indices")

    B, TQ, HQ, Kdim = q.shape
    _, TK, H, _ = k.shape
    Vdim = v.shape[-1]
    S = block_indices.shape[-1]
    assert TK % block_size == 0
    M = TK // block_size

    if scale is None:
        scale = Kdim ** -0.5

    BK = min(256, _next_pow2(Kdim))
    BV = min(256, _next_pow2(Vdim))
    BS_PAD = _next_pow2(block_size)
    NV = triton.cdiv(Vdim, BV)
    G = HQ // H

    # Avoid compiling any code path that calls tl.dot with dims < 16 (notably when G=1).
    USE_DOT = (G >= 16) and (BK >= 16) and (BV >= 16) and (block_size >= 16)

    # delta
    delta = memory_cross_attn_preprocess_delta(o, do)

    # ---- dQ ----
    # Like NSA: if NV==1, compute dq directly without NV dimension; else accumulate then sum.
    if NV == 1:
        dq = torch.empty_like(q)
        grid = (TQ, NV, B * H)
        mem_xattn_bwd_dq_kernel[grid](
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
        dq_stride_nv = dq_part.stride(0)  # in elements
        grid = (TQ, NV, B * H)
        mem_xattn_bwd_dq_kernel[grid](
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

    # ---- dK / dV ----
    # We'll keep dk/dv in float32 accumulators, then cast to input dtype at the end.
    if dkv_strategy == "a":
        # block mask
        block_mask = torch.zeros((B, TQ, H, M), device=q.device, dtype=torch.int32)
        mem_xattn_block_mask_kernel[(TQ, B, H * S)](
            block_indices=block_indices,
            block_mask=block_mask,
            TQ=TQ, H=H, S=S, M=M,
        )
        dk_part = torch.empty((NV, *k.shape), device=q.device, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        grid = (NV, M, B * H)
        mem_xattn_bwd_dkv_a_kernel[grid](
            q=q, k=k, v=v,
            lse=lse, delta=delta, do=do,
            dk=dk_part, dv=dv,
            block_mask=block_mask,
            scale=scale,
            TQ=TQ, TK=TK,
            B=B, H=H, HQ=HQ, G=G,
            Kdim=Kdim, Vdim=Vdim,
            M=M, BS=block_size, BS_PAD=BS_PAD,
            BK=BK, BV=BV,
            USE_DOT=USE_DOT,
        )
        dk = dk_part.sum(0)

    elif dkv_strategy == "b":
        dk_part = torch.zeros((NV, *k.shape), device=q.device, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        num_q_chunks = (TQ + q_chunk_size - 1) // q_chunk_size
        grid = (NV, M, num_q_chunks * B * H)
        mem_xattn_bwd_dkv_b_atomic_kernel[grid](
            q=q, k=k, v=v,
            lse=lse, delta=delta, do=do,
            dk=dk_part, dv=dv,
            block_indices=block_indices,
            scale=scale,
            TQ=TQ, TK=TK,
            B=B, H=H, HQ=HQ, G=G,
            Kdim=Kdim, Vdim=Vdim,
            S=S, BS=block_size, BS_PAD=BS_PAD,
            BK=BK, BV=BV,
            CHUNK_Q=q_chunk_size,
            USE_DOT=USE_DOT,
        )
        dk = dk_part.sum(0)

    elif dkv_strategy == "c":
        inv = build_inverted_index(block_indices, num_blocks=M)
        dk_part = torch.empty((NV, *k.shape), device=q.device, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        grid = (NV, M, B * H)
        mem_xattn_bwd_dkv_c_invidx_kernel[grid](
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

    elif dkv_strategy == "d":
        inv = build_inverted_index(block_indices, num_blocks=M)
        cidx = build_chunked_index(inv, chunk_size=d_chunk_size)
        num_chunks = cidx.num_chunks

        dk_part = torch.empty((NV, *k.shape), device=q.device, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        if num_chunks == 0:
            dk = torch.zeros_like(k, dtype=torch.float32)
        else:
            # partial buffers (float32; can change to float16 to save memory if needed)
            dk_buf = torch.zeros((NV, num_chunks, block_size, BK), device=q.device, dtype=torch.float32)
            dv_buf = torch.zeros((NV, num_chunks, block_size, BV), device=q.device, dtype=torch.float32)

            # Keep large dimension on grid-x (CUDA grid-y is limited to ~65k).
            grid = (num_chunks, NV)
            mem_xattn_bwd_dkv_d_partial_kernel[grid](
                q=q, k=k, v=v,
                lse=lse, delta=delta, do=do,
                dk_part=dk_buf, dv_part=dv_buf,
                sorted_q=inv.sorted_q,
                chunk_group=cidx.chunk_group,
                chunk_start=cidx.chunk_start,
                chunk_end=cidx.chunk_end,
                scale=scale,
                TQ=TQ, TK=TK,
                B=B, H=H, HQ=HQ, G=G,
                Kdim=Kdim, Vdim=Vdim,
                M=M, BS=block_size, BS_PAD=BS_PAD,
                BK=BK, BV=BV,
                CHUNK_Q=d_chunk_size,
                USE_DOT=USE_DOT,
                NUM_CHUNKS=num_chunks,
            )

            # reduction
            G_total = B * H * M
            grid = (G_total, NV)
            mem_xattn_bwd_dkv_d_reduce_kernel[grid](
                dk_part=dk_buf, dv_part=dv_buf,
                dk=dk_part, dv=dv,
                chunk_offsets=cidx.chunk_offsets,
                scale_dummy=0.0,
                num_chunks=num_chunks,
                TQ=TQ, TK=TK,
                B=B, H=H,
                Kdim=Kdim, Vdim=Vdim,
                M=M, BS=block_size, BS_PAD=BS_PAD,
                BK=BK, BV=BV,
            )
            dk = dk_part.sum(0)
    else:
        raise ValueError(f"Unknown dkv_strategy={dkv_strategy}")

    # cast grads to input dtype
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
    return dq.to(q.dtype), dk, dv


# ---------------------------
# autograd.Function wrapper
# ---------------------------

class MemoryCrossAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_size: int, scale: Optional[float], dkv_strategy: str,
                q_chunk_size: int, d_chunk_size: int):
        o, lse = memory_cross_attn_forward(q, k, v, block_indices, block_size, scale)
        ctx.save_for_backward(q, k, v, o, lse, block_indices)
        ctx.block_size = block_size
        ctx.scale = scale
        ctx.dkv_strategy = dkv_strategy
        ctx.q_chunk_size = q_chunk_size
        ctx.d_chunk_size = d_chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices = ctx.saved_tensors
        dq, dk, dv = memory_cross_attn_backward(
            q=q, k=k, v=v,
            o=o, lse=lse, do=do,
            block_indices=block_indices,
            block_size=ctx.block_size,
            scale=ctx.scale,
            dkv_strategy=ctx.dkv_strategy,
            q_chunk_size=ctx.q_chunk_size,
            d_chunk_size=ctx.d_chunk_size,
        )
        return dq, dk, dv, None, None, None, None, None, None


def memory_cross_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
    dkv_strategy: Literal["a", "b", "c", "d"] = "b",
    q_chunk_size: int = 1024,
    d_chunk_size: int = 256,
) -> torch.Tensor:
    return MemoryCrossAttnFunction.apply(
        q, k, v, block_indices, block_size, scale, dkv_strategy, q_chunk_size, d_chunk_size
    )
