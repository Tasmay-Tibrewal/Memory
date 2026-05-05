"""
Exact weighted MoE sparse attention for token routing.

Design:
- Unweighted path delegates to the stable v1 wrapper unchanged.
- Weighted path is a single logical fused kernel path with exact MoE semantics:

      output = sum_s weight_s * softmax(Q K_s^T / sqrt(d)) V_s

- Each routed slot keeps its own softmax normalization domain.
- Forward saves compact slot-local logsumexp state only.
- Backward uses custom Triton kernels for dQ, dK/dV, and dWeights.

This replaces the previous v4 wrapper that internally called v1 once per slot.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl


_V1_MODULE = None


def _load_v1_module():
    global _V1_MODULE
    if _V1_MODULE is not None:
        return _V1_MODULE

    kernel_file = Path(__file__).resolve().with_name("kernel_v1.py")
    module_name = "_memory_token_routing_v4_base_v1"
    spec = importlib.util.spec_from_file_location(module_name, str(kernel_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build module spec for {kernel_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _V1_MODULE = module
    return module


def FSA_topk_sparse_attention_bthd(*args, **kwargs):
    """Delegate unweighted sparse attention to the stable v1 implementation."""
    module = _load_v1_module()
    return module.FSA_topk_sparse_attention_bthd(*args, **kwargs)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _parse_num_warps(raw: Optional[str], default: tuple[int, ...]) -> tuple[int, ...]:
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


@triton.jit
def mm_or_fallback(a, b, USE_DOT: tl.constexpr):
    if USE_DOT:
        return tl.dot(a, b).to(tl.float32)
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)
    return tl.sum(a32[:, :, None] * b32[None, :, :], axis=1)


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** (int(x - 1).bit_length())


def _validate_qkv(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    block_size: int,
):
    if q_bthd.ndim != 4 or k_bthd.ndim != 4 or v_bthd.ndim != 4:
        raise ValueError("q/k/v must be rank-4 [B,T,H,D].")
    if k_bthd.shape != v_bthd.shape:
        raise ValueError("k and v must have identical [B,Tk,H,D] shape.")
    if q_bthd.shape[0] != k_bthd.shape[0]:
        raise ValueError("q and k/v batch size must match.")
    if q_bthd.shape[3] != k_bthd.shape[3]:
        raise ValueError("q and k/v head dim must match.")
    if q_bthd.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Weighted FSA requires fp16/bf16; got {q_bthd.dtype}.")
    if k_bthd.dtype != q_bthd.dtype or v_bthd.dtype != q_bthd.dtype:
        raise ValueError("q/k/v dtype must match.")
    if block_size not in {32, 64, 128, 256, 512, 1024}:
        raise ValueError(
            f"Weighted FSA supports block_size in {{32,64,128,256,512,1024}}; got {block_size}."
        )
    if not q_bthd.is_cuda:
        raise ValueError("Weighted fused v4 requires CUDA tensors.")
    if not q_bthd.is_contiguous() or not k_bthd.is_contiguous() or not v_bthd.is_contiguous():
        raise ValueError("q/k/v must be contiguous.")


def _canonicalize_block_indices(
    *,
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    block_indices_bths: Optional[torch.Tensor],
    topk_idx_hns: Optional[torch.Tensor],
) -> torch.Tensor:
    B, Tq, HQ, _ = q_bthd.shape
    _, _, HK, _ = k_bthd.shape
    if HQ % HK != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HK ({HK}) for GQA.")
    gqa_deg = HQ // HK

    if topk_idx_hns is not None:
        if topk_idx_hns.ndim != 3:
            raise ValueError("topk_idx_hns must be rank-3 [HK or HQ, B*Tq, topk].")
        if topk_idx_hns.shape[1] != (B * Tq):
            raise ValueError(
                f"topk_idx_hns shape mismatch: expected second dim B*Tq={B*Tq}, got {tuple(topk_idx_hns.shape)}."
            )
        if topk_idx_hns.shape[0] == HK:
            topk_idx = topk_idx_hns
        elif topk_idx_hns.shape[0] == HQ:
            grouped = topk_idx_hns.view(HK, gqa_deg, B * Tq, topk_idx_hns.shape[-1]).contiguous()
            ref = grouped[:, 0:1, :, :]
            if not torch.equal(grouped, ref.expand_as(grouped)):
                raise ValueError(
                    "Weighted FSA requires shared routing inside each GQA group when HQ routes are provided."
                )
            topk_idx = grouped[:, 0, :, :]
        else:
            raise ValueError(
                f"topk_idx_hns first dim must be HK={HK} or HQ={HQ}, got {topk_idx_hns.shape[0]}."
            )
        return topk_idx.transpose(0, 1).reshape(B, Tq, HK, topk_idx.shape[-1]).contiguous().to(torch.int32)

    if block_indices_bths is None:
        raise ValueError("Either block_indices_bths or topk_idx_hns must be provided.")
    if block_indices_bths.ndim != 4:
        raise ValueError("block_indices_bths must be rank-4 [B,Tq,H,topk].")
    if block_indices_bths.shape[0] != B or block_indices_bths.shape[1] != Tq:
        raise ValueError("block_indices_bths prefix must match q [B,Tq,...].")

    if block_indices_bths.shape[2] == HK:
        return block_indices_bths.contiguous().to(torch.int32)

    if block_indices_bths.shape[2] != HQ:
        raise ValueError(
            f"block_indices_bths third dim must be HK={HK} or HQ={HQ}, got {block_indices_bths.shape[2]}."
        )

    grouped = block_indices_bths.view(B, Tq, HK, gqa_deg, block_indices_bths.shape[-1]).contiguous()
    ref = grouped[:, :, :, 0:1, :]
    if not torch.equal(grouped, ref.expand_as(grouped)):
        raise ValueError(
            "Weighted FSA requires shared routing inside each GQA group when HQ routes are provided."
        )
    return grouped[:, :, :, 0, :].contiguous().to(torch.int32)


@dataclass
class WeightedInvertedIndex:
    sorted_q: torch.Tensor
    sorted_s: torch.Tensor
    group_offsets: torch.Tensor
    B: int
    TQ: int
    H: int
    M: int
    S: int


@dataclass
class WeightedChunkedIndex:
    inv: WeightedInvertedIndex
    chunk_group: torch.Tensor
    chunk_start: torch.Tensor
    chunk_end: torch.Tensor
    chunk_offsets: torch.Tensor
    num_chunks: int


def build_weighted_inverted_index(
    block_indices: torch.Tensor,
    num_blocks: int,
) -> WeightedInvertedIndex:
    B, TQ, H, S = block_indices.shape
    device = block_indices.device

    chapters = block_indices.reshape(-1).to(torch.int32)
    t_ids = torch.arange(TQ, device=device, dtype=torch.int32).repeat(B).repeat_interleave(H * S)
    h_ids = torch.arange(H, device=device, dtype=torch.int32).repeat_interleave(S).repeat(B * TQ)
    s_ids = torch.arange(S, device=device, dtype=torch.int32).repeat(H).repeat(B * TQ)
    b_ids = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(TQ * H * S)

    valid = (chapters >= 0) & (chapters < num_blocks)
    chapters = chapters[valid]
    t_ids = t_ids[valid]
    h_ids = h_ids[valid]
    s_ids = s_ids[valid]
    b_ids = b_ids[valid]

    bh = b_ids * H + h_ids
    group = bh * num_blocks + chapters

    key_scale = int(TQ * S) + 1
    sort_key = group.to(torch.int64) * key_scale + (t_ids.to(torch.int64) * int(S) + s_ids.to(torch.int64))
    sort_idx = torch.argsort(sort_key)
    group_sorted = group[sort_idx]
    sorted_q = t_ids[sort_idx]
    sorted_s = s_ids[sort_idx]

    g_total = B * H * num_blocks
    group_sorted64 = group_sorted.to(torch.int64)
    needles = torch.arange(g_total + 1, device=device, dtype=torch.int64)
    group_offsets = torch.searchsorted(group_sorted64, needles)

    return WeightedInvertedIndex(
        sorted_q=sorted_q,
        sorted_s=sorted_s,
        group_offsets=group_offsets.to(torch.int32),
        B=B,
        TQ=TQ,
        H=H,
        M=num_blocks,
        S=S,
    )


def build_weighted_chunked_index(inv: WeightedInvertedIndex, chunk_size: int) -> WeightedChunkedIndex:
    device = inv.sorted_q.device
    g_total = inv.B * inv.H * inv.M
    offsets_cpu = inv.group_offsets.cpu().tolist()

    chunk_group = []
    chunk_start = []
    chunk_end = []
    chunk_offsets = [0]

    for g in range(g_total):
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

    chunk_group_t = torch.tensor(chunk_group, device=device, dtype=torch.int32)
    chunk_start_t = torch.tensor(chunk_start, device=device, dtype=torch.int32)
    chunk_end_t = torch.tensor(chunk_end, device=device, dtype=torch.int32)
    chunk_offsets_t = torch.tensor(chunk_offsets, device=device, dtype=torch.int32)

    return WeightedChunkedIndex(
        inv=inv,
        chunk_group=chunk_group_t,
        chunk_start=chunk_start_t,
        chunk_end=chunk_end_t,
        chunk_offsets=chunk_offsets_t,
        num_chunks=len(chunk_group),
    )


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV", "G", "USE_DOT"])
@triton.jit
def mem_xattn_weighted_fwd_kernel(
    q,
    k,
    v,
    o,
    lse_slots,
    chapter_weights,
    scale,
    block_indices,
    TQ: tl.constexpr,
    TK: tl.constexpr,
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
):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_tok = i_b * TQ + i_t
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim
    bi_base = (i_b * TQ + i_t) * H * S + i_h * S
    w_base = (i_b * TQ + i_t) * S

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

    b_o = tl.zeros([G, BV], dtype=tl.float32)
    offs_local = tl.arange(0, BS_PAD)

    for s in range(S):
        chap = tl.load(block_indices + bi_base + s).to(tl.int32)
        tok0 = chap * BS
        if tok0 >= 0 and tok0 < TK:
            p_k = tl.make_block_ptr(
                base=k + k_head_base + tok0 * H * Kdim,
                shape=(Kdim, BS),
                strides=(1, H * Kdim),
                offsets=(0, 0),
                block_shape=(BK, BS_PAD),
                order=(0, 1),
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

            b_s = mm_or_fallback(b_q, b_k, USE_DOT=USE_DOT)
            offs_global = tok0 + offs_local
            valid = (offs_local < BS) & (offs_global < TK)
            b_s = tl.where(valid[None, :], b_s, float("-inf"))

            b_m = tl.max(b_s, axis=1)
            b_p = tl.exp(b_s - b_m[:, None])
            b_p = tl.where(valid[None, :], b_p, 0.0)
            b_acc = tl.sum(b_p, axis=1)
            b_slot_o = tl.where(
                b_acc[:, None] > 0,
                mm_or_fallback(b_p.to(b_q.dtype), b_v, USE_DOT=USE_DOT) / b_acc[:, None],
                0.0,
            )

            w = tl.load(chapter_weights + w_base + s).to(tl.float32)
            b_o += b_slot_o * w

            if i_v == 0:
                b_lse = tl.where(b_acc > 0, b_m + tl.log(b_acc), float("-inf"))
                lse_ptrs = lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + s
                tl.store(lse_ptrs, b_lse.to(lse_slots.dtype.element_ty))
        elif i_v == 0:
            lse_ptrs = lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + s
            tl.store(lse_ptrs, tl.full((G,), float("-inf"), dtype=tl.float32))

    p_o = tl.make_block_ptr(
        base=o + q_tok * HQ * Vdim,
        shape=(HQ, Vdim),
        strides=(Vdim, 1),
        offsets=(i_h * G, i_v * BV),
        block_shape=(G, BV),
        order=(1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV", "G", "USE_DOT"])
@triton.jit
def mem_xattn_weighted_bwd_dq_kernel(
    q,
    k,
    v,
    lse_slots,
    do,
    dq,
    dw,
    chapter_weights,
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
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_tok = i_b * TQ + i_t
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim
    bi_base = (i_b * TQ + i_t) * H * S + i_h * S
    w_base = (i_b * TQ + i_t) * S

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
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)

    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    offs_local = tl.arange(0, BS_PAD)

    for s in range(S):
        chap = tl.load(block_indices + bi_base + s).to(tl.int32)
        tok0 = chap * BS
        if tok0 >= 0 and tok0 < TK:
            w = tl.load(chapter_weights + w_base + s).to(tl.float32)
            p_k = tl.make_block_ptr(
                base=k + k_head_base + tok0 * H * Kdim,
                shape=(Kdim, BS),
                strides=(1, H * Kdim),
                offsets=(0, 0),
                block_shape=(BK, BS_PAD),
                order=(0, 1),
            )
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
            b_lse = tl.load(lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + s)

            b_s = mm_or_fallback(b_q, b_k, USE_DOT=USE_DOT)
            offs_global = tok0 + offs_local
            valid = (offs_local < BS) & (offs_global < TK)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(valid[None, :], b_p, 0.0)

            b_out = mm_or_fallback(b_p.to(b_vt.dtype), tl.trans(b_vt), USE_DOT=USE_DOT)
            b_dw_part = tl.sum(b_out.to(tl.float32) * b_do, axis=1)
            tl.atomic_add(dw + (i_b * TQ + i_t) * S + s, tl.sum(b_dw_part))

            b_dp = mm_or_fallback((b_do * w).to(b_vt.dtype), b_vt, USE_DOT=USE_DOT)
            b_delta = tl.sum(b_p * b_dp.to(tl.float32), axis=1)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += mm_or_fallback(b_ds.to(b_k.dtype), tl.trans(b_k), USE_DOT=USE_DOT)

    b_dq *= scale

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


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV", "G", "USE_DOT"])
@triton.jit
def mem_xattn_weighted_bwd_dkv_invidx_kernel(
    q,
    k,
    v,
    lse_slots,
    do,
    dk,
    dv,
    chapter_weights,
    sorted_q,
    sorted_s,
    group_offsets,
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
    USE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_blk = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H
    m_blocks = TK // BS
    group = (i_bh * m_blocks + i_blk).to(tl.int32)

    start = tl.load(group_offsets + group)
    end = tl.load(group_offsets + group + 1)
    tok0 = i_blk * BS

    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

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
    offs_bs = tl.arange(0, BS_PAD)

    for idx in range(start, end):
        i_t = tl.load(sorted_q + idx).to(tl.int32)
        i_s = tl.load(sorted_s + idx).to(tl.int32)
        q_tok = i_b * TQ + i_t
        w = tl.load(chapter_weights + (i_b * TQ + i_t) * S + i_s).to(tl.float32)

        p_q = tl.make_block_ptr(
            base=q + q_tok * HQ * Kdim,
            shape=(HQ, Kdim),
            strides=(Kdim, 1),
            offsets=(i_h * G, 0),
            block_shape=(G, BK),
            order=(1, 0),
        )
        p_do = tl.make_block_ptr(
            base=do + q_tok * HQ * Vdim,
            shape=(HQ, Vdim),
            strides=(Vdim, 1),
            offsets=(i_h * G, i_v * BV),
            block_shape=(G, BV),
            order=(1, 0),
        )

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_lse = tl.load(lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + i_s)

        b_s = mm_or_fallback(b_k, tl.trans(b_q.to(b_k.dtype)), USE_DOT=USE_DOT)
        row_valid = (offs_bs < BS)[:, None]
        b_p = tl.exp(b_s - b_lse[None, :])
        b_p = tl.where(row_valid, b_p, 0.0)

        b_do_w = b_do * w
        if G == 1:
            b_p_vec = tl.sum(b_p, axis=1)
            b_do_vec = tl.sum(b_do_w, axis=0)
            b_dv += b_p_vec[:, None] * b_do_vec[None, :]
        else:
            b_dv += mm_or_fallback(b_p.to(b_do_w.dtype), b_do_w, USE_DOT=USE_DOT)

        b_dp = mm_or_fallback(b_v, tl.trans(b_do_w.to(b_v.dtype)), USE_DOT=USE_DOT)
        b_delta = tl.sum(b_p * b_dp.to(tl.float32), axis=0)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
        if G == 1:
            b_ds_vec = tl.sum(b_ds, axis=1)
            b_q_vec = tl.sum(b_q, axis=0)
            b_dk += b_ds_vec[:, None] * b_q_vec[None, :]
        else:
            b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q.to(b_q.dtype), USE_DOT=USE_DOT)

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


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV", "CHUNK_Q", "G", "USE_DOT"])
@triton.jit
def mem_xattn_weighted_bwd_dkv_chunked_partial_kernel(
    q,
    k,
    v,
    lse_slots,
    do,
    dk_part,
    dv_part,
    chapter_weights,
    sorted_q,
    sorted_s,
    chunk_group,
    chunk_start,
    chunk_end,
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
    NUM_CHUNKS: tl.constexpr,
):
    c_id = tl.program_id(0)
    i_v = tl.program_id(1)

    group = tl.load(chunk_group + c_id).to(tl.int32)
    start = tl.load(chunk_start + c_id).to(tl.int32)
    end = tl.load(chunk_end + c_id).to(tl.int32)

    M = TK // BS
    bh = group // M
    i_blk = group - bh * M
    i_b = bh // H
    i_h = bh - i_b * H

    tok0 = i_blk * BS

    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim

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
    offs_bs = tl.arange(0, BS_PAD)

    for idx in range(start, end):
        i_t = tl.load(sorted_q + idx).to(tl.int32)
        i_s = tl.load(sorted_s + idx).to(tl.int32)
        q_tok = i_b * TQ + i_t
        w = tl.load(chapter_weights + (i_b * TQ + i_t) * S + i_s).to(tl.float32)

        p_q = tl.make_block_ptr(
            base=q + q_tok * HQ * Kdim,
            shape=(HQ, Kdim),
            strides=(Kdim, 1),
            offsets=(i_h * G, 0),
            block_shape=(G, BK),
            order=(1, 0),
        )
        p_do = tl.make_block_ptr(
            base=do + q_tok * HQ * Vdim,
            shape=(HQ, Vdim),
            strides=(Vdim, 1),
            offsets=(i_h * G, i_v * BV),
            block_shape=(G, BV),
            order=(1, 0),
        )

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_lse = tl.load(lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + i_s)

        b_s = mm_or_fallback(b_k, tl.trans(b_q.to(b_k.dtype)), USE_DOT=USE_DOT)
        row_valid = (offs_bs < BS)[:, None]
        b_p = tl.exp(b_s - b_lse[None, :])
        b_p = tl.where(row_valid, b_p, 0.0)

        b_do_w = b_do * w
        if G == 1:
            b_p_vec = tl.sum(b_p, axis=1)
            b_do_vec = tl.sum(b_do_w, axis=0)
            b_dv += b_p_vec[:, None] * b_do_vec[None, :]
        else:
            b_dv += mm_or_fallback(b_p.to(b_do_w.dtype), b_do_w, USE_DOT=USE_DOT)

        b_dp = mm_or_fallback(b_v, tl.trans(b_do_w.to(b_v.dtype)), USE_DOT=USE_DOT)
        b_delta = tl.sum(b_p * b_dp.to(tl.float32), axis=0)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
        if G == 1:
            b_ds_vec = tl.sum(b_ds, axis=1)
            b_q_vec = tl.sum(b_q, axis=0)
            b_dk += b_ds_vec[:, None] * b_q_vec[None, :]
        else:
            b_dk += mm_or_fallback(b_ds.to(b_q.dtype), b_q.to(b_q.dtype), USE_DOT=USE_DOT)

    dk_base = dk_part + (i_v * NUM_CHUNKS + c_id) * (BS * BK)
    dv_base = dv_part + (i_v * NUM_CHUNKS + c_id) * (BS * BV)
    offs_t = tl.arange(0, BS_PAD)[:, None]
    offs_k = tl.arange(0, BK)[None, :]
    offs_v = tl.arange(0, BV)[None, :]
    row_mask = offs_t < BS
    tl.store(dk_base + offs_t * BK + offs_k, b_dk, mask=row_mask)
    tl.store(dv_base + offs_t * BV + offs_v, b_dv, mask=row_mask)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV", "G", "USE_DOT"])
@triton.jit
def mem_xattn_weighted_bwd_dw_kernel(
    q,
    k,
    v,
    lse_slots,
    do,
    dw,
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
    USE_DOT: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_tok = i_b * TQ + i_t
    k_head_base = (i_b * TK * H + i_h) * Kdim
    v_head_base = (i_b * TK * H + i_h) * Vdim
    bi_base = (i_b * TQ + i_t) * H * S + i_h * S

    p_q = tl.make_block_ptr(
        base=q + q_tok * HQ * Kdim,
        shape=(HQ, Kdim),
        strides=(Kdim, 1),
        offsets=(i_h * G, 0),
        block_shape=(G, BK),
        order=(1, 0),
    )
    p_do = tl.make_block_ptr(
        base=do + q_tok * HQ * Vdim,
        shape=(HQ, Vdim),
        strides=(Vdim, 1),
        offsets=(i_h * G, i_v * BV),
        block_shape=(G, BV),
        order=(1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    offs_local = tl.arange(0, BS_PAD)

    for s in range(S):
        chap = tl.load(block_indices + bi_base + s).to(tl.int32)
        tok0 = chap * BS
        if tok0 >= 0 and tok0 < TK:
            p_k = tl.make_block_ptr(
                base=k + k_head_base + tok0 * H * Kdim,
                shape=(Kdim, BS),
                strides=(1, H * Kdim),
                offsets=(0, 0),
                block_shape=(BK, BS_PAD),
                order=(0, 1),
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
            b_lse = tl.load(lse_slots + (q_tok * HQ + i_h * G + tl.arange(0, G)) * S + s)

            b_s = mm_or_fallback(b_q, b_k, USE_DOT=USE_DOT)
            offs_global = tok0 + offs_local
            valid = (offs_local < BS) & (offs_global < TK)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(valid[None, :], b_p, 0.0)
            b_out = mm_or_fallback(b_p.to(b_q.dtype), b_v, USE_DOT=USE_DOT)
            b_contrib = tl.sum(b_out.to(tl.float32) * b_do, axis=1)
            tl.atomic_add(dw + (i_b * TQ + i_t) * S + s, tl.sum(b_contrib))


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["BS", "BK", "BV"])
@triton.jit
def mem_xattn_weighted_bwd_dkv_reduce_kernel(
    dk_part,
    dv_part,
    dk,
    dv,
    chunk_offsets,
    scale_dummy,
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
    group = tl.program_id(0).to(tl.int32)
    i_v = tl.program_id(1)

    start = tl.load(chunk_offsets + group)
    end = tl.load(chunk_offsets + group + 1)

    bh = group // M
    i_blk = group - bh * M
    i_b = bh // H
    i_h = bh - i_b * H
    tok0 = i_blk * BS

    b_dk = tl.zeros([BS_PAD, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS_PAD, BV], dtype=tl.float32)

    for c_id in range(start, end):
        dk_base = dk_part + (i_v * num_chunks + c_id) * (BS * BK)
        dv_base = dv_part + (i_v * num_chunks + c_id) * (BS * BV)
        offs_t = tl.arange(0, BS_PAD)[:, None]
        offs_k = tl.arange(0, BK)[None, :]
        offs_v = tl.arange(0, BV)[None, :]
        row_mask = offs_t < BS
        b_dk += tl.load(dk_base + offs_t * BK + offs_k, mask=row_mask, other=0.0).to(tl.float32)
        b_dv += tl.load(dv_base + offs_t * BV + offs_v, mask=row_mask, other=0.0).to(tl.float32)

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


def _weighted_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    chapter_weights: torch.Tensor,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, TQ, HQ, Kdim = q.shape
    _, TK, H, _ = k.shape
    Vdim = v.shape[-1]
    S = block_indices.shape[-1]
    if TK % block_size != 0:
        raise ValueError(
            f"Weighted v4 requires routed memory length TK ({TK}) to be divisible by block_size ({block_size})."
        )
    if HQ % H != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HK ({H}) for GQA.")

    BK = min(256, _next_pow2(Kdim))
    BV = min(256, _next_pow2(Vdim))
    BS_PAD = _next_pow2(block_size)
    G = HQ // H
    NV = triton.cdiv(Vdim, BV)
    USE_DOT = (G >= 16) and (BK >= 16) and (BV >= 16) and (block_size >= 16)

    o = torch.empty((B, TQ, HQ, Vdim), device=q.device, dtype=v.dtype)
    lse_slots = torch.empty((B, TQ, HQ, S), device=q.device, dtype=torch.float32)

    grid = (TQ, NV, B * H)
    mem_xattn_weighted_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse_slots=lse_slots,
        chapter_weights=chapter_weights,
        scale=scale,
        block_indices=block_indices,
        TQ=TQ,
        TK=TK,
        H=H,
        HQ=HQ,
        G=G,
        Kdim=Kdim,
        Vdim=Vdim,
        S=S,
        BS=block_size,
        BS_PAD=BS_PAD,
        BK=BK,
        BV=BV,
        USE_DOT=USE_DOT,
    )
    return o, lse_slots


def _weighted_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    chapter_weights: torch.Tensor,
    lse_slots: torch.Tensor,
    do: torch.Tensor,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, TQ, HQ, Kdim = q.shape
    _, TK, H, _ = k.shape
    Vdim = v.shape[-1]
    S = block_indices.shape[-1]
    G = HQ // H

    BK = min(256, _next_pow2(Kdim))
    BV = min(256, _next_pow2(Vdim))
    BS_PAD = _next_pow2(block_size)
    NV = triton.cdiv(Vdim, BV)
    USE_DOT = (G >= 16) and (BK >= 16) and (BV >= 16) and (block_size >= 16)

    if NV == 1:
        dq = torch.empty_like(q)
        dw = torch.zeros((B, TQ, S), device=q.device, dtype=torch.float32)
        grid = (TQ, NV, B * H)
        mem_xattn_weighted_bwd_dq_kernel[grid](
            q=q,
            k=k,
            v=v,
            lse_slots=lse_slots,
            do=do,
            dq=dq,
            dw=dw,
            chapter_weights=chapter_weights,
            scale=scale,
            block_indices=block_indices,
            TQ=TQ,
            TK=TK,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            Kdim=Kdim,
            Vdim=Vdim,
            S=S,
            BS=block_size,
            BS_PAD=BS_PAD,
            BK=BK,
            BV=BV,
            USE_DOT=USE_DOT,
            DQ_HAS_NV=False,
            DQ_STRIDE_NV=0,
        )
    else:
        dq_part = torch.empty((NV, *q.shape), device=q.device, dtype=torch.float32)
        dw = torch.zeros((B, TQ, S), device=q.device, dtype=torch.float32)
        grid = (TQ, NV, B * H)
        mem_xattn_weighted_bwd_dq_kernel[grid](
            q=q,
            k=k,
            v=v,
            lse_slots=lse_slots,
            do=do,
            dq=dq_part,
            dw=dw,
            chapter_weights=chapter_weights,
            scale=scale,
            block_indices=block_indices,
            TQ=TQ,
            TK=TK,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            Kdim=Kdim,
            Vdim=Vdim,
            S=S,
            BS=block_size,
            BS_PAD=BS_PAD,
            BK=BK,
            BV=BV,
            USE_DOT=USE_DOT,
            DQ_HAS_NV=True,
            DQ_STRIDE_NV=dq_part.stride(0),
        )
        dq = dq_part.sum(0).to(q.dtype)

    M = TK // block_size
    dkv_mode = os.getenv("MEM_XATTN_WEIGHTED_DKV_MODE", "direct").strip().lower()
    if dkv_mode not in ("direct", "chunked"):
        dkv_mode = "direct"
    inv = build_weighted_inverted_index(block_indices, num_blocks=M)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dk_part = torch.empty((NV, *k.shape), device=q.device, dtype=torch.float32)

    if dkv_mode == "direct":
        grid = (NV, M, B * H)
        mem_xattn_weighted_bwd_dkv_invidx_kernel[grid](
            q=q,
            k=k,
            v=v,
            lse_slots=lse_slots,
            do=do,
            dk=dk_part,
            dv=dv,
            chapter_weights=chapter_weights,
            sorted_q=inv.sorted_q,
            sorted_s=inv.sorted_s,
            group_offsets=inv.group_offsets,
            scale=scale,
            TQ=TQ,
            TK=TK,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            Kdim=Kdim,
            Vdim=Vdim,
            S=S,
            BS=block_size,
            BS_PAD=BS_PAD,
            BK=BK,
            BV=BV,
            USE_DOT=USE_DOT,
        )
        dk = dk_part.sum(0)
    else:
        edge_chunk_size = int(os.getenv("MEM_XATTN_WEIGHTED_DKV_EDGE_CHUNK", "256"))
        if edge_chunk_size not in (64, 128, 256, 512, 1024):
            edge_chunk_size = 256
        cidx = build_weighted_chunked_index(inv, chunk_size=edge_chunk_size)

        if cidx.num_chunks == 0:
            dk = torch.zeros_like(k, dtype=torch.float32)
        else:
            dk_buf = torch.zeros((NV, cidx.num_chunks, block_size, BK), device=q.device, dtype=torch.float32)
            dv_buf = torch.zeros((NV, cidx.num_chunks, block_size, BV), device=q.device, dtype=torch.float32)

            grid = (cidx.num_chunks, NV)
            mem_xattn_weighted_bwd_dkv_chunked_partial_kernel[grid](
                q=q,
                k=k,
                v=v,
                lse_slots=lse_slots,
                do=do,
                dk_part=dk_buf,
                dv_part=dv_buf,
                chapter_weights=chapter_weights,
                sorted_q=inv.sorted_q,
                sorted_s=inv.sorted_s,
                chunk_group=cidx.chunk_group,
                chunk_start=cidx.chunk_start,
                chunk_end=cidx.chunk_end,
                scale=scale,
                TQ=TQ,
                TK=TK,
                B=B,
                H=H,
                HQ=HQ,
                G=G,
                Kdim=Kdim,
                Vdim=Vdim,
                S=S,
                BS=block_size,
                BS_PAD=BS_PAD,
                BK=BK,
                BV=BV,
                CHUNK_Q=edge_chunk_size,
                USE_DOT=USE_DOT,
                NUM_CHUNKS=cidx.num_chunks,
            )

            g_total = B * H * M
            grid = (g_total, NV)
            mem_xattn_weighted_bwd_dkv_reduce_kernel[grid](
                dk_part=dk_buf,
                dv_part=dv_buf,
                dk=dk_part,
                dv=dv,
                chunk_offsets=cidx.chunk_offsets,
                scale_dummy=0.0,
                num_chunks=cidx.num_chunks,
                TQ=TQ,
                TK=TK,
                B=B,
                H=H,
                Kdim=Kdim,
                Vdim=Vdim,
                M=M,
                BS=block_size,
                BS_PAD=BS_PAD,
                BK=BK,
                BV=BV,
            )
            dk = dk_part.sum(0)

    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    return dq.to(q.dtype), dk, dv, dw.to(chapter_weights.dtype)


class _WeightedFSATopkSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_bthd: torch.Tensor,
        k_bthd: torch.Tensor,
        v_bthd: torch.Tensor,
        block_indices_bths: Optional[torch.Tensor],
        chapter_weights_bts: torch.Tensor,
        block_size: int,
        softmax_scale: Optional[float] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        topk_idx_hns: Optional[torch.Tensor] = None,
        assume_sorted_topk: bool = False,
        disable_causal_mask: bool = False,
    ):
        del assume_sorted_topk, disable_causal_mask

        _validate_qkv(q_bthd, k_bthd, v_bthd, block_size)
        if chapter_weights_bts.ndim != 3:
            raise ValueError(
                f"chapter_weights_bts must be rank-3 [B,Tq,topk], got {tuple(chapter_weights_bts.shape)}."
            )
        if chapter_weights_bts.shape[0] != q_bthd.shape[0] or chapter_weights_bts.shape[1] != q_bthd.shape[1]:
            raise ValueError(
                "chapter_weights_bts prefix must match q [B,Tq,...]. "
                f"got q={tuple(q_bthd.shape)}, chapter_weights={tuple(chapter_weights_bts.shape)}"
            )
        if not chapter_weights_bts.is_contiguous():
            chapter_weights_bts = chapter_weights_bts.contiguous()

        if cu_seqlens_q is not None or cu_seqlens_k is not None:
            raise ValueError("Weighted fused v4 currently supports only dense BTHD inputs without cu_seqlens.")

        block_indices = _canonicalize_block_indices(
            q_bthd=q_bthd,
            k_bthd=k_bthd,
            block_indices_bths=block_indices_bths,
            topk_idx_hns=topk_idx_hns,
        )
        if chapter_weights_bts.shape[-1] != block_indices.shape[-1]:
            raise ValueError(
                f"chapter_weights topk ({chapter_weights_bts.shape[-1]}) must match routing topk ({block_indices.shape[-1]})."
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q_bthd.shape[-1])

        output, lse_slots = _weighted_forward(
            q=q_bthd,
            k=k_bthd,
            v=v_bthd,
            block_indices=block_indices,
            chapter_weights=chapter_weights_bts,
            block_size=int(block_size),
            scale=float(softmax_scale),
        )

        ctx.save_for_backward(q_bthd, k_bthd, v_bthd, block_indices, chapter_weights_bts, lse_slots)
        ctx.block_size = int(block_size)
        ctx.softmax_scale = float(softmax_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, block_indices, chapter_weights, lse_slots = ctx.saved_tensors
        dq, dk, dv, dw = _weighted_backward(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            chapter_weights=chapter_weights,
            lse_slots=lse_slots,
            do=grad_output.contiguous(),
            block_size=ctx.block_size,
            scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, dw, None, None, None, None, None, None, None


def FSA_topk_sparse_attention_weighted_bthd(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    block_indices_bths: Optional[torch.Tensor],
    chapter_weights_bts: torch.Tensor,
    block_size: int,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    topk_idx_hns: Optional[torch.Tensor] = None,
    assume_sorted_topk: bool = False,
    disable_causal_mask: bool = False,
) -> torch.Tensor:
    return _WeightedFSATopkSparseAttention.apply(
        q_bthd,
        k_bthd,
        v_bthd,
        block_indices_bths,
        chapter_weights_bts,
        int(block_size),
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        topk_idx_hns,
        bool(assume_sorted_topk),
        bool(disable_causal_mask),
    )
