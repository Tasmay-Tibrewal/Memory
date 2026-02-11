import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _is_env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _resolve_use_triton(
    use_triton: Optional[bool],
    *,
    q: torch.Tensor,
) -> bool:
    if use_triton is not None:
        return bool(use_triton)
    raw = os.getenv("FSA_CHAPTER_USE_TRITON", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(_HAS_TRITON and q.is_cuda)


def _resolve_triton_backward_recompute(flag: Optional[bool]) -> bool:
    if flag is not None:
        return bool(flag)
    return _is_env_enabled("FSA_CHAPTER_TRITON_BWD_RECOMPUTE", "1")


def _can_use_triton_chunk(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
) -> bool:
    if not _HAS_TRITON:
        return False
    if (not q_chunk.is_cuda) or (not k_chunk.is_cuda) or (not v_chunk.is_cuda):
        return False
    if q_chunk.dtype not in (torch.float16, torch.bfloat16):
        return False
    if (k_chunk.dtype != q_chunk.dtype) or (v_chunk.dtype != q_chunk.dtype):
        return False
    if q_chunk.shape[-1] > 128:
        return False
    return True


def _chapter_attn_stats_torch(
    q_chunk: torch.Tensor,      # [Qc, G, D]
    k_chunk: torch.Tensor,      # [Kc, D]
    v_chunk: torch.Tensor,      # [Kc, D]
    q_token_ids: torch.Tensor,  # [Qc]
    *,
    softmax_scale: float,
    disable_causal_mask: bool,
    k_start: int,
):
    qf = q_chunk.to(torch.float32)
    kf = k_chunk.to(torch.float32)
    vf = v_chunk.to(torch.float32)
    scores = torch.einsum("qgd,kd->qgk", qf, kf) * float(softmax_scale)
    if not disable_causal_mask:
        k_positions = torch.arange(
            int(k_start),
            int(k_start) + int(kf.shape[0]),
            device=q_chunk.device,
            dtype=torch.int64,
        )
        q_pos = q_token_ids.view(-1, 1)
        causal_ok = k_positions.view(1, -1) <= q_pos
        scores = scores.masked_fill(~causal_ok.unsqueeze(1), float("-inf"))
    m_local = scores.amax(dim=-1)
    finite = torch.isfinite(m_local)
    shifted = scores - torch.where(finite, m_local, torch.zeros_like(m_local)).unsqueeze(-1)
    p = torch.exp(shifted) * finite.unsqueeze(-1).to(shifted.dtype)
    l_local = p.sum(dim=-1)
    acc_local = torch.einsum("qgk,kd->qgd", p, vf)
    return m_local, l_local, acc_local


if _HAS_TRITON:
    @triton.jit
    def _chapter_attn_stats_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_pos_ptr,
        out_m_ptr,
        out_l_ptr,
        out_acc_ptr,
        n_rows,
        k_len,
        head_dim,
        softmax_scale,
        k_start,
        stride_qn,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_outn,
        stride_outd,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_ids < n_rows
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = q_ptr + row_ids[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(
            q_ptrs,
            mask=row_mask[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        if CAUSAL:
            q_pos = tl.load(q_pos_ptr + row_ids, mask=row_mask, other=0).to(tl.int32)
        else:
            q_pos = tl.zeros((BLOCK_M,), dtype=tl.int32)

        m = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for n0 in range(0, k_len, BLOCK_N):
            n_ids = n0 + tl.arange(0, BLOCK_N)
            n_mask = n_ids < k_len

            k_ptrs = k_ptr + n_ids[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + n_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(
                k_ptrs,
                mask=n_mask[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=n_mask[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)

            # qk: [BM, BN]
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * softmax_scale

            valid = row_mask[:, None] & n_mask[None, :]
            if CAUSAL:
                k_pos = (k_start + n_ids).to(tl.int32)
                valid = valid & (q_pos[:, None] >= k_pos[None, :])
            qk = tl.where(valid, qk, float("-inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m, m_ij)
            alpha = tl.exp(m - m_new)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            l = alpha * l + tl.sum(p, axis=1)
            acc = alpha[:, None] * acc + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m = m_new

        out_m_ptrs = out_m_ptr + row_ids
        out_l_ptrs = out_l_ptr + row_ids
        out_acc_ptrs = out_acc_ptr + row_ids[:, None] * stride_outn + offs_d[None, :] * stride_outd
        tl.store(out_m_ptrs, m, mask=row_mask)
        tl.store(out_l_ptrs, l, mask=row_mask)
        tl.store(out_acc_ptrs, acc, mask=row_mask[:, None] & (offs_d[None, :] < head_dim))


def _chapter_attn_stats_triton_rows(
    q_rows: torch.Tensor,       # [N, D]
    k_chunk: torch.Tensor,      # [Kc, D]
    v_chunk: torch.Tensor,      # [Kc, D]
    q_pos_rows: torch.Tensor,   # [N]
    *,
    softmax_scale: float,
    disable_causal_mask: bool,
    k_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    n_rows, d = q_rows.shape
    k_len = int(k_chunk.shape[0])
    out_m = torch.empty((n_rows,), dtype=torch.float32, device=q_rows.device)
    out_l = torch.empty((n_rows,), dtype=torch.float32, device=q_rows.device)
    out_acc = torch.empty((n_rows, d), dtype=torch.float32, device=q_rows.device)
    if n_rows == 0 or k_len == 0:
        if n_rows > 0:
            out_m.fill_(float("-inf"))
            out_l.zero_()
            out_acc.zero_()
        return out_m, out_l, out_acc

    block_d = 16
    while block_d < d:
        block_d *= 2
    if block_d > 128:
        raise RuntimeError(f"Triton chapter kernel supports head_dim <= 128, got {d}.")
    block_m = 32 if d <= 64 else 16
    block_n = 64 if k_len >= 64 else 32
    grid = (triton.cdiv(n_rows, block_m),)
    num_warps = 4 if d <= 64 else 8
    num_stages = 3
    _chapter_attn_stats_kernel[grid](
        q_rows,
        k_chunk,
        v_chunk,
        q_pos_rows,
        out_m,
        out_l,
        out_acc,
        n_rows,
        k_len,
        d,
        float(softmax_scale),
        int(k_start),
        q_rows.stride(0),
        q_rows.stride(1),
        k_chunk.stride(0),
        k_chunk.stride(1),
        v_chunk.stride(0),
        v_chunk.stride(1),
        out_acc.stride(0),
        out_acc.stride(1),
        CAUSAL=not bool(disable_causal_mask),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_m, out_l, out_acc


class _ChapterAttnStatsTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_chunk: torch.Tensor,      # [Qc, G, D]
        k_chunk: torch.Tensor,      # [Kc, D]
        v_chunk: torch.Tensor,      # [Kc, D]
        q_token_ids: torch.Tensor,  # [Qc]
        softmax_scale: float,
        disable_causal_mask: bool,
        k_start: int,
    ):
        q_contig = q_chunk.contiguous()
        k_contig = k_chunk.contiguous()
        v_contig = v_chunk.contiguous()
        q_ids = q_token_ids.to(dtype=torch.int64, device=q_chunk.device).contiguous()
        q_rows = q_contig.reshape(-1, q_contig.shape[-1]).contiguous()
        g = int(q_contig.shape[1])
        q_pos_rows = q_ids.view(-1, 1).expand(-1, g).reshape(-1).contiguous()
        m_row, l_row, acc_row = _chapter_attn_stats_triton_rows(
            q_rows=q_rows,
            k_chunk=k_contig,
            v_chunk=v_contig,
            q_pos_rows=q_pos_rows,
            softmax_scale=float(softmax_scale),
            disable_causal_mask=bool(disable_causal_mask),
            k_start=int(k_start),
        )
        m_local = m_row.view(q_contig.shape[0], g)
        l_local = l_row.view(q_contig.shape[0], g)
        acc_local = acc_row.view(q_contig.shape[0], g, q_contig.shape[-1])
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(q_contig, k_contig, v_contig, q_ids)
            ctx.softmax_scale = float(softmax_scale)
            ctx.disable_causal_mask = bool(disable_causal_mask)
            ctx.k_start = int(k_start)
        return m_local, l_local, acc_local

    @staticmethod
    def backward(ctx, grad_m, grad_l, grad_acc):
        if not hasattr(ctx, "saved_tensors") or len(ctx.saved_tensors) == 0:
            return None, None, None, None, None, None, None
        q_chunk, k_chunk, v_chunk, q_token_ids = ctx.saved_tensors
        with torch.enable_grad():
            q = q_chunk.detach().requires_grad_(True)
            k = k_chunk.detach().requires_grad_(True)
            v = v_chunk.detach().requires_grad_(True)
            m_local, l_local, acc_local = _chapter_attn_stats_torch(
                q,
                k,
                v,
                q_token_ids,
                softmax_scale=float(ctx.softmax_scale),
                disable_causal_mask=bool(ctx.disable_causal_mask),
                k_start=int(ctx.k_start),
            )
            dq, dk, dv = torch.autograd.grad(
                outputs=(m_local, l_local, acc_local),
                inputs=(q, k, v),
                grad_outputs=(grad_m, grad_l, grad_acc),
                allow_unused=True,
            )
        return dq, dk, dv, None, None, None, None


def _chapter_attn_stats_dispatch(
    q_chunk: torch.Tensor,      # [Qc, G, D]
    k_chunk: torch.Tensor,      # [Kc, D]
    v_chunk: torch.Tensor,      # [Kc, D]
    q_token_ids: torch.Tensor,  # [Qc]
    *,
    softmax_scale: float,
    disable_causal_mask: bool,
    k_start: int,
    use_triton: bool,
    triton_backward_recompute: bool,
):
    if use_triton and _can_use_triton_chunk(q_chunk, k_chunk, v_chunk):
        need_grad = bool(q_chunk.requires_grad or k_chunk.requires_grad or v_chunk.requires_grad)
        if need_grad and triton_backward_recompute:
            return _ChapterAttnStatsTritonFn.apply(
                q_chunk,
                k_chunk,
                v_chunk,
                q_token_ids,
                float(softmax_scale),
                bool(disable_causal_mask),
                int(k_start),
            )
        if not need_grad:
            q_rows = q_chunk.contiguous().reshape(-1, q_chunk.shape[-1]).contiguous()
            q_pos_rows = (
                q_token_ids.to(dtype=torch.int64, device=q_chunk.device)
                .view(-1, 1)
                .expand(-1, q_chunk.shape[1])
                .reshape(-1)
                .contiguous()
            )
            m_row, l_row, acc_row = _chapter_attn_stats_triton_rows(
                q_rows=q_rows,
                k_chunk=k_chunk.contiguous(),
                v_chunk=v_chunk.contiguous(),
                q_pos_rows=q_pos_rows,
                softmax_scale=float(softmax_scale),
                disable_causal_mask=bool(disable_causal_mask),
                k_start=int(k_start),
            )
            m_local = m_row.view(q_chunk.shape[0], q_chunk.shape[1])
            l_local = l_row.view(q_chunk.shape[0], q_chunk.shape[1])
            acc_local = acc_row.view(q_chunk.shape[0], q_chunk.shape[1], q_chunk.shape[2])
            return m_local, l_local, acc_local
    return _chapter_attn_stats_torch(
        q_chunk,
        k_chunk,
        v_chunk,
        q_token_ids,
        softmax_scale=float(softmax_scale),
        disable_causal_mask=bool(disable_causal_mask),
        k_start=int(k_start),
    )


def _collapse_hq_routes_to_hk(
    topk_idx_q: torch.Tensor,
    hk: int,
    gqa_deg: int,
    mode: str,
) -> torch.Tensor:
    """
    Collapse per-query-head routes [HQ, N, S] to per-kv-head routes [HK, N, S].

    Modes:
      - strict/validate/error: require identical routes inside each GQA group
      - first/head0: take first query-head route from each GQA group
      - auto: validate, then fallback to first with one warning
    """
    grouped = topk_idx_q.view(hk, gqa_deg, topk_idx_q.shape[1], topk_idx_q.shape[2]).contiguous()
    mode = mode.strip().lower()

    if mode in ("first", "head0"):
        return grouped[:, 0, :, :]

    ref = grouped[:, 0:1, :, :]
    shared = bool(torch.equal(grouped, ref.expand_as(grouped)))
    if shared:
        return grouped[:, 0, :, :]

    if mode in ("strict", "validate", "error"):
        raise ValueError(
            "HQ routes differ inside at least one GQA group. "
            "Use route_collapse='first' for best-effort collapse."
        )

    # auto / unknown: best-effort fallback
    print(
        "chapter-routed warning: HQ routes differ within GQA groups; "
        "falling back to first query-head route per group."
    )
    return grouped[:, 0, :, :]


def _prepare_routes_hk(
    *,
    block_indices_bths: Optional[torch.Tensor],
    topk_idx_hns: Optional[torch.Tensor],
    B: int,
    Tq: int,
    HQ: int,
    HK: int,
    route_collapse: str,
) -> torch.Tensor:
    """
    Return chapter routes as int32 tensor [B, Tq, HK, topk].
    Accepts either:
      - block_indices_bths: [B, Tq, HK or HQ, topk]
      - topk_idx_hns: [HK or HQ, B*Tq, topk]
    """
    gqa_deg = HQ // HK

    if topk_idx_hns is not None:
        if topk_idx_hns.ndim != 3:
            raise ValueError("topk_idx_hns must be rank-3 [H, B*Tq, topk].")
        if topk_idx_hns.shape[1] != B * Tq:
            raise ValueError(
                f"topk_idx_hns second dim must be B*Tq={B*Tq}, got {topk_idx_hns.shape[1]}"
            )
        if topk_idx_hns.shape[0] == HK:
            topk_hk = topk_idx_hns
        elif topk_idx_hns.shape[0] == HQ:
            topk_hk = _collapse_hq_routes_to_hk(topk_idx_hns, hk=HK, gqa_deg=gqa_deg, mode=route_collapse)
        else:
            raise ValueError(f"topk_idx_hns first dim must be HK={HK} or HQ={HQ}.")
        routes = topk_hk.view(HK, B, Tq, -1).permute(1, 2, 0, 3).contiguous()
    else:
        if block_indices_bths is None:
            raise ValueError("Provide either block_indices_bths or topk_idx_hns.")
        if block_indices_bths.ndim != 4:
            raise ValueError("block_indices_bths must be rank-4 [B, Tq, H, topk].")
        if block_indices_bths.shape[0] != B or block_indices_bths.shape[1] != Tq:
            raise ValueError("block_indices_bths must have prefix [B, Tq, ...] matching q.")
        if block_indices_bths.shape[2] == HK:
            routes = block_indices_bths.contiguous()
        elif block_indices_bths.shape[2] == HQ:
            topk_hq = block_indices_bths.permute(2, 0, 1, 3).reshape(HQ, B * Tq, -1).contiguous()
            topk_hk = _collapse_hq_routes_to_hk(topk_hq, hk=HK, gqa_deg=gqa_deg, mode=route_collapse)
            routes = topk_hk.view(HK, B, Tq, -1).permute(1, 2, 0, 3).contiguous()
        else:
            raise ValueError(
                f"block_indices_bths third dim must be HK={HK} or HQ={HQ}, got {block_indices_bths.shape[2]}."
            )

    if routes.dtype != torch.int32:
        routes = routes.to(torch.int32)
    return routes


def _invert_routes_for_chapters(
    routes_tqk: torch.Tensor,
    num_chapters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Invert routes for one (batch, kv_head):
      routes_tqk: [Tq, topk]

    Returns flattened grouped metadata:
      chapter_ids_sorted: [N]
      query_ids_sorted: [N]
      unique_chapters: [C]
      chapter_offsets: [C+1]
    """
    if routes_tqk.ndim != 2:
        raise ValueError("routes_tqk must be [Tq, topk].")
    tq, topk = routes_tqk.shape
    if tq == 0 or topk == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=routes_tqk.device)
        return empty, empty, empty, torch.zeros((1,), dtype=torch.int64, device=routes_tqk.device)

    q_ids = torch.arange(tq, device=routes_tqk.device, dtype=torch.int64).view(tq, 1).expand(tq, topk)
    valid = (routes_tqk >= 0) & (routes_tqk < num_chapters)
    if not bool(torch.any(valid)):
        empty = torch.empty((0,), dtype=torch.int64, device=routes_tqk.device)
        return empty, empty, empty, torch.zeros((1,), dtype=torch.int64, device=routes_tqk.device)

    chapter_ids = routes_tqk[valid].to(torch.int64)
    query_ids = q_ids[valid]

    order = torch.argsort(chapter_ids, stable=True)
    chapter_ids = chapter_ids.index_select(0, order)
    query_ids = query_ids.index_select(0, order)

    unique_chapters, counts = torch.unique_consecutive(chapter_ids, return_counts=True)
    chapter_offsets = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=routes_tqk.device),
            counts.to(torch.int64).cumsum(0),
        ],
        dim=0,
    )
    return chapter_ids, query_ids, unique_chapters, chapter_offsets


def _build_chapter_dispatch(
    routes_hk: torch.Tensor,  # [B, Tq, HK, topk]
    num_chapters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build flattened chapter-dispatch worklist across all (batch, kv_head) groups.

    Returns:
      dispatch_keys: [S] int64 where key = bh * num_chapters + chapter_id
      dispatch_offsets: [S+1] int64 segment offsets into q_ids_sorted
      q_ids_sorted: [N] int64 query token ids sorted by (key, qid)
    """
    if routes_hk.ndim != 4:
        raise ValueError("routes_hk must be [B, Tq, HK, topk].")
    B, Tq, HK, topk = routes_hk.shape
    if B <= 0 or Tq <= 0 or HK <= 0 or topk <= 0:
        empty = torch.empty((0,), dtype=torch.int64, device=routes_hk.device)
        return empty, torch.zeros((1,), dtype=torch.int64, device=routes_hk.device), empty

    routes_bh = routes_hk.permute(0, 2, 1, 3).contiguous()  # [B, HK, Tq, topk]
    bh = torch.arange(B * HK, device=routes_hk.device, dtype=torch.int64).view(B, HK, 1, 1).expand(B, HK, Tq, topk)
    qid = torch.arange(Tq, device=routes_hk.device, dtype=torch.int64).view(1, 1, Tq, 1).expand(B, HK, Tq, topk)

    valid = (routes_bh >= 0) & (routes_bh < num_chapters)
    if not bool(torch.any(valid)):
        empty = torch.empty((0,), dtype=torch.int64, device=routes_hk.device)
        return empty, torch.zeros((1,), dtype=torch.int64, device=routes_hk.device), empty

    chap = routes_bh[valid].to(torch.int64)
    bh = bh[valid]
    qid = qid[valid]

    key = bh * int(num_chapters) + chap
    # Sort by (key, qid) so each segment has q-ids already grouped for cheap dedupe/count.
    sort_key = key * int(Tq) + qid
    order = torch.argsort(sort_key, stable=True)
    key = key.index_select(0, order)
    qid = qid.index_select(0, order)

    dispatch_keys, counts = torch.unique_consecutive(key, return_counts=True)
    dispatch_offsets = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=routes_hk.device),
            counts.to(torch.int64).cumsum(0),
        ],
        dim=0,
    )
    return dispatch_keys, dispatch_offsets, qid


def FSA_topk_sparse_attention_chapter_routed_bthd(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    block_indices_bths: Optional[torch.Tensor],
    block_size: int,
    *,
    softmax_scale: Optional[float] = None,
    topk_idx_hns: Optional[torch.Tensor] = None,
    disable_causal_mask: bool = True,
    route_collapse: Optional[str] = None,
    chapter_query_chunk_size: int = 4096,
    dedupe_queries_per_chapter: bool = False,
    use_triton: Optional[bool] = None,
    triton_backward_recompute: Optional[bool] = None,
) -> torch.Tensor:
    """
    Chapter-routed sparse attention (MoE-style dispatch), BTHD wrapper.

    This implements an inverted routing pipeline:
      1) route queries to chapters
      2) run dense chapter-local attention for each routed chapter bucket
      3) merge per-chapter contributions via online softmax state
      4) gather final output back to [B, Tq, HQ, D]

    Shapes:
      q: [B, Tq, HQ, D]
      k/v: [B, Tk, HK, D]
      block_indices: [B, Tq, HK, topk] (preferred) or [B, Tq, HQ, topk]

    Notes:
      - Handles non-uniform number of routed queries per chapter without padding.
      - Forward is fully differentiable in PyTorch; backward is handled by autograd.
    """
    if q_bthd.ndim != 4 or k_bthd.ndim != 4 or v_bthd.ndim != 4:
        raise ValueError("q/k/v must be rank-4 [B, T, H, D].")
    if k_bthd.shape != v_bthd.shape:
        raise ValueError("k and v must have identical shape.")
    if q_bthd.shape[0] != k_bthd.shape[0]:
        raise ValueError("Batch size mismatch between q and k/v.")
    if q_bthd.shape[3] != k_bthd.shape[3]:
        raise ValueError("Head dim mismatch between q and k/v.")
    if q_bthd.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype {q_bthd.dtype}; expected fp16/bf16/fp32.")

    B, Tq, HQ, D = q_bthd.shape
    _, Tk, HK, _ = k_bthd.shape
    if HQ % HK != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HK ({HK}).")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")
    if chapter_query_chunk_size <= 0:
        raise ValueError("chapter_query_chunk_size must be > 0.")

    gqa_deg = HQ // HK
    if softmax_scale is None:
        softmax_scale = float(D) ** -0.5

    if route_collapse is None:
        route_collapse = os.getenv("FSA_LOCAL_GQA_ROUTE_COLLAPSE", "auto")
    use_triton_resolved = _resolve_use_triton(use_triton, q=q_bthd)
    triton_bwd_recompute = _resolve_triton_backward_recompute(triton_backward_recompute)

    routes_hk = _prepare_routes_hk(
        block_indices_bths=block_indices_bths,
        topk_idx_hns=topk_idx_hns,
        B=B,
        Tq=Tq,
        HQ=HQ,
        HK=HK,
        route_collapse=route_collapse,
    )

    num_chapters = (Tk + block_size - 1) // block_size
    bh = B * HK
    q_flat = (
        q_bthd.view(B, Tq, HK, gqa_deg, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(bh, Tq, gqa_deg, D)
        .contiguous()
    )
    k_flat = k_bthd.permute(0, 2, 1, 3).reshape(bh, Tk, D).contiguous()
    v_flat = v_bthd.permute(0, 2, 1, 3).reshape(bh, Tk, D).contiguous()

    m_state = torch.full((bh, Tq, gqa_deg), float("-inf"), dtype=torch.float32, device=q_bthd.device)
    l_state = torch.zeros((bh, Tq, gqa_deg), dtype=torch.float32, device=q_bthd.device)
    acc_state = torch.zeros((bh, Tq, gqa_deg, D), dtype=torch.float32, device=q_bthd.device)

    dispatch_keys, dispatch_offsets, dispatch_qids = _build_chapter_dispatch(routes_hk=routes_hk, num_chapters=num_chapters)
    if dispatch_keys.numel() == 0:
        return torch.zeros_like(q_bthd)

    chapter_starts = torch.arange(num_chapters, device=q_bthd.device, dtype=torch.int64) * int(block_size)
    chapter_ends = torch.clamp(chapter_starts + int(block_size), max=int(Tk))

    for seg in range(int(dispatch_keys.numel())):
        key = int(dispatch_keys[seg].item())
        seg_start = int(dispatch_offsets[seg].item())
        seg_end = int(dispatch_offsets[seg + 1].item())
        if seg_end <= seg_start:
            continue

        bh_idx = key // int(num_chapters)
        chapter_id = key - bh_idx * int(num_chapters)
        k_start = int(chapter_starts[chapter_id].item())
        k_end = int(chapter_ends[chapter_id].item())
        if k_end <= k_start:
            continue

        q_ids_raw = dispatch_qids[seg_start:seg_end]
        if q_ids_raw.numel() == 0:
            continue
        # q_ids_raw is already sorted by q-id due dispatch sort.
        if dedupe_queries_per_chapter:
            q_ids = torch.unique_consecutive(q_ids_raw)
            q_mult = None
        else:
            q_ids, q_counts = torch.unique_consecutive(q_ids_raw, return_counts=True)
            q_mult = q_counts.to(torch.float32)
        if q_ids.numel() == 0:
            continue

        q_head = q_flat[bh_idx]            # [Tq, G, D]
        m_head = m_state[bh_idx]           # [Tq, G]
        l_head = l_state[bh_idx]           # [Tq, G]
        acc_head = acc_state[bh_idx]       # [Tq, G, D]
        k_chunk = k_flat[bh_idx, k_start:k_end]  # [Kc, D]
        v_chunk = v_flat[bh_idx, k_start:k_end]  # [Kc, D]
        for q0 in range(0, int(q_ids.numel()), chapter_query_chunk_size):
            q1 = min(int(q_ids.numel()), q0 + chapter_query_chunk_size)
            q_chunk_ids = q_ids[q0:q1]
            q_chunk = q_head.index_select(0, q_chunk_ids)  # [Qc, G, D]

            m_local, l_local, acc_local = _chapter_attn_stats_dispatch(
                q_chunk,
                k_chunk,
                v_chunk,
                q_chunk_ids,
                softmax_scale=float(softmax_scale),
                disable_causal_mask=bool(disable_causal_mask),
                k_start=int(k_start),
                use_triton=bool(use_triton_resolved),
                triton_backward_recompute=bool(triton_bwd_recompute),
            )
            finite = torch.isfinite(m_local)
            if not bool(torch.any(finite)):
                continue

            if not dedupe_queries_per_chapter:
                # Preserve exact semantics when the same chapter appears multiple times
                # in a query's top-k list by scaling this chapter contribution.
                q_chunk_mult = q_mult[q0:q1]
                l_local = l_local * q_chunk_mult.unsqueeze(-1)
                acc_local = acc_local * q_chunk_mult.unsqueeze(-1).unsqueeze(-1)

            m_old = m_head[q_chunk_ids]
            l_old = l_head[q_chunk_ids]
            acc_old = acc_head[q_chunk_ids]

            m_new = torch.maximum(m_old, m_local)
            alpha = torch.exp(torch.where(torch.isfinite(m_old), m_old - m_new, torch.full_like(m_new, -float("inf"))))
            beta = torch.exp(torch.where(finite, m_local - m_new, torch.full_like(m_new, -float("inf"))))

            l_new = alpha * l_old + beta * l_local
            acc_new = alpha.unsqueeze(-1) * acc_old + beta.unsqueeze(-1) * acc_local

            m_head[q_chunk_ids] = m_new
            l_head[q_chunk_ids] = l_new
            acc_head[q_chunk_ids] = acc_new

    out_grouped = acc_state / l_state.clamp_min(1e-9).unsqueeze(-1)
    out_grouped = torch.where(l_state.unsqueeze(-1) > 0, out_grouped, torch.zeros_like(out_grouped))
    out = (
        out_grouped.view(B, HK, Tq, gqa_deg, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, Tq, HQ, D)
        .to(dtype=q_bthd.dtype)
    )
    return out.contiguous()


def chapter_routed_sparse_attention_bthd(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    block_indices_bths: Optional[torch.Tensor],
    block_size: int,
    **kwargs,
) -> torch.Tensor:
    """Alias for FSA_topk_sparse_attention_chapter_routed_bthd."""
    return FSA_topk_sparse_attention_chapter_routed_bthd(
        q_bthd=q_bthd,
        k_bthd=k_bthd,
        v_bthd=v_bthd,
        block_indices_bths=block_indices_bths,
        block_size=block_size,
        **kwargs,
    )
