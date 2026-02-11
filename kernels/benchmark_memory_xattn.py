# benchmark_memory_xattn.py
# Run on H100 Colab (CUDA). Requires:
#   - your custom kernels module: memory_cross_attn.py (the one with memory_cross_attn(..., dkv_strategy=...))
#   - (optional) flash-attn for baseline (ii)
#   - (optional) native-sparse-attention for baseline (iv)

import os, time, math, gc, importlib, sys, types
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F

# Cold-start defaults (can be overridden by user-provided env vars).
os.environ.setdefault("MEM_XATTN_FAST_START", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str((Path(__file__).resolve().parent / ".triton_cache")))


# ----------------------------
# Small helpers
# ----------------------------

def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bytes_gb(n: int) -> float:
    return n / (1024**3)

def cuda_mem_gb() -> Tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return bytes_gb(free), bytes_gb(total)

def clear_cuda_cache():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.synchronize()

@dataclass
class Config:
    # User inputs
    mem_tokens: int          # TK
    chapter_size: int        # BS
    topk: int                # S
    batch_size: int          # input batch (will be collapsed)
    seq_len: int             # input seq
    hidden_dim: int          # model dim = heads * head_dim
    heads: int               # HQ (query heads)
    kv_heads: Optional[int] = None  # HK (KV heads); None -> same as heads
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    # Derived
    @property
    def TQ(self) -> int:
        return self.batch_size * self.seq_len  # collapse into 1 mega sequence

    @property
    def TK(self) -> int:
        return self.mem_tokens

    @property
    def BS(self) -> int:
        return self.chapter_size

    @property
    def H(self) -> int:
        return self.heads

    @property
    def HK(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def G(self) -> int:
        assert self.H % self.HK == 0, f"heads ({self.H}) must be divisible by kv_heads ({self.HK})"
        return self.H // self.HK

    @property
    def D(self) -> int:
        assert self.hidden_dim % self.heads == 0
        return self.hidden_dim // self.heads

    @property
    def M(self) -> int:
        assert self.mem_tokens % self.chapter_size == 0
        return self.mem_tokens // self.chapter_size

    @property
    def scale(self) -> float:
        return self.D ** -0.5

    def __post_init__(self):
        if self.hidden_dim % self.heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by heads ({self.heads}).")
        hk = self.kv_heads if self.kv_heads is not None else self.heads
        if self.heads % hk != 0:
            raise ValueError(f"heads ({self.heads}) must be divisible by kv_heads ({hk}).")


def make_inputs(cfg: Config, seed: int, *, per_query_random_topk: bool = True) -> Dict[str, torch.Tensor]:
    """
    Shapes match your prompt:
      q: [1, TQ, HQ, D]
      k: [1, TK, HK, D]
      v: [1, TK, HK, D]
      block_indices: [1, TQ, HK, topk]
    """
    set_all_seeds(seed)
    device, dtype = cfg.device, cfg.dtype

    q = torch.randn(1, cfg.TQ, cfg.H, cfg.D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, cfg.TK, cfg.HK, cfg.D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, cfg.TK, cfg.HK, cfg.D, device=device, dtype=dtype, requires_grad=True)

    # Per-query different chapters: sample unique topk chapters per (t,h),
    # matching real top-k router behavior (no duplicates within a single query/head).
    if per_query_random_topk:
        if cfg.topk > cfg.M:
            raise ValueError(f"topk ({cfg.topk}) cannot exceed number of chapters M ({cfg.M}).")
        block_indices = torch.empty((1, cfg.TQ, cfg.HK, cfg.topk), device=device, dtype=torch.int32)
        # Chunk over TQ to avoid a very large temporary [1,TQ,H,M] tensor.
        chunk_t = 4096
        for t0 in range(0, cfg.TQ, chunk_t):
            t1 = min(cfg.TQ, t0 + chunk_t)
            scores = torch.rand((1, t1 - t0, cfg.HK, cfg.M), device=device, dtype=torch.float32)
            idx = torch.topk(scores, k=cfg.topk, dim=-1, largest=True, sorted=False).indices.to(torch.int32)
            block_indices[:, t0:t1, :, :] = idx
    else:
        # same chapters for all queries (not used for sparse tests; used for dense baselines)
        global_ch = torch.randperm(cfg.M, device=device)[:cfg.topk].to(torch.int32)  # [topk]
        block_indices = global_ch.view(1, 1, 1, cfg.topk).expand(1, cfg.TQ, cfg.HK, cfg.topk).contiguous()

    return dict(q=q, k=k, v=v, block_indices=block_indices)


# ----------------------------
# Naive reference (slow, correctness only)
# ----------------------------

# @torch.no_grad()
def _gather_kv_for_one(q_idx: int, h: int, block_indices_th: torch.Tensor, k: torch.Tensor, v: torch.Tensor, BS: int):
    # block_indices_th: [topk] int32 on GPU
    # returns kk, vv: [topk*BS, D] float32
    ch = block_indices_th.to(torch.int64)  # [S]
    # token ids for each chapter
    tok = ch[:, None] * BS + torch.arange(BS, device=ch.device)[None, :]  # [S, BS]
    tok = tok.reshape(-1)  # [S*BS]
    kk = k[0, tok, h, :].float()
    vv = v[0, tok, h, :].float()
    return kk, vv

def naive_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.Tensor, BS: int, scale: float) -> torch.Tensor:
    """
    Very slow correctness reference.
    q: [1,TQ,HQ,D], k/v: [1,TK,HK,D], block_indices: [1,TQ,HK,S]
    returns o: [1,TQ,HQ,D]
    """
    assert q.shape[0] == 1
    _, TQ, HQ, D = q.shape
    _, TK, HK, Dk = k.shape
    _, TKv, HKv, Dv = v.shape
    assert TK == TKv and HK == HKv and D == Dk and D == Dv
    assert block_indices.shape[:3] == (1, TQ, HK), "block_indices must be [1,TQ,HK,S]"
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    G = HQ // HK
    out = torch.zeros((1, TQ, HQ, D), device=q.device, dtype=torch.float32)

    for hk in range(HK):
        for t in range(TQ):
            kk, vv = _gather_kv_for_one(t, hk, block_indices[0, t, hk, :], k, v, BS)  # [S*BS,D]
            for g in range(G):
                hq = hk * G + g
                q_t = (q[0, t, hq, :].float() * scale)  # [D]
                scores = q_t @ kk.T  # [S*BS]
                p = torch.softmax(scores, dim=-1)
                out[0, t, hq, :] = p @ vv
    return out.to(q.dtype)


def check_correctness_5x(cfg: Config, *, sanity_TQ: int = 128, sanity_M: int = 16, sanity_BS: Optional[int] = None):
    """
    Runs 5 correctness trials (forward + backward) against naive PyTorch reference
    for each custom dKV strategy: a,b,c,d.

    For safety, we do correctness on a reduced config by default.
    """
    from memory_cross_attn import memory_cross_attn
    try:
        from memory_cross_attn_fsa_opt import memory_cross_attn_fsa_opt
        has_fsa_opt = True
        fsa_opt_import_err = None
    except Exception as e:
        memory_cross_attn_fsa_opt = None
        has_fsa_opt = False
        fsa_opt_import_err = e
    fsa_local_bthd_fn, fsa_local_import_err = try_import_fsa_local()
    has_fsa_local = fsa_local_bthd_fn is not None

    BS = sanity_BS if sanity_BS is not None else min(cfg.BS, 64)
    M = min(cfg.M, sanity_M)
    TQ = min(cfg.TQ, sanity_TQ)
    # topk = min(cfg.topk, 4) if cfg.topk > 4 else cfg.topk  # keep naive manageable
    topk = min(cfg.topk, 4)
    H = cfg.H
    HK = cfg.HK
    D = max(16, cfg.D // 2) if cfg.D > 16 else cfg.D

    # Build a reduced config (keep dtype/heads/head_dim)
    red = Config(
        mem_tokens=M * BS,
        chapter_size=BS,
        topk=topk,
        batch_size=1,
        seq_len=TQ,              # since we collapse anyway, just make TQ=seq_len
        hidden_dim=H * D,
        heads=H,
        kv_heads=HK,
        dtype=cfg.dtype,
        device=cfg.device,
    )

    print("\n=== Correctness (5 trials) on reduced config ===")
    print(
        f"Reduced: TQ={red.TQ}, TK={red.TK}, M={red.M}, BS={red.BS}, topk={red.topk}, "
        f"HQ={red.H}, HK={red.HK}, G={red.G}, D={red.D}, dtype={red.dtype}\n"
    )

    # tolerances (bf16 is noisy; atomics in B add a bit more noise)
    if red.dtype in (torch.float16, torch.bfloat16):
        atol_fwd, rtol_fwd = 2e-2, 2e-2
        atol_bwd, rtol_bwd = 3e-2, 3e-2
    else:
        atol_fwd, rtol_fwd = 1e-4, 1e-4
        atol_bwd, rtol_bwd = 5e-4, 5e-4
    relax_large_bwd = red.dtype in (torch.float16, torch.bfloat16) and red.TQ >= 8192
    if relax_large_bwd:
        print("Note: using relaxed backward tolerance at large TQ (bf16/fp16 reduction-order noise).")

    strategies = ["a", "b", "c", "d"]
    if not has_fsa_opt:
        print(f"Note: skipping fsa_opt correctness (import failed: {fsa_opt_import_err})")
    if not has_fsa_local:
        print(f"Note: skipping fsa_local correctness (import failed: {fsa_local_import_err})")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(f"Note: skipping fsa_local correctness (unsupported BS={red.BS}; requires 32/64/128/256/512/1024)")
        has_fsa_local = False

    for trial in range(5):
        print(f"Running trial {trial + 1}/5")

        seed = 1234 + trial
        x = make_inputs(red, seed=seed, per_query_random_topk=True)

        # naive forward (no grad)
        with torch.no_grad():
            o_ref = naive_reference(x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], red.BS, red.scale)

        # random dO (fixed for ref vs custom)
        set_all_seeds(999 + trial)
        dO = torch.randn_like(o_ref).to(red.dtype)

        # reference backward: use autograd on naive graph (must recompute with grad enabled)
        q2 = x["q"].detach().clone().requires_grad_(True)
        k2 = x["k"].detach().clone().requires_grad_(True)
        v2 = x["v"].detach().clone().requires_grad_(True)

        o2 = naive_reference(q2, k2, v2, x["block_indices"], red.BS, red.scale)
        loss2 = (o2 * dO).sum()
        loss2.backward()
        dq_ref, dk_ref, dv_ref = q2.grad.detach(), k2.grad.detach(), v2.grad.detach()

        # test each strategy
        for strat in strategies:
            print(f"Running strategy {strat}")
            q = x["q"].detach().clone().requires_grad_(True)
            k = x["k"].detach().clone().requires_grad_(True)
            v = x["v"].detach().clone().requires_grad_(True)
            bi = x["block_indices"]
            atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
            if relax_large_bwd:
                atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                rtol_bwd_use = max(rtol_bwd_use, 1e-1)

            o = memory_cross_attn(q, k, v, bi, red.BS, scale=red.scale, dkv_strategy=strat, q_chunk_size=32, d_chunk_size=32)
            # forward compare
            try:
                torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy '{strat}': forward mismatch\n{exc}") from exc

            loss = (o * dO).sum()
            loss.backward()

            try:
                torch.testing.assert_close(q.grad, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(k.grad, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(v.grad, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy '{strat}': backward mismatch\n{exc}") from exc

        if has_fsa_opt:
            print("Running strategy fsa_opt")
            q = x["q"].detach().clone().requires_grad_(True)
            k = x["k"].detach().clone().requires_grad_(True)
            v = x["v"].detach().clone().requires_grad_(True)
            bi = x["block_indices"]
            atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
            if relax_large_bwd:
                atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                rtol_bwd_use = max(rtol_bwd_use, 1e-1)

            o = memory_cross_attn_fsa_opt(q, k, v, bi, red.BS, scale=red.scale)
            try:
                torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy 'fsa_opt': forward mismatch\n{exc}") from exc

            loss = (o * dO).sum()
            loss.backward()

            try:
                torch.testing.assert_close(q.grad, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(k.grad, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(v.grad, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy 'fsa_opt': backward mismatch\n{exc}") from exc

        if has_fsa_local:
            print("Running strategy fsa_local")
            q = x["q"].detach().clone().requires_grad_(True)
            k = x["k"].detach().clone().requires_grad_(True)
            v = x["v"].detach().clone().requires_grad_(True)
            bi = x["block_indices"]
            q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
                q.detach(), k.detach(), v.detach(), bi, red.TK
            )
            cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
            cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
            atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
            if relax_large_bwd:
                atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                rtol_bwd_use = max(rtol_bwd_use, 1e-1)

            o_full = fsa_local_bthd_fn(
                q_bthd=q_full,
                k_bthd=k_full[:, :red.TK, :, :],
                v_bthd=v_full[:, :red.TK, :, :],
                block_indices_bths=bi_full,
                block_size=red.BS,
                softmax_scale=red.scale,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                disable_causal_mask=True,
            )
            o = o_full[:, red.TK:, :, :]
            try:
                torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local': forward mismatch\n{exc}") from exc

            loss = (o * dO).sum()
            loss.backward()
            dq_loc = q_full.grad[:, red.TK:, :, :]
            dk_loc = k_full.grad[:, :red.TK, :, :]
            dv_loc = v_full.grad[:, :red.TK, :, :]

            try:
                torch.testing.assert_close(dq_loc, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(dk_loc, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
                torch.testing.assert_close(dv_loc, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
            except AssertionError as exc:
                raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local': backward mismatch\n{exc}") from exc

        print(f"Trial {trial+1}/5: OK")

    if has_fsa_opt and has_fsa_local:
        print("\nCorrectness: ALL strategies (a,b,c,d,fsa_opt,fsa_local) passed vs naive reference.\n")
    elif has_fsa_opt:
        print("\nCorrectness: ALL strategies (a,b,c,d,fsa_opt) passed vs naive reference.\n")
    elif has_fsa_local:
        print("\nCorrectness: ALL strategies (a,b,c,d,fsa_local) passed vs naive reference.\n")
    else:
        print("\nCorrectness: ALL strategies (a,b,c,d) passed vs naive reference.\n")


# ----------------------------
# Timing helpers
# ----------------------------

def time_fwd_bwd(
    forward_fn,
    make_loss_fn,
    *,
    iters: int = 5,
    warmup: int = 2,
    clear_cache_each_iter: bool = True,
    reinit_fn=None,
    reinit_each_iter: bool = False,
    zero_grad_fn=None,
) -> Tuple[float, float]:
    """
    Returns (avg_fwd_ms, avg_bwd_ms) over `iters` timed iterations.
    warmup iterations are excluded.

    clear_cache_each_iter:
      If True, empty CUDA cache between timed iterations (NOT included in timing).
    reinit_each_iter:
      If False, call reinit_fn once before warmup and once before timed loop.
    """
    if reinit_fn is not None and not reinit_each_iter:
        if clear_cache_each_iter:
            clear_cuda_cache()
        reinit_fn()

    # warmup (includes compilation)
    for _ in range(warmup):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if reinit_fn is not None and reinit_each_iter:
            reinit_fn()
        if zero_grad_fn is not None:
            zero_grad_fn()
        out = forward_fn()
        loss = make_loss_fn(out)
        loss.backward()
        torch.cuda.synchronize()

    if reinit_fn is not None and not reinit_each_iter:
        if clear_cache_each_iter:
            clear_cuda_cache()
        reinit_fn()

    # timed
    fwd_times = []
    bwd_times = []

    for _ in range(iters):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if reinit_fn is not None and reinit_each_iter:
            reinit_fn()
        if zero_grad_fn is not None:
            zero_grad_fn()

        torch.cuda.synchronize()
        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        start_f.record()
        out = forward_fn()
        end_f.record()

        loss = make_loss_fn(out)

        start_b.record()
        loss.backward()
        end_b.record()

        torch.cuda.synchronize()
        fwd_times.append(start_f.elapsed_time(end_f))
        bwd_times.append(start_b.elapsed_time(end_b))

    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times)


def time_fwd_only(
    forward_fn,
    *,
    iters: int = 5,
    warmup: int = 2,
    clear_cache_each_iter: bool = True,
    reinit_fn=None,
    reinit_each_iter: bool = False,
) -> float:
    """
    Returns avg forward ms over `iters` timed iterations.
    warmup iterations are excluded.
    """
    if reinit_fn is not None and not reinit_each_iter:
        if clear_cache_each_iter:
            clear_cuda_cache()
        reinit_fn()

    for _ in range(warmup):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if reinit_fn is not None and reinit_each_iter:
            reinit_fn()
        with torch.no_grad():
            _ = forward_fn()
        torch.cuda.synchronize()

    if reinit_fn is not None and not reinit_each_iter:
        if clear_cache_each_iter:
            clear_cuda_cache()
        reinit_fn()

    fwd_times = []
    for _ in range(iters):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if reinit_fn is not None and reinit_each_iter:
            reinit_fn()

        torch.cuda.synchronize()
        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)

        start_f.record()
        with torch.no_grad():
            _ = forward_fn()
        end_f.record()

        torch.cuda.synchronize()
        fwd_times.append(start_f.elapsed_time(end_f))

    return sum(fwd_times) / len(fwd_times)


def reset_grads(*tensors: torch.Tensor):
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def reset_grads_in_state(state: Dict[str, torch.Tensor], keys: List[str]):
    tensors: List[torch.Tensor] = []
    for key in keys:
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.requires_grad:
            tensors.append(value)
    if tensors:
        reset_grads(*tensors)


# ----------------------------
# Baselines (i) dense math, (ii) flash/sdpa
# ----------------------------

def dense_math_attention(q_bthd, k_bLhd, v_bLhd, scale: float):
    """
    q: [B, TQ, HQ, D]
    k/v: [B, L,  HK, D]
    returns: [B, TQ, HQ, D]
    """
    B, TQ, HQ, D = q_bthd.shape
    _, L, HK, Dk = k_bLhd.shape
    assert D == Dk and v_bLhd.shape == (B, L, HK, D)
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if HK != HQ:
        G = HQ // HK
        k_bLhd = k_bLhd.repeat_interleave(G, dim=2).contiguous()
        v_bLhd = v_bLhd.repeat_interleave(G, dim=2).contiguous()
        HK = HQ
    L = k_bLhd.shape[1]

    q = (q_bthd.float() * scale).permute(0, 2, 1, 3)   # [B,HQ,TQ,D]
    k = k_bLhd.float().permute(0, 2, 1, 3)             # [B,HQ,L,D]
    v = v_bLhd.float().permute(0, 2, 1, 3)             # [B,HQ,L,D]

    scores = torch.matmul(q, k.transpose(-2, -1))       # [B,H,TQ,L]
    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, v)                           # [B,H,TQ,D]
    return out.permute(0, 2, 1, 3).to(q_bthd.dtype)     # [B,TQ,H,D]


def dense_sdpa_flash_attention(q_bthd, k_bLhd, v_bLhd, scale: float):
    """
    Uses PyTorch SDPA, forcing flash if possible.
    q: [B,TQ,HQ,D], k/v: [B,L,HK,D]
    """
    HQ = q_bthd.shape[2]
    HK = k_bLhd.shape[2]
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if HK != HQ:
        G = HQ // HK
        k_bLhd = k_bLhd.repeat_interleave(G, dim=2).contiguous()
        v_bLhd = v_bLhd.repeat_interleave(G, dim=2).contiguous()

    q = (q_bthd * scale).permute(0, 2, 1, 3)  # [B,H,TQ,D]
    k = k_bLhd.permute(0, 2, 1, 3)            # [B,H,L,D]
    v = v_bLhd.permute(0, 2, 1, 3)

    # Force flash path if supported. If not supported, fall back.
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    except Exception:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    return out.permute(0, 2, 1, 3).contiguous()  # [B,TQ,H,D]


def try_flash_attn_pkg(q_bthd, k_bLhd, v_bLhd, scale: float):
    """
    If flash-attn package is installed, use it.
    flash_attn_func expects [B,T,H,D] for q,k,v.
    """
    try:
        from flash_attn import flash_attn_func
    except Exception as e:
        return None, f"flash-attn not available: {e}"

    HQ = q_bthd.shape[2]
    HK = k_bLhd.shape[2]
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if HK != HQ:
        G = HQ // HK
        k_bLhd = k_bLhd.repeat_interleave(G, dim=2).contiguous()
        v_bLhd = v_bLhd.repeat_interleave(G, dim=2).contiguous()

    # flash_attn_func internally applies scaling; we pass softmax_scale
    out = flash_attn_func(q_bthd, k_bLhd, v_bLhd, causal=False, softmax_scale=scale)
    return out, "flash_attn_func"


def try_flash_attn_with_kvcache_pkg(q_bthd, k_bLhd, v_bLhd, scale: float):
    """
    Best-effort wrapper for flash_attn_with_kvcache across flash-attn versions.
    Returns (out, tag) on success, (None, reason) on failure.
    """
    try:
        from flash_attn import flash_attn_with_kvcache
    except Exception as e:
        return None, f"flash-attn not available: {e}"

    bsz = q_bthd.shape[0]
    HQ = q_bthd.shape[2]
    HK = k_bLhd.shape[2]
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if HK != HQ:
        G = HQ // HK
        k_bLhd = k_bLhd.repeat_interleave(G, dim=2).contiguous()
        v_bLhd = v_bLhd.repeat_interleave(G, dim=2).contiguous()

    cache_len = int(k_bLhd.shape[1])
    cache_seqlens = torch.full((bsz,), cache_len, device=q_bthd.device, dtype=torch.int32)

    # Most common signature in flash-attn 2.x
    try:
        out = flash_attn_with_kvcache(
            q_bthd, k_bLhd, v_bLhd,
            cache_seqlens=cache_seqlens,
            softmax_scale=scale,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out, "flash_attn_with_kvcache"
    except Exception as first_error:
        # Fallback: some builds accept scalar cache_seqlens.
        try:
            out = flash_attn_with_kvcache(
                q_bthd, k_bLhd, v_bLhd,
                cache_seqlens=cache_len,
                softmax_scale=scale,
                causal=False,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out, "flash_attn_with_kvcache"
        except Exception as second_error:
            return None, f"{first_error}; fallback failed: {second_error}"


def try_flash_attn_with_kvcache_sparse_topk_pkg(
    q_bthd: torch.Tensor,
    k_bLhd: torch.Tensor,
    v_bLhd: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    chunk_qh: Optional[int] = None,
):
    """
    Token-level sparse top-k (chapter) emulation using flash_attn_with_kvcache.

    This materializes per-(query,head) selected KV slices in chunks, then calls
    flash_attn_with_kvcache with cache_batch_idx to map each query row to its
    own temporary cache row.

    Forward-only helper for benchmarking feasibility/throughput.
    """
    try:
        from flash_attn import flash_attn_with_kvcache
    except Exception as e:
        return None, f"flash-attn not available: {e}"

    if q_bthd.shape[0] != 1 or k_bLhd.shape[0] != 1 or v_bLhd.shape[0] != 1:
        return None, "sparse kvcache helper expects batch size 1 inputs [1,T,H,D]"
    _, TQ, HQ, D = q_bthd.shape
    HK = int(k_bLhd.shape[2])
    if HQ % HK != 0:
        return None, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if block_indices.shape[:3] != (1, TQ, HK):
        return None, "block_indices shape must be [1, TQ, HK, S]"

    S = int(block_indices.shape[-1])
    Lsel = S * int(block_size)
    if Lsel <= 0:
        return None, "invalid selected KV length"

    G = HQ // HK
    q_flat = q_bthd[0].reshape(TQ * HQ, D).contiguous()
    k_src = k_bLhd[0]  # [TK, HK, D]
    v_src = v_bLhd[0]  # [TK, HK, D]
    out_flat = torch.empty((TQ * HQ, D), device=q_bthd.device, dtype=q_bthd.dtype)

    # Auto chunk size to keep temporary K/V under a reasonable bound.
    if chunk_qh is None:
        target_bytes = 384 * 1024 * 1024
        bytes_per_qh = max(1, Lsel * D * q_bthd.element_size() * 2)  # K + V
        chunk_qh = max(1, min(4096, target_bytes // bytes_per_qh))

    tok_offsets = torch.arange(block_size, device=q_bthd.device, dtype=torch.int64)[None, None, :]

    for qh0 in range(0, TQ * HQ, int(chunk_qh)):
        qh1 = min(TQ * HQ, qh0 + int(chunk_qh))
        n_chunk = qh1 - qh0

        qh_ids = torch.arange(qh0, qh1, device=q_bthd.device, dtype=torch.int64)
        t_ids = torch.div(qh_ids, HQ, rounding_mode="floor")
        q_heads = qh_ids - t_ids * HQ
        kv_heads = torch.div(q_heads, G, rounding_mode="floor")

        bi_chunk = block_indices[0, t_ids, kv_heads, :].to(torch.int64).contiguous()  # [chunk, S]
        tok_chunk = (bi_chunk[:, :, None] * block_size + tok_offsets).reshape(n_chunk, Lsel)  # [chunk, Lsel]

        h_idx = kv_heads[:, None].expand(n_chunk, Lsel)

        # Materialize sparse-selected cache rows: [chunk, Lsel, 1, D]
        k_cache = k_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)
        v_cache = v_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)
        q_chunk = q_flat[qh0:qh1].contiguous().unsqueeze(1).unsqueeze(2)  # [chunk,1,1,D]

        cache_seqlens = torch.full((n_chunk,), Lsel, device=q_bthd.device, dtype=torch.int32)
        cache_batch_idx = torch.arange(n_chunk, device=q_bthd.device, dtype=torch.int32)

        try:
            out_chunk = flash_attn_with_kvcache(
                q_chunk, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                softmax_scale=scale,
                causal=False,
            )
        except Exception:
            # Fallback for builds without cache_batch_idx in Python binding.
            out_chunk = flash_attn_with_kvcache(
                q_chunk, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale,
                causal=False,
            )

        if isinstance(out_chunk, tuple):
            out_chunk = out_chunk[0]
        out_flat[qh0:qh1] = out_chunk[:, 0, 0, :]

    out = out_flat.view(TQ, HQ, D).unsqueeze(0).contiguous()
    return out, "flash_attn_with_kvcache_sparse_topk"


def try_flash_sparse_algo_sparse_topk_pkg(
    fs_algo_fn,
    q_bthd: torch.Tensor,
    k_bLhd: torch.Tensor,
    v_bLhd: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    chunk_qh: Optional[int] = None,
):
    """
    Token-level sparse top-k (chapter) emulation using flash-sparse-attn API.

    Similar to try_flash_attn_with_kvcache_sparse_topk_pkg: materialize per-(query,head)
    selected KV rows in chunks and run attention with batch=n_chunk, seq_q=1, heads=1.

    Forward-only helper for benchmarking feasibility/throughput.
    """
    if q_bthd.shape[0] != 1 or k_bLhd.shape[0] != 1 or v_bLhd.shape[0] != 1:
        return None, "flash-sparse sparse helper expects batch size 1 inputs [1,T,H,D]"
    _, TQ, HQ, D = q_bthd.shape
    HK = int(k_bLhd.shape[2])
    if HQ % HK != 0:
        return None, f"HQ ({HQ}) must be divisible by HK ({HK})"
    if block_indices.shape[:3] != (1, TQ, HK):
        return None, "block_indices shape must be [1, TQ, HK, S]"

    S = int(block_indices.shape[-1])
    Lsel = S * int(block_size)
    if Lsel <= 0:
        return None, "invalid selected KV length"

    G = HQ // HK
    q_flat = q_bthd[0].reshape(TQ * HQ, D).contiguous()
    k_src = k_bLhd[0]  # [TK, HK, D]
    v_src = v_bLhd[0]  # [TK, HK, D]
    out_flat = torch.empty((TQ * HQ, D), device=q_bthd.device, dtype=q_bthd.dtype)

    if chunk_qh is None:
        target_bytes = 384 * 1024 * 1024
        bytes_per_qh = max(1, Lsel * D * q_bthd.element_size() * 2)  # K + V
        chunk_qh = max(1, min(4096, target_bytes // bytes_per_qh))

    tok_offsets = torch.arange(block_size, device=q_bthd.device, dtype=torch.int64)[None, None, :]

    for qh0 in range(0, TQ * HQ, int(chunk_qh)):
        qh1 = min(TQ * HQ, qh0 + int(chunk_qh))
        n_chunk = qh1 - qh0

        qh_ids = torch.arange(qh0, qh1, device=q_bthd.device, dtype=torch.int64)
        t_ids = torch.div(qh_ids, HQ, rounding_mode="floor")
        q_heads = qh_ids - t_ids * HQ
        kv_heads = torch.div(q_heads, G, rounding_mode="floor")

        bi_chunk = block_indices[0, t_ids, kv_heads, :].to(torch.int64).contiguous()  # [chunk, S]
        tok_chunk = (bi_chunk[:, :, None] * block_size + tok_offsets).reshape(n_chunk, Lsel)  # [chunk, Lsel]
        h_idx = kv_heads[:, None].expand(n_chunk, Lsel)

        # Materialize sparse-selected rows: [chunk, Lsel, 1, D]
        k_chunk = k_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)
        v_chunk = v_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)
        q_chunk = q_flat[qh0:qh1].contiguous().unsqueeze(1).unsqueeze(2)  # [chunk,1,1,D]

        # Bias API is supported across CUDA/Triton/Flex in this package.
        attn_bias = torch.zeros((n_chunk, 1, 1, Lsel), device=q_bthd.device, dtype=q_bthd.dtype)
        try:
            out_chunk = fs_algo_fn(
                query=q_chunk,
                key=k_chunk,
                value=v_chunk,
                attn_bias=attn_bias,
                softmax_scale=scale,
                is_causal=False,
            )
        except RuntimeError as first_error:
            # Some extension builds require float bias.
            try:
                out_chunk = fs_algo_fn(
                    query=q_chunk,
                    key=k_chunk,
                    value=v_chunk,
                    attn_bias=attn_bias.float(),
                    softmax_scale=scale,
                    is_causal=False,
                )
            except Exception as second_error:
                return None, f"{first_error}; fallback failed: {second_error}"
        except Exception as error:
            return None, str(error)

        if out_chunk is None:
            return None, "flash-sparse-attn returned None"
        if isinstance(out_chunk, tuple):
            out_chunk = out_chunk[0]

        out_flat[qh0:qh1] = out_chunk[:, 0, 0, :]

    out = out_flat.view(TQ, HQ, D).unsqueeze(0).contiguous()
    return out, "flash_sparse_attn_sparse_topk"


# ----------------------------
# NSA baseline (iv)
# ----------------------------

def build_nsa_inputs(q, k, v, block_indices, TK: int):
    """
    NSA is causal sparse attention over same-length q/k/v sequences.
    Trick: build a single sequence where memory is first, queries are later.
    Causality then allows queries to attend to all memory tokens with no masking.
    """
    assert q.shape[0] == 1
    _, TQ, HQ, D = q.shape
    _, TK2, HK, D2 = k.shape
    assert TK2 == TK and D2 == D
    assert v.shape == k.shape
    assert HQ % HK == 0, f"HQ ({HQ}) must be divisible by HK ({HK})"
    assert block_indices.shape[:3] == (1, TQ, HK), "block_indices must be [1,TQ,HK,S]"

    Tfull = TK + TQ
    # q_full: [1, Tfull, HQ, D], k/v_full: [1, Tfull, HK, D]
    q_full = torch.zeros((1, Tfull, HQ, D), device=q.device, dtype=q.dtype, requires_grad=True)
    k_full = torch.zeros((1, Tfull, HK, D), device=k.device, dtype=k.dtype, requires_grad=True)
    v_full = torch.zeros((1, Tfull, HK, D), device=v.device, dtype=v.dtype, requires_grad=True)

    # Copy memory and query segments
    with torch.no_grad():
        q_full[:, TK:, :, :] = q.detach()
        k_full[:, :TK, :, :] = k.detach()
        v_full[:, :TK, :, :] = v.detach()

    # block_indices_full: [1, Tfull, H, topk]
    bi_full = torch.full((1, Tfull, HK, block_indices.shape[-1]), -1, device=q.device, dtype=torch.int32)
    bi_full[:, TK:, :, :] = block_indices  # memory chapters are 0..M-1 (token offsets chap*BS)

    return q_full, k_full, v_full, bi_full, Tfull


def try_import_nsa():
    """
    Attempt to import NSA wrappers.
    """
    try:
        # adjust if your install path differs
        from native_sparse_attention.ops.parallel import (
            ParallelNSAFunction,
            parallel_nsa_bwd,
            parallel_nsa_fwd,
        )

        def parallel_nsa_selected_autograd(
            q, k, v, block_indices, block_counts, block_size, scale, offsets=None
        ):
            # Autograd-enabled selected-attention path.
            return ParallelNSAFunction.apply(
                q, k, v, block_indices, block_counts, block_size, scale, offsets
            )

        return (parallel_nsa_fwd, parallel_nsa_bwd, parallel_nsa_selected_autograd), None
    except Exception as first_error:
        # Fallback: bypass package __init__ side effects (e.g., Transformers config-name collisions).
        # We inject a lightweight package shell and import only ops.parallel.
        try:
            candidates = []
            here = Path(__file__).resolve().parent
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

            # Remove partially loaded modules from failed import attempts.
            for name in list(sys.modules.keys()):
                if name == "native_sparse_attention" or name.startswith("native_sparse_attention."):
                    sys.modules.pop(name, None)

            # Install package stubs so importing submodules does not execute top-level __init__.py.
            nsa_pkg = types.ModuleType("native_sparse_attention")
            nsa_pkg.__path__ = [str(pkg_dir)]
            ops_pkg = types.ModuleType("native_sparse_attention.ops")
            ops_pkg.__path__ = [str(pkg_dir / "ops")]
            sys.modules["native_sparse_attention"] = nsa_pkg
            sys.modules["native_sparse_attention.ops"] = ops_pkg

            parallel_mod = importlib.import_module("native_sparse_attention.ops.parallel")

            def parallel_nsa_selected_autograd(
                q, k, v, block_indices, block_counts, block_size, scale, offsets=None
            ):
                return parallel_mod.ParallelNSAFunction.apply(
                    q, k, v, block_indices, block_counts, block_size, scale, offsets
                )

            return (
                parallel_mod.parallel_nsa_fwd,
                parallel_mod.parallel_nsa_bwd,
                parallel_nsa_selected_autograd,
            ), None
        except Exception as fallback_error:
            return None, f"{first_error}; fallback failed: {fallback_error}"


def try_import_fsa_local():
    """
    Import local FSA copy (root-level) without modifying upstream source tree.
    """
    try:
        from fsa_topk_sparse_attention_local_optimized import FSA_topk_sparse_attention_bthd
        return FSA_topk_sparse_attention_bthd, None
    except Exception as e:
        try:
            from fsa_topk_sparse_attention_local import FSA_topk_sparse_attention_bthd
            return (
                FSA_topk_sparse_attention_bthd,
                f"optimized import failed; using legacy local kernel: {e}",
            )
        except Exception as legacy_error:
            return None, f"optimized import failed: {e}; legacy fallback failed: {legacy_error}"


def try_import_fsa_upstream():
    """
    Import upstream FSA implementation from Flash-Sparse-Attention.
    """
    try:
        from fsa.ops import FSA_topk_sparse_attention
        return FSA_topk_sparse_attention, None
    except Exception as first_error:
        # Fallback: discover local Flash-Sparse-Attention checkout and add it to sys.path.
        try:
            candidates = []
            here = Path(__file__).resolve().parent
            candidates.append(here / "Flash-Sparse-Attention")
            for p in sys.path:
                candidates.append(Path(p) / "Flash-Sparse-Attention")

            repo_root = None
            for cand in candidates:
                if (cand / "fsa" / "ops" / "FSA_topk_sparse_attention.py").exists() and (
                    cand / "nsa_ref" / "ops" / "topk_sparse_attention.py"
                ).exists():
                    repo_root = cand
                    break

            if repo_root is None:
                raise RuntimeError("could not locate Flash-Sparse-Attention package directory")

            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)

            # Clear partially loaded modules from failed import attempts.
            for name in list(sys.modules.keys()):
                if (
                    name == "fsa"
                    or name.startswith("fsa.")
                    or name == "nsa_ref"
                    or name.startswith("nsa_ref.")
                ):
                    sys.modules.pop(name, None)

            from fsa.ops import FSA_topk_sparse_attention
            return FSA_topk_sparse_attention, None
        except Exception as fallback_error:
            return None, f"{first_error}; fallback failed: {fallback_error}"


def try_import_fsa_local_varlen():
    """
    Import local copied FSA varlen API (same signature as upstream) as compatibility fallback.
    """
    try:
        from fsa_topk_sparse_attention_local_optimized import FSA_topk_sparse_attention_varlen_qk
        return FSA_topk_sparse_attention_varlen_qk, None
    except Exception as e:
        try:
            from fsa_topk_sparse_attention_local import FSA_topk_sparse_attention_varlen_qk
            return (
                FSA_topk_sparse_attention_varlen_qk,
                f"optimized varlen import failed; using legacy varlen kernel: {e}",
            )
        except Exception as legacy_error:
            return None, f"optimized varlen import failed: {e}; legacy varlen fallback failed: {legacy_error}"


def try_import_flex_attention():
    """
    Import PyTorch FlexAttention APIs if available.
    """
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        return (flex_attention, create_block_mask), None
    except Exception as e:
        return None, str(e)


def try_import_flash_sparse_algo():
    """
    Import flash-sparse-attention-flash-algo public API.
    Returns:
      ((flash_sparse_attn_func_auto, get_available_backends), None) on success
      (None, error_string) on failure
    """
    try:
        from flash_sparse_attn import flash_sparse_attn_func_auto, get_available_backends
        return (flash_sparse_attn_func_auto, get_available_backends), None
    except Exception as first_error:
        # Fallback: discover local checkout paths.
        try:
            candidates = []
            env_repo = os.getenv("FLASH_SPARSE_ALGO_REPO_DIR", "").strip()
            if env_repo:
                candidates.append(Path(env_repo))

            here = Path(__file__).resolve().parent
            candidates.extend(
                [
                    here / "flash-sparse-attention-flash-algo",
                    here / "flash_sparse_attention_flash_algo",
                    here / "flash_sparse_attn",
                ]
            )

            for parent in [here, *here.parents]:
                candidates.extend(
                    [
                        parent / "flash-sparse-attention-flash-algo",
                        parent / "flash_sparse_attention_flash_algo",
                        parent / "flash_sparse_attn",
                    ]
                )

            for p in sys.path:
                pp = Path(p)
                candidates.extend(
                    [
                        pp / "flash-sparse-attention-flash-algo",
                        pp / "flash_sparse_attention_flash_algo",
                        pp / "flash_sparse_attn",
                    ]
                )

            repo_root = None
            seen = set()
            for cand in candidates:
                key = str(cand.resolve()) if cand.exists() else str(cand)
                if key in seen:
                    continue
                seen.add(key)
                if (cand / "flash_sparse_attn" / "__init__.py").exists():
                    repo_root = cand
                    break

            if repo_root is None:
                raise RuntimeError(
                    "could not locate flash-sparse-attention-flash-algo checkout; "
                    "set FLASH_SPARSE_ALGO_REPO_DIR to its path"
                )

            repo_root_str = str(repo_root.resolve())
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)

            for name in list(sys.modules.keys()):
                if name == "flash_sparse_attn" or name.startswith("flash_sparse_attn."):
                    sys.modules.pop(name, None)

            from flash_sparse_attn import flash_sparse_attn_func_auto, get_available_backends
            return (flash_sparse_attn_func_auto, get_available_backends), None
        except Exception as fallback_error:
            return None, f"{first_error}; fallback failed: {fallback_error}"


# ----------------------------
# Full benchmark (b)
# ----------------------------

def run_benchmarks(cfg: Config, *, iters: int = 5, warmup: int = 2):
    print("\n=== Benchmark config ===")
    print(f"TQ={cfg.TQ} (=batch*seq={cfg.batch_size}*{cfg.seq_len}), TK={cfg.TK}, M={cfg.M}, BS={cfg.BS}, topk={cfg.topk}")
    print(f"HQ={cfg.H}, HK={cfg.HK}, G={cfg.G}, hidden_dim={cfg.hidden_dim} => head_dim D={cfg.D}, dtype={cfg.dtype}")
    free_gb, total_gb = cuda_mem_gb()
    print(f"GPU mem free/total: {free_gb:.1f} / {total_gb:.1f} GB")

    from memory_cross_attn import memory_cross_attn
    try:
        from memory_cross_attn_fsa_opt import memory_cross_attn_fsa_opt
        has_fsa_opt = True
        fsa_opt_import_err = None
    except Exception as e:
        memory_cross_attn_fsa_opt = None
        has_fsa_opt = False
        fsa_opt_import_err = e
    fsa_local_bthd_fn, fsa_local_import_err = try_import_fsa_local()
    has_fsa_local = fsa_local_bthd_fn is not None
    fsa_upstream_fn, fsa_upstream_import_err = try_import_fsa_upstream()
    has_fsa_upstream = fsa_upstream_fn is not None
    fsa_local_varlen_fn, fsa_local_varlen_import_err = try_import_fsa_local_varlen()
    has_fsa_local_varlen = fsa_local_varlen_fn is not None
    flex_mod, flex_import_err = try_import_flex_attention()
    has_flex = flex_mod is not None
    fs_algo_mod, fs_algo_import_err = try_import_flash_sparse_algo()
    has_flash_sparse_algo = fs_algo_mod is not None

    # Shared dO / loss masks
    set_all_seeds(77)
    dO_sparse = torch.randn((1, cfg.TQ, cfg.H, cfg.D), device=cfg.device, dtype=cfg.dtype)
    dO_dense = dO_sparse.view(cfg.batch_size, cfg.seq_len, cfg.H, cfg.D).contiguous()
    token_offsets = torch.arange(cfg.BS, device=cfg.device)[None, :]

    def loss_from_out_sparse(out):
        return (out * dO_sparse).sum()

    def loss_from_out_dense(out):
        return (out * dO_dense).sum()

    def build_dense_inputs(seed: int):
        x_sparse_local = make_inputs(cfg, seed=seed, per_query_random_topk=True)
        # One global top-k chapter set per original batch.
        global_ch = torch.stack(
            [torch.randperm(cfg.M, device=cfg.device)[:cfg.topk] for _ in range(cfg.batch_size)],
            dim=0,
        ).to(torch.int64)  # [B, topk]
        tok = (global_ch[:, :, None] * cfg.BS + token_offsets[None, :, :]).reshape(cfg.batch_size, -1)  # [B, topk*BS]

        q = (
            x_sparse_local["q"]
            .detach()
            .view(cfg.batch_size, cfg.seq_len, cfg.H, cfg.D)
            .contiguous()
            .clone()
            .requires_grad_(True)
        )  # [B, L, H, D]
        k_src = x_sparse_local["k"].detach()[0]  # [TK, H, D]
        v_src = x_sparse_local["v"].detach()[0]
        k_sel = k_src[tok, :, :].contiguous().clone().requires_grad_(True)  # [B, topk*BS, HK, D]
        v_sel = v_src[tok, :, :].contiguous().clone().requires_grad_(True)
        return q, k_sel, v_sel

    def build_dense_full_inputs(seed: int):
        x_sparse_local = make_inputs(cfg, seed=seed, per_query_random_topk=True)
        # One global top-k chapter set per original batch.
        global_ch = torch.stack(
            [torch.randperm(cfg.M, device=cfg.device)[:cfg.topk] for _ in range(cfg.batch_size)],
            dim=0,
        ).to(torch.int64)  # [B, topk]
        tok = (global_ch[:, :, None] * cfg.BS + token_offsets[None, :, :]).reshape(cfg.batch_size, -1)  # [B, topk*BS]
        # Keep full K/V so gather cost is included inside timed forward.
        q = (
            x_sparse_local["q"]
            .detach()
            .view(cfg.batch_size, cfg.seq_len, cfg.H, cfg.D)
            .contiguous()
            .clone()
            .requires_grad_(True)
        )  # [B, L, H, D]
        k = x_sparse_local["k"].detach().clone().requires_grad_(True)  # [1, TK, H, D]
        v = x_sparse_local["v"].detach().clone().requires_grad_(True)
        return q, k, v, tok

    # ---- (i) PyTorch dense math ----
    clear_cuda_cache()
    dense_math_state = {}
    dense_math_seed = [20_260]

    def reinit_dense_math():
        q, k_sel, v_sel = build_dense_inputs(dense_math_seed[0])
        dense_math_seed[0] += 1
        dense_math_state["q"] = q
        dense_math_state["k_sel"] = k_sel
        dense_math_state["v_sel"] = v_sel

    def fwd_math():
        return dense_math_attention(
            dense_math_state["q"], dense_math_state["k_sel"], dense_math_state["v_sel"], cfg.scale
        )
    # math_f_ms, math_b_ms = time_fwd_bwd(
    #     fwd_math, loss_from_out, iters=iters, warmup=warmup,
    #     clear_cache_each_iter=True, reinit_fn=reinit_dense_math
    # )
    # print(f"\n(i) Dense PyTorch math  fwd: {math_f_ms:.3f} ms | bwd: {math_b_ms:.3f} ms")

    # ---- (i.g) Dense math + gather (timed) ----
    clear_cuda_cache()
    dense_math_g_state = {}
    dense_math_g_seed = [20_760]

    def reinit_dense_math_gather():
        q, k, v, tok = build_dense_full_inputs(dense_math_g_seed[0])
        dense_math_g_seed[0] += 1
        dense_math_g_state["q"] = q
        dense_math_g_state["k"] = k
        dense_math_g_state["v"] = v
        dense_math_g_state["tok"] = tok

    def fwd_math_gather():
        k_sel = dense_math_g_state["k"][0, dense_math_g_state["tok"], :, :].contiguous()
        v_sel = dense_math_g_state["v"][0, dense_math_g_state["tok"], :, :].contiguous()
        return dense_math_attention(dense_math_g_state["q"], k_sel, v_sel, cfg.scale)

    # math_g_f_ms, math_g_b_ms = time_fwd_bwd(
    #     fwd_math_gather, loss_from_out_dense, iters=iters, warmup=warmup,
    #     clear_cache_each_iter=True, reinit_fn=reinit_dense_math_gather
    # )
    # print(f"(i.g) Dense math + gather fwd: {math_g_f_ms:.3f} ms | bwd: {math_g_b_ms:.3f} ms")

    # ---- (ii) Flash attention baseline ----
    clear_cuda_cache()
    sdpa_state = {}
    sdpa_seed = [30_260]

    def reinit_sdpa():
        q, k_sel, v_sel = build_dense_inputs(sdpa_seed[0])
        sdpa_seed[0] += 1
        sdpa_state["q"] = q
        sdpa_state["k_sel"] = k_sel
        sdpa_state["v_sel"] = v_sel

    def fwd_sdpa():
        return dense_sdpa_flash_attention(
            sdpa_state["q"], sdpa_state["k_sel"], sdpa_state["v_sel"], cfg.scale
        )
    def zero_sdpa():
        reset_grads_in_state(sdpa_state, ["q", "k_sel", "v_sel"])
    # sdpa_f_ms, sdpa_b_ms = time_fwd_bwd(
    #     fwd_sdpa, loss_from_out_dense, iters=iters, warmup=warmup,
    #     clear_cache_each_iter=True, reinit_fn=reinit_sdpa, zero_grad_fn=zero_sdpa
    # )
    # print(f"(ii) Dense SDPA/Flash     fwd: {sdpa_f_ms:.3f} ms | bwd: {sdpa_b_ms:.3f} ms")

    # ---- (ii.g) Dense SDPA/Flash + gather (timed) ----
    clear_cuda_cache()
    sdpa_g_state = {}
    sdpa_g_seed = [30_760]

    def reinit_sdpa_gather():
        q, k, v, tok = build_dense_full_inputs(sdpa_g_seed[0])
        sdpa_g_seed[0] += 1
        sdpa_g_state["q"] = q
        sdpa_g_state["k"] = k
        sdpa_g_state["v"] = v
        sdpa_g_state["tok"] = tok

    def fwd_sdpa_gather():
        k_sel = sdpa_g_state["k"][0, sdpa_g_state["tok"], :, :].contiguous()
        v_sel = sdpa_g_state["v"][0, sdpa_g_state["tok"], :, :].contiguous()
        return dense_sdpa_flash_attention(sdpa_g_state["q"], k_sel, v_sel, cfg.scale)
    def zero_sdpa_gather():
        reset_grads_in_state(sdpa_g_state, ["q", "k", "v"])

    # sdpa_g_f_ms, sdpa_g_b_ms = time_fwd_bwd(
    #     fwd_sdpa_gather, loss_from_out_dense, iters=iters, warmup=warmup,
    #     clear_cache_each_iter=True, reinit_fn=reinit_sdpa_gather, zero_grad_fn=zero_sdpa_gather
    # )
    # print(f"(ii.g) Dense SDPA + gather fwd: {sdpa_g_f_ms:.3f} ms | bwd: {sdpa_g_b_ms:.3f} ms")

    # If flash-attn package is available, measure it too
    clear_cuda_cache()
    flash_state = {}
    flash_seed = [40_260]

    def reinit_flash():
        q, k_sel, v_sel = build_dense_inputs(flash_seed[0])
        flash_seed[0] += 1
        flash_state["q"] = q
        flash_state["k_sel"] = k_sel
        flash_state["v_sel"] = v_sel

    def fwd_flash_attn_pkg():
        out, msg = try_flash_attn_pkg(flash_state["q"], flash_state["k_sel"], flash_state["v_sel"], cfg.scale)
        if out is None:
            raise RuntimeError(msg)
        return out
    def zero_flash():
        reset_grads_in_state(flash_state, ["q", "k_sel", "v_sel"])
    # try:
    #     fa_f_ms, fa_b_ms = time_fwd_bwd(
    #         fwd_flash_attn_pkg, loss_from_out_dense, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_flash, zero_grad_fn=zero_flash
    #     )
    #     print(f"(ii.b) flash_attn_func     fwd: {fa_f_ms:.3f} ms | bwd: {fa_b_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(ii.b) flash_attn_func: skipped ({e})")

    # # ---- (ii.b.g) flash_attn_func + gather (timed) ----
    # clear_cuda_cache()
    # flash_g_state = {}
    # flash_g_seed = [40_760]

    # def reinit_flash_gather():
    #     q, k, v, tok = build_dense_full_inputs(flash_g_seed[0])
    #     flash_g_seed[0] += 1
    #     flash_g_state["q"] = q
    #     flash_g_state["k"] = k
    #     flash_g_state["v"] = v
    #     flash_g_state["tok"] = tok

    # def fwd_flash_attn_pkg_gather():
    #     k_sel = flash_g_state["k"][0, flash_g_state["tok"], :, :].contiguous()
    #     v_sel = flash_g_state["v"][0, flash_g_state["tok"], :, :].contiguous()
    #     out, msg = try_flash_attn_pkg(flash_g_state["q"], k_sel, v_sel, cfg.scale)
    #     if out is None:
    #         raise RuntimeError(msg)
    #     return out
    # def zero_flash_gather():
    #     reset_grads_in_state(flash_g_state, ["q", "k", "v"])

    # try:
    #     fa_g_f_ms, fa_g_b_ms = time_fwd_bwd(
    #         fwd_flash_attn_pkg_gather, loss_from_out_dense, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_flash_gather, zero_grad_fn=zero_flash_gather
    #     )
    #     print(f"(ii.b.g) flash_attn + gather fwd: {fa_g_f_ms:.3f} ms | bwd: {fa_g_b_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(ii.b.g) flash_attn + gather: skipped ({e})")

    # ---- (v) flash_attn_with_kvcache forward-only ----
    clear_cuda_cache()
    kvc_state = {}
    kvc_seed = [45_260]

    def reinit_flash_kvcache():
        q, k_sel, v_sel = build_dense_inputs(kvc_seed[0])
        kvc_seed[0] += 1
        kvc_state["q"] = q
        kvc_state["k_sel"] = k_sel
        kvc_state["v_sel"] = v_sel

    def fwd_flash_kvcache():
        out, msg = try_flash_attn_with_kvcache_pkg(
            kvc_state["q"], kvc_state["k_sel"], kvc_state["v_sel"], cfg.scale
        )
        if out is None:
            raise RuntimeError(msg)
        return out

    # try:
    #     kvc_f_ms = time_fwd_only(
    #         fwd_flash_kvcache, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_flash_kvcache
    #     )
    #     print(f"(v) flash_attn_with_kvcache fwd: {kvc_f_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(v) flash_attn_with_kvcache: skipped ({e})")

    # ---- (v.g) flash_attn_with_kvcache + gather (timed, fwd only) ----
    clear_cuda_cache()
    kvc_g_state = {}
    kvc_g_seed = [45_760]

    def reinit_flash_kvcache_gather():
        q, k, v, tok = build_dense_full_inputs(kvc_g_seed[0])
        kvc_g_seed[0] += 1
        kvc_g_state["q"] = q
        kvc_g_state["k"] = k
        kvc_g_state["v"] = v
        kvc_g_state["tok"] = tok

    def fwd_flash_kvcache_gather():
        k_sel = kvc_g_state["k"][0, kvc_g_state["tok"], :, :].contiguous()
        v_sel = kvc_g_state["v"][0, kvc_g_state["tok"], :, :].contiguous()
        out, msg = try_flash_attn_with_kvcache_pkg(kvc_g_state["q"], k_sel, v_sel, cfg.scale)
        if out is None:
            raise RuntimeError(msg)
        return out

    # try:
    #     kvc_g_f_ms = time_fwd_only(
    #         fwd_flash_kvcache_gather, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_flash_kvcache_gather
    #     )
    #     print(f"(v.g) flash_attn_with_kvcache + gather fwd: {kvc_g_f_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(v.g) flash_attn_with_kvcache + gather: skipped ({e})")

    # ---- (self) Sequence self-attention (B x L, LxL per batch) ----
    # print(f"\n(self) Sequence self-attn on [B={cfg.batch_size}, L={cfg.seq_len}] (each batch does {cfg.seq_len}x{cfg.seq_len})")
    # set_all_seeds(88)
    dO_self = torch.randn((cfg.batch_size, cfg.seq_len, cfg.H, cfg.D), device=cfg.device, dtype=cfg.dtype)

    def loss_from_out_self(out):
        return (out * dO_self).sum()

    def build_self_inputs(seed: int):
        set_all_seeds(seed)
        x = torch.randn((cfg.batch_size, cfg.seq_len, cfg.H, cfg.D), device=cfg.device, dtype=cfg.dtype, requires_grad=True)
        return x

    def build_self_full_inputs(seed: int):
        set_all_seeds(seed)
        l_full = cfg.seq_len + max(256, cfg.seq_len // 4)
        x_full = torch.randn((cfg.batch_size, l_full, cfg.H, cfg.D), device=cfg.device, dtype=cfg.dtype, requires_grad=True)
        idx = torch.stack(
            [torch.randperm(l_full, device=cfg.device)[:cfg.seq_len] for _ in range(cfg.batch_size)],
            dim=0,
        ).to(torch.int64)  # [B, L]
        return x_full, idx

    batch_idx = torch.arange(cfg.batch_size, device=cfg.device, dtype=torch.int64)[:, None]

    # (self.i) PyTorch SDPA self-attn
    clear_cuda_cache()
    self_sdpa_state = {}
    self_sdpa_seed = [70_260]

    def reinit_self_sdpa():
        x = build_self_inputs(self_sdpa_seed[0])
        self_sdpa_seed[0] += 1
        self_sdpa_state["x"] = x

    def fwd_self_sdpa():
        x = self_sdpa_state["x"]
        return dense_sdpa_flash_attention(x, x, x, cfg.scale)
    def zero_self_sdpa():
        reset_grads_in_state(self_sdpa_state, ["x"])

    # try:
    #     self_sdpa_f_ms, self_sdpa_b_ms = time_fwd_bwd(
    #         fwd_self_sdpa, loss_from_out_self, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_self_sdpa, zero_grad_fn=zero_self_sdpa
    #     )
    #     print(f"(self.i) PyTorch SDPA self-attn       fwd: {self_sdpa_f_ms:.3f} ms | bwd: {self_sdpa_b_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(self.i) PyTorch SDPA self-attn: skipped ({e})")

    # (self.i.g) PyTorch SDPA self-attn + gather
    clear_cuda_cache()
    self_sdpa_g_state = {}
    self_sdpa_g_seed = [70_760]

    def reinit_self_sdpa_gather():
        x_full, idx = build_self_full_inputs(self_sdpa_g_seed[0])
        self_sdpa_g_seed[0] += 1
        self_sdpa_g_state["x_full"] = x_full
        self_sdpa_g_state["idx"] = idx

    def fwd_self_sdpa_gather():
        x = self_sdpa_g_state["x_full"][batch_idx, self_sdpa_g_state["idx"], :, :].contiguous()
        return dense_sdpa_flash_attention(x, x, x, cfg.scale)
    def zero_self_sdpa_gather():
        reset_grads_in_state(self_sdpa_g_state, ["x_full"])

    # try:
    #     self_sdpa_g_f_ms, self_sdpa_g_b_ms = time_fwd_bwd(
    #         fwd_self_sdpa_gather, loss_from_out_self, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_self_sdpa_gather, zero_grad_fn=zero_self_sdpa_gather
    #     )
    #     print(f"(self.i.g) PyTorch SDPA + gather      fwd: {self_sdpa_g_f_ms:.3f} ms | bwd: {self_sdpa_g_b_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(self.i.g) PyTorch SDPA + gather: skipped ({e})")

    # (self.ii) flash_attn_func self-attn
    clear_cuda_cache()
    self_fa_state = {}
    self_fa_seed = [71_260]

    def reinit_self_flash():
        x = build_self_inputs(self_fa_seed[0])
        self_fa_seed[0] += 1
        self_fa_state["x"] = x

    def fwd_self_flash():
        x = self_fa_state["x"]
        out, msg = try_flash_attn_pkg(x, x, x, cfg.scale)
        if out is None:
            raise RuntimeError(msg)
        return out
    def zero_self_flash():
        reset_grads_in_state(self_fa_state, ["x"])

    # try:
    #     self_fa_f_ms, self_fa_b_ms = time_fwd_bwd(
    #         fwd_self_flash, loss_from_out_self, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_self_flash, zero_grad_fn=zero_self_flash
    #     )
    #     print(f"(self.ii) flash_attn_func self-attn    fwd: {self_fa_f_ms:.3f} ms | bwd: {self_fa_b_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(self.ii) flash_attn_func self-attn: skipped ({e})")

    # (self.ii.g) flash_attn_func self-attn + gather
    clear_cuda_cache()
    self_fa_g_state = {}
    self_fa_g_seed = [71_760]

    def reinit_self_flash_gather():
        x_full, idx = build_self_full_inputs(self_fa_g_seed[0])
        self_fa_g_seed[0] += 1
        self_fa_g_state["x_full"] = x_full
        self_fa_g_state["idx"] = idx

    def fwd_self_flash_gather():
        x = self_fa_g_state["x_full"][batch_idx, self_fa_g_state["idx"], :, :].contiguous()
        out, msg = try_flash_attn_pkg(x, x, x, cfg.scale)
        if out is None:
            raise RuntimeError(msg)
        return out
    def zero_self_flash_gather():
        reset_grads_in_state(self_fa_g_state, ["x_full"])

    try:
        self_fa_g_f_ms, self_fa_g_b_ms = time_fwd_bwd(
            fwd_self_flash_gather, loss_from_out_self, iters=iters, warmup=warmup,
            clear_cache_each_iter=True, reinit_fn=reinit_self_flash_gather, zero_grad_fn=zero_self_flash_gather
        )
        print(f"(self.ii.g) flash_attn + gather       fwd: {self_fa_g_f_ms:.3f} ms | bwd: {self_fa_g_b_ms:.3f} ms")
    except Exception as e:
        print(f"(self.ii.g) flash_attn + gather: skipped ({e})")
    
    # ---- (ii.b.g) flash_attn_func + gather (timed) ----
    clear_cuda_cache()
    flash_g_state = {}
    flash_g_seed = [40_760]

    def reinit_flash_gather():
        q, k, v, tok = build_dense_full_inputs(flash_g_seed[0])
        flash_g_seed[0] += 1
        flash_g_state["q"] = q
        flash_g_state["k"] = k
        flash_g_state["v"] = v
        flash_g_state["tok"] = tok

    def fwd_flash_attn_pkg_gather():
        k_sel = flash_g_state["k"][0, flash_g_state["tok"], :, :].contiguous()
        v_sel = flash_g_state["v"][0, flash_g_state["tok"], :, :].contiguous()
        out, msg = try_flash_attn_pkg(flash_g_state["q"], k_sel, v_sel, cfg.scale)
        if out is None:
            raise RuntimeError(msg)
        return out
    def zero_flash_gather():
        reset_grads_in_state(flash_g_state, ["q", "k", "v"])

    try:
        fa_g_f_ms, fa_g_b_ms = time_fwd_bwd(
            fwd_flash_attn_pkg_gather, loss_from_out_dense, iters=iters, warmup=warmup,
            clear_cache_each_iter=True, reinit_fn=reinit_flash_gather, zero_grad_fn=zero_flash_gather
        )
        print(f"(ii.b.g) flash_attn + gather fwd: {fa_g_f_ms:.3f} ms | bwd: {fa_g_b_ms:.3f} ms")
    except Exception as e:
        print(f"(ii.b.g) flash_attn + gather: skipped ({e})")

    # ---- (ii.c.g) flash-sparse-attention-flash-algo + gather (timed) ----
    clear_cuda_cache()
    if not has_flash_sparse_algo:
        print(f"(ii.c.g) flash-sparse-attn + gather: skipped (import failed: {fs_algo_import_err})")
    else:
        flash_sparse_attn_func_auto, get_available_backends = fs_algo_mod
        fs_algo_state = {}
        fs_algo_seed = [42_260]
        requested_backend = os.getenv("FLASH_SPARSE_ALGO_BACKEND", "auto").strip().lower()
        if requested_backend in ("", "auto", "none"):
            requested_backend = None

        try:
            available_backends = get_available_backends()
        except Exception:
            available_backends = []

        if requested_backend is not None and available_backends and requested_backend not in available_backends:
            print(
                f"(ii.c.g) flash-sparse-attn + gather: skipped "
                f"(requested backend '{requested_backend}' not available; available={available_backends})"
            )
        else:
            try:
                fs_algo_fn = flash_sparse_attn_func_auto(backend=requested_backend)
                fs_algo_backend = (
                    requested_backend
                    if requested_backend is not None
                    else (available_backends[0] if available_backends else "auto")
                )
            except Exception as e:
                fs_algo_fn = None
                fs_algo_backend = "unknown"
                print(f"(ii.c.g) flash-sparse-attn + gather: skipped (backend init failed: {e})")

            if fs_algo_fn is not None:
                def reinit_flash_sparse_algo_gather():
                    q, k, v, tok = build_dense_full_inputs(fs_algo_seed[0])
                    fs_algo_seed[0] += 1
                    fs_algo_state["q"] = q
                    fs_algo_state["k"] = k
                    fs_algo_state["v"] = v
                    fs_algo_state["tok"] = tok
                    fs_algo_state["attn_bias"] = torch.zeros((1, 1, 1, 1), device=q.device, dtype=q.dtype)

                def fwd_flash_sparse_algo_gather():
                    k_sel = fs_algo_state["k"][0, fs_algo_state["tok"], :, :].contiguous()
                    v_sel = fs_algo_state["v"][0, fs_algo_state["tok"], :, :].contiguous()
                    try:
                        return fs_algo_fn(
                            query=fs_algo_state["q"],
                            key=k_sel,
                            value=v_sel,
                            attn_bias=fs_algo_state["attn_bias"],
                            softmax_scale=cfg.scale,
                            is_causal=False,
                        )
                    except RuntimeError as err:
                        if "expected scalar type Float but found BFloat16" in str(err):
                            return fs_algo_fn(
                                query=fs_algo_state["q"],
                                key=k_sel,
                                value=v_sel,
                                attn_bias=fs_algo_state["attn_bias"].float(),
                                softmax_scale=cfg.scale,
                                is_causal=False,
                            )
                        raise

                def zero_flash_sparse_algo_gather():
                    reset_grads_in_state(fs_algo_state, ["q", "k", "v"])

                try:
                    fs_algo_f_ms, fs_algo_b_ms = time_fwd_bwd(
                        fwd_flash_sparse_algo_gather,
                        loss_from_out_dense,
                        iters=iters,
                        warmup=warmup,
                        clear_cache_each_iter=True,
                        reinit_fn=reinit_flash_sparse_algo_gather,
                        zero_grad_fn=zero_flash_sparse_algo_gather,
                    )
                    print(
                        f"(ii.c.g) flash-sparse-attn ({fs_algo_backend}) + gather "
                        f"fwd: {fs_algo_f_ms:.3f} ms | bwd: {fs_algo_b_ms:.3f} ms"
                    )
                except Exception as e:
                    print(f"(ii.c.g) flash-sparse-attn ({fs_algo_backend}) + gather: skipped ({e})")

    # ---- (mlp) Transformer FFN layer on full batch ----
    print(
        f"\n(mlp) FFN on [B={cfg.batch_size}, L={cfg.seq_len}, d_model={cfg.hidden_dim}]"
    )
    clear_cuda_cache()
    mlp_state = {}
    mlp_seed = [72_260]
    mlp_ffn_dim = 4 * cfg.hidden_dim
    dO_mlp = torch.randn(
        (cfg.batch_size, cfg.seq_len, cfg.hidden_dim),
        device=cfg.device,
        dtype=cfg.dtype,
    )

    def loss_from_out_mlp(out):
        return (out * dO_mlp).sum()

    def reinit_mlp():
        set_all_seeds(mlp_seed[0])
        mlp_seed[0] += 1
        # F.linear expects weight shapes [out_features, in_features].
        mlp_state["x"] = torch.randn(
            (cfg.batch_size, cfg.seq_len, cfg.hidden_dim),
            device=cfg.device,
            dtype=cfg.dtype,
            requires_grad=True,
        )
        mlp_state["w1"] = (
            torch.randn(
                (mlp_ffn_dim, cfg.hidden_dim),
                device=cfg.device,
                dtype=cfg.dtype,
            )
            / math.sqrt(cfg.hidden_dim)
        ).requires_grad_(True)
        mlp_state["b1"] = torch.zeros(
            (mlp_ffn_dim,),
            device=cfg.device,
            dtype=cfg.dtype,
            requires_grad=True,
        )
        mlp_state["w2"] = (
            torch.randn(
                (cfg.hidden_dim, mlp_ffn_dim),
                device=cfg.device,
                dtype=cfg.dtype,
            )
            / math.sqrt(mlp_ffn_dim)
        ).requires_grad_(True)
        mlp_state["b2"] = torch.zeros(
            (cfg.hidden_dim,),
            device=cfg.device,
            dtype=cfg.dtype,
            requires_grad=True,
        )

    def fwd_mlp():
        h = F.linear(mlp_state["x"], mlp_state["w1"], mlp_state["b1"])
        h = F.gelu(h, approximate="tanh")
        return F.linear(h, mlp_state["w2"], mlp_state["b2"])

    def zero_mlp():
        reset_grads_in_state(mlp_state, ["x", "w1", "b1", "w2", "b2"])

    try:
        mlp_f_ms, mlp_b_ms = time_fwd_bwd(
            fwd_mlp,
            loss_from_out_mlp,
            iters=iters,
            warmup=warmup,
            clear_cache_each_iter=True,
            reinit_fn=reinit_mlp,
            zero_grad_fn=zero_mlp,
        )
        print(
            f"(mlp.i) PyTorch Linear-GELU-Linear fwd: {mlp_f_ms:.3f} ms | bwd: {mlp_b_ms:.3f} ms"
        )
    except Exception as e:
        print(f"(mlp.i) PyTorch Linear-GELU-Linear: skipped ({e})")

    # ---- (vi) flash_attn_with_kvcache sparse top-k (timed, fwd only) ----
    clear_cuda_cache()
    kvc_sparse_state = {}
    kvc_sparse_seed = [46_260]

    def reinit_flash_kvcache_sparse():
        x = make_inputs(cfg, seed=kvc_sparse_seed[0], per_query_random_topk=True)
        kvc_sparse_seed[0] += 1
        kvc_sparse_state["q"] = x["q"]
        kvc_sparse_state["k"] = x["k"]
        kvc_sparse_state["v"] = x["v"]
        kvc_sparse_state["bi"] = x["block_indices"]

    def fwd_flash_kvcache_sparse():
        out, msg = try_flash_attn_with_kvcache_sparse_topk_pkg(
            kvc_sparse_state["q"],
            kvc_sparse_state["k"],
            kvc_sparse_state["v"],
            kvc_sparse_state["bi"],
            block_size=cfg.BS,
            scale=cfg.scale,
            chunk_qh=None,  # auto
        )
        if out is None:
            raise RuntimeError(msg)
        return out

    try:
        kvc_sparse_f_ms = time_fwd_only(
            fwd_flash_kvcache_sparse, iters=iters, warmup=warmup,
            clear_cache_each_iter=True, reinit_fn=reinit_flash_kvcache_sparse
        )
        print(f"(vi) flash_attn_with_kvcache sparse-topk fwd: {kvc_sparse_f_ms:.3f} ms")
    except Exception as e:
        print(f"(vi) flash_attn_with_kvcache sparse-topk: skipped ({e})")

    # ---- (vi.c) flash-sparse-attn sparse top-k (timed, fwd only) ----
    clear_cuda_cache()
    if not has_flash_sparse_algo:
        print(f"(vi.c) flash-sparse-attn sparse-topk: skipped (import failed: {fs_algo_import_err})")
    else:
        flash_sparse_attn_func_auto, get_available_backends = fs_algo_mod
        requested_backend = os.getenv("FLASH_SPARSE_ALGO_BACKEND", "auto").strip().lower()
        if requested_backend in ("", "auto", "none"):
            requested_backend = None
        try:
            available_backends = get_available_backends()
        except Exception:
            available_backends = []

        if requested_backend is not None and available_backends and requested_backend not in available_backends:
            print(
                f"(vi.c) flash-sparse-attn sparse-topk: skipped "
                f"(requested backend '{requested_backend}' not available; available={available_backends})"
            )
        else:
            try:
                fs_algo_fn = flash_sparse_attn_func_auto(backend=requested_backend)
                fs_algo_backend = (
                    requested_backend
                    if requested_backend is not None
                    else (available_backends[0] if available_backends else "auto")
                )
            except Exception as e:
                fs_algo_fn = None
                fs_algo_backend = "unknown"
                print(f"(vi.c) flash-sparse-attn sparse-topk: skipped (backend init failed: {e})")

            if fs_algo_fn is not None:
                fs_algo_sparse_state = {}
                fs_algo_sparse_seed = [46_760]
                sparse_chunk_qh_raw = os.getenv("FLASH_SPARSE_ALGO_SPARSE_CHUNK_QH", "auto").strip().lower()
                sparse_chunk_qh = None if sparse_chunk_qh_raw in ("", "auto", "none") else int(sparse_chunk_qh_raw)

                def reinit_flash_sparse_algo_sparse():
                    x = make_inputs(cfg, seed=fs_algo_sparse_seed[0], per_query_random_topk=True)
                    fs_algo_sparse_seed[0] += 1
                    fs_algo_sparse_state["q"] = x["q"]
                    fs_algo_sparse_state["k"] = x["k"]
                    fs_algo_sparse_state["v"] = x["v"]
                    fs_algo_sparse_state["bi"] = x["block_indices"]

                def fwd_flash_sparse_algo_sparse():
                    out, msg = try_flash_sparse_algo_sparse_topk_pkg(
                        fs_algo_fn,
                        fs_algo_sparse_state["q"],
                        fs_algo_sparse_state["k"],
                        fs_algo_sparse_state["v"],
                        fs_algo_sparse_state["bi"],
                        block_size=cfg.BS,
                        scale=cfg.scale,
                        chunk_qh=sparse_chunk_qh,
                    )
                    if out is None:
                        raise RuntimeError(msg)
                    return out

                try:
                    fs_algo_sparse_f_ms = time_fwd_only(
                        fwd_flash_sparse_algo_sparse,
                        iters=iters,
                        warmup=warmup,
                        clear_cache_each_iter=True,
                        reinit_fn=reinit_flash_sparse_algo_sparse,
                    )
                    print(
                        f"(vi.c) flash-sparse-attn ({fs_algo_backend}) sparse-topk fwd: "
                        f"{fs_algo_sparse_f_ms:.3f} ms"
                    )
                except Exception as e:
                    print(f"(vi.c) flash-sparse-attn ({fs_algo_backend}) sparse-topk: skipped ({e})")

    # ---- (iii) Custom sparse kernel (variable per-query topk) ----
    print("\n(iii) Custom sparse (variable per-query topk):")
    if not has_fsa_opt:
        print(f"  fsa_opt: skipped (import failed: {fsa_opt_import_err})")
    # Initialize once globally and reuse for all sparse strategies.
    sparse_x = make_inputs(cfg, seed=50_260, per_query_random_topk=True)
    sparse_state = {
        "q": sparse_x["q"],
        "k": sparse_x["k"],
        "v": sparse_x["v"],
        "bi": sparse_x["block_indices"],
    }

    def zero_sparse():
        reset_grads_in_state(sparse_state, ["q", "k", "v"])

    # for strat_idx, strat in enumerate(["a", "b", "c", "d"]):
    #     clear_cuda_cache()

    #     def fwd_custom():
    #         return memory_cross_attn(
    #             sparse_state["q"], sparse_state["k"], sparse_state["v"], sparse_state["bi"],
    #             cfg.BS, scale=cfg.scale,
    #             dkv_strategy=strat,
    #             q_chunk_size=1024,
    #             d_chunk_size=256,
    #         )

    #     f_ms, b_ms = time_fwd_bwd(
    #         fwd_custom, loss_from_out_sparse, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=None, zero_grad_fn=zero_sparse
    #     )
    #     print(f"  strategy={strat}  fwd: {f_ms:.3f} ms | bwd: {b_ms:.3f} ms")

    # if has_fsa_opt:
    #     clear_cuda_cache()

    #     def fwd_custom_fsa_opt():
    #         return memory_cross_attn_fsa_opt(
    #             sparse_state["q"], sparse_state["k"], sparse_state["v"], sparse_state["bi"],
    #             cfg.BS, scale=cfg.scale
    #         )

    #     f_ms, b_ms = time_fwd_bwd(
    #         fwd_custom_fsa_opt, loss_from_out_sparse, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=None, zero_grad_fn=zero_sparse
    #     )
    #     print(f"  strategy=fsa_opt  fwd: {f_ms:.3f} ms | bwd: {b_ms:.3f} ms")

    # ---- (viii) PyTorch FlexAttention block-sparse baseline ----
    # Disabled by request.
    # if not has_flex:
    #     print(f"  strategy=flex_attention: skipped (import failed: {flex_import_err})")
    # else:
    #     ...
    print("  strategy=flex_attention: skipped (disabled)")

    # ---- (vii) FSA local copy baseline (inner API modified, prefix trick) ----
    print("\n(vii) FSA local-copy baseline (prefix memory timeline)")
    if not has_fsa_local:
        print(f"  FSA local-copy: skipped (import failed: {fsa_local_import_err})")
    elif cfg.BS not in {32, 64, 128, 256, 512, 1024}:
        print(f"  FSA local-copy: skipped (unsupported block_size BS={cfg.BS}; requires one of 32/64/128/256/512/1024)")
    else:
        clear_cuda_cache()
        fsa_dkdv_bq = os.getenv("FSA_LOCAL_BWD_DKDV_BQ", "auto")
        fsa_dq_bq = os.getenv("FSA_LOCAL_BWD_DQ_BQ", "auto")
        fsa_dq_loops = os.getenv("FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS", "auto")
        fsa_nsa_fwd = os.getenv("FSA_LOCAL_USE_NSA_STYLE_FWD", "1")
        fsa_native_nsa_fwd = os.getenv("FSA_LOCAL_USE_NSA_NATIVE_FWD", "0")
        fsa_pad_g16 = os.getenv("FSA_LOCAL_PAD_G_TO_16", "1")
        fsa_small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "pad")
        fsa_torch_chunk = os.getenv("FSA_LOCAL_TORCH_CHUNK_TOKENS", "512")
        fsa_fwd_chunk = os.getenv("FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL", "auto")
        fsa_head_tile = os.getenv("FSA_LOCAL_HEAD_TILE", "auto")
        fsa_sort_qidx = os.getenv("FSA_LOCAL_SORT_TOPK_Q_IDX", "auto")
        fsa_dq_accum_mode = os.getenv("FSA_LOCAL_DQ_ACCUM_MODE", "atomic")
        fsa_compact_blocks = os.getenv("FSA_LOCAL_COMPACT_ACTIVE_BLOCKS", "auto")
        fsa_max_kblk = os.getenv("FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE", "128")
        fsa_dkdv_mode = os.getenv("FSA_LOCAL_DKDV_MODE", "auto")
        print(
            f"  FSA local tuning: disable_causal_mask=True, "
            f"FSA_LOCAL_BWD_DKDV_BQ={fsa_dkdv_bq}, "
            f"FSA_LOCAL_BWD_DQ_BQ={fsa_dq_bq}, "
            f"FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS={fsa_dq_loops}, "
            f"FSA_LOCAL_USE_NSA_STYLE_FWD={fsa_nsa_fwd}, "
            f"FSA_LOCAL_USE_NSA_NATIVE_FWD={fsa_native_nsa_fwd}, "
            f"FSA_LOCAL_PAD_G_TO_16={fsa_pad_g16}, "
            f"FSA_LOCAL_SMALL_G_MODE={fsa_small_g_mode}, "
            f"FSA_LOCAL_TORCH_CHUNK_TOKENS={fsa_torch_chunk}, "
            f"FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL={fsa_fwd_chunk}, "
            f"FSA_LOCAL_HEAD_TILE={fsa_head_tile}, "
            f"FSA_LOCAL_SORT_TOPK_Q_IDX={fsa_sort_qidx}, "
            f"FSA_LOCAL_DQ_ACCUM_MODE={fsa_dq_accum_mode}, "
            f"FSA_LOCAL_COMPACT_ACTIVE_BLOCKS={fsa_compact_blocks}, "
            f"FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE={fsa_max_kblk}, "
            f"FSA_LOCAL_DKDV_MODE={fsa_dkdv_mode}"
        )
        fsa_local_state = {}
        fsa_local_seed = [55_260]

        def reinit_fsa_local():
            x = make_inputs(cfg, seed=fsa_local_seed[0], per_query_random_topk=True)
            fsa_local_seed[0] += 1
            q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
                x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
            )
            bi_full_sorted = bi_full.sort(dim=-1).values.contiguous()
            topk_idx_hns = (
                bi_full_sorted
                .permute(0, 2, 1, 3)
                .reshape(bi_full_sorted.shape[2], bi_full_sorted.shape[0] * bi_full_sorted.shape[1], bi_full_sorted.shape[3])
                .to(torch.int32)
                .contiguous()
            )
            fsa_local_state["q_full"] = q_full
            fsa_local_state["k_full"] = k_full
            fsa_local_state["v_full"] = v_full
            fsa_local_state["bi_full"] = bi_full
            fsa_local_state["topk_idx_hns"] = topk_idx_hns
            fsa_local_state["cu_q"] = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
            fsa_local_state["cu_k"] = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
            fsa_local_state["Tfull"] = Tfull

        def fwd_fsa_local():
            return fsa_local_bthd_fn(
                q_bthd=fsa_local_state["q_full"],
                k_bthd=fsa_local_state["k_full"][:, :cfg.TK, :, :],
                v_bthd=fsa_local_state["v_full"][:, :cfg.TK, :, :],
                block_indices_bths=None,
                block_size=cfg.BS,
                softmax_scale=cfg.scale,
                cu_seqlens_q=fsa_local_state["cu_q"],
                cu_seqlens_k=fsa_local_state["cu_k"],
                topk_idx_hns=fsa_local_state["topk_idx_hns"],
                assume_sorted_topk=True,
                disable_causal_mask=True,
            )

        def loss_query_only_fsa_local(o_full):
            o_q = o_full[:, cfg.TK:, :, :]
            return (o_q * dO_sparse).sum()

        def zero_fsa_local():
            reset_grads_in_state(fsa_local_state, ["q_full", "k_full", "v_full"])

        try:
            fsa_local_f_ms, fsa_local_b_ms = time_fwd_bwd(
                fwd_fsa_local, loss_query_only_fsa_local, iters=iters, warmup=warmup,
                clear_cache_each_iter=True, reinit_fn=reinit_fsa_local, zero_grad_fn=zero_fsa_local
            )
            print(f"  FSA local-copy (query loss only)  fwd: {fsa_local_f_ms:.3f} ms | bwd: {fsa_local_b_ms:.3f} ms")
        except Exception as e:
            print(f"  FSA local-copy: skipped (runtime/compile failure: {e})")
            if "illegal memory access" in str(e).lower():
                print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
                return

    # ---- (iv) NSA baseline ----
    clear_cuda_cache()
    nsa_mod, err = try_import_nsa()
    if nsa_mod is None:
        print(f"\n(iv) NSA baseline: skipped (could not import native_sparse_attention: {err})")
        nsa_mod = None

    if nsa_mod is not None:
        parallel_nsa_fwd, parallel_nsa_bwd, parallel_nsa_selected_autograd = nsa_mod
        print("\n(iv) NSA selected-attn baseline (causal trick: memory first, queries later)")
        nsa_state = {}
        nsa_seed = [60_260]

        def reinit_nsa():
            x = make_inputs(cfg, seed=nsa_seed[0], per_query_random_topk=True)
            nsa_seed[0] += 1
            q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
                x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
            )
            nsa_state["q_full"] = q_full
            nsa_state["k_full"] = k_full
            nsa_state["v_full"] = v_full
            nsa_state["bi_full"] = bi_full
            nsa_state["Tfull"] = Tfull

        def check_nsa_feasibility() -> Optional[str]:
            # Build one sample to inspect the actual head/group layout used in this benchmark.
            reinit_nsa()
            hq = int(nsa_state["q_full"].shape[2])
            hk = int(nsa_state["k_full"].shape[2])
            g = hq // hk if hk > 0 and hq % hk == 0 else -1
            if g < 1:
                return f"invalid NSA head ratio HQ/HK: HQ={hq}, HK={hk}"
            # Current NSA Triton kernels require tl.dot tile minima; in practice G must be >=16.
            if g < 16:
                return f"unsupported head-group ratio G={g} (requires >=16 for current NSA kernels)"
            if (g & (g - 1)) != 0:
                return f"unsupported head-group ratio G={g} (must be power-of-two)"
            if cfg.BS < 16:
                return f"unsupported block_size BS={cfg.BS} (requires >=16)"
            if cfg.D < 16:
                return f"unsupported head_dim D={cfg.D} (requires >=16)"
            return None

        def loss_query_only(o_full):
            o_q = o_full[:, cfg.TK:, :, :]
            return (o_q * dO_sparse).sum()

        def fwd_nsa():
            # Use autograd-backed selected path; raw `parallel_nsa_fwd` is forward-only.
            return parallel_nsa_selected_autograd(
                q=nsa_state["q_full"],
                k=nsa_state["k_full"],
                v=nsa_state["v_full"],
                block_indices=nsa_state["bi_full"],
                block_counts=cfg.topk,
                block_size=cfg.BS,
                scale=cfg.scale,
                offsets=None,
            )

        def zero_nsa():
            reset_grads_in_state(nsa_state, ["q_full", "k_full", "v_full"])

        nsa_skip_reason = check_nsa_feasibility()
        if nsa_skip_reason is not None:
            print(f"  NSA baseline: skipped ({nsa_skip_reason})")
        else:
            clear_cuda_cache()
            try:
                nsa_f_ms, nsa_b_ms = time_fwd_bwd(
                    fwd_nsa, loss_query_only, iters=iters, warmup=warmup,
                    clear_cache_each_iter=True, reinit_fn=reinit_nsa, zero_grad_fn=zero_nsa
                )
                print(
                    f"  NSA (includes computing outputs for memory positions too): "
                    f"fwd {nsa_f_ms:.3f} ms | bwd {nsa_b_ms:.3f} ms"
                )
            except Exception as e:
                print(f"  NSA baseline: skipped (runtime/compile failure: {e})")

    # ---- (viii) FLA FSA upstream baseline ----
    print("\n(viii) FLA FSA upstream baseline (prefix memory timeline)")
    if not has_fsa_upstream:
        print(f"  FLA FSA upstream: skipped (import failed: {fsa_upstream_import_err})")
        return
    if cfg.BS not in {32, 64, 128, 256}:
        print(f"  FLA FSA upstream: skipped (unsupported block_size BS={cfg.BS}; requires one of 32/64/128/256)")
        return

    fla_fsa_state = {}
    fla_fsa_seed = [65_260]

    def reinit_fla_fsa():
        x = make_inputs(cfg, seed=fla_fsa_seed[0], per_query_random_topk=True)
        fla_fsa_seed[0] += 1
        q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
            x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
        )
        bi_full_sorted = bi_full.sort(dim=-1).values.contiguous()
        topk_idx_hns = (
            bi_full_sorted
            .permute(0, 2, 1, 3)
            .reshape(
                bi_full_sorted.shape[2],
                bi_full_sorted.shape[0] * bi_full_sorted.shape[1],
                bi_full_sorted.shape[3],
            )
            .to(torch.int32)
            .contiguous()
        )
        fla_fsa_state["q_full"] = q_full
        fla_fsa_state["k_full"] = k_full
        fla_fsa_state["v_full"] = v_full
        fla_fsa_state["topk_idx_hns"] = topk_idx_hns
        fla_fsa_state["cu_q"] = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
        fla_fsa_state["Tfull"] = Tfull

    def fwd_fla_fsa():
        o = fsa_upstream_fn(
            q=fla_fsa_state["q_full"][0],
            k=fla_fsa_state["k_full"][0],
            v=fla_fsa_state["v_full"][0],
            topk_idx=fla_fsa_state["topk_idx_hns"],
            block_size=cfg.BS,
            cu_seqlens=fla_fsa_state["cu_q"],
            softmax_scale=cfg.scale,
        )
        return o.unsqueeze(0)

    def loss_query_only_fla(o_full):
        o_q = o_full[:, cfg.TK:, :, :]
        return (o_q * dO_sparse).sum()

    def zero_fla_fsa():
        reset_grads_in_state(fla_fsa_state, ["q_full", "k_full", "v_full"])

    clear_cuda_cache()
    try:
        fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
            fwd_fla_fsa,
            loss_query_only_fla,
            iters=iters,
            warmup=warmup,
            clear_cache_each_iter=True,
            reinit_fn=reinit_fla_fsa,
            zero_grad_fn=zero_fla_fsa,
        )
        print(f"  FLA FSA upstream (query loss only)  fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms")
    except Exception as e:
        err_text = str(e)
        # Known Triton incompatibility in upstream file:
        #   lse_ptrs = (ptr,) tuple form can fail to compile on newer Triton.
        known_ptr_tuple_issue = (
            "lse_ptrs = (lse_ptr + pid_q_j * stride_lse_n,)" in err_text
            or "at 73:10" in err_text
        )
        if known_ptr_tuple_issue and has_fsa_local_varlen:
            print("  FLA FSA upstream hit known Triton pointer-tuple compile issue; retrying with local varlen compatibility copy.")
            clear_cuda_cache()

            def fwd_fla_fsa_local_varlen():
                o = fsa_local_varlen_fn(
                    q=fla_fsa_state["q_full"][0],
                    k=fla_fsa_state["k_full"][0],
                    v=fla_fsa_state["v_full"][0],
                    topk_idx=fla_fsa_state["topk_idx_hns"],
                    block_size=cfg.BS,
                    cu_seqlens=fla_fsa_state["cu_q"],
                    softmax_scale=cfg.scale,
                    disable_causal_mask=True,
                )
                return o.unsqueeze(0)

            try:
                fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
                    fwd_fla_fsa_local_varlen,
                    loss_query_only_fla,
                    iters=iters,
                    warmup=warmup,
                    clear_cache_each_iter=True,
                    reinit_fn=reinit_fla_fsa,
                    zero_grad_fn=zero_fla_fsa,
                )
                print(
                    f"  FLA FSA local-varlen compatibility (query loss only)  "
                    f"fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms"
                )
            except Exception as e2:
                print(f"  FLA FSA upstream: skipped (runtime/compile failure: {e}; fallback failed: {e2})")
                if not has_fsa_local_varlen:
                    print(f"  local-varlen fallback unavailable: {fsa_local_varlen_import_err}")
        else:
            print(f"  FLA FSA upstream: skipped (runtime/compile failure: {e})")
            if known_ptr_tuple_issue and not has_fsa_local_varlen:
                print(f"  local-varlen fallback unavailable: {fsa_local_varlen_import_err}")


# ----------------------------
# Main entry (edit these)
# ----------------------------

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name()
    print("CUDA device:", device_name)
    print(
        "Cold-start mode:",
        f"MEM_XATTN_FAST_START={os.getenv('MEM_XATTN_FAST_START')},",
        f"TRITON_CACHE_DIR={os.getenv('TRITON_CACHE_DIR')}",
    )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # EDIT THESE PARAMETERS (what you said you'll pass)
    # mem_tokens = total memory tokens (TK) = num_chapters * chapter_size
    cfg = Config(
        mem_tokens=65536,        # TK
        chapter_size=128,         # BS
        topk=8,
        batch_size=32,           # input batch (collapsed)
        seq_len=8192,             # input seq (collapsed)
        hidden_dim=1024,           # heads * head_dim, with query heads below
        heads=16,                  # HQ
        kv_heads=1,               # HK (set < heads for GQA, e.g. 2 or 4)
        dtype=torch.bfloat16,
        device="cuda",
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # (a) correctness against naive, 5 times
    # Runs on a reduced config automatically (so it finishes).
    check_correctness_5x(cfg, sanity_TQ=16384, sanity_M=16384, sanity_BS=32)

    # (b) timing benchmarks (dense baselines, custom sparse, NSA)
    # WARNING: with your full sizes (TQ=128*2048=262k), this is heavy but should run on H100.
    run_benchmarks(cfg, iters=5, warmup=5)
