# benchmark_kernels_final.py
# Benchmarks kernels-final v1, v2, v3 for correctness and performance.
#
# Two operation modes:
#   (i)  Unweighted: single kernel call with all top-k chapters (joint softmax)
#   (ii) Weighted (MoE-style): per-chapter kernel calls on CUDA streams,
#        independent softmax per chapter, weighted output accumulation.
#        Mirrors the real routing path in memory_transformer/memory_attention.py.
#
# Usage:
#   python benchmark_kernels_final.py --mode all
#   python benchmark_kernels_final.py --mode correctness --num-checks 5
#   python benchmark_kernels_final.py --mode timing --iters 10 --warmup 3
#   python benchmark_kernels_final.py --kernels v1,v3 --no-weighted
#   python benchmark_kernels_final.py --preset large

import os
import sys
import gc
import math
import time
import random
import argparse
import importlib
import importlib.util
import traceback
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Callable
from pathlib import Path

import torch
import torch.nn.functional as F

# Avoid Triton recompilation noise.
os.environ.setdefault("TRITON_CACHE_DIR", str((Path(__file__).resolve().parent / ".triton_cache")))


# ===========================================================================
#  Helpers
# ===========================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bytes_gb(n: int) -> float:
    return n / (1024 ** 3)


def cuda_mem_gb() -> Tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return bytes_gb(free), bytes_gb(total)


def clear_cuda_cache():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.synchronize()


# ===========================================================================
#  Config
# ===========================================================================

@dataclass
class Config:
    """Configuration for benchmark shapes.  Mirrors the layout used by
    ``kernels/benchmark_memory_xattn_optimized_import.py``."""

    mem_tokens: int          # TK  – total memory tokens
    chapter_size: int        # BS  – tokens per chapter (block size)
    topk: int                # S   – number of routed chapters per query
    batch_size: int          # input batch (collapsed to mega-sequence of B*seq_len)
    seq_len: int             # input sequence length
    hidden_dim: int          # model dim = heads * head_dim
    heads: int               # HQ  (query heads)
    kv_heads: Optional[int] = None   # HK  (KV heads); None => same as heads
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    # ---- derived properties ------------------------------------------------
    @property
    def TQ(self) -> int:
        return self.batch_size * self.seq_len

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


# ===========================================================================
#  Input generation
# ===========================================================================

def make_inputs(cfg: Config, seed: int, *, per_query_random_topk: bool = True) -> Dict[str, torch.Tensor]:
    """Generate random q/k/v and per-query-per-KV-head block_indices.

    Shapes:
      q:             [1, TQ, HQ, D]
      k, v:          [1, TK, HK, D]
      block_indices: [1, TQ, HK, topk]   (int32, unique chapters per query)
    """
    set_all_seeds(seed)
    device, dtype = cfg.device, cfg.dtype

    q = torch.randn(1, cfg.TQ, cfg.H, cfg.D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, cfg.TK, cfg.HK, cfg.D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, cfg.TK, cfg.HK, cfg.D, device=device, dtype=dtype, requires_grad=True)

    if per_query_random_topk:
        if cfg.topk > cfg.M:
            raise ValueError(f"topk ({cfg.topk}) cannot exceed number of chapters M ({cfg.M}).")
        block_indices = torch.empty((1, cfg.TQ, cfg.HK, cfg.topk), device=device, dtype=torch.int32)
        chunk_t = 4096
        for t0 in range(0, cfg.TQ, chunk_t):
            t1 = min(cfg.TQ, t0 + chunk_t)
            scores = torch.rand((1, t1 - t0, cfg.HK, cfg.M), device=device, dtype=torch.float32)
            idx = torch.topk(scores, k=cfg.topk, dim=-1, largest=True, sorted=False).indices.to(torch.int32)
            block_indices[:, t0:t1, :, :] = idx
    else:
        global_ch = torch.randperm(cfg.M, device=device)[:cfg.topk].to(torch.int32)
        block_indices = global_ch.view(1, 1, 1, cfg.topk).expand(1, cfg.TQ, cfg.HK, cfg.topk).contiguous()

    return dict(q=q, k=k, v=v, block_indices=block_indices)


def make_chapter_weights(cfg: Config, seed: int) -> torch.Tensor:
    """Generate random normalised chapter weights matching router output.

    Shape: [1, TQ, topk]   (softmax-normalised, requires_grad for backward test)

    This mirrors the router output: softmax → top-k → renormalize with
    clamp_min(1e-12) to avoid div-by-zero in fp16.
    """
    set_all_seeds(seed + 9999)
    raw = torch.rand(1, cfg.TQ, cfg.topk, device=cfg.device, dtype=torch.float32)
    weights = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights.to(cfg.dtype).requires_grad_(True)
    return weights


# ===========================================================================
#  Naive reference implementations
# ===========================================================================

def _gather_kv_for_one_query(
    t: int, hk: int, block_indices_topk: torch.Tensor,
    k: torch.Tensor, v: torch.Tensor, BS: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V tokens for one query position, one KV-head, all selected chapters.

    block_indices_topk: [topk] int32
    k, v: [1, TK, HK, D]
    Returns kk, vv: [topk*BS, D] in float32
    """
    ch = block_indices_topk.to(torch.int64)   # [topk]
    tok = ch[:, None] * BS + torch.arange(BS, device=ch.device)[None, :]   # [topk, BS]
    tok = tok.reshape(-1)                      # [topk * BS]
    kk = k[0, tok, hk, :].float()
    vv = v[0, tok, hk, :].float()
    return kk, vv


def naive_reference_unweighted(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, BS: int, scale: float,
) -> torch.Tensor:
    """Reference for the unweighted (joint-softmax) case.

    All top-k chapters' tokens are gathered together, then a single softmax
    is computed across them.  This is what a single kernel call with all topk
    chapters produces.

    q: [1,TQ,HQ,D], k/v: [1,TK,HK,D], block_indices: [1,TQ,HK,topk]
    Returns: [1,TQ,HQ,D]
    """
    assert q.shape[0] == 1
    _, TQ, HQ, D = q.shape
    _, TK, HK, _ = k.shape
    assert HQ % HK == 0
    G = HQ // HK
    out = torch.zeros((1, TQ, HQ, D), device=q.device, dtype=torch.float32)

    for hk in range(HK):
        for t in range(TQ):
            kk, vv = _gather_kv_for_one_query(t, hk, block_indices[0, t, hk, :], k, v, BS)
            for g in range(G):
                hq = hk * G + g
                q_t = q[0, t, hq, :].float() * scale          # [D]
                scores = q_t @ kk.T                             # [topk*BS]
                p = torch.softmax(scores, dim=-1)               # [topk*BS]
                out[0, t, hq, :] = p @ vv                       # [D]

    return out.to(q.dtype)


def naive_reference_weighted(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, chapter_weights: torch.Tensor,
    BS: int, scale: float,
) -> torch.Tensor:
    """Reference for the weighted (MoE-style) case.

    Each chapter is attended INDEPENDENTLY (its own softmax), then outputs
    are combined with router weights:

        For each chapter i in top-k:
            output_i = softmax(Q @ K_i^T / sqrt(d)) @ V_i   ← per-chapter softmax
        final = Σ_i  w_i · output_i                          ← weighted combination

    This matches the MoE-style logic in memory_attention.py:
    _forward_token_routed_attention (lines 591-650).

    q: [1,TQ,HQ,D], k/v: [1,TK,HK,D], block_indices: [1,TQ,HK,topk]
    chapter_weights: [1,TQ,topk]
    Returns: [1,TQ,HQ,D]
    """
    assert q.shape[0] == 1
    _, TQ, HQ, D = q.shape
    _, TK, HK, _ = k.shape
    topk = block_indices.shape[-1]
    assert HQ % HK == 0
    G = HQ // HK
    out = torch.zeros((1, TQ, HQ, D), device=q.device, dtype=torch.float32)

    for hk in range(HK):
        for t in range(TQ):
            for g in range(G):
                hq = hk * G + g
                q_t = q[0, t, hq, :].float() * scale        # [D]

                accum = torch.zeros(D, device=q.device, dtype=torch.float32)
                for i in range(topk):
                    # Gather KV for this single chapter
                    ch_idx = block_indices[0, t, hk, i].to(torch.int64)
                    tok_start = ch_idx * BS
                    tok_end = tok_start + BS
                    kk_i = k[0, tok_start:tok_end, hk, :].float()   # [BS, D]
                    vv_i = v[0, tok_start:tok_end, hk, :].float()   # [BS, D]

                    # Independent softmax for this chapter
                    scores_i = q_t @ kk_i.T                  # [BS]
                    p_i = torch.softmax(scores_i, dim=-1)     # [BS]
                    out_i = p_i @ vv_i                        # [D]

                    # Weight by router probability
                    w_i = chapter_weights[0, t, i].float()
                    accum = accum + out_i * w_i

                out[0, t, hq, :] = accum

    return out.to(q.dtype)


def naive_reference_weighted_joint_bias(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, chapter_weights: torch.Tensor,
    BS: int, scale: float,
) -> torch.Tensor:
    """Reference for single-softmax weighted routing with log-weight bias.

    logits(token in chapter i) = qk * scale + log(w_i)
    """
    assert q.shape[0] == 1
    _, TQ, HQ, D = q.shape
    _, _, HK, _ = k.shape
    topk = block_indices.shape[-1]
    assert HQ % HK == 0
    G = HQ // HK
    out = torch.zeros((1, TQ, HQ, D), device=q.device, dtype=torch.float32)
    eps = 1e-20

    for hk in range(HK):
        for t in range(TQ):
            for g in range(G):
                hq = hk * G + g
                q_t = q[0, t, hq, :].float() * scale
                logits = []
                values = []
                for i in range(topk):
                    ch_idx = block_indices[0, t, hk, i].to(torch.int64)
                    tok_start = ch_idx * BS
                    tok_end = tok_start + BS
                    kk_i = k[0, tok_start:tok_end, hk, :].float()
                    vv_i = v[0, tok_start:tok_end, hk, :].float()
                    scores_i = q_t @ kk_i.T
                    scores_i = scores_i + torch.log(chapter_weights[0, t, i].float().clamp_min(eps))
                    logits.append(scores_i)
                    values.append(vv_i)
                logits_cat = torch.cat(logits, dim=0)
                values_cat = torch.cat(values, dim=0)
                p = torch.softmax(logits_cat, dim=-1)
                out[0, t, hq, :] = p @ values_cat

    return out.to(q.dtype)


# ===========================================================================
#  Kernel import helpers
# ===========================================================================

def _import_kernel(version: str):
    """Import unweighted/weighted sparse attention entrypoints for a kernel version."""
    mod_name = f"kernel_{version}"
    root_dir = Path(__file__).resolve().parent
    mod_file = root_dir / f"{mod_name}.py"

    if not mod_file.exists():
        return None, f"{mod_file} not found"

    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(mod_file))
        if spec is None or spec.loader is None:
            return None, f"failed to create module spec for {mod_file}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        fn = getattr(mod, "FSA_topk_sparse_attention_bthd", None)
        if fn is None:
            return None, f"{mod_name} has no FSA_topk_sparse_attention_bthd"
        weighted_fn = getattr(mod, "FSA_topk_sparse_attention_weighted_bthd", None)
        weighted_semantics = getattr(mod, "FSA_WEIGHTED_SEMANTICS", "exact_moe")
        return {"unweighted": fn, "weighted": weighted_fn, "weighted_semantics": weighted_semantics}, None
    except Exception as e:
        return None, f"import error for {mod_name}: {e}"


def import_kernels(versions: List[str]) -> Dict[str, Tuple[Optional[Dict[str, Optional[Callable]]], Optional[str]]]:
    """Import requested kernel versions. Returns {version: (kernel_info_or_None, err_or_None)}."""
    result = {}
    for v in versions:
        fn, err = _import_kernel(v)
        result[v] = (fn, err)
    return result


# ===========================================================================
#  Kernel execution wrappers  (unweighted + weighted MoE)
# ===========================================================================

def kernel_forward_unweighted(
    kernel_info: Dict[str, Optional[Callable]],
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, BS: int, scale: float,
) -> torch.Tensor:
    """Single kernel call with all top-k chapters (joint softmax)."""
    kernel_fn = kernel_info["unweighted"]
    if kernel_fn is None:
        raise RuntimeError("Kernel info is missing unweighted entrypoint.")
    return kernel_fn(
        q_bthd=q,
        k_bthd=k,
        v_bthd=v,
        block_indices_bths=block_indices,
        block_size=BS,
        softmax_scale=scale,
        disable_causal_mask=True,
    )


def kernel_forward_weighted_legacy(
    kernel_info: Dict[str, Optional[Callable]],
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, chapter_weights: torch.Tensor,
    BS: int, scale: float,
) -> torch.Tensor:
    """MoE-style per-chapter kernel calls with CUDA stream parallelism.

    This replicates the exact logic from memory_attention.py
    ``_forward_token_routed_attention`` (lines 591-650):

    1. Each chapter's kernel is launched on a separate CUDA stream.
    2. Event-based synchronisation waits for all streams.
    3. Weighted accumulation runs on the default stream.

    chapter_weights: [B, TQ, topk]
    """
    kernel_fn = kernel_info["unweighted"]
    if kernel_fn is None:
        raise RuntimeError("Kernel info is missing unweighted entrypoint.")
    B, T, HQ, D = q.shape
    topk = block_indices.shape[-1]

    use_streams = q.is_cuda and topk > 1

    if use_streams:
        # -- Parallel CUDA stream path (matches production code) --
        streams = [torch.cuda.Stream(device=q.device) for _ in range(topk)]
        chapter_outputs: List[Optional[torch.Tensor]] = [None] * topk
        events: List[Optional[torch.cuda.Event]] = [None] * topk

        for i in range(topk):
            single_idx = block_indices[:, :, :, i:i + 1]   # [B, TQ, HK, 1]
            with torch.cuda.stream(streams[i]):
                chapter_outputs[i] = kernel_fn(
                    q_bthd=q,
                    k_bthd=k,
                    v_bthd=v,
                    block_indices_bths=single_idx,
                    block_size=BS,
                    softmax_scale=scale,
                    disable_causal_mask=True,
                )   # [B, TQ, HQ, D]
                events[i] = torch.cuda.Event()
                events[i].record()

        # Synchronise: default stream waits for all chapter events
        current_stream = torch.cuda.current_stream(q.device)
        for e in events:
            current_stream.wait_event(e)

        # Weighted accumulation on default stream
        accum = torch.zeros((B, T, HQ, D), device=q.device, dtype=torch.float32)
        for i in range(topk):
            w_i = chapter_weights[:, :, i].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
            accum = accum + chapter_outputs[i].to(torch.float32) * w_i

    else:
        # -- Sequential / single-chapter path --
        accum = torch.zeros((B, T, HQ, D), device=q.device, dtype=torch.float32)
        for i in range(topk):
            single_idx = block_indices[:, :, :, i:i + 1]
            chapter_out = kernel_fn(
                q_bthd=q,
                k_bthd=k,
                v_bthd=v,
                block_indices_bths=single_idx,
                block_size=BS,
                softmax_scale=scale,
                disable_causal_mask=True,
            )
            w_i = chapter_weights[:, :, i].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
            accum = accum + chapter_out.to(torch.float32) * w_i

    return accum.to(dtype=q.dtype)


def kernel_forward_weighted_fused(
    kernel_info: Dict[str, Optional[Callable]],
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, chapter_weights: torch.Tensor,
    BS: int, scale: float,
) -> torch.Tensor:
    """Single-call exact weighted MoE path for kernels that expose it."""
    weighted_fn = kernel_info.get("weighted")
    if weighted_fn is None:
        raise RuntimeError("Kernel does not expose FSA_topk_sparse_attention_weighted_bthd.")
    return weighted_fn(
        q_bthd=q,
        k_bthd=k,
        v_bthd=v,
        block_indices_bths=block_indices,
        chapter_weights_bts=chapter_weights,
        block_size=BS,
        softmax_scale=scale,
        disable_causal_mask=True,
    )


def kernel_forward_weighted(
    kernel_info: Dict[str, Optional[Callable]],
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.Tensor, chapter_weights: torch.Tensor,
    BS: int, scale: float,
    *,
    weighted_impl: str = "auto",
    return_impl: bool = False,
):
    """Dispatch weighted MoE attention to legacy-loop or fused-weighted implementation."""
    impl = str(weighted_impl).strip().lower()
    if impl not in {"auto", "legacy_loop", "fused_weighted"}:
        raise ValueError(f"Unsupported weighted_impl='{weighted_impl}'.")

    weighted_fn = kernel_info.get("weighted")
    use_fused = (impl == "fused_weighted") or (impl == "auto" and weighted_fn is not None)
    if use_fused:
        out = kernel_forward_weighted_fused(kernel_info, q, k, v, block_indices, chapter_weights, BS, scale)
        impl_name = "fused_weighted"
    else:
        out = kernel_forward_weighted_legacy(kernel_info, q, k, v, block_indices, chapter_weights, BS, scale)
        impl_name = "legacy_loop"

    if return_impl:
        return out, impl_name
    return out


# ===========================================================================
#  Timing helpers (CUDA event-based, carried from existing benchmark)
# ===========================================================================

def time_fwd_only(
    forward_fn: Callable,
    *,
    iters: int = 5,
    warmup: int = 2,
    clear_cache_each_iter: bool = True,
) -> float:
    """Returns average forward time in ms over ``iters`` timed iterations."""
    # Warmup (includes JIT / Triton compilation)
    for _ in range(warmup):
        if clear_cache_each_iter:
            clear_cuda_cache()
        with torch.no_grad():
            _ = forward_fn()
        torch.cuda.synchronize()

    fwd_times: List[float] = []
    for _ in range(iters):
        if clear_cache_each_iter:
            clear_cuda_cache()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = forward_fn()
        end.record()

        torch.cuda.synchronize()
        fwd_times.append(start.elapsed_time(end))

    return sum(fwd_times) / len(fwd_times)


def time_fwd_bwd(
    forward_fn: Callable,
    make_loss_fn: Callable,
    *,
    iters: int = 5,
    warmup: int = 2,
    clear_cache_each_iter: bool = True,
    zero_grad_fn: Optional[Callable] = None,
) -> Tuple[float, float]:
    """Returns (avg_fwd_ms, avg_bwd_ms) over ``iters`` timed iterations."""
    # Warmup
    for _ in range(warmup):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if zero_grad_fn is not None:
            zero_grad_fn()
        out = forward_fn()
        loss = make_loss_fn(out)
        loss.backward()
        torch.cuda.synchronize()

    fwd_times: List[float] = []
    bwd_times: List[float] = []

    for _ in range(iters):
        if clear_cache_each_iter:
            clear_cuda_cache()
        if zero_grad_fn is not None:
            zero_grad_fn()

        torch.cuda.synchronize()
        sf = torch.cuda.Event(enable_timing=True)
        ef = torch.cuda.Event(enable_timing=True)
        sb = torch.cuda.Event(enable_timing=True)
        eb = torch.cuda.Event(enable_timing=True)

        sf.record()
        out = forward_fn()
        ef.record()

        loss = make_loss_fn(out)

        sb.record()
        loss.backward()
        eb.record()

        torch.cuda.synchronize()
        fwd_times.append(sf.elapsed_time(ef))
        bwd_times.append(sb.elapsed_time(eb))

    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times)


# ===========================================================================
#  Correctness checks
# ===========================================================================

def check_correctness(
    cfg: Config,
    kernels: Dict[str, Dict[str, Optional[Callable]]],
    *,
    run_weighted: bool = True,
    weighted_impl: str = "auto",
    sanity_TQ: int = 128,
    sanity_M: int = 16,
    sanity_BS: Optional[int] = None,
    num_checks: int = 5,
):
    """Run forward + backward correctness checks for all provided kernels.

    Two sub-checks per kernel:
      (a) Unweighted:  kernel(all topk) vs naive_reference_unweighted
      (b) Weighted:    MoE-style kernel loop vs naive_reference_weighted
    """
    if not kernels:
        print("\nCorrectness: skipped (no kernels available).\n")
        return

    # ---- Build reduced config for fast correctness ----
    BS = sanity_BS if sanity_BS is not None else min(cfg.BS, 64)
    M = min(cfg.M, sanity_M)
    TQ = min(cfg.TQ, sanity_TQ)
    topk = min(cfg.topk, 4)
    H, HK, D = cfg.H, cfg.HK, max(16, cfg.D // 2) if cfg.D > 16 else cfg.D

    red = Config(
        mem_tokens=M * BS,
        chapter_size=BS,
        topk=topk,
        batch_size=1,
        seq_len=TQ,
        hidden_dim=H * D,
        heads=H,
        kv_heads=HK,
        dtype=cfg.dtype,
        device=cfg.device,
    )

    print(f"\n{'='*70}")
    print(f"  Correctness ({num_checks} trials) — reduced config")
    print(f"{'='*70}")
    print(
        f"  TQ={red.TQ}, TK={red.TK}, M={red.M}, BS={red.BS}, topk={red.topk}, "
        f"HQ={red.H}, HK={red.HK}, G={red.G}, D={red.D}, dtype={red.dtype}\n"
    )

    # Tolerances (bf16 is noisy)
    if red.dtype in (torch.float16, torch.bfloat16):
        atol_fwd, rtol_fwd = 2e-2, 2e-2
        atol_bwd, rtol_bwd = 3e-2, 3e-2
    else:
        atol_fwd, rtol_fwd = 1e-4, 1e-4
        atol_bwd, rtol_bwd = 5e-4, 5e-4

    # Track results
    results: Dict[str, Dict[str, str]] = {}  # {kernel: {mode: PASS/FAIL}}

    for vname, kernel_info in kernels.items():
        results[vname] = {}

        # ----------------------------------------------------------
        # (a) Unweighted correctness
        # ----------------------------------------------------------
        print(f"  [{vname}] Unweighted correctness...", end=" ", flush=True)
        ok_unweighted = True
        for trial in range(num_checks):
            seed = 1234 + trial
            x = make_inputs(red, seed=seed)

            # Forward reference (no grad)
            with torch.no_grad():
                o_ref = naive_reference_unweighted(
                    x["q"].detach(), x["k"].detach(), x["v"].detach(),
                    x["block_indices"], red.BS, red.scale,
                )

            # Forward kernel
            try:
                with torch.no_grad():
                    o_kern = kernel_forward_unweighted(
                        kernel_info, x["q"].detach(), x["k"].detach(), x["v"].detach(),
                        x["block_indices"], red.BS, red.scale,
                    )
                torch.testing.assert_close(o_kern, o_ref, atol=atol_fwd, rtol=rtol_fwd)
            except Exception as e:
                print(f"FAIL (trial {trial+1}, forward)")
                print(f"    {e}")
                ok_unweighted = False
                break

            # Backward reference
            set_all_seeds(999 + trial)
            dO = torch.randn_like(o_ref).to(red.dtype)

            q2 = x["q"].detach().clone().requires_grad_(True)
            k2 = x["k"].detach().clone().requires_grad_(True)
            v2 = x["v"].detach().clone().requires_grad_(True)
            o2 = naive_reference_unweighted(q2, k2, v2, x["block_indices"], red.BS, red.scale)
            (o2 * dO).sum().backward()
            dq_ref, dk_ref, dv_ref = q2.grad.detach(), k2.grad.detach(), v2.grad.detach()

            # Backward kernel
            try:
                q3 = x["q"].detach().clone().requires_grad_(True)
                k3 = x["k"].detach().clone().requires_grad_(True)
                v3 = x["v"].detach().clone().requires_grad_(True)
                o3 = kernel_forward_unweighted(
                    kernel_info, q3, k3, v3, x["block_indices"], red.BS, red.scale,
                )
                (o3 * dO).sum().backward()
                torch.testing.assert_close(q3.grad, dq_ref, atol=atol_bwd, rtol=rtol_bwd)
                torch.testing.assert_close(k3.grad, dk_ref, atol=atol_bwd, rtol=rtol_bwd)
                torch.testing.assert_close(v3.grad, dv_ref, atol=atol_bwd, rtol=rtol_bwd)
            except Exception as e:
                print(f"FAIL (trial {trial+1}, backward)")
                print(f"    {e}")
                ok_unweighted = False
                break

        if ok_unweighted:
            print("PASS")
            results[vname]["unweighted"] = "PASS"
        else:
            results[vname]["unweighted"] = "FAIL"

        # ----------------------------------------------------------
        # (b) Weighted (MoE) correctness
        # ----------------------------------------------------------
        if not run_weighted:
            results[vname]["weighted"] = "SKIP"
            continue

        print(f"  [{vname}] Weighted (MoE) correctness...", end=" ", flush=True)
        ok_weighted = True
        weighted_impl_name = None
        weighted_semantics = kernel_info.get("weighted_semantics", "exact_moe")
        for trial in range(num_checks):
            seed = 5678 + trial
            x = make_inputs(red, seed=seed)
            cw = make_chapter_weights(red, seed=seed)

            # Forward reference (no grad)
            with torch.no_grad():
                if weighted_semantics == "joint_bias":
                    o_ref_w = naive_reference_weighted_joint_bias(
                        x["q"].detach(), x["k"].detach(), x["v"].detach(),
                        x["block_indices"], cw.detach(), red.BS, red.scale,
                    )
                else:
                    o_ref_w = naive_reference_weighted(
                        x["q"].detach(), x["k"].detach(), x["v"].detach(),
                        x["block_indices"], cw.detach(), red.BS, red.scale,
                    )

            # Forward kernel (weighted)
            try:
                with torch.no_grad():
                    o_kern_w = kernel_forward_weighted(
                        kernel_info,
                        x["q"].detach(), x["k"].detach(), x["v"].detach(),
                        x["block_indices"], cw.detach(), red.BS, red.scale,
                        weighted_impl=weighted_impl,
                        return_impl=True,
                    )
                    if isinstance(o_kern_w, tuple):
                        o_kern_w, weighted_impl_name = o_kern_w
                torch.testing.assert_close(o_kern_w, o_ref_w, atol=atol_fwd, rtol=rtol_fwd)
            except Exception as e:
                print(f"FAIL (trial {trial+1}, forward)")
                print(f"    {e}")
                ok_weighted = False
                break

            # Backward reference (weighted)
            set_all_seeds(999 + trial)
            dO_w = torch.randn_like(o_ref_w).to(red.dtype)

            q2 = x["q"].detach().clone().requires_grad_(True)
            k2 = x["k"].detach().clone().requires_grad_(True)
            v2 = x["v"].detach().clone().requires_grad_(True)
            cw2 = cw.detach().clone().requires_grad_(True)
            if weighted_semantics == "joint_bias":
                o2_w = naive_reference_weighted_joint_bias(q2, k2, v2, x["block_indices"], cw2, red.BS, red.scale)
            else:
                o2_w = naive_reference_weighted(q2, k2, v2, x["block_indices"], cw2, red.BS, red.scale)
            (o2_w * dO_w).sum().backward()
            dq_ref_w = q2.grad.detach()
            dk_ref_w = k2.grad.detach()
            dv_ref_w = v2.grad.detach()
            dcw_ref = cw2.grad.detach()

            # Backward kernel (weighted)
            try:
                q3 = x["q"].detach().clone().requires_grad_(True)
                k3 = x["k"].detach().clone().requires_grad_(True)
                v3 = x["v"].detach().clone().requires_grad_(True)
                cw3 = cw.detach().clone().requires_grad_(True)
                o3_w = kernel_forward_weighted(
                    kernel_info, q3, k3, v3, x["block_indices"], cw3, red.BS, red.scale,
                    weighted_impl=weighted_impl,
                )
                (o3_w * dO_w).sum().backward()

                torch.testing.assert_close(q3.grad, dq_ref_w, atol=atol_bwd, rtol=rtol_bwd)
                torch.testing.assert_close(k3.grad, dk_ref_w, atol=atol_bwd, rtol=rtol_bwd)
                torch.testing.assert_close(v3.grad, dv_ref_w, atol=atol_bwd, rtol=rtol_bwd)
                torch.testing.assert_close(cw3.grad, dcw_ref, atol=atol_bwd, rtol=rtol_bwd)
            except Exception as e:
                print(f"FAIL (trial {trial+1}, backward)")
                print(f"    {e}")
                ok_weighted = False
                break

        if ok_weighted:
            if weighted_impl_name is not None:
                print(f"PASS ({weighted_impl_name})")
            else:
                print("PASS")
            results[vname]["weighted"] = "PASS"
        else:
            results[vname]["weighted"] = "FAIL"

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("  Correctness Summary")
    print(f"{'='*70}")
    print(f"  {'Kernel':<10} {'Unweighted':<15} {'Weighted (MoE)':<15}")
    print(f"  {'------':<10} {'----------':<15} {'--------------':<15}")
    for vname in kernels:
        uw = results.get(vname, {}).get("unweighted", "N/A")
        wt = results.get(vname, {}).get("weighted", "N/A")
        print(f"  {vname:<10} {uw:<15} {wt:<15}")
    print()


# ===========================================================================
#  Timing benchmarks
# ===========================================================================

def run_timing(
    cfg: Config,
    kernels: Dict[str, Dict[str, Optional[Callable]]],
    *,
    run_weighted: bool = True,
    weighted_impl: str = "auto",
    iters: int = 5,
    warmup: int = 2,
    fwd_only: bool = False,
):
    """Time forward (+ optional backward) for each kernel, unweighted and weighted."""
    if not kernels:
        print("\nTiming: skipped (no kernels available).\n")
        return

    seed = 42
    x = make_inputs(cfg, seed=seed)
    cw = make_chapter_weights(cfg, seed=seed)

    print(f"\n{'='*70}")
    print(f"  Timing Benchmark")
    print(f"{'='*70}")
    print(
        f"  TQ={cfg.TQ}, TK={cfg.TK}, M={cfg.M}, BS={cfg.BS}, topk={cfg.topk}, "
        f"HQ={cfg.H}, HK={cfg.HK}, G={cfg.G}, D={cfg.D}, dtype={cfg.dtype}"
    )
    print(f"  warmup={warmup}, iters={iters}, fwd_only={fwd_only}\n")

    rows: List[Tuple[str, str, float, Optional[float]]] = []

    for vname, kernel_info in kernels.items():
        # ---- Unweighted ----
        print(f"  [{vname}] Timing unweighted...", end=" ", flush=True)
        try:
            def fwd_uw():
                return kernel_forward_unweighted(
                    kernel_info,
                    x["q"], x["k"], x["v"], x["block_indices"],
                    cfg.BS, cfg.scale,
                )

            if fwd_only:
                fwd_ms = time_fwd_only(fwd_uw, iters=iters, warmup=warmup)
                rows.append((vname, "unweighted", fwd_ms, None))
                print(f"fwd={fwd_ms:.3f}ms")
            else:
                dO = torch.randn(1, cfg.TQ, cfg.H, cfg.D, device=cfg.device, dtype=cfg.dtype)

                def fwd_uw_grad():
                    q = x["q"].detach().clone().requires_grad_(True)
                    k = x["k"].detach().clone().requires_grad_(True)
                    v = x["v"].detach().clone().requires_grad_(True)
                    return kernel_forward_unweighted(
                        kernel_info, q, k, v, x["block_indices"], cfg.BS, cfg.scale,
                    )

                def loss_fn(o):
                    return (o * dO).sum()

                fwd_ms, bwd_ms = time_fwd_bwd(fwd_uw_grad, loss_fn, iters=iters, warmup=warmup)
                rows.append((vname, "unweighted", fwd_ms, bwd_ms))
                print(f"fwd={fwd_ms:.3f}ms  bwd={bwd_ms:.3f}ms")
        except Exception as e:
            print(f"ERROR: {e}")
            rows.append((vname, "unweighted", float("nan"), float("nan")))

        if not run_weighted:
            continue

        # ---- Weighted (MoE) ----
        print(f"  [{vname}] Timing weighted (MoE)...", end=" ", flush=True)
        try:
            _, active_weighted_impl = kernel_forward_weighted(
                kernel_info,
                x["q"].detach(), x["k"].detach(), x["v"].detach(),
                x["block_indices"], cw.detach(), cfg.BS, cfg.scale,
                weighted_impl=weighted_impl,
                return_impl=True,
            )
            def fwd_wt():
                return kernel_forward_weighted(
                    kernel_info,
                    x["q"], x["k"], x["v"], x["block_indices"],
                    cw, cfg.BS, cfg.scale,
                    weighted_impl=weighted_impl,
                )

            if fwd_only:
                fwd_ms = time_fwd_only(fwd_wt, iters=iters, warmup=warmup)
                rows.append((vname, "weighted", fwd_ms, None))
                print(f"fwd={fwd_ms:.3f}ms  impl={active_weighted_impl}")
            else:
                dO = torch.randn(1, cfg.TQ, cfg.H, cfg.D, device=cfg.device, dtype=cfg.dtype)

                def fwd_wt_grad():
                    q = x["q"].detach().clone().requires_grad_(True)
                    k = x["k"].detach().clone().requires_grad_(True)
                    v = x["v"].detach().clone().requires_grad_(True)
                    cw_g = cw.detach().clone().requires_grad_(True)
                    return kernel_forward_weighted(
                        kernel_info, q, k, v, x["block_indices"], cw_g, cfg.BS, cfg.scale,
                        weighted_impl=weighted_impl,
                    )

                def loss_fn(o):
                    return (o * dO).sum()

                fwd_ms, bwd_ms = time_fwd_bwd(fwd_wt_grad, loss_fn, iters=iters, warmup=warmup)
                rows.append((vname, "weighted", fwd_ms, bwd_ms))
                print(f"fwd={fwd_ms:.3f}ms  bwd={bwd_ms:.3f}ms  impl={active_weighted_impl}")
        except Exception as e:
            print(f"ERROR: {e}")
            rows.append((vname, "weighted", float("nan"), float("nan")))

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("  Timing Summary (ms)")
    print(f"{'='*70}")
    if fwd_only:
        print(f"  {'Kernel':<10} {'Mode':<15} {'Fwd':>10}")
        print(f"  {'------':<10} {'----':<15} {'---':>10}")
        for vname, mode, fwd_ms, _ in rows:
            print(f"  {vname:<10} {mode:<15} {fwd_ms:>10.3f}")
    else:
        print(f"  {'Kernel':<10} {'Mode':<15} {'Fwd':>10} {'Bwd':>10} {'Total':>10}")
        print(f"  {'------':<10} {'----':<15} {'---':>10} {'---':>10} {'-----':>10}")
        for vname, mode, fwd_ms, bwd_ms in rows:
            total = (fwd_ms or 0) + (bwd_ms or 0)
            print(f"  {vname:<10} {mode:<15} {fwd_ms:>10.3f} {bwd_ms:>10.3f} {total:>10.3f}")
    print()


# ===========================================================================
#  Presets
# ===========================================================================

PRESETS = {
    "small": Config(
        mem_tokens=1024, chapter_size=64, topk=4,
        batch_size=1, seq_len=512,
        hidden_dim=768, heads=12, kv_heads=4,
    ),
    "medium": Config(
        mem_tokens=4096, chapter_size=64, topk=4,
        batch_size=1, seq_len=2048,
        hidden_dim=2048, heads=16, kv_heads=4,
    ),
    "large": Config(
        mem_tokens=16384, chapter_size=128, topk=4,
        batch_size=1, seq_len=4096,
        hidden_dim=3584, heads=28, kv_heads=4,
    ),
    "qwen-7b": Config(
        mem_tokens=16384, chapter_size=128, topk=4,
        batch_size=1, seq_len=4096,
        hidden_dim=3584, heads=28, kv_heads=4,
    ),
}


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark kernels-final v1/v2/v3/v4/v5 (unweighted + weighted variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_kernels_final.py --mode all
  python benchmark_kernels_final.py --mode correctness --num-checks 3
  python benchmark_kernels_final.py --mode timing --iters 10 --warmup 3
  python benchmark_kernels_final.py --kernels v1,v3 --no-weighted
  python benchmark_kernels_final.py --preset large --mode timing --fwd-only
  python benchmark_kernels_final.py --mem-tokens 8192 --chapter-size 64 --topk 4 \\
      --seq-len 2048 --hidden-dim 2048 --heads 16 --kv-heads 4
""",
    )

    parser.add_argument("--mode", choices=["all", "correctness", "timing"], default="all",
                        help="Run mode (default: all)")
    parser.add_argument("--kernels", default="v1,v2,v3,v4,v5",
                        help="Comma-separated kernel versions to test (default: v1,v2,v3,v4,v5)")
    parser.add_argument("--weighted", action="store_true", default=True,
                        help="Include weighted (MoE-style) tests (default: True)")
    parser.add_argument("--no-weighted", dest="weighted", action="store_false",
                        help="Skip weighted (MoE-style) tests")
    parser.add_argument(
        "--weighted-impl",
        choices=["auto", "legacy_loop", "fused_weighted"],
        default="auto",
        help="Weighted implementation selection: auto, legacy loop, or fused weighted kernel.",
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="Use a predefined config (overridden by explicit shape args)")
    parser.add_argument("--fwd-only", action="store_true", default=False,
                        help="Time forward pass only (skip backward)")

    # Shape args
    parser.add_argument("--mem-tokens", type=int, default=None)
    parser.add_argument("--chapter-size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")

    # Correctness args
    parser.add_argument("--num-checks", type=int, default=5)
    parser.add_argument("--sanity-tq", type=int, default=128)
    parser.add_argument("--sanity-m", type=int, default=16)

    # Timing args
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)

    args = parser.parse_args()

    # ---- Build config ----
    if args.preset:
        cfg = PRESETS[args.preset]
    else:
        cfg = PRESETS["small"]  # default base

    # Override with explicit args
    overrides = {}
    if args.mem_tokens is not None:
        overrides["mem_tokens"] = args.mem_tokens
    if args.chapter_size is not None:
        overrides["chapter_size"] = args.chapter_size
    if args.topk is not None:
        overrides["topk"] = args.topk
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.heads is not None:
        overrides["heads"] = args.heads
    if args.kv_heads is not None:
        overrides["kv_heads"] = args.kv_heads
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if overrides:
        cfg = Config(
            mem_tokens=overrides.get("mem_tokens", cfg.mem_tokens),
            chapter_size=overrides.get("chapter_size", cfg.chapter_size),
            topk=overrides.get("topk", cfg.topk),
            batch_size=overrides.get("batch_size", cfg.batch_size),
            seq_len=overrides.get("seq_len", cfg.seq_len),
            hidden_dim=overrides.get("hidden_dim", cfg.hidden_dim),
            heads=overrides.get("heads", cfg.heads),
            kv_heads=overrides.get("kv_heads", cfg.kv_heads),
            dtype=dtype,
        )
    elif args.dtype != "bf16":
        cfg = Config(
            mem_tokens=cfg.mem_tokens,
            chapter_size=cfg.chapter_size,
            topk=cfg.topk,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            heads=cfg.heads,
            kv_heads=cfg.kv_heads,
            dtype=dtype,
        )

    # ---- Check CUDA ----
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("  Kernels-Final Benchmark")
    print(f"{'='*70}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free_gb, total_gb = cuda_mem_gb()
    print(f"  Memory: {free_gb:.1f} / {total_gb:.1f} GB free")
    print(f"  Config: TQ={cfg.TQ}, TK={cfg.TK}, M={cfg.M}, BS={cfg.BS}, topk={cfg.topk}")
    print(f"          HQ={cfg.H}, HK={cfg.HK}, G={cfg.G}, D={cfg.D}, dtype={cfg.dtype}")
    print(f"  Mode: {args.mode}, weighted={args.weighted}, weighted_impl={args.weighted_impl}")

    # ---- Import kernels ----
    versions = [v.strip() for v in args.kernels.split(",") if v.strip()]
    imported = import_kernels(versions)

    available_kernels: Dict[str, Dict[str, Optional[Callable]]] = {}
    for v in versions:
        kernel_info, err = imported[v]
        if kernel_info is not None:
            available_kernels[v] = kernel_info
            weighted_status = "YES" if kernel_info.get("weighted") is not None else "NO"
            semantics = kernel_info.get("weighted_semantics", "exact_moe")
            print(f"  Kernel {v}: OK (weighted_fused={weighted_status}, weighted_semantics={semantics})")
        else:
            print(f"  Kernel {v}: UNAVAILABLE ({err})")

    if not available_kernels:
        print("\nERROR: No kernels available. Exiting.")
        sys.exit(1)

    # ---- Run ----
    if args.mode in ("all", "correctness"):
        check_correctness(
            cfg, available_kernels,
            run_weighted=args.weighted,
            weighted_impl=args.weighted_impl,
            sanity_TQ=args.sanity_tq,
            sanity_M=args.sanity_m,
            num_checks=args.num_checks,
        )

    if args.mode in ("all", "timing"):
        run_timing(
            cfg, available_kernels,
            run_weighted=args.weighted,
            weighted_impl=args.weighted_impl,
            iters=args.iters,
            warmup=args.warmup,
            fwd_only=args.fwd_only,
        )

    print("Done.\n")


if __name__ == "__main__":
    main()
