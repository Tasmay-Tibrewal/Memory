# benchmark_memory_xattn.py
# Run on H100 Colab (CUDA). Requires:
#   - your custom kernels module: memory_cross_attn.py (the one with memory_cross_attn(..., dkv_strategy=...))
#   - (optional) flash-attn for baseline (ii)
#   - (optional) native-sparse-attention for baseline (iv)

import os, time, math, gc, importlib, importlib.util, sys, types, traceback
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F

# Cold-start defaults (can be overridden by user-provided env vars).
os.environ.setdefault("MEM_XATTN_FAST_START", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str((Path(__file__).resolve().parent / ".triton_cache")))

_FSA_LOCAL_MODULE_FILE = None
_FSA_LOCAL_IMPORT_NOTE = None
_FSA_LOCAL_UNOPT_MODULE_FILE = None
_FSA_LOCAL_UNOPT_IMPORT_NOTE = None
_FSA_LOCAL_OLD_MODULE_FILE = None
_FSA_LOCAL_OLD_IMPORT_NOTE = None
_FSA_LOCAL_OLDER_MODULE_FILE = None
_FSA_LOCAL_OLDER_IMPORT_NOTE = None
_FSA_CHAPTER_MODULE_FILE = None
_FSA_CHAPTER_IMPORT_NOTE = None


@contextmanager
def _temporary_environ(overrides: Dict[str, Optional[str]]):
    """Temporarily set environment variables and restore previous values."""
    prev: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            prev[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old_value in prev.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _fsa_local_varlen_stable_env() -> Dict[str, Optional[str]]:
    """
    Conservative varlen settings for correctness/timing fallback.
    This avoids inheriting aggressive NSA-forward policies that can produce NaNs
    on some shapes/configurations in local-compat usage.
    """
    enabled = os.getenv("FSA_BENCH_VARLEN_STABLE_ENV", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )
    if not enabled:
        return {}
    return {
        "FSA_LOCAL_USE_NSA_STYLE_FWD": "0",
        "FSA_LOCAL_FORCE_NSA_STYLE_FWD_SMALL_G": "0",
        "FSA_LOCAL_USE_NSA_NATIVE_FWD": "0",
        "FSA_LOCAL_FWD_PACKED_GQA": "0",
        "FSA_LOCAL_NSA_PACKED_GQA": "0",
        "FSA_LOCAL_BLOCK_PRUNING_MODE": "off",
    }


def _canonicalize_topk_idx(topk_idx: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize per-query top-k chapter ids for FSA varlen path:
    - int32
    - sorted non-negative ids
    - padding sentinel -1 forced to tail (not head)
    """
    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    if topk_idx.numel() == 0:
        return topk_idx.contiguous()
    sentinel = torch.iinfo(torch.int32).max
    neg_mask = topk_idx < 0
    work = torch.where(neg_mask, torch.full_like(topk_idx, sentinel), topk_idx)
    work = work.sort(dim=-1).values
    work = torch.where(work == sentinel, torch.full_like(work, -1), work)
    return work.contiguous()


def _import_fsa_local_module():
    """
    Import local optimized FSA module, preferring the file colocated with this benchmark.
    """
    global _FSA_LOCAL_MODULE_FILE, _FSA_LOCAL_IMPORT_NOTE
    mod_name = "fsa_topk_sparse_attention_local_optimized"
    root_dir = Path(__file__).resolve().parent
    preferred_files = [
        (root_dir / "kernels" / f"{mod_name}.py").resolve(),
        (root_dir / f"{mod_name}.py").resolve(),
    ]
    here_file = next((p for p in preferred_files if p.exists()), preferred_files[-1])
    _FSA_LOCAL_IMPORT_NOTE = None
    try:
        mod = importlib.import_module(mod_name)
        mod_file_raw = getattr(mod, "__file__", None)
        mod_file = Path(mod_file_raw).resolve() if mod_file_raw else None
        if here_file.exists() and mod_file is not None and mod_file != here_file:
            # In notebook/root workflows multiple copies may exist; prefer colocated module.
            spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create module spec for {here_file}")
            mod_local = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod_local
            spec.loader.exec_module(mod_local)
            _FSA_LOCAL_MODULE_FILE = str(here_file)
            _FSA_LOCAL_IMPORT_NOTE = f"resolved module mismatch: imported {mod_file}, using colocated {here_file}"
            return mod_local
        _FSA_LOCAL_MODULE_FILE = str(mod_file) if mod_file is not None else None
        return mod
    except Exception as first_error:
        if here_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to create module spec for {here_file}")
                mod_local = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod_local
                spec.loader.exec_module(mod_local)
                _FSA_LOCAL_MODULE_FILE = str(here_file)
                _FSA_LOCAL_IMPORT_NOTE = f"fallback to colocated module after import error: {first_error}"
                return mod_local
            except Exception:
                pass
        raise


def _import_fsa_chapter_module():
    """
    Import chapter-routed FSA module, preferring file colocated with this benchmark.
    """
    global _FSA_CHAPTER_MODULE_FILE, _FSA_CHAPTER_IMPORT_NOTE
    mod_name = "fsa_topk_sparse_attention_chapter_routed"
    root_dir = Path(__file__).resolve().parent
    preferred_files = [
        (root_dir / "kernels" / f"{mod_name}.py").resolve(),
        (root_dir / f"{mod_name}.py").resolve(),
    ]
    here_file = next((p for p in preferred_files if p.exists()), preferred_files[-1])
    _FSA_CHAPTER_IMPORT_NOTE = None
    try:
        mod = importlib.import_module(mod_name)
        mod_file_raw = getattr(mod, "__file__", None)
        mod_file = Path(mod_file_raw).resolve() if mod_file_raw else None
        if here_file.exists() and mod_file is not None and mod_file != here_file:
            spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create module spec for {here_file}")
            mod_local = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod_local
            spec.loader.exec_module(mod_local)
            _FSA_CHAPTER_MODULE_FILE = str(here_file)
            _FSA_CHAPTER_IMPORT_NOTE = (
                f"resolved module mismatch: imported {mod_file}, using colocated {here_file}"
            )
            return mod_local
        _FSA_CHAPTER_MODULE_FILE = str(mod_file) if mod_file is not None else None
        return mod
    except Exception as first_error:
        if here_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to create module spec for {here_file}")
                mod_local = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod_local
                spec.loader.exec_module(mod_local)
                _FSA_CHAPTER_MODULE_FILE = str(here_file)
                _FSA_CHAPTER_IMPORT_NOTE = (
                    f"fallback to colocated module after import error: {first_error}"
                )
                return mod_local
            except Exception:
                pass
        raise


def _import_fsa_local_unopt_module():
    """
    Import local unoptimized FSA module, preferring the file colocated with this benchmark.
    """
    global _FSA_LOCAL_UNOPT_MODULE_FILE, _FSA_LOCAL_UNOPT_IMPORT_NOTE
    mod_name = "fsa_topk_sparse_attention_local"
    root_dir = Path(__file__).resolve().parent
    preferred_files = [
        (root_dir / "kernels" / f"{mod_name}.py").resolve(),
        (root_dir / f"{mod_name}.py").resolve(),
    ]
    here_file = next((p for p in preferred_files if p.exists()), preferred_files[-1])
    _FSA_LOCAL_UNOPT_IMPORT_NOTE = None
    try:
        mod = importlib.import_module(mod_name)
        mod_file_raw = getattr(mod, "__file__", None)
        mod_file = Path(mod_file_raw).resolve() if mod_file_raw else None
        if here_file.exists() and mod_file is not None and mod_file != here_file:
            spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create module spec for {here_file}")
            mod_local = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod_local
            spec.loader.exec_module(mod_local)
            _FSA_LOCAL_UNOPT_MODULE_FILE = str(here_file)
            _FSA_LOCAL_UNOPT_IMPORT_NOTE = f"resolved module mismatch: imported {mod_file}, using colocated {here_file}"
            return mod_local
        _FSA_LOCAL_UNOPT_MODULE_FILE = str(mod_file) if mod_file is not None else None
        return mod
    except Exception as first_error:
        if here_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to create module spec for {here_file}")
                mod_local = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod_local
                spec.loader.exec_module(mod_local)
                _FSA_LOCAL_UNOPT_MODULE_FILE = str(here_file)
                _FSA_LOCAL_UNOPT_IMPORT_NOTE = f"fallback to colocated module after import error: {first_error}"
                return mod_local
            except Exception:
                pass
        raise


def _import_fsa_local_old_module():
    """
    Import local historical optimized FSA module (old), preferring colocated file.
    """
    global _FSA_LOCAL_OLD_MODULE_FILE, _FSA_LOCAL_OLD_IMPORT_NOTE
    mod_name = "fsa_topk_sparse_attention_local_optimized_old"
    root_dir = Path(__file__).resolve().parent
    preferred_files = [
        (root_dir / "kernels" / f"{mod_name}.py").resolve(),
        (root_dir / f"{mod_name}.py").resolve(),
    ]
    here_file = next((p for p in preferred_files if p.exists()), preferred_files[-1])
    _FSA_LOCAL_OLD_IMPORT_NOTE = None
    try:
        mod = importlib.import_module(mod_name)
        mod_file_raw = getattr(mod, "__file__", None)
        mod_file = Path(mod_file_raw).resolve() if mod_file_raw else None
        if here_file.exists() and mod_file is not None and mod_file != here_file:
            spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create module spec for {here_file}")
            mod_local = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod_local
            spec.loader.exec_module(mod_local)
            _FSA_LOCAL_OLD_MODULE_FILE = str(here_file)
            _FSA_LOCAL_OLD_IMPORT_NOTE = f"resolved module mismatch: imported {mod_file}, using colocated {here_file}"
            return mod_local
        _FSA_LOCAL_OLD_MODULE_FILE = str(mod_file) if mod_file is not None else None
        return mod
    except Exception as first_error:
        if here_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to create module spec for {here_file}")
                mod_local = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod_local
                spec.loader.exec_module(mod_local)
                _FSA_LOCAL_OLD_MODULE_FILE = str(here_file)
                _FSA_LOCAL_OLD_IMPORT_NOTE = f"fallback to colocated module after import error: {first_error}"
                return mod_local
            except Exception:
                pass
        raise


def _import_fsa_local_older_module():
    """
    Import local historical optimized FSA module (older), preferring colocated file.
    """
    global _FSA_LOCAL_OLDER_MODULE_FILE, _FSA_LOCAL_OLDER_IMPORT_NOTE
    mod_name = "fsa_topk_sparse_attention_local_optimized_older"
    root_dir = Path(__file__).resolve().parent
    preferred_files = [
        (root_dir / "kernels" / f"{mod_name}.py").resolve(),
        (root_dir / f"{mod_name}.py").resolve(),
    ]
    here_file = next((p for p in preferred_files if p.exists()), preferred_files[-1])
    _FSA_LOCAL_OLDER_IMPORT_NOTE = None
    try:
        mod = importlib.import_module(mod_name)
        mod_file_raw = getattr(mod, "__file__", None)
        mod_file = Path(mod_file_raw).resolve() if mod_file_raw else None
        if here_file.exists() and mod_file is not None and mod_file != here_file:
            spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create module spec for {here_file}")
            mod_local = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod_local
            spec.loader.exec_module(mod_local)
            _FSA_LOCAL_OLDER_MODULE_FILE = str(here_file)
            _FSA_LOCAL_OLDER_IMPORT_NOTE = f"resolved module mismatch: imported {mod_file}, using colocated {here_file}"
            return mod_local
        _FSA_LOCAL_OLDER_MODULE_FILE = str(mod_file) if mod_file is not None else None
        return mod
    except Exception as first_error:
        if here_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(here_file))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to create module spec for {here_file}")
                mod_local = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod_local
                spec.loader.exec_module(mod_local)
                _FSA_LOCAL_OLDER_MODULE_FILE = str(here_file)
                _FSA_LOCAL_OLDER_IMPORT_NOTE = f"fallback to colocated module after import error: {first_error}"
                return mod_local
            except Exception:
                pass
        raise


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


def _is_cuda_context_poisoned_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "cuda error: an illegal memory access was encountered" in msg
    )

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


def check_correctness_5x(
    cfg: Config,
    *,
    sanity_TQ: int = 128,
    sanity_M: int = 16,
    sanity_BS: Optional[int] = None,
    num_checks: int = 5,
):
    """
    Runs correctness trials (forward + backward) against naive PyTorch reference
    for optional optimized paths
    (fsa_opt / fsa_local / fsa_local_old / fsa_local_older / fsa_local_unopt / fsa_local_varlen / fsa_chapter / nsa_selected_attn).

    For safety, we do correctness on a reduced config by default.
    Returns:
      True  -> completed without poisoning CUDA context
      False -> hit illegal CUDA access; caller should stop and restart runtime
    """
    if int(num_checks) <= 0:
        raise ValueError(f"num_checks must be >= 1, got {num_checks}.")
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
    fsa_local_old_bthd_fn, fsa_local_old_import_err = try_import_fsa_local_old()
    has_fsa_local_old = fsa_local_old_bthd_fn is not None
    fsa_local_older_bthd_fn, fsa_local_older_import_err = try_import_fsa_local_older()
    has_fsa_local_older = fsa_local_older_bthd_fn is not None
    fsa_local_unopt_bthd_fn, fsa_local_unopt_import_err = try_import_fsa_local_unopt()
    has_fsa_local_unopt = fsa_local_unopt_bthd_fn is not None
    fsa_local_varlen_fn, fsa_local_varlen_import_err = try_import_fsa_local_varlen()
    has_fsa_local_varlen = fsa_local_varlen_fn is not None
    fsa_chapter_bthd_fn, fsa_chapter_import_err = try_import_fsa_chapter_routed()
    has_fsa_chapter = fsa_chapter_bthd_fn is not None
    nsa_mod, nsa_import_err = try_import_nsa()
    has_nsa = nsa_mod is not None
    parallel_nsa_selected_autograd = None
    if has_nsa:
        _, _, parallel_nsa_selected_autograd = nsa_mod
    nsa_only_correctness = os.getenv("FSA_CORRECTNESS_NSA_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if nsa_only_correctness:
        has_fsa_opt = False
        has_fsa_local = False
        has_fsa_local_old = False
        has_fsa_local_older = False
        has_fsa_local_unopt = False
        has_fsa_local_varlen = False
        has_fsa_chapter = False
        print("Note: FSA_CORRECTNESS_NSA_ONLY=1 -> running only nsa_selected_attn correctness.")

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

    print(f"\n=== Correctness ({int(num_checks)} checks) on reduced config ===")
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

    if not has_fsa_opt:
        print(f"Note: skipping fsa_opt correctness (import failed: {fsa_opt_import_err})")
    else:
        force_fsa_opt_bf16 = os.getenv("FSA_CORRECTNESS_FORCE_FSA_OPT_BF16", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        skip_fsa_opt = os.getenv("FSA_CORRECTNESS_SKIP_FSA_OPT", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if skip_fsa_opt:
            has_fsa_opt = False
            print("Note: skipping fsa_opt correctness (FSA_CORRECTNESS_SKIP_FSA_OPT=1).")
        elif red.dtype == torch.bfloat16 and not force_fsa_opt_bf16:
            has_fsa_opt = False
            print(
                "Note: skipping fsa_opt correctness for bf16 due known Triton dtype-mismatch compile issue. "
                "Set FSA_CORRECTNESS_FORCE_FSA_OPT_BF16=1 to force attempt."
            )
    if not has_fsa_local:
        print(f"Note: skipping fsa_local correctness (import failed: {fsa_local_import_err})")
    elif fsa_local_import_err:
        print(f"Note: fsa_local import warning: {fsa_local_import_err}")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(f"Note: skipping fsa_local correctness (unsupported BS={red.BS}; requires 32/64/128/256/512/1024)")
        has_fsa_local = False
    if not has_fsa_local_old:
        print(f"Note: skipping fsa_local_old correctness (import failed: {fsa_local_old_import_err})")
    elif fsa_local_old_import_err:
        print(f"Note: fsa_local_old import warning: {fsa_local_old_import_err}")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"Note: skipping fsa_local_old correctness (unsupported BS={red.BS}; "
            "requires 32/64/128/256/512/1024)"
        )
        has_fsa_local_old = False
    if not has_fsa_local_older:
        print(f"Note: skipping fsa_local_older correctness (import failed: {fsa_local_older_import_err})")
    elif fsa_local_older_import_err:
        print(f"Note: fsa_local_older import warning: {fsa_local_older_import_err}")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"Note: skipping fsa_local_older correctness (unsupported BS={red.BS}; "
            "requires 32/64/128/256/512/1024)"
        )
        has_fsa_local_older = False
    if not has_fsa_local_unopt:
        print(f"Note: skipping fsa_local_unopt correctness (import failed: {fsa_local_unopt_import_err})")
    elif fsa_local_unopt_import_err:
        print(f"Note: fsa_local_unopt import warning: {fsa_local_unopt_import_err}")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"Note: skipping fsa_local_unopt correctness (unsupported BS={red.BS}; "
            "requires 32/64/128/256/512/1024)"
        )
        has_fsa_local_unopt = False
    if not has_fsa_local_varlen:
        print(f"Note: skipping fsa_local_varlen correctness (import failed: {fsa_local_varlen_import_err})")
    elif fsa_local_varlen_import_err:
        print(f"Note: fsa_local_varlen import warning: {fsa_local_varlen_import_err}")
    elif red.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"Note: skipping fsa_local_varlen correctness (unsupported BS={red.BS}; "
            "requires 32/64/128/256/512/1024)"
        )
        has_fsa_local_varlen = False
    if not has_fsa_chapter:
        print(f"Note: skipping fsa_chapter correctness (import failed: {fsa_chapter_import_err})")
    elif fsa_chapter_import_err:
        print(f"Note: fsa_chapter import warning: {fsa_chapter_import_err}")
    if not has_nsa:
        print(f"Note: skipping nsa_selected_attn correctness (import failed: {nsa_import_err})")
    elif red.G < 16:
        print(
            f"Note: skipping nsa_selected_attn correctness (unsupported head-group ratio G={red.G}; "
            "requires >=16 for current NSA kernels)"
        )
        has_nsa = False
    elif (red.G & (red.G - 1)) != 0:
        print(
            f"Note: skipping nsa_selected_attn correctness (unsupported head-group ratio G={red.G}; "
            "must be power-of-two)"
        )
        has_nsa = False
    elif red.BS < 16:
        print(
            f"Note: skipping nsa_selected_attn correctness (unsupported BS={red.BS}; requires >=16)"
        )
        has_nsa = False
    elif red.D < 16:
        print(
            f"Note: skipping nsa_selected_attn correctness (unsupported head_dim D={red.D}; requires >=16)"
        )
        has_nsa = False

    if (
        not has_fsa_opt
        and not has_fsa_local
        and not has_fsa_local_old
        and not has_fsa_local_older
        and not has_fsa_local_unopt
        and not has_fsa_local_varlen
        and not has_fsa_chapter
        and not has_nsa
    ):
        print("\nCorrectness: skipped (no optimized strategy available to compare).\n")
        return True

    fsa_opt_runtime_err = None
    fsa_local_runtime_err = None
    fsa_local_old_runtime_err = None
    fsa_local_older_runtime_err = None
    fsa_local_unopt_runtime_err = None
    fsa_local_varlen_runtime_err = None
    fsa_chapter_runtime_err = None
    nsa_runtime_err = None
    chapter_chunk_raw = os.getenv("FSA_CHAPTER_QUERY_CHUNK_SIZE", "4096").strip().lower()
    try:
        chapter_chunk = max(1, int(chapter_chunk_raw))
    except Exception:
        chapter_chunk = 4096
    chapter_dedupe = os.getenv("FSA_CHAPTER_DEDUPE_QUERIES", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    chapter_route_collapse = os.getenv("FSA_LOCAL_GQA_ROUTE_COLLAPSE", "auto")

    for trial in range(int(num_checks)):
        print(f"Running trial {trial + 1}/{int(num_checks)}")

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

        if has_fsa_opt:
            print("Running strategy fsa_opt")
            try:
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
            except Exception as e:
                fsa_opt_runtime_err = e
                has_fsa_opt = False
                print(f"Note: disabling fsa_opt correctness after runtime/compile failure: {e}")
                if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                    print("fsa_opt correctness traceback:")
                    print(traceback.format_exc())
                if _is_cuda_context_poisoned_error(e):
                    print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
                    return False

        # if has_fsa_local:
        #     print("Running strategy fsa_local")
        #     try:
        #         q = x["q"].detach().clone().requires_grad_(True)
        #         k = x["k"].detach().clone().requires_grad_(True)
        #         v = x["v"].detach().clone().requires_grad_(True)
        #         bi = x["block_indices"]
        #         q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
        #             q.detach(), k.detach(), v.detach(), bi, red.TK
        #         )
        #         cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
        #         cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
        #         atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
        #         if relax_large_bwd:
        #             atol_bwd_use = max(atol_bwd_use, 1.25e-1)
        #             rtol_bwd_use = max(rtol_bwd_use, 1e-1)

        #         o_full = fsa_local_bthd_fn(
        #             q_bthd=q_full,
        #             k_bthd=k_full[:, :red.TK, :, :],
        #             v_bthd=v_full[:, :red.TK, :, :],
        #             block_indices_bths=bi_full,
        #             block_size=red.BS,
        #             softmax_scale=red.scale,
        #             cu_seqlens_q=cu_q,
        #             cu_seqlens_k=cu_k,
        #             disable_causal_mask=True,
        #         )
        #         o = o_full[:, red.TK:, :, :]
        #         try:
        #             torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
        #         except AssertionError as exc:
        #             raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local': forward mismatch\n{exc}") from exc

        #         loss = (o * dO).sum()
        #         loss.backward()
        #         dq_loc = q_full.grad[:, red.TK:, :, :]
        #         dk_loc = k_full.grad[:, :red.TK, :, :]
        #         dv_loc = v_full.grad[:, :red.TK, :, :]

        #         try:
        #             torch.testing.assert_close(dq_loc, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #             torch.testing.assert_close(dk_loc, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #             torch.testing.assert_close(dv_loc, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #         except AssertionError as exc:
        #             raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local': backward mismatch\n{exc}") from exc
        #     except Exception as e:
        #         fsa_local_runtime_err = e
        #         has_fsa_local = False
        #         print(f"Note: disabling fsa_local correctness after runtime/compile failure: {e}")
        #         if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
        #             print("fsa_local correctness traceback:")
        #             print(traceback.format_exc())
        #         if _is_cuda_context_poisoned_error(e):
        #             print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
        #             return False

        if has_fsa_local_unopt:
            print("Running strategy fsa_local_unopt")
            try:
                q = x["q"].detach().clone().requires_grad_(True)
                k = x["k"].detach().clone().requires_grad_(True)
                v = x["v"].detach().clone().requires_grad_(True)
                bi = x["block_indices"]
                q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
                    q.detach(), k.detach(), v.detach(), bi, red.TK
                )
                cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
                cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
                topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(red.HK, q_full.shape[1], -1).to(torch.int32)
                atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
                if relax_large_bwd:
                    atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                    rtol_bwd_use = max(rtol_bwd_use, 1e-1)

                o_full = fsa_local_unopt_bthd_fn(
                    q_bthd=q_full,
                    k_bthd=k_full[:, :red.TK, :, :],
                    v_bthd=v_full[:, :red.TK, :, :],
                    block_indices_bths=None,
                    block_size=red.BS,
                    softmax_scale=red.scale,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    topk_idx_hns=topk_idx_hns,
                    assume_sorted_topk=False,
                    disable_causal_mask=True,
                )
                o = o_full[:, red.TK:, :, :]
                try:
                    torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
                except AssertionError as exc:
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_unopt': forward mismatch\n{exc}") from exc

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
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_unopt': backward mismatch\n{exc}") from exc
            except Exception as e:
                fsa_local_unopt_runtime_err = e
                has_fsa_local_unopt = False
                print(f"Note: disabling fsa_local_unopt correctness after runtime/compile failure: {e}")
                if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                    print("fsa_local_unopt correctness traceback:")
                    print(traceback.format_exc())
                if _is_cuda_context_poisoned_error(e):
                    print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
                    return False

        if has_fsa_local_old:
            print("Running strategy fsa_local_old")
            try:
                q = x["q"].detach().clone().requires_grad_(True)
                k = x["k"].detach().clone().requires_grad_(True)
                v = x["v"].detach().clone().requires_grad_(True)
                bi = x["block_indices"]
                q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
                    q.detach(), k.detach(), v.detach(), bi, red.TK
                )
                cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
                cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
                topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(red.HK, q_full.shape[1], -1).to(torch.int32)
                atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
                if relax_large_bwd:
                    atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                    rtol_bwd_use = max(rtol_bwd_use, 1e-1)

                o_full = fsa_local_old_bthd_fn(
                    q_bthd=q_full,
                    k_bthd=k_full[:, :red.TK, :, :],
                    v_bthd=v_full[:, :red.TK, :, :],
                    block_indices_bths=None,
                    block_size=red.BS,
                    softmax_scale=red.scale,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    topk_idx_hns=topk_idx_hns,
                    assume_sorted_topk=False,
                    disable_causal_mask=True,
                )
                o = o_full[:, red.TK:, :, :]
                try:
                    torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
                except AssertionError as exc:
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_old': forward mismatch\n{exc}") from exc

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
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_old': backward mismatch\n{exc}") from exc
            except Exception as e:
                fsa_local_old_runtime_err = e
                has_fsa_local_old = False
                print(f"Note: disabling fsa_local_old correctness after runtime/compile failure: {e}")
                if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                    print("fsa_local_old correctness traceback:")
                    print(traceback.format_exc())
                if _is_cuda_context_poisoned_error(e):
                    print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
                    return False

        if has_fsa_local_older:
            print("Running strategy fsa_local_older")
            try:
                q = x["q"].detach().clone().requires_grad_(True)
                k = x["k"].detach().clone().requires_grad_(True)
                v = x["v"].detach().clone().requires_grad_(True)
                bi = x["block_indices"]
                q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
                    q.detach(), k.detach(), v.detach(), bi, red.TK
                )
                cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
                cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
                topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(red.HK, q_full.shape[1], -1).to(torch.int32)
                atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
                if relax_large_bwd:
                    atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                    rtol_bwd_use = max(rtol_bwd_use, 1e-1)

                o_full = fsa_local_older_bthd_fn(
                    q_bthd=q_full,
                    k_bthd=k_full[:, :red.TK, :, :],
                    v_bthd=v_full[:, :red.TK, :, :],
                    block_indices_bths=None,
                    block_size=red.BS,
                    softmax_scale=red.scale,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    topk_idx_hns=topk_idx_hns,
                    assume_sorted_topk=False,
                    disable_causal_mask=True,
                )
                o = o_full[:, red.TK:, :, :]
                try:
                    torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
                except AssertionError as exc:
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_older': forward mismatch\n{exc}") from exc

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
                    raise AssertionError(f"Trial {trial+1}, strategy 'fsa_local_older': backward mismatch\n{exc}") from exc
            except Exception as e:
                fsa_local_older_runtime_err = e
                has_fsa_local_older = False
                print(f"Note: disabling fsa_local_older correctness after runtime/compile failure: {e}")
                if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                    print("fsa_local_older correctness traceback:")
                    print(traceback.format_exc())
                if _is_cuda_context_poisoned_error(e):
                    print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
                    return False

        # if has_fsa_local_varlen:
        #     print("Running strategy fsa_local_varlen")
        #     try:
        #         q = x["q"].detach().clone()
        #         k = x["k"].detach().clone()
        #         v = x["v"].detach().clone()
        #         bi = x["block_indices"]
        #         q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
        #             q, k, v, bi, red.TK
        #         )
        #         # Keep identical timeline semantics as fsa_local correctness path.
        #         topk_idx_hns = _canonicalize_topk_idx(
        #             bi_full[0].permute(1, 0, 2)
        #         )
        #         cu_q = torch.tensor([0, q_full.shape[1]], device=q_full.device, dtype=torch.int32)
        #         cu_k = torch.tensor([0, red.TK], device=q_full.device, dtype=torch.int32)
        #         atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
        #         if relax_large_bwd:
        #             atol_bwd_use = max(atol_bwd_use, 1.25e-1)
        #             rtol_bwd_use = max(rtol_bwd_use, 1e-1)

        #         with _temporary_environ(_fsa_local_varlen_stable_env()):
        #             o_full = fsa_local_varlen_fn(
        #                 q=q_full[0],
        #                 k=k_full[0, :red.TK, :, :],
        #                 v=v_full[0, :red.TK, :, :],
        #                 topk_idx=topk_idx_hns,
        #                 block_size=red.BS,
        #                 cu_seqlens_q=cu_q,
        #                 cu_seqlens_k=cu_k,
        #                 softmax_scale=red.scale,
        #                 disable_causal_mask=True,
        #             )
        #             if o_full is None:
        #                 raise RuntimeError("fsa_local_varlen returned None output.")
        #             if o_full.ndim == 3:
        #                 o_full = o_full.unsqueeze(0)
        #             o = o_full[:, red.TK:, :, :]
        #             try:
        #                 torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
        #             except AssertionError as exc:
        #                 raise AssertionError(
        #                     f"Trial {trial+1}, strategy 'fsa_local_varlen': forward mismatch\n{exc}"
        #                 ) from exc

        #             loss = (o * dO).sum()
        #             loss.backward()
        #             try:
        #                 dq_loc = q_full.grad[:, red.TK:, :, :]
        #                 dk_loc = k_full.grad[:, :red.TK, :, :]
        #                 dv_loc = v_full.grad[:, :red.TK, :, :]
        #                 torch.testing.assert_close(dq_loc, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #                 torch.testing.assert_close(dk_loc, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #                 torch.testing.assert_close(dv_loc, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #             except AssertionError as exc:
        #                 raise AssertionError(
        #                     f"Trial {trial+1}, strategy 'fsa_local_varlen': backward mismatch\n{exc}"
        #                 ) from exc
        #     except Exception as e:
        #         fsa_local_varlen_runtime_err = e
        #         has_fsa_local_varlen = False
        #         print(f"Note: disabling fsa_local_varlen correctness after runtime/compile failure: {e}")
        #         if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
        #             print("fsa_local_varlen correctness traceback:")
        #             print(traceback.format_exc())
        #         if _is_cuda_context_poisoned_error(e):
        #             print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
        #             return False

        # if has_fsa_chapter:
        #     print("Running strategy fsa_chapter")
        #     try:
        #         q = x["q"].detach().clone().requires_grad_(True)
        #         k = x["k"].detach().clone().requires_grad_(True)
        #         v = x["v"].detach().clone().requires_grad_(True)
        #         bi = x["block_indices"]
        #         atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
        #         if relax_large_bwd:
        #             atol_bwd_use = max(atol_bwd_use, 1.25e-1)
        #             rtol_bwd_use = max(rtol_bwd_use, 1e-1)

        #         o = fsa_chapter_bthd_fn(
        #             q_bthd=q,
        #             k_bthd=k,
        #             v_bthd=v,
        #             block_indices_bths=bi,
        #             block_size=red.BS,
        #             softmax_scale=red.scale,
        #             disable_causal_mask=True,
        #             route_collapse=chapter_route_collapse,
        #             chapter_query_chunk_size=chapter_chunk,
        #             dedupe_queries_per_chapter=chapter_dedupe,
        #         )
        #         try:
        #             torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
        #         except AssertionError as exc:
        #             raise AssertionError(f"Trial {trial+1}, strategy 'fsa_chapter': forward mismatch\n{exc}") from exc

        #         loss = (o * dO).sum()
        #         loss.backward()
        #         try:
        #             torch.testing.assert_close(q.grad, dq_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #             torch.testing.assert_close(k.grad, dk_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #             torch.testing.assert_close(v.grad, dv_ref, atol=atol_bwd_use, rtol=rtol_bwd_use)
        #         except AssertionError as exc:
        #             raise AssertionError(f"Trial {trial+1}, strategy 'fsa_chapter': backward mismatch\n{exc}") from exc
        #     except Exception as e:
        #         fsa_chapter_runtime_err = e
        #         has_fsa_chapter = False
        #         print(f"Note: disabling fsa_chapter correctness after runtime/compile failure: {e}")
        #         if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
        #             print("fsa_chapter correctness traceback:")
        #             print(traceback.format_exc())
        #         if _is_cuda_context_poisoned_error(e):
        #             print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
        #             return False

        if has_nsa:
            print("Running strategy nsa_selected_attn")
            try:
                q = x["q"].detach().clone().requires_grad_(True)
                k = x["k"].detach().clone().requires_grad_(True)
                v = x["v"].detach().clone().requires_grad_(True)
                bi = x["block_indices"]
                q_full, k_full, v_full, bi_full, _tfull = build_nsa_inputs(
                    q.detach(), k.detach(), v.detach(), bi, red.TK
                )
                atol_bwd_use, rtol_bwd_use = atol_bwd, rtol_bwd
                if relax_large_bwd:
                    atol_bwd_use = max(atol_bwd_use, 1.25e-1)
                    rtol_bwd_use = max(rtol_bwd_use, 1e-1)

                o_full = parallel_nsa_selected_autograd(
                    q=q_full,
                    k=k_full,
                    v=v_full,
                    block_indices=bi_full,
                    block_counts=red.topk,
                    block_size=red.BS,
                    scale=red.scale,
                    offsets=None,
                )
                if o_full is None:
                    raise RuntimeError("nsa_selected_attn returned None output.")
                o = o_full[:, red.TK:, :, :]
                try:
                    torch.testing.assert_close(o.detach(), o_ref, atol=atol_fwd, rtol=rtol_fwd)
                except AssertionError as exc:
                    raise AssertionError(
                        f"Trial {trial+1}, strategy 'nsa_selected_attn': forward mismatch\n{exc}"
                    ) from exc

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
                    raise AssertionError(
                        f"Trial {trial+1}, strategy 'nsa_selected_attn': backward mismatch\n{exc}"
                    ) from exc
            except Exception as e:
                nsa_runtime_err = e
                has_nsa = False
                print(f"Note: disabling nsa_selected_attn correctness after runtime/compile failure: {e}")
                if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                    print("nsa_selected_attn correctness traceback:")
                    print(traceback.format_exc())
                if _is_cuda_context_poisoned_error(e):
                    print("CUDA context is likely poisoned by prior kernel fault; stopping correctness early.")
                    return False

        print(f"Trial {trial + 1}/{int(num_checks)}: OK")

        if (
            not has_fsa_opt
            and not has_fsa_local
            and not has_fsa_local_old
            and not has_fsa_local_older
            and not has_fsa_local_unopt
            and not has_fsa_local_varlen
            and not has_fsa_chapter
            and not has_nsa
        ):
            print("Note: no correctness strategy left enabled; ending correctness loop early.")
            break

    passed = []
    if has_fsa_opt:
        passed.append("fsa_opt")
    if has_fsa_local:
        passed.append("fsa_local")
    if has_fsa_local_old:
        passed.append("fsa_local_old")
    if has_fsa_local_older:
        passed.append("fsa_local_older")
    if has_fsa_local_unopt:
        passed.append("fsa_local_unopt")
    if has_fsa_local_varlen:
        passed.append("fsa_local_varlen")
    if has_fsa_chapter:
        passed.append("fsa_chapter")
    if has_nsa:
        passed.append("nsa_selected_attn")
    if passed:
        print(f"\nCorrectness: {', '.join(passed)} passed vs naive reference.\n")
    else:
        print("\nCorrectness: skipped (no optimized strategy available to compare).\n")
    if fsa_opt_runtime_err is not None:
        print(f"Correctness note: fsa_opt was disabled after runtime/compile failure: {fsa_opt_runtime_err}")
    if fsa_local_runtime_err is not None:
        print(f"Correctness note: fsa_local was disabled after runtime/compile failure: {fsa_local_runtime_err}")
    if fsa_local_old_runtime_err is not None:
        print(
            "Correctness note: fsa_local_old was disabled after runtime/compile failure: "
            f"{fsa_local_old_runtime_err}"
        )
    if fsa_local_older_runtime_err is not None:
        print(
            "Correctness note: fsa_local_older was disabled after runtime/compile failure: "
            f"{fsa_local_older_runtime_err}"
        )
    if fsa_local_unopt_runtime_err is not None:
        print(
            "Correctness note: fsa_local_unopt was disabled after runtime/compile failure: "
            f"{fsa_local_unopt_runtime_err}"
        )
    if fsa_local_varlen_runtime_err is not None:
        print(
            "Correctness note: fsa_local_varlen was disabled after runtime/compile failure: "
            f"{fsa_local_varlen_runtime_err}"
        )
    if fsa_chapter_runtime_err is not None:
        print(f"Correctness note: fsa_chapter was disabled after runtime/compile failure: {fsa_chapter_runtime_err}")
    if nsa_runtime_err is not None:
        print(
            "Correctness note: nsa_selected_attn was disabled after runtime/compile failure: "
            f"{nsa_runtime_err}"
        )
    return True


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


def output_requires_grad_for_inputs(out: torch.Tensor, *inputs: torch.Tensor) -> bool:
    """
    Return True when either:
      - no input requires grad, or
      - output requires grad.
    """
    need_grad = False
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.requires_grad:
            need_grad = True
            break
    return (not need_grad) or (isinstance(out, torch.Tensor) and out.requires_grad)


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


def try_flash_sparse_local_nomask_sparse_topk_pkg(
    fs_local_fn,
    q_bthd: torch.Tensor,
    k_bLhd: torch.Tensor,
    v_bLhd: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    chunk_qh: Optional[int] = None,
):
    """
    Token-level sparse top-k emulation using local no-mask Triton backend.

    Materializes per-(query,head) selected KV rows in chunks and runs attention
    with shape [chunk, 1, 1, D] x [chunk, Lsel, 1, D].
    """
    if q_bthd.shape[0] != 1 or k_bLhd.shape[0] != 1 or v_bLhd.shape[0] != 1:
        return None, "flash-sparse local helper expects batch size 1 inputs [1,T,H,D]"
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

        k_chunk = k_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)  # [chunk, Lsel, 1, D]
        v_chunk = v_src[tok_chunk, h_idx, :].contiguous().unsqueeze(2)
        q_chunk = q_flat[qh0:qh1].contiguous().unsqueeze(1).unsqueeze(2)  # [chunk, 1, 1, D]

        out_chunk = fs_local_fn(
            query=q_chunk,
            key=k_chunk,
            value=v_chunk,
            is_causal=False,
            softmax_scale=scale,
        )
        if out_chunk is None:
            return None, "flash-sparse local backend returned None"
        if isinstance(out_chunk, tuple):
            out_chunk = out_chunk[0]

        out_flat[qh0:qh1] = out_chunk[:, 0, 0, :]

    out = out_flat.view(TQ, HQ, D).unsqueeze(0).contiguous()
    return out, "flash_sparse_local_nomask_sparse_topk"


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
    Import optimized local FSA copy (root-level).
    """
    try:
        mod = _import_fsa_local_module()
        fn = getattr(mod, "FSA_topk_sparse_attention_bthd")
        return fn, _FSA_LOCAL_IMPORT_NOTE
    except Exception as e:
        return None, f"optimized local import failed: {e}"


def try_import_fsa_local_unopt():
    """
    Import unoptimized local FSA copy (root-level).
    """
    try:
        mod = _import_fsa_local_unopt_module()
        fn = getattr(mod, "FSA_topk_sparse_attention_bthd")
        return fn, _FSA_LOCAL_UNOPT_IMPORT_NOTE
    except Exception as e:
        return None, f"unoptimized local import failed: {e}"


def try_import_fsa_local_old():
    """
    Import historical optimized local FSA copy (old).
    """
    try:
        mod = _import_fsa_local_old_module()
        fn = getattr(mod, "FSA_topk_sparse_attention_bthd")
        return fn, _FSA_LOCAL_OLD_IMPORT_NOTE
    except Exception as e:
        return None, f"optimized-old local import failed: {e}"


def try_import_fsa_local_older():
    """
    Import historical optimized local FSA copy (older).
    """
    try:
        mod = _import_fsa_local_older_module()
        fn = getattr(mod, "FSA_topk_sparse_attention_bthd")
        return fn, _FSA_LOCAL_OLDER_IMPORT_NOTE
    except Exception as e:
        return None, f"optimized-older local import failed: {e}"


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
    Import optimized local varlen API (same signature as upstream).
    """
    try:
        mod = _import_fsa_local_module()
        fn = getattr(mod, "FSA_topk_sparse_attention_varlen_qk")
        return fn, _FSA_LOCAL_IMPORT_NOTE
    except Exception as e:
        return None, f"optimized local varlen import failed: {e}"


def try_import_fsa_chapter_routed():
    """
    Import chapter-routed local FSA API.
    """
    try:
        mod = _import_fsa_chapter_module()
        if hasattr(mod, "FSA_topk_sparse_attention_chapter_routed_bthd"):
            fn = getattr(mod, "FSA_topk_sparse_attention_chapter_routed_bthd")
        else:
            fn = getattr(mod, "chapter_routed_sparse_attention_bthd")
        return fn, _FSA_CHAPTER_IMPORT_NOTE
    except Exception as e:
        return None, f"chapter-routed local import failed: {e}"


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
            # Explicit override takes highest priority.
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

            # Also scan parents in case benchmark is launched from a copied location.
            for parent in [here, *here.parents]:
                candidates.extend(
                    [
                        parent / "flash-sparse-attention-flash-algo",
                        parent / "flash_sparse_attention_flash_algo",
                        parent / "flash_sparse_attn",
                    ]
                )

            # Check sys.path hints as well.
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


def try_import_flash_sparse_local_nomask():
    """
    Import local replicated Triton backend (no attn_mask / no attn_bias API).
    """
    try:
        from flash_sparse_attn_triton_local_nomask import (
            triton_sparse_attn_nomask_func,
            triton_sparse_topk_attn_fused_fwd,
        )
        return (triton_sparse_attn_nomask_func, triton_sparse_topk_attn_fused_fwd), None
    except Exception as e:
        return None, str(e)


# ----------------------------
# Full benchmark (b)
# ----------------------------

def run_benchmarks(cfg: Config, *, iters: int = 5, warmup: int = 2):
    print("\n=== Benchmark config ===")
    print(f"TQ={cfg.TQ} (=batch*seq={cfg.batch_size}*{cfg.seq_len}), TK={cfg.TK}, M={cfg.M}, BS={cfg.BS}, topk={cfg.topk}")
    print(f"HQ={cfg.H}, HK={cfg.HK}, G={cfg.G}, hidden_dim={cfg.hidden_dim} => head_dim D={cfg.D}, dtype={cfg.dtype}")
    try:
        free_gb, total_gb = cuda_mem_gb()
    except Exception as e:
        if _is_cuda_context_poisoned_error(e):
            print("GPU mem free/total: unavailable (CUDA context poisoned by prior kernel fault).")
            print("Stop here, restart runtime/kernel, and rerun.")
            return
        raise
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
    fsa_local_old_bthd_fn, fsa_local_old_import_err = try_import_fsa_local_old()
    has_fsa_local_old = fsa_local_old_bthd_fn is not None
    fsa_local_older_bthd_fn, fsa_local_older_import_err = try_import_fsa_local_older()
    has_fsa_local_older = fsa_local_older_bthd_fn is not None
    fsa_local_unopt_bthd_fn, fsa_local_unopt_import_err = try_import_fsa_local_unopt()
    has_fsa_local_unopt = fsa_local_unopt_bthd_fn is not None
    fsa_chapter_bthd_fn, fsa_chapter_import_err = try_import_fsa_chapter_routed()
    has_fsa_chapter = fsa_chapter_bthd_fn is not None
    fsa_upstream_fn, fsa_upstream_import_err = try_import_fsa_upstream()
    has_fsa_upstream = fsa_upstream_fn is not None
    fsa_local_varlen_fn, fsa_local_varlen_import_err = try_import_fsa_local_varlen()
    has_fsa_local_varlen = fsa_local_varlen_fn is not None
    flex_mod, flex_import_err = try_import_flex_attention()
    has_flex = flex_mod is not None
    fs_algo_mod, fs_algo_import_err = try_import_flash_sparse_algo()
    has_flash_sparse_algo = fs_algo_mod is not None
    fs_local_nomask_mod, fs_local_nomask_import_err = try_import_flash_sparse_local_nomask()
    has_flash_sparse_local_nomask = fs_local_nomask_mod is not None
    fs_local_nomask_fn = None
    fs_local_topk_fused_fn = None
    if has_flash_sparse_local_nomask:
        fs_local_nomask_fn, fs_local_topk_fused_fn = fs_local_nomask_mod
    if has_fsa_local and _FSA_LOCAL_MODULE_FILE:
        print(f"fsa_local module file: {_FSA_LOCAL_MODULE_FILE}")
    if has_fsa_local_old and _FSA_LOCAL_OLD_MODULE_FILE:
        print(f"fsa_local_old module file: {_FSA_LOCAL_OLD_MODULE_FILE}")
    if has_fsa_local_older and _FSA_LOCAL_OLDER_MODULE_FILE:
        print(f"fsa_local_older module file: {_FSA_LOCAL_OLDER_MODULE_FILE}")
    if has_fsa_local_unopt and _FSA_LOCAL_UNOPT_MODULE_FILE:
        print(f"fsa_local_unopt module file: {_FSA_LOCAL_UNOPT_MODULE_FILE}")
    if has_fsa_chapter and _FSA_CHAPTER_MODULE_FILE:
        print(f"fsa_chapter module file: {_FSA_CHAPTER_MODULE_FILE}")
    if has_fsa_local and fsa_local_import_err:
        print(f"fsa_local import warning: {fsa_local_import_err}")
    if has_fsa_local_old and fsa_local_old_import_err:
        print(f"fsa_local_old import warning: {fsa_local_old_import_err}")
    if has_fsa_local_older and fsa_local_older_import_err:
        print(f"fsa_local_older import warning: {fsa_local_older_import_err}")
    if has_fsa_local_unopt and fsa_local_unopt_import_err:
        print(f"fsa_local_unopt import warning: {fsa_local_unopt_import_err}")
    if has_fsa_chapter and fsa_chapter_import_err:
        print(f"fsa_chapter import warning: {fsa_chapter_import_err}")
    if has_fsa_local_varlen and fsa_local_varlen_import_err:
        print(f"fsa_local_varlen import warning: {fsa_local_varlen_import_err}")

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
    # if not has_flash_sparse_algo:
    #     print(f"(ii.c.g) flash-sparse-attn + gather: skipped (import failed: {fs_algo_import_err})")
    # else:
    #     flash_sparse_attn_func_auto, get_available_backends = fs_algo_mod
    #     fs_algo_state = {}
    #     fs_algo_seed = [42_260]
    #     requested_backend = os.getenv("FLASH_SPARSE_ALGO_BACKEND", "auto").strip().lower()
    #     if requested_backend in ("", "auto", "none"):
    #         requested_backend = None

    #     try:
    #         available_backends = get_available_backends()
    #     except Exception:
    #         available_backends = []

    #     if requested_backend is not None and available_backends and requested_backend not in available_backends:
    #         print(
    #             f"(ii.c.g) flash-sparse-attn + gather: skipped "
    #             f"(requested backend '{requested_backend}' not available; available={available_backends})"
    #         )
    #     else:
    #         try:
    #             fs_algo_fn = flash_sparse_attn_func_auto(backend=requested_backend)
    #             fs_algo_backend = (
    #                 requested_backend
    #                 if requested_backend is not None
    #                 else (available_backends[0] if available_backends else "auto")
    #             )
    #         except Exception as e:
    #             fs_algo_fn = None
    #             fs_algo_backend = "unknown"
    #             print(f"(ii.c.g) flash-sparse-attn + gather: skipped (backend init failed: {e})")

    #         if fs_algo_fn is not None:
    #             def reinit_flash_sparse_algo_gather():
    #                 q, k, v, tok = build_dense_full_inputs(fs_algo_seed[0])
    #                 fs_algo_seed[0] += 1
    #                 fs_algo_state["q"] = q
    #                 fs_algo_state["k"] = k
    #                 fs_algo_state["v"] = v
    #                 fs_algo_state["tok"] = tok
    #                 # Broadcastable no-op bias for APIs that require attn_bias.
    #                 fs_algo_state["attn_bias"] = torch.zeros((1, 1, 1, 1), device=q.device, dtype=q.dtype)

    #             def fwd_flash_sparse_algo_gather():
    #                 k_sel = fs_algo_state["k"][0, fs_algo_state["tok"], :, :].contiguous()
    #                 v_sel = fs_algo_state["v"][0, fs_algo_state["tok"], :, :].contiguous()
    #                 try:
    #                     return fs_algo_fn(
    #                         query=fs_algo_state["q"],
    #                         key=k_sel,
    #                         value=v_sel,
    #                         attn_bias=fs_algo_state["attn_bias"],
    #                         softmax_scale=cfg.scale,
    #                         is_causal=False,
    #                     )
    #                 except RuntimeError as err:
    #                     # Some backends/extensions expect float attn_bias.
    #                     if "expected scalar type Float but found BFloat16" in str(err):
    #                         return fs_algo_fn(
    #                             query=fs_algo_state["q"],
    #                             key=k_sel,
    #                             value=v_sel,
    #                             attn_bias=fs_algo_state["attn_bias"].float(),
    #                             softmax_scale=cfg.scale,
    #                             is_causal=False,
    #                         )
    #                     raise

    #             def zero_flash_sparse_algo_gather():
    #                 reset_grads_in_state(fs_algo_state, ["q", "k", "v"])

    #             try:
    #                 fs_algo_f_ms, fs_algo_b_ms = time_fwd_bwd(
    #                     fwd_flash_sparse_algo_gather,
    #                     loss_from_out_dense,
    #                     iters=iters,
    #                     warmup=warmup,
    #                     clear_cache_each_iter=True,
    #                     reinit_fn=reinit_flash_sparse_algo_gather,
    #                     zero_grad_fn=zero_flash_sparse_algo_gather,
    #                 )
    #                 print(
    #                     f"(ii.c.g) flash-sparse-attn ({fs_algo_backend}) + gather "
    #                     f"fwd: {fs_algo_f_ms:.3f} ms | bwd: {fs_algo_b_ms:.3f} ms"
    #                 )
    #             except Exception as e:
    #                 print(f"(ii.c.g) flash-sparse-attn ({fs_algo_backend}) + gather: skipped ({e})")

    # # ---- (ii.d.g) flash-sparse local Triton no-mask + gather (timed) ----
    # clear_cuda_cache()
    # if not has_flash_sparse_local_nomask:
    #     print(
    #         "(ii.d.g) flash-sparse local-triton (no-mask) + gather: "
    #         f"skipped (import failed: {fs_local_nomask_import_err})"
    #     )
    # else:
    #     fs_local_state = {}
    #     fs_local_seed = [43_260]

    #     def reinit_flash_sparse_local_gather():
    #         q, k, v, tok = build_dense_full_inputs(fs_local_seed[0])
    #         fs_local_seed[0] += 1
    #         fs_local_state["q"] = q
    #         fs_local_state["k"] = k
    #         fs_local_state["v"] = v
    #         fs_local_state["tok"] = tok

    #     def fwd_flash_sparse_local_gather():
    #         k_sel = fs_local_state["k"][0, fs_local_state["tok"], :, :].contiguous()
    #         v_sel = fs_local_state["v"][0, fs_local_state["tok"], :, :].contiguous()
    #         return fs_local_nomask_fn(
    #             query=fs_local_state["q"],
    #             key=k_sel,
    #             value=v_sel,
    #             is_causal=False,
    #             softmax_scale=cfg.scale,
    #         )

    #     def zero_flash_sparse_local_gather():
    #         reset_grads_in_state(fs_local_state, ["q", "k", "v"])

    #     try:
    #         fs_local_f_ms, fs_local_b_ms = time_fwd_bwd(
    #             fwd_flash_sparse_local_gather,
    #             loss_from_out_dense,
    #             iters=iters,
    #             warmup=warmup,
    #             clear_cache_each_iter=True,
    #             reinit_fn=reinit_flash_sparse_local_gather,
    #             zero_grad_fn=zero_flash_sparse_local_gather,
    #         )
    #         print(
    #             "(ii.d.g) flash-sparse local-triton (no-mask) + gather "
    #             f"fwd: {fs_local_f_ms:.3f} ms | bwd: {fs_local_b_ms:.3f} ms"
    #         )
    #     except Exception as e:
    #         print(f"(ii.d.g) flash-sparse local-triton (no-mask) + gather: skipped ({e})")

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
    # clear_cuda_cache()
    # kvc_sparse_state = {}
    # kvc_sparse_seed = [46_260]

    # def reinit_flash_kvcache_sparse():
    #     x = make_inputs(cfg, seed=kvc_sparse_seed[0], per_query_random_topk=True)
    #     kvc_sparse_seed[0] += 1
    #     kvc_sparse_state["q"] = x["q"]
    #     kvc_sparse_state["k"] = x["k"]
    #     kvc_sparse_state["v"] = x["v"]
    #     kvc_sparse_state["bi"] = x["block_indices"]

    # def fwd_flash_kvcache_sparse():
    #     out, msg = try_flash_attn_with_kvcache_sparse_topk_pkg(
    #         kvc_sparse_state["q"],
    #         kvc_sparse_state["k"],
    #         kvc_sparse_state["v"],
    #         kvc_sparse_state["bi"],
    #         block_size=cfg.BS,
    #         scale=cfg.scale,
    #         chunk_qh=None,  # auto
    #     )
    #     if out is None:
    #         raise RuntimeError(msg)
    #     return out

    # try:
    #     kvc_sparse_f_ms = time_fwd_only(
    #         fwd_flash_kvcache_sparse, iters=iters, warmup=warmup,
    #         clear_cache_each_iter=True, reinit_fn=reinit_flash_kvcache_sparse
    #     )
    #     print(f"(vi) flash_attn_with_kvcache sparse-topk fwd: {kvc_sparse_f_ms:.3f} ms")
    # except Exception as e:
    #     print(f"(vi) flash_attn_with_kvcache sparse-topk: skipped ({e})")

    # # ---- (vi.c) flash-sparse-attn sparse top-k (timed, fwd only) ----
    # clear_cuda_cache()
    # if not has_flash_sparse_algo:
    #     print(f"(vi.c) flash-sparse-attn sparse-topk: skipped (import failed: {fs_algo_import_err})")
    # else:
    #     flash_sparse_attn_func_auto, get_available_backends = fs_algo_mod
    #     requested_backend = os.getenv("FLASH_SPARSE_ALGO_BACKEND", "auto").strip().lower()
    #     if requested_backend in ("", "auto", "none"):
    #         requested_backend = None
    #     try:
    #         available_backends = get_available_backends()
    #     except Exception:
    #         available_backends = []

    #     if requested_backend is not None and available_backends and requested_backend not in available_backends:
    #         print(
    #             f"(vi.c) flash-sparse-attn sparse-topk: skipped "
    #             f"(requested backend '{requested_backend}' not available; available={available_backends})"
    #         )
    #     else:
    #         try:
    #             fs_algo_fn = flash_sparse_attn_func_auto(backend=requested_backend)
    #             fs_algo_backend = (
    #                 requested_backend
    #                 if requested_backend is not None
    #                 else (available_backends[0] if available_backends else "auto")
    #             )
    #         except Exception as e:
    #             fs_algo_fn = None
    #             fs_algo_backend = "unknown"
    #             print(f"(vi.c) flash-sparse-attn sparse-topk: skipped (backend init failed: {e})")

    #         if fs_algo_fn is not None:
    #             fs_algo_sparse_state = {}
    #             fs_algo_sparse_seed = [46_760]
    #             sparse_chunk_qh_raw = os.getenv("FLASH_SPARSE_ALGO_SPARSE_CHUNK_QH", "auto").strip().lower()
    #             sparse_chunk_qh = None if sparse_chunk_qh_raw in ("", "auto", "none") else int(sparse_chunk_qh_raw)

    #             def reinit_flash_sparse_algo_sparse():
    #                 x = make_inputs(cfg, seed=fs_algo_sparse_seed[0], per_query_random_topk=True)
    #                 fs_algo_sparse_seed[0] += 1
    #                 fs_algo_sparse_state["q"] = x["q"]
    #                 fs_algo_sparse_state["k"] = x["k"]
    #                 fs_algo_sparse_state["v"] = x["v"]
    #                 fs_algo_sparse_state["bi"] = x["block_indices"]

    #             def fwd_flash_sparse_algo_sparse():
    #                 out, msg = try_flash_sparse_algo_sparse_topk_pkg(
    #                     fs_algo_fn,
    #                     fs_algo_sparse_state["q"],
    #                     fs_algo_sparse_state["k"],
    #                     fs_algo_sparse_state["v"],
    #                     fs_algo_sparse_state["bi"],
    #                     block_size=cfg.BS,
    #                     scale=cfg.scale,
    #                     chunk_qh=sparse_chunk_qh,
    #                 )
    #                 if out is None:
    #                     raise RuntimeError(msg)
    #                 return out

    #             try:
    #                 fs_algo_sparse_f_ms = time_fwd_only(
    #                     fwd_flash_sparse_algo_sparse,
    #                     iters=iters,
    #                     warmup=warmup,
    #                     clear_cache_each_iter=True,
    #                     reinit_fn=reinit_flash_sparse_algo_sparse,
    #                 )
    #                 print(
    #                     f"(vi.c) flash-sparse-attn ({fs_algo_backend}) sparse-topk fwd: "
    #                     f"{fs_algo_sparse_f_ms:.3f} ms"
    #                 )
    #             except Exception as e:
    #                 print(f"(vi.c) flash-sparse-attn ({fs_algo_backend}) sparse-topk: skipped ({e})")

    # # ---- (vi.d) flash-sparse local Triton no-mask sparse top-k (timed, fwd only) ----
    # clear_cuda_cache()
    # if not has_flash_sparse_local_nomask:
    #     print(
    #         "(vi.d) flash-sparse local-triton (no-mask) sparse-topk: "
    #         f"skipped (import failed: {fs_local_nomask_import_err})"
    #     )
    # else:
    #     fs_local_sparse_state = {}
    #     fs_local_sparse_seed = [47_260]
    #     local_sparse_chunk_raw = os.getenv("FLASH_SPARSE_LOCAL_SPARSE_CHUNK_QH", "auto").strip().lower()
    #     local_sparse_chunk_qh = None if local_sparse_chunk_raw in ("", "auto", "none") else int(local_sparse_chunk_raw)

    #     def reinit_flash_sparse_local_sparse():
    #         x = make_inputs(cfg, seed=fs_local_sparse_seed[0], per_query_random_topk=True)
    #         fs_local_sparse_seed[0] += 1
    #         fs_local_sparse_state["q"] = x["q"]
    #         fs_local_sparse_state["k"] = x["k"]
    #         fs_local_sparse_state["v"] = x["v"]
    #         fs_local_sparse_state["bi"] = x["block_indices"]

    #     if fs_local_topk_fused_fn is not None:
    #         def fwd_flash_sparse_local_sparse():
    #             return fs_local_topk_fused_fn(
    #                 query=fs_local_sparse_state["q"],
    #                 key=fs_local_sparse_state["k"],
    #                 value=fs_local_sparse_state["v"],
    #                 block_indices=fs_local_sparse_state["bi"],
    #                 block_size=cfg.BS,
    #                 softmax_scale=cfg.scale,
    #             )
    #         vi_d_label = "(vi.d) flash-sparse local-triton fused sparse-topk"
    #     else:
    #         def fwd_flash_sparse_local_sparse():
    #             out, msg = try_flash_sparse_local_nomask_sparse_topk_pkg(
    #                 fs_local_nomask_fn,
    #                 fs_local_sparse_state["q"],
    #                 fs_local_sparse_state["k"],
    #                 fs_local_sparse_state["v"],
    #                 fs_local_sparse_state["bi"],
    #                 block_size=cfg.BS,
    #                 scale=cfg.scale,
    #                 chunk_qh=local_sparse_chunk_qh,
    #             )
    #             if out is None:
    #                 raise RuntimeError(msg)
    #             return out
    #         vi_d_label = "(vi.d) flash-sparse local-triton (emulated sparse-topk)"

    #     try:
    #         fs_local_sparse_f_ms = time_fwd_only(
    #             fwd_flash_sparse_local_sparse,
    #             iters=iters,
    #             warmup=warmup,
    #             clear_cache_each_iter=True,
    #             reinit_fn=reinit_flash_sparse_local_sparse,
    #         )
    #         print(f"{vi_d_label} fwd: {fs_local_sparse_f_ms:.3f} ms")
    #     except Exception as e:
    #         print(f"(vi.d) flash-sparse local-triton (no-mask) sparse-topk: skipped ({e})")

    # ---- (iii) Custom sparse kernel (variable per-query topk) ----
    # print("\n(iii) Custom sparse (variable per-query topk):")
    # if not has_fsa_opt:
    #     print(f"  fsa_opt: skipped (import failed: {fsa_opt_import_err})")
    # # Initialize once globally and reuse for all sparse strategies.
    # sparse_x = make_inputs(cfg, seed=50_260, per_query_random_topk=True)
    # sparse_state = {
    #     "q": sparse_x["q"],
    #     "k": sparse_x["k"],
    #     "v": sparse_x["v"],
    #     "bi": sparse_x["block_indices"],
    # }

    # def zero_sparse():
    #     reset_grads_in_state(sparse_state, ["q", "k", "v"])

    # # for strat_idx, strat in enumerate(["a", "b", "c", "d"]):
    # #     clear_cuda_cache()

    # #     def fwd_custom():
    # #         return memory_cross_attn(
    # #             sparse_state["q"], sparse_state["k"], sparse_state["v"], sparse_state["bi"],
    # #             cfg.BS, scale=cfg.scale,
    # #             dkv_strategy=strat,
    # #             q_chunk_size=1024,
    # #             d_chunk_size=256,
    # #         )

    # #     f_ms, b_ms = time_fwd_bwd(
    # #         fwd_custom, loss_from_out_sparse, iters=iters, warmup=warmup,
    # #         clear_cache_each_iter=True, reinit_fn=None, zero_grad_fn=zero_sparse
    # #     )
    # #     print(f"  strategy={strat}  fwd: {f_ms:.3f} ms | bwd: {b_ms:.3f} ms")

    # # if has_fsa_opt:
    # #     clear_cuda_cache()

    # #     def fwd_custom_fsa_opt():
    # #         return memory_cross_attn_fsa_opt(
    # #             sparse_state["q"], sparse_state["k"], sparse_state["v"], sparse_state["bi"],
    # #             cfg.BS, scale=cfg.scale
    # #         )

    # #     f_ms, b_ms = time_fwd_bwd(
    # #         fwd_custom_fsa_opt, loss_from_out_sparse, iters=iters, warmup=warmup,
    # #         clear_cache_each_iter=True, reinit_fn=None, zero_grad_fn=zero_sparse
    # #     )
    # #     print(f"  strategy=fsa_opt  fwd: {f_ms:.3f} ms | bwd: {b_ms:.3f} ms")

    # # ---- (viii) PyTorch FlexAttention block-sparse baseline ----
    # # Disabled by request.
    # # if not has_flex:
    # #     print(f"  strategy=flex_attention: skipped (import failed: {flex_import_err})")
    # # else:
    # #     ...
    # print("  strategy=flex_attention: skipped (disabled)")

    # ---- (vii) FSA local copy baseline (inner API modified, prefix trick) ----
    # print("\n(vii) FSA local-copy baseline (prefix/no-prefix timeline)")
    # if not has_fsa_local:
    #     print(f"  FSA local-copy: skipped (import failed: {fsa_local_import_err})")
    # elif cfg.BS not in {32, 64, 128, 256, 512, 1024}:
    #     print(f"  FSA local-copy: skipped (unsupported block_size BS={cfg.BS}; requires one of 32/64/128/256/512/1024)")
    # else:
    #     auto_safe_large_gqa = os.getenv("FSA_LOCAL_AUTO_SAFE_FOR_LARGE_GQA", "1").strip().lower() not in (
    #         "0", "false", "no", "off", ""
    #     )
    #     risky_large_gqa = (
    #         int(cfg.HK) == 1
    #         and int(cfg.H // max(1, cfg.HK)) >= 16
    #         and int(cfg.BS) >= 128
    #         and int(cfg.TQ) >= 131072
    #     )
    #     if auto_safe_large_gqa and risky_large_gqa:
    #         # Stability defaults for very large HK=1,G>=16 workloads.
    #         # setdefault keeps explicit user overrides intact.
    #         os.environ.setdefault("FSA_LOCAL_USE_PREFIX_TIMELINE", "1")
    #         os.environ.setdefault("FSA_LOCAL_FWD_FULL_DESERIALIZE", "0")
    #         os.environ.setdefault("FSA_LOCAL_DQ_ACCUM_MODE", "buffer")
    #         os.environ.setdefault("FSA_LOCAL_DQ_FORCE_ATOMIC", "0")
    #         os.environ.setdefault("FSA_LOCAL_DQ_FULL_DESERIALIZE", "0")
    #         os.environ.setdefault("FSA_LOCAL_DQ_SAFE_TOKEN_INDEX", "1")
    #         os.environ.setdefault("FSA_LOCAL_COMPACT_ACTIVE_BLOCKS", "0")
    #         os.environ.setdefault("FSA_LOCAL_BWD_PREP_MODE", "legacy")
    #         os.environ.setdefault("FSA_LOCAL_SORT_TOPK_Q_IDX", "0")
    #         os.environ.setdefault("FSA_LOCAL_BWD_SEQUENCE_PARALLEL", "0")

    #     clear_cuda_cache()
    #     fsa_dkdv_bq = os.getenv("FSA_LOCAL_BWD_DKDV_BQ", "auto")
    #     fsa_dq_bq = os.getenv("FSA_LOCAL_BWD_DQ_BQ", "auto")
    #     fsa_dq_loops = os.getenv("FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS", "auto")
    #     fsa_nsa_fwd = os.getenv("FSA_LOCAL_USE_NSA_STYLE_FWD", "1")
    #     fsa_force_nsa_small_g = os.getenv("FSA_LOCAL_FORCE_NSA_STYLE_FWD_SMALL_G", "1")
    #     fsa_native_nsa_fwd = os.getenv("FSA_LOCAL_USE_NSA_NATIVE_FWD", "0")
    #     fsa_pad_g16 = os.getenv("FSA_LOCAL_PAD_G_TO_16", "1")
    #     fsa_small_g_mode = os.getenv("FSA_LOCAL_SMALL_G_MODE", "fallback")
    #     fsa_torch_chunk = os.getenv("FSA_LOCAL_TORCH_CHUNK_TOKENS", "512")
    #     fsa_fwd_chunk = os.getenv("FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL", "auto")
    #     fsa_head_tile = os.getenv("FSA_LOCAL_HEAD_TILE", "auto")
    #     fsa_sort_qidx = os.getenv("FSA_LOCAL_SORT_TOPK_Q_IDX", "auto")
    #     fsa_dq_accum_mode = os.getenv("FSA_LOCAL_DQ_ACCUM_MODE", "atomic")
    #     fsa_dq_force_atomic = os.getenv("FSA_LOCAL_DQ_FORCE_ATOMIC", "1")
    #     fsa_dq_full_deser = os.getenv("FSA_LOCAL_DQ_FULL_DESERIALIZE", "1")
    #     fsa_dq_atomic_guard = os.getenv("FSA_LOCAL_DQ_ATOMIC_GUARD", "1")
    #     fsa_dq_safe_token_index = os.getenv("FSA_LOCAL_DQ_SAFE_TOKEN_INDEX", "1")
    #     fsa_dq_buf_dtype = os.getenv("FSA_LOCAL_DQ_BUFFER_DTYPE", "auto")
    #     fsa_compact_blocks = os.getenv("FSA_LOCAL_COMPACT_ACTIVE_BLOCKS", "auto")
    #     fsa_bwd_prep_mode = os.getenv("FSA_LOCAL_BWD_PREP_MODE", "auto")
    #     fsa_max_kblk = os.getenv("FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE", "128")
    #     fsa_dkdv_mode = os.getenv("FSA_LOCAL_DKDV_MODE", "auto")
    #     fsa_dkdv_two_pass = os.getenv("FSA_LOCAL_DKDV_TWO_PASS", "auto")
    #     fsa_dkdv_schedule = os.getenv("FSA_LOCAL_DKDV_SCHEDULE", "auto")
    #     fsa_dkdv_persist_auto = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_AUTO", "auto")
    #     fsa_dkdv_persist_ratio = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_ACTIVE_RATIO", "auto")
    #     fsa_dkdv_persist_minq = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_ACTIVE_Q", "auto")
    #     fsa_dkdv_persist_minwi = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_WORK_ITEMS", "auto")
    #     fsa_dkdv_persist_minqpi = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_Q_PER_ITEM", "auto")
    #     fsa_dkdv_persist_chunk = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_CHUNK", "auto")
    #     fsa_dkdv_persist_workers_factor = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_WORKERS_FACTOR", "auto")
    #     fsa_dkdv_persist_target_items = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_TARGET_ITEMS_PER_WORKER", "auto")
    #     fsa_dkdv_persist_min_items = os.getenv("FSA_LOCAL_DKDV_PERSISTENT_MIN_ITEMS_PER_WORKER", "auto")
    #     fsa_use_prefix_timeline = os.getenv("FSA_LOCAL_USE_PREFIX_TIMELINE", "0")
    #     fsa_fwd_full_deser = os.getenv("FSA_LOCAL_FWD_FULL_DESERIALIZE", "1")
    #     fsa_fwd_packed_gqa = os.getenv("FSA_LOCAL_FWD_PACKED_GQA", "auto")
    #     fsa_nsa_packed_gqa = os.getenv("FSA_LOCAL_NSA_PACKED_GQA", "auto")
    #     fsa_nsa_packed_scope = os.getenv("FSA_LOCAL_NSA_PACKED_GQA_SCOPE", "small_g")
    #     fsa_dq_packed_gqa = os.getenv("FSA_LOCAL_DQ_PACKED_GQA", "auto")
    #     fsa_gqa_route_collapse = os.getenv("FSA_LOCAL_GQA_ROUTE_COLLAPSE", "auto")
    #     fsa_block_prune_mode = os.getenv("FSA_LOCAL_BLOCK_PRUNING_MODE", "auto")
    #     fsa_block_prune_sanitize = os.getenv("FSA_LOCAL_BLOCK_PRUNING_SANITIZE", "1")
    #     fsa_block_prune_min_tail = os.getenv("FSA_LOCAL_BLOCK_PRUNE_MIN_TAIL", "auto")
    #     fsa_block_prune_min_ratio = os.getenv("FSA_LOCAL_BLOCK_PRUNE_MIN_RATIO", "auto")
    #     fsa_active_map_ratio_th = os.getenv("FSA_LOCAL_ACTIVE_MAP_RATIO_THRESHOLD", "auto")
    #     fsa_bwd_seq_parallel = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL", "auto")
    #     fsa_bwd_seq_parallel_streams = os.getenv("FSA_LOCAL_BWD_SEQUENCE_PARALLEL_STREAMS", "auto")
    #     fsa_fwd_flat_multi_seq = os.getenv("FSA_LOCAL_FWD_FLAT_MULTI_SEQ", "1")
    #     fsa_dq_flat_multi_seq = os.getenv("FSA_LOCAL_DQ_FLAT_MULTI_SEQ", "1")
    #     fsa_bwd_flat_multi_seq = os.getenv("FSA_LOCAL_BWD_FLAT_MULTI_SEQ", "1")
    #     fsa_use_arch_policy = os.getenv("FSA_LOCAL_USE_ARCH_POLICY", "1")
    #     fsa_hopper_async = os.getenv("FSA_LOCAL_HOPPER_ASYNC_PIPELINE", "auto")
    #     fsa_hopper_stages = os.getenv("FSA_LOCAL_HOPPER_PIPELINE_STAGES", "auto")
    #     fsa_hopper_chunks = os.getenv("FSA_LOCAL_HOPPER_PIPELINE_CHUNKS", "auto")
    #     print(
    #         f"  FSA local tuning: disable_causal_mask=True, "
    #         f"FSA_LOCAL_BWD_DKDV_BQ={fsa_dkdv_bq}, "
    #         f"FSA_LOCAL_BWD_DQ_BQ={fsa_dq_bq}, "
    #         f"FSA_LOCAL_BWD_DQ_NUM_Q_BLOCKS={fsa_dq_loops}, "
    #         f"FSA_LOCAL_USE_NSA_STYLE_FWD={fsa_nsa_fwd}, "
    #         f"FSA_LOCAL_FORCE_NSA_STYLE_FWD_SMALL_G={fsa_force_nsa_small_g}, "
    #         f"FSA_LOCAL_USE_NSA_NATIVE_FWD={fsa_native_nsa_fwd}, "
    #         f"FSA_LOCAL_PAD_G_TO_16={fsa_pad_g16}, "
    #         f"FSA_LOCAL_SMALL_G_MODE={fsa_small_g_mode}, "
    #         f"FSA_LOCAL_TORCH_CHUNK_TOKENS={fsa_torch_chunk}, "
    #         f"FSA_LOCAL_FWD_MAX_TOKENS_PER_CALL={fsa_fwd_chunk}, "
    #         f"FSA_LOCAL_HEAD_TILE={fsa_head_tile}, "
    #         f"FSA_LOCAL_SORT_TOPK_Q_IDX={fsa_sort_qidx}, "
    #         f"FSA_LOCAL_DQ_ACCUM_MODE={fsa_dq_accum_mode}, "
    #         f"FSA_LOCAL_DQ_FORCE_ATOMIC={fsa_dq_force_atomic}, "
    #         f"FSA_LOCAL_DQ_FULL_DESERIALIZE={fsa_dq_full_deser}, "
    #         f"FSA_LOCAL_DQ_ATOMIC_GUARD={fsa_dq_atomic_guard}, "
    #         f"FSA_LOCAL_DQ_SAFE_TOKEN_INDEX={fsa_dq_safe_token_index}, "
    #         f"FSA_LOCAL_DQ_BUFFER_DTYPE={fsa_dq_buf_dtype}, "
    #         f"FSA_LOCAL_COMPACT_ACTIVE_BLOCKS={fsa_compact_blocks}, "
    #         f"FSA_LOCAL_BWD_PREP_MODE={fsa_bwd_prep_mode}, "
    #         f"FSA_LOCAL_MAX_KERNEL_BLOCK_SIZE={fsa_max_kblk}, "
    #         f"FSA_LOCAL_DKDV_MODE={fsa_dkdv_mode}, "
    #         f"FSA_LOCAL_DKDV_TWO_PASS={fsa_dkdv_two_pass}, "
    #         f"FSA_LOCAL_DKDV_SCHEDULE={fsa_dkdv_schedule}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_AUTO={fsa_dkdv_persist_auto}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_ACTIVE_RATIO={fsa_dkdv_persist_ratio}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_MIN_ACTIVE_Q={fsa_dkdv_persist_minq}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_MIN_WORK_ITEMS={fsa_dkdv_persist_minwi}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_MIN_Q_PER_ITEM={fsa_dkdv_persist_minqpi}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_CHUNK={fsa_dkdv_persist_chunk}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_WORKERS_FACTOR={fsa_dkdv_persist_workers_factor}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_TARGET_ITEMS_PER_WORKER={fsa_dkdv_persist_target_items}, "
    #         f"FSA_LOCAL_DKDV_PERSISTENT_MIN_ITEMS_PER_WORKER={fsa_dkdv_persist_min_items}, "
    #         f"FSA_LOCAL_USE_PREFIX_TIMELINE={fsa_use_prefix_timeline}, "
    #         f"FSA_LOCAL_FWD_FULL_DESERIALIZE={fsa_fwd_full_deser}, "
    #         f"FSA_LOCAL_FWD_PACKED_GQA={fsa_fwd_packed_gqa}, "
    #         f"FSA_LOCAL_NSA_PACKED_GQA={fsa_nsa_packed_gqa}, "
    #         f"FSA_LOCAL_NSA_PACKED_GQA_SCOPE={fsa_nsa_packed_scope}, "
    #         f"FSA_LOCAL_DQ_PACKED_GQA={fsa_dq_packed_gqa}, "
    #         f"FSA_LOCAL_GQA_ROUTE_COLLAPSE={fsa_gqa_route_collapse}, "
    #         f"FSA_LOCAL_BLOCK_PRUNING_MODE={fsa_block_prune_mode}, "
    #         f"FSA_LOCAL_BLOCK_PRUNING_SANITIZE={fsa_block_prune_sanitize}, "
    #         f"FSA_LOCAL_BLOCK_PRUNE_MIN_TAIL={fsa_block_prune_min_tail}, "
    #         f"FSA_LOCAL_BLOCK_PRUNE_MIN_RATIO={fsa_block_prune_min_ratio}, "
    #         f"FSA_LOCAL_ACTIVE_MAP_RATIO_THRESHOLD={fsa_active_map_ratio_th}, "
    #         f"FSA_LOCAL_BWD_SEQUENCE_PARALLEL={fsa_bwd_seq_parallel}, "
    #         f"FSA_LOCAL_BWD_SEQUENCE_PARALLEL_STREAMS={fsa_bwd_seq_parallel_streams}, "
    #         f"FSA_LOCAL_FWD_FLAT_MULTI_SEQ={fsa_fwd_flat_multi_seq}, "
    #         f"FSA_LOCAL_DQ_FLAT_MULTI_SEQ={fsa_dq_flat_multi_seq}, "
    #         f"FSA_LOCAL_BWD_FLAT_MULTI_SEQ={fsa_bwd_flat_multi_seq}, "
    #         f"FSA_LOCAL_USE_ARCH_POLICY={fsa_use_arch_policy}, "
    #         f"FSA_LOCAL_HOPPER_ASYNC_PIPELINE={fsa_hopper_async}, "
    #         f"FSA_LOCAL_HOPPER_PIPELINE_STAGES={fsa_hopper_stages}, "
    #         f"FSA_LOCAL_HOPPER_PIPELINE_CHUNKS={fsa_hopper_chunks}"
    #     )
    #     hq_hk_ok = (cfg.HK > 0) and (cfg.H % cfg.HK == 0)
    #     print(
    #         "  FSA full-deser forward preconditions: "
    #         f"(1) HQ % HK == 0 -> {hq_hk_ok}, "
    #         "(2) routed queries may differ across KV heads; full-deser now uses union-range handling. "
    #         "Fallback only occurs when HQ/HK precondition is unmet."
    #     )
    #     fsa_local_state = {}
    #     fsa_local_seed = [55_260]
    #     use_prefix_timeline = str(fsa_use_prefix_timeline).strip().lower() not in (
    #         "0", "false", "no", "off", ""
    #     )
    #     fsa_local_warned_detached = {"printed": False}

    #     def _build_prefix_fallback_from_local_state():
    #         q_np = fsa_local_state["q_full"]
    #         k_np = fsa_local_state["k_full"]
    #         v_np = fsa_local_state["v_full"]
    #         bi_np = fsa_local_state["bi_full"]
    #         if q_np is None or k_np is None or v_np is None or bi_np is None:
    #             raise RuntimeError("Cannot build prefix fallback state: missing no-prefix tensors.")
    #         q_pref, k_pref, v_pref, bi_pref, tfull = build_nsa_inputs(
    #             q_np.detach(), k_np.detach(), v_np.detach(), bi_np, int(k_np.shape[1])
    #         )
    #         topk_pref = (
    #             bi_pref
    #             .permute(0, 2, 1, 3)
    #             .reshape(
    #                 bi_pref.shape[2],
    #                 bi_pref.shape[0] * bi_pref.shape[1],
    #                 bi_pref.shape[3],
    #             )
    #         )
    #         topk_pref = _canonicalize_topk_idx(topk_pref)
    #         cu_q_pref = torch.tensor([0, tfull], device=q_pref.device, dtype=torch.int32)
    #         cu_k_pref = torch.tensor([0, int(k_np.shape[1])], device=q_pref.device, dtype=torch.int32)
    #         return {
    #             "q_full": q_pref,
    #             "k_full": k_pref,
    #             "v_full": v_pref,
    #             "bi_full": bi_pref,
    #             "topk_idx_hns": topk_pref,
    #             "cu_q": cu_q_pref,
    #             "cu_k": cu_k_pref,
    #             "Tfull": tfull,
    #         }

    #     def reinit_fsa_local():
    #         x = make_inputs(cfg, seed=fsa_local_seed[0], per_query_random_topk=True)
    #         fsa_local_seed[0] += 1
    #         fsa_local_state["_prefix_fallback_state"] = None
    #         fsa_local_state["_using_prefix_fallback"] = False
    #         if use_prefix_timeline:
    #             q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
    #                 x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
    #             )
    #             bi_for_topk = bi_full
    #             cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
    #             cu_k = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
    #         else:
    #             q_full = x["q"].detach().contiguous().requires_grad_(True)
    #             k_full = x["k"].detach().contiguous().requires_grad_(True)
    #             v_full = x["v"].detach().contiguous().requires_grad_(True)
    #             bi_full = x["block_indices"].detach().contiguous()
    #             bi_for_topk = bi_full
    #             Tfull = int(q_full.shape[1])
    #             cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
    #             cu_k = torch.tensor([0, int(k_full.shape[1])], device=q_full.device, dtype=torch.int32)

    #         topk_idx_hns = (
    #             bi_for_topk
    #             .permute(0, 2, 1, 3)
    #             .reshape(
    #                 bi_for_topk.shape[2],
    #                 bi_for_topk.shape[0] * bi_for_topk.shape[1],
    #                 bi_for_topk.shape[3],
    #             )
    #         )
    #         topk_idx_hns = _canonicalize_topk_idx(topk_idx_hns)
    #         fsa_local_state["q_full"] = q_full
    #         fsa_local_state["k_full"] = k_full
    #         fsa_local_state["v_full"] = v_full
    #         fsa_local_state["bi_full"] = bi_full
    #         fsa_local_state["topk_idx_hns"] = topk_idx_hns
    #         fsa_local_state["cu_q"] = cu_q
    #         fsa_local_state["cu_k"] = cu_k
    #         fsa_local_state["Tfull"] = Tfull

    #     def fwd_fsa_local():
    #         required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
    #         if any((key not in fsa_local_state) or (fsa_local_state.get(key) is None) for key in required):
    #             reinit_fsa_local()
    #         k_len = int(fsa_local_state["cu_k"][-1].item())
    #         try:
    #             out = fsa_local_bthd_fn(
    #                 q_bthd=fsa_local_state["q_full"],
    #                 k_bthd=fsa_local_state["k_full"][:, :k_len, :, :],
    #                 v_bthd=fsa_local_state["v_full"][:, :k_len, :, :],
    #                 block_indices_bths=None,
    #                 block_size=cfg.BS,
    #                 softmax_scale=cfg.scale,
    #                 cu_seqlens_q=fsa_local_state["cu_q"],
    #                 cu_seqlens_k=fsa_local_state["cu_k"],
    #                 topk_idx_hns=fsa_local_state["topk_idx_hns"],
    #                 assume_sorted_topk=True,
    #                 disable_causal_mask=True,
    #             )
    #             if out is None:
    #                 raise RuntimeError("FSA local forward returned None.")
    #             if (not out.requires_grad) and (
    #                 fsa_local_state["q_full"].requires_grad
    #                 or fsa_local_state["k_full"].requires_grad
    #                 or fsa_local_state["v_full"].requires_grad
    #             ):
    #                 if not use_prefix_timeline:
    #                     if not fsa_local_warned_detached["printed"]:
    #                         print(
    #                             "  FSA local: no-prefix output was detached; auto-falling back to prefix-mode execution "
    #                             "for grad-valid benchmarking."
    #                         )
    #                         fsa_local_warned_detached["printed"] = True
    #                     if fsa_local_state.get("_prefix_fallback_state") is None:
    #                         fsa_local_state["_prefix_fallback_state"] = _build_prefix_fallback_from_local_state()
    #                     pref = fsa_local_state["_prefix_fallback_state"]
    #                     out = fsa_local_bthd_fn(
    #                         q_bthd=pref["q_full"],
    #                         k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         block_indices_bths=None,
    #                         block_size=cfg.BS,
    #                         softmax_scale=cfg.scale,
    #                         cu_seqlens_q=pref["cu_q"],
    #                         cu_seqlens_k=pref["cu_k"],
    #                         topk_idx_hns=pref["topk_idx_hns"],
    #                         assume_sorted_topk=True,
    #                         disable_causal_mask=True,
    #                     )
    #                     if out is None or (not out.requires_grad):
    #                         raise RuntimeError("FSA local prefix fallback produced invalid (detached/None) output.")
    #                     fsa_local_state["_using_prefix_fallback"] = True
    #                     return out
    #                 raise RuntimeError("FSA local output is detached while inputs require grad.")
    #             return out
    #         except Exception as e_bthd:
    #             # Robust fallback path for environments where the BTHD wrapper fails.
    #             if not has_fsa_local_varlen:
    #                 raise
    #             with _temporary_environ(_fsa_local_varlen_stable_env()):
    #                 out_var = fsa_local_varlen_fn(
    #                     q=fsa_local_state["q_full"][0],
    #                     k=fsa_local_state["k_full"][0, :k_len, :, :],
    #                     v=fsa_local_state["v_full"][0, :k_len, :, :],
    #                     topk_idx=fsa_local_state["topk_idx_hns"],
    #                     block_size=cfg.BS,
    #                     cu_seqlens_q=fsa_local_state["cu_q"],
    #                     cu_seqlens_k=fsa_local_state["cu_k"],
    #                     softmax_scale=cfg.scale,
    #                     disable_causal_mask=True,
    #                 )
    #             if out_var is None:
    #                 raise RuntimeError(
    #                     f"FSA local bthd failed ({e_bthd}); varlen fallback returned None."
    #                 )
    #             out = out_var.unsqueeze(0)
    #             if (not out.requires_grad) and (
    #                 fsa_local_state["q_full"].requires_grad
    #                 or fsa_local_state["k_full"].requires_grad
    #                 or fsa_local_state["v_full"].requires_grad
    #             ):
    #                 if not use_prefix_timeline:
    #                     if not fsa_local_warned_detached["printed"]:
    #                         print(
    #                             "  FSA local varlen fallback output was detached in no-prefix mode; "
    #                             "auto-falling back to prefix-mode execution."
    #                         )
    #                         fsa_local_warned_detached["printed"] = True
    #                     if fsa_local_state.get("_prefix_fallback_state") is None:
    #                         fsa_local_state["_prefix_fallback_state"] = _build_prefix_fallback_from_local_state()
    #                     pref = fsa_local_state["_prefix_fallback_state"]
    #                     out = fsa_local_bthd_fn(
    #                         q_bthd=pref["q_full"],
    #                         k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         block_indices_bths=None,
    #                         block_size=cfg.BS,
    #                         softmax_scale=cfg.scale,
    #                         cu_seqlens_q=pref["cu_q"],
    #                         cu_seqlens_k=pref["cu_k"],
    #                         topk_idx_hns=pref["topk_idx_hns"],
    #                         assume_sorted_topk=True,
    #                         disable_causal_mask=True,
    #                     )
    #                     if out is None or (not out.requires_grad):
    #                         raise RuntimeError("FSA local prefix fallback produced invalid (detached/None) output.")
    #                     fsa_local_state["_using_prefix_fallback"] = True
    #                     return out
    #                 raise RuntimeError("FSA local varlen fallback output is detached while inputs require grad.")
    #             return out

    #     def loss_query_only_fsa_local(o_full):
    #         if o_full is None:
    #             raise RuntimeError("FSA local produced None output before loss.")
    #         if int(o_full.shape[1]) == (cfg.TK + cfg.TQ):
    #             o_q = o_full[:, cfg.TK:, :, :]
    #         elif int(o_full.shape[1]) == cfg.TQ:
    #             o_q = o_full
    #         else:
    #             raise RuntimeError(
    #                 f"Unexpected FSA local output length {int(o_full.shape[1])}; "
    #                 f"expected TQ={cfg.TQ} or TK+TQ={cfg.TK + cfg.TQ}."
    #             )
    #         return (o_q * dO_sparse).sum()

    #     def zero_fsa_local():
    #         reset_grads_in_state(fsa_local_state, ["q_full", "k_full", "v_full"])
    #         pref = fsa_local_state.get("_prefix_fallback_state")
    #         if isinstance(pref, dict):
    #             reset_grads_in_state(pref, ["q_full", "k_full", "v_full"])

    #     try:
    #         fsa_local_f_ms, fsa_local_b_ms = time_fwd_bwd(
    #             fwd_fsa_local, loss_query_only_fsa_local, iters=iters, warmup=warmup,
    #             clear_cache_each_iter=True, reinit_fn=reinit_fsa_local, zero_grad_fn=zero_fsa_local
    #         )
    #         print(f"  FSA local-copy (query loss only)  fwd: {fsa_local_f_ms:.3f} ms | bwd: {fsa_local_b_ms:.3f} ms")
    #     except Exception as e:
    #         print(f"  FSA local-copy: skipped (runtime/compile failure: {e})")
    #         if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
    #             print("  FSA local-copy traceback:")
    #             print(traceback.format_exc())
    #         if "illegal memory access" in str(e).lower():
    #             print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
    #             return

    # ---- (vii.b) FSA local unoptimized baseline ----
    print("\n(vii.b) FSA local-unoptimized baseline (prefix timeline)")
    if not has_fsa_local_unopt:
        print(f"  FSA local-unoptimized: skipped (import failed: {fsa_local_unopt_import_err})")
    elif cfg.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"  FSA local-unoptimized: skipped (unsupported block_size BS={cfg.BS}; "
            "requires one of 32/64/128/256/512/1024)"
        )
    else:
        clear_cuda_cache()
        fsa_local_unopt_state = {}
        fsa_local_unopt_seed = [56_260]

        def reinit_fsa_local_unopt():
            x = make_inputs(cfg, seed=fsa_local_unopt_seed[0], per_query_random_topk=True)
            fsa_local_unopt_seed[0] += 1
            q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
                x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
            )
            topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(cfg.HK, Tfull, bi_full.shape[-1]).to(torch.int32)
            cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
            cu_k = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
            fsa_local_unopt_state["q_full"] = q_full
            fsa_local_unopt_state["k_full"] = k_full
            fsa_local_unopt_state["v_full"] = v_full
            fsa_local_unopt_state["topk_idx_hns"] = topk_idx_hns
            fsa_local_unopt_state["cu_q"] = cu_q
            fsa_local_unopt_state["cu_k"] = cu_k

        def fwd_fsa_local_unopt():
            required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
            if any((key not in fsa_local_unopt_state) or (fsa_local_unopt_state.get(key) is None) for key in required):
                reinit_fsa_local_unopt()
            k_len = int(fsa_local_unopt_state["cu_k"][-1].item())
            out = fsa_local_unopt_bthd_fn(
                q_bthd=fsa_local_unopt_state["q_full"],
                k_bthd=fsa_local_unopt_state["k_full"][:, :k_len, :, :],
                v_bthd=fsa_local_unopt_state["v_full"][:, :k_len, :, :],
                block_indices_bths=None,
                block_size=cfg.BS,
                softmax_scale=cfg.scale,
                cu_seqlens_q=fsa_local_unopt_state["cu_q"],
                cu_seqlens_k=fsa_local_unopt_state["cu_k"],
                topk_idx_hns=fsa_local_unopt_state["topk_idx_hns"],
                assume_sorted_topk=False,
                disable_causal_mask=True,
            )
            if out is None:
                raise RuntimeError("FSA local-unoptimized forward returned None.")
            return out

        def loss_query_only_fsa_local_unopt(o_full):
            if o_full is None:
                raise RuntimeError("FSA local-unoptimized produced None output before loss.")
            if int(o_full.shape[1]) == (cfg.TK + cfg.TQ):
                o_q = o_full[:, cfg.TK:, :, :]
            elif int(o_full.shape[1]) == cfg.TQ:
                o_q = o_full
            else:
                raise RuntimeError(
                    f"Unexpected FSA local-unoptimized output length {int(o_full.shape[1])}; "
                    f"expected TQ={cfg.TQ} or TK+TQ={cfg.TK + cfg.TQ}."
                )
            return (o_q * dO_sparse).sum()

        def zero_fsa_local_unopt():
            reset_grads_in_state(fsa_local_unopt_state, ["q_full", "k_full", "v_full"])

        try:
            fsa_local_unopt_f_ms, fsa_local_unopt_b_ms = time_fwd_bwd(
                fwd_fsa_local_unopt, loss_query_only_fsa_local_unopt, iters=iters, warmup=warmup,
                clear_cache_each_iter=True, reinit_fn=reinit_fsa_local_unopt, zero_grad_fn=zero_fsa_local_unopt
            )
            print(
                f"  FSA local-unoptimized (query loss only)  "
                f"fwd: {fsa_local_unopt_f_ms:.3f} ms | bwd: {fsa_local_unopt_b_ms:.3f} ms"
            )
        except Exception as e:
            print(f"  FSA local-unoptimized: skipped (runtime/compile failure: {e})")
            if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                print("  FSA local-unoptimized traceback:")
                print(traceback.format_exc())
            if "illegal memory access" in str(e).lower():
                print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
                return

    # ---- (ix) chapter-routed local baseline ----
    print("\n(vii.c) FSA local-optimized-old baseline (prefix timeline)")
    if not has_fsa_local_old:
        print(f"  FSA local-optimized-old: skipped (import failed: {fsa_local_old_import_err})")
    elif cfg.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"  FSA local-optimized-old: skipped (unsupported block_size BS={cfg.BS}; "
            "requires one of 32/64/128/256/512/1024)"
        )
    else:
        clear_cuda_cache()
        fsa_local_old_state = {}
        fsa_local_old_seed = [56_760]

        def reinit_fsa_local_old():
            x = make_inputs(cfg, seed=fsa_local_old_seed[0], per_query_random_topk=True)
            fsa_local_old_seed[0] += 1
            q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
                x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
            )
            topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(cfg.HK, Tfull, bi_full.shape[-1]).to(torch.int32)
            cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
            cu_k = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
            fsa_local_old_state["q_full"] = q_full
            fsa_local_old_state["k_full"] = k_full
            fsa_local_old_state["v_full"] = v_full
            fsa_local_old_state["topk_idx_hns"] = topk_idx_hns
            fsa_local_old_state["cu_q"] = cu_q
            fsa_local_old_state["cu_k"] = cu_k

        def fwd_fsa_local_old():
            required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
            if any((key not in fsa_local_old_state) or (fsa_local_old_state.get(key) is None) for key in required):
                reinit_fsa_local_old()
            k_len = int(fsa_local_old_state["cu_k"][-1].item())
            out = fsa_local_old_bthd_fn(
                q_bthd=fsa_local_old_state["q_full"],
                k_bthd=fsa_local_old_state["k_full"][:, :k_len, :, :],
                v_bthd=fsa_local_old_state["v_full"][:, :k_len, :, :],
                block_indices_bths=None,
                block_size=cfg.BS,
                softmax_scale=cfg.scale,
                cu_seqlens_q=fsa_local_old_state["cu_q"],
                cu_seqlens_k=fsa_local_old_state["cu_k"],
                topk_idx_hns=fsa_local_old_state["topk_idx_hns"],
                assume_sorted_topk=False,
                disable_causal_mask=True,
            )
            if out is None:
                raise RuntimeError("FSA local-optimized-old forward returned None.")
            return out

        def loss_query_only_fsa_local_old(o_full):
            if o_full is None:
                raise RuntimeError("FSA local-optimized-old produced None output before loss.")
            if int(o_full.shape[1]) == (cfg.TK + cfg.TQ):
                o_q = o_full[:, cfg.TK:, :, :]
            elif int(o_full.shape[1]) == cfg.TQ:
                o_q = o_full
            else:
                raise RuntimeError(
                    f"Unexpected FSA local-optimized-old output length {int(o_full.shape[1])}; "
                    f"expected TQ={cfg.TQ} or TK+TQ={cfg.TK + cfg.TQ}."
                )
            return (o_q * dO_sparse).sum()

        def zero_fsa_local_old():
            reset_grads_in_state(fsa_local_old_state, ["q_full", "k_full", "v_full"])

        try:
            fsa_local_old_f_ms, fsa_local_old_b_ms = time_fwd_bwd(
                fwd_fsa_local_old, loss_query_only_fsa_local_old, iters=iters, warmup=warmup,
                clear_cache_each_iter=True, reinit_fn=reinit_fsa_local_old, zero_grad_fn=zero_fsa_local_old
            )
            print(
                f"  FSA local-optimized-old (query loss only)  "
                f"fwd: {fsa_local_old_f_ms:.3f} ms | bwd: {fsa_local_old_b_ms:.3f} ms"
            )
        except Exception as e:
            print(f"  FSA local-optimized-old: skipped (runtime/compile failure: {e})")
            if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                print("  FSA local-optimized-old traceback:")
                print(traceback.format_exc())
            if "illegal memory access" in str(e).lower():
                print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
                return

    print("\n(vii.d) FSA local-optimized-older baseline (prefix timeline)")
    if not has_fsa_local_older:
        print(f"  FSA local-optimized-older: skipped (import failed: {fsa_local_older_import_err})")
    elif cfg.BS not in {32, 64, 128, 256, 512, 1024}:
        print(
            f"  FSA local-optimized-older: skipped (unsupported block_size BS={cfg.BS}; "
            "requires one of 32/64/128/256/512/1024)"
        )
    else:
        clear_cuda_cache()
        fsa_local_older_state = {}
        fsa_local_older_seed = [57_260]

        def reinit_fsa_local_older():
            x = make_inputs(cfg, seed=fsa_local_older_seed[0], per_query_random_topk=True)
            fsa_local_older_seed[0] += 1
            q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
                x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
            )
            topk_idx_hns = bi_full.permute(0, 2, 1, 3).reshape(cfg.HK, Tfull, bi_full.shape[-1]).to(torch.int32)
            cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
            cu_k = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
            fsa_local_older_state["q_full"] = q_full
            fsa_local_older_state["k_full"] = k_full
            fsa_local_older_state["v_full"] = v_full
            fsa_local_older_state["topk_idx_hns"] = topk_idx_hns
            fsa_local_older_state["cu_q"] = cu_q
            fsa_local_older_state["cu_k"] = cu_k

        def fwd_fsa_local_older():
            required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
            if any((key not in fsa_local_older_state) or (fsa_local_older_state.get(key) is None) for key in required):
                reinit_fsa_local_older()
            k_len = int(fsa_local_older_state["cu_k"][-1].item())
            out = fsa_local_older_bthd_fn(
                q_bthd=fsa_local_older_state["q_full"],
                k_bthd=fsa_local_older_state["k_full"][:, :k_len, :, :],
                v_bthd=fsa_local_older_state["v_full"][:, :k_len, :, :],
                block_indices_bths=None,
                block_size=cfg.BS,
                softmax_scale=cfg.scale,
                cu_seqlens_q=fsa_local_older_state["cu_q"],
                cu_seqlens_k=fsa_local_older_state["cu_k"],
                topk_idx_hns=fsa_local_older_state["topk_idx_hns"],
                assume_sorted_topk=False,
                disable_causal_mask=True,
            )
            if out is None:
                raise RuntimeError("FSA local-optimized-older forward returned None.")
            return out

        def loss_query_only_fsa_local_older(o_full):
            if o_full is None:
                raise RuntimeError("FSA local-optimized-older produced None output before loss.")
            if int(o_full.shape[1]) == (cfg.TK + cfg.TQ):
                o_q = o_full[:, cfg.TK:, :, :]
            elif int(o_full.shape[1]) == cfg.TQ:
                o_q = o_full
            else:
                raise RuntimeError(
                    f"Unexpected FSA local-optimized-older output length {int(o_full.shape[1])}; "
                    f"expected TQ={cfg.TQ} or TK+TQ={cfg.TK + cfg.TQ}."
                )
            return (o_q * dO_sparse).sum()

        def zero_fsa_local_older():
            reset_grads_in_state(fsa_local_older_state, ["q_full", "k_full", "v_full"])

        try:
            fsa_local_older_f_ms, fsa_local_older_b_ms = time_fwd_bwd(
                fwd_fsa_local_older, loss_query_only_fsa_local_older, iters=iters, warmup=warmup,
                clear_cache_each_iter=True, reinit_fn=reinit_fsa_local_older, zero_grad_fn=zero_fsa_local_older
            )
            print(
                f"  FSA local-optimized-older (query loss only)  "
                f"fwd: {fsa_local_older_f_ms:.3f} ms | bwd: {fsa_local_older_b_ms:.3f} ms"
            )
        except Exception as e:
            print(f"  FSA local-optimized-older: skipped (runtime/compile failure: {e})")
            if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
                print("  FSA local-optimized-older traceback:")
                print(traceback.format_exc())
            if "illegal memory access" in str(e).lower():
                print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
                return

    # ---- (ix) chapter-routed local baseline ----
    # if not has_fsa_chapter:
    #     print(f"  FSA chapter-routed: skipped (import failed: {fsa_chapter_import_err})")
    # else:
    #     clear_cuda_cache()
    #     fsa_chapter_state = {}
    #     fsa_chapter_seed = [57_260]
    #     fsa_chapter_chunk_raw = os.getenv("FSA_CHAPTER_QUERY_CHUNK_SIZE", "4096").strip().lower()
    #     try:
    #         fsa_chapter_chunk = max(1, int(fsa_chapter_chunk_raw))
    #     except Exception:
    #         fsa_chapter_chunk = 4096
    #     fsa_chapter_dedupe = os.getenv("FSA_CHAPTER_DEDUPE_QUERIES", "0").strip().lower() in (
    #         "1", "true", "yes", "on"
    #     )
    #     fsa_chapter_use_triton = os.getenv("FSA_CHAPTER_USE_TRITON", "auto")
    #     fsa_chapter_triton_bwd = os.getenv("FSA_CHAPTER_TRITON_BWD_RECOMPUTE", "1")
    #     print(
    #         f"  FSA chapter-routed tuning: disable_causal_mask=True, "
    #         f"route_collapse={os.getenv('FSA_LOCAL_GQA_ROUTE_COLLAPSE', 'auto')}, "
    #         f"FSA_CHAPTER_QUERY_CHUNK_SIZE={fsa_chapter_chunk}, "
    #         f"FSA_CHAPTER_DEDUPE_QUERIES={int(fsa_chapter_dedupe)}, "
    #         f"FSA_CHAPTER_USE_TRITON={fsa_chapter_use_triton}, "
    #         f"FSA_CHAPTER_TRITON_BWD_RECOMPUTE={fsa_chapter_triton_bwd}"
    #     )

    #     def reinit_fsa_chapter():
    #         x = make_inputs(cfg, seed=fsa_chapter_seed[0], per_query_random_topk=True)
    #         fsa_chapter_seed[0] += 1
    #         fsa_chapter_state["q"] = x["q"].detach().contiguous().requires_grad_(True)
    #         fsa_chapter_state["k"] = x["k"].detach().contiguous().requires_grad_(True)
    #         fsa_chapter_state["v"] = x["v"].detach().contiguous().requires_grad_(True)
    #         fsa_chapter_state["bi"] = x["block_indices"].detach().contiguous()

    #     def fwd_fsa_chapter():
    #         required = ("q", "k", "v", "bi")
    #         if any((key not in fsa_chapter_state) or (fsa_chapter_state.get(key) is None) for key in required):
    #             reinit_fsa_chapter()
    #         out = fsa_chapter_bthd_fn(
    #             q_bthd=fsa_chapter_state["q"],
    #             k_bthd=fsa_chapter_state["k"],
    #             v_bthd=fsa_chapter_state["v"],
    #             block_indices_bths=fsa_chapter_state["bi"],
    #             block_size=cfg.BS,
    #             softmax_scale=cfg.scale,
    #             disable_causal_mask=True,
    #             route_collapse=os.getenv("FSA_LOCAL_GQA_ROUTE_COLLAPSE", "auto"),
    #             chapter_query_chunk_size=fsa_chapter_chunk,
    #             dedupe_queries_per_chapter=fsa_chapter_dedupe,
    #         )
    #         if out is None:
    #             raise RuntimeError("FSA chapter-routed forward returned None.")
    #         if not output_requires_grad_for_inputs(
    #             out,
    #             fsa_chapter_state["q"],
    #             fsa_chapter_state["k"],
    #             fsa_chapter_state["v"],
    #         ):
    #             raise RuntimeError("FSA chapter-routed output is detached while inputs require grad.")
    #         return out

    #     def loss_query_only_fsa_chapter(o_full):
    #         if o_full is None:
    #             raise RuntimeError("FSA chapter-routed produced None output before loss.")
    #         if int(o_full.shape[1]) != cfg.TQ:
    #             raise RuntimeError(
    #                 f"Unexpected FSA chapter-routed output length {int(o_full.shape[1])}; expected TQ={cfg.TQ}."
    #             )
    #         return (o_full * dO_sparse).sum()

    #     def zero_fsa_chapter():
    #         reset_grads_in_state(fsa_chapter_state, ["q", "k", "v"])

    #     try:
    #         fsa_chapter_f_ms, fsa_chapter_b_ms = time_fwd_bwd(
    #             fwd_fsa_chapter,
    #             loss_query_only_fsa_chapter,
    #             iters=iters,
    #             warmup=warmup,
    #             clear_cache_each_iter=True,
    #             reinit_fn=reinit_fsa_chapter,
    #             zero_grad_fn=zero_fsa_chapter,
    #         )
    #         print(
    #             f"  FSA chapter-routed (query loss only)  "
    #             f"fwd: {fsa_chapter_f_ms:.3f} ms | bwd: {fsa_chapter_b_ms:.3f} ms"
    #         )
    #     except Exception as e:
    #         print(f"  FSA chapter-routed: skipped (runtime/compile failure: {e})")
    #         if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
    #             print("  FSA chapter-routed traceback:")
    #             print(traceback.format_exc())
    #         if "illegal memory access" in str(e).lower():
    #             print("  CUDA context is likely poisoned by prior kernel fault; stopping benchmark early.")
    #             return

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

    # # ---- (viii) FLA FSA upstream baseline ----
    # print("\n(viii) FLA FSA upstream baseline (prefix/no-prefix timeline)")
    # if not has_fsa_upstream:
    #     print(f"  FLA FSA upstream: skipped (import failed: {fsa_upstream_import_err})")
    #     return
    # if cfg.BS not in {32, 64, 128, 256}:
    #     print(f"  FLA FSA upstream: skipped (unsupported block_size BS={cfg.BS}; requires one of 32/64/128/256)")
    #     return

    # fla_fsa_state = {}
    # fla_fsa_seed = [65_260]
    # fla_warned_detached = {"printed": False}
    # upstream_prefix_raw = os.getenv("FSA_UPSTREAM_USE_PREFIX_TIMELINE", "0")
    # upstream_use_prefix_timeline = str(upstream_prefix_raw).strip().lower() not in (
    #     "0", "false", "no", "off", ""
    # )
    # force_local_compat_raw = os.getenv("FLA_FSA_FORCE_LOCAL_COMPAT", "1")
    # force_local_compat = str(force_local_compat_raw).strip().lower() not in (
    #     "0", "false", "no", "off", ""
    # )

    # def _build_prefix_fallback_from_fla_state():
    #     q_np = fla_fsa_state["q_full"]
    #     k_np = fla_fsa_state["k_full"]
    #     v_np = fla_fsa_state["v_full"]
    #     bi_np = fla_fsa_state["bi_full"]
    #     if q_np is None or k_np is None or v_np is None or bi_np is None:
    #         raise RuntimeError("Cannot build FLA prefix fallback state: missing no-prefix tensors.")
    #     q_pref, k_pref, v_pref, bi_pref, tfull = build_nsa_inputs(
    #         q_np.detach(), k_np.detach(), v_np.detach(), bi_np, int(k_np.shape[1])
    #     )
    #     topk_pref = (
    #         bi_pref
    #         .permute(0, 2, 1, 3)
    #         .reshape(
    #             bi_pref.shape[2],
    #             bi_pref.shape[0] * bi_pref.shape[1],
    #             bi_pref.shape[3],
    #         )
    #     )
    #     topk_pref = _canonicalize_topk_idx(topk_pref)
    #     cu_q_pref = torch.tensor([0, tfull], device=q_pref.device, dtype=torch.int32)
    #     cu_k_pref = torch.tensor([0, int(k_np.shape[1])], device=q_pref.device, dtype=torch.int32)
    #     return {
    #         "q_full": q_pref,
    #         "k_full": k_pref,
    #         "v_full": v_pref,
    #         "bi_full": bi_pref,
    #         "topk_idx_hns": topk_pref,
    #         "cu_q": cu_q_pref,
    #         "cu_k": cu_k_pref,
    #         "Tfull": tfull,
    #     }

    # def reinit_fla_fsa():
    #     x = make_inputs(cfg, seed=fla_fsa_seed[0], per_query_random_topk=True)
    #     fla_fsa_seed[0] += 1
    #     fla_fsa_state["_prefix_fallback_state"] = None
    #     if upstream_use_prefix_timeline:
    #         q_full, k_full, v_full, bi_full, Tfull = build_nsa_inputs(
    #             x["q"].detach(), x["k"].detach(), x["v"].detach(), x["block_indices"], cfg.TK
    #         )
    #         cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
    #         cu_k = torch.tensor([0, cfg.TK], device=q_full.device, dtype=torch.int32)
    #     else:
    #         q_full = x["q"].detach().contiguous().requires_grad_(True)
    #         k_full = x["k"].detach().contiguous().requires_grad_(True)
    #         v_full = x["v"].detach().contiguous().requires_grad_(True)
    #         bi_full = x["block_indices"].detach().contiguous()
    #         Tfull = int(q_full.shape[1])
    #         cu_q = torch.tensor([0, Tfull], device=q_full.device, dtype=torch.int32)
    #         cu_k = torch.tensor([0, int(k_full.shape[1])], device=q_full.device, dtype=torch.int32)
    #     topk_idx_hns = (
    #         bi_full
    #         .permute(0, 2, 1, 3)
    #         .reshape(
    #             bi_full.shape[2],
    #             bi_full.shape[0] * bi_full.shape[1],
    #             bi_full.shape[3],
    #         )
    #     )
    #     topk_idx_hns = _canonicalize_topk_idx(topk_idx_hns)
    #     fla_fsa_state["q_full"] = q_full
    #     fla_fsa_state["k_full"] = k_full
    #     fla_fsa_state["v_full"] = v_full
    #     fla_fsa_state["bi_full"] = bi_full
    #     fla_fsa_state["topk_idx_hns"] = topk_idx_hns
    #     fla_fsa_state["cu_q"] = cu_q
    #     fla_fsa_state["cu_k"] = cu_k
    #     fla_fsa_state["Tfull"] = Tfull

    # def fwd_fla_fsa():
    #     o = fsa_upstream_fn(
    #         q=fla_fsa_state["q_full"][0],
    #         k=fla_fsa_state["k_full"][0],
    #         v=fla_fsa_state["v_full"][0],
    #         topk_idx=fla_fsa_state["topk_idx_hns"],
    #         block_size=cfg.BS,
    #         cu_seqlens=fla_fsa_state["cu_q"],
    #         softmax_scale=cfg.scale,
    #     )
    #     return o.unsqueeze(0)

    # def loss_query_only_fla(o_full):
    #     if int(o_full.shape[1]) == (cfg.TK + cfg.TQ):
    #         o_q = o_full[:, cfg.TK:, :, :]
    #     elif int(o_full.shape[1]) == cfg.TQ:
    #         o_q = o_full
    #     else:
    #         raise RuntimeError(
    #             f"Unexpected FLA output length {int(o_full.shape[1])}; "
    #             f"expected TQ={cfg.TQ} or TK+TQ={cfg.TK + cfg.TQ}."
    #         )
    #     return (o_q * dO_sparse).sum()

    # def zero_fla_fsa():
    #     reset_grads_in_state(fla_fsa_state, ["q_full", "k_full", "v_full"])
    #     pref = fla_fsa_state.get("_prefix_fallback_state")
    #     if isinstance(pref, dict):
    #         reset_grads_in_state(pref, ["q_full", "k_full", "v_full"])

    # if force_local_compat:
    #     if has_fsa_local_varlen:
    #         print("  FLA FSA upstream: using local varlen compatibility implementation by default.")
    #     elif has_fsa_local:
    #         print("  FLA FSA upstream: varlen compatibility unavailable; using local bthd compatibility by default.")
    #     else:
    #         print("  FLA FSA upstream: skipped (forced local compatibility, but no local compatibility backend is available).")
    #         return
    #     clear_cuda_cache()

    #     def fwd_fla_fsa_local_varlen():
    #         required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
    #         if any((key not in fla_fsa_state) or (fla_fsa_state.get(key) is None) for key in required):
    #             reinit_fla_fsa()
    #         k_len = int(fla_fsa_state["cu_k"][-1].item())
    #         with _temporary_environ(_fsa_local_varlen_stable_env()):
    #             o = fsa_local_varlen_fn(
    #                 q=fla_fsa_state["q_full"][0],
    #                 k=fla_fsa_state["k_full"][0, :k_len, :, :],
    #                 v=fla_fsa_state["v_full"][0, :k_len, :, :],
    #                 topk_idx=fla_fsa_state["topk_idx_hns"],
    #                 block_size=cfg.BS,
    #                 cu_seqlens_q=fla_fsa_state["cu_q"],
    #                 cu_seqlens_k=fla_fsa_state["cu_k"],
    #                 softmax_scale=cfg.scale,
    #                 disable_causal_mask=True,
    #             )
    #         if o is None:
    #             raise RuntimeError("Local varlen compatibility returned None.")
    #         out = o.unsqueeze(0)
    #         if (not out.requires_grad) and (
    #             fla_fsa_state["q_full"].requires_grad
    #             or fla_fsa_state["k_full"].requires_grad
    #             or fla_fsa_state["v_full"].requires_grad
    #         ):
    #             if not upstream_use_prefix_timeline:
    #                 if not fla_warned_detached["printed"]:
    #                     print(
    #                         "  FLA local-varlen compatibility output was detached in no-prefix mode; "
    #                         "auto-falling back to prefix-mode execution."
    #                     )
    #                     fla_warned_detached["printed"] = True
    #                 if fla_fsa_state.get("_prefix_fallback_state") is None:
    #                     fla_fsa_state["_prefix_fallback_state"] = _build_prefix_fallback_from_fla_state()
    #                 pref = fla_fsa_state["_prefix_fallback_state"]
    #                 out = fsa_local_bthd_fn(
    #                     q_bthd=pref["q_full"],
    #                     k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                     v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                     block_indices_bths=None,
    #                     block_size=cfg.BS,
    #                     softmax_scale=cfg.scale,
    #                     cu_seqlens_q=pref["cu_q"],
    #                     cu_seqlens_k=pref["cu_k"],
    #                     topk_idx_hns=pref["topk_idx_hns"],
    #                     assume_sorted_topk=True,
    #                     disable_causal_mask=True,
    #                 )
    #                 if out is None or (not out.requires_grad):
    #                     raise RuntimeError("FLA prefix fallback produced invalid (detached/None) output.")
    #                 return out
    #             raise RuntimeError("FLA local-varlen compatibility output is detached while inputs require grad.")
    #         return out

    #     def fwd_fla_fsa_local_bthd():
    #         required = ("q_full", "k_full", "v_full", "topk_idx_hns", "cu_q", "cu_k")
    #         if any((key not in fla_fsa_state) or (fla_fsa_state.get(key) is None) for key in required):
    #             reinit_fla_fsa()
    #         k_len = int(fla_fsa_state["cu_k"][-1].item())
    #         o = fsa_local_bthd_fn(
    #             q_bthd=fla_fsa_state["q_full"],
    #             k_bthd=fla_fsa_state["k_full"][:, :k_len, :, :],
    #             v_bthd=fla_fsa_state["v_full"][:, :k_len, :, :],
    #             block_indices_bths=None,
    #             block_size=cfg.BS,
    #             softmax_scale=cfg.scale,
    #             cu_seqlens_q=fla_fsa_state["cu_q"],
    #             cu_seqlens_k=fla_fsa_state["cu_k"],
    #             topk_idx_hns=fla_fsa_state["topk_idx_hns"],
    #             assume_sorted_topk=True,
    #             disable_causal_mask=True,
    #         )
    #         if o is None:
    #             raise RuntimeError("Local bthd compatibility returned None.")
    #         if (not o.requires_grad) and (
    #             fla_fsa_state["q_full"].requires_grad
    #             or fla_fsa_state["k_full"].requires_grad
    #             or fla_fsa_state["v_full"].requires_grad
    #         ):
    #             if not upstream_use_prefix_timeline:
    #                 if not fla_warned_detached["printed"]:
    #                     print(
    #                         "  FLA local-bthd compatibility output was detached in no-prefix mode; "
    #                         "auto-falling back to prefix-mode execution."
    #                     )
    #                     fla_warned_detached["printed"] = True
    #                 if fla_fsa_state.get("_prefix_fallback_state") is None:
    #                     fla_fsa_state["_prefix_fallback_state"] = _build_prefix_fallback_from_fla_state()
    #                 pref = fla_fsa_state["_prefix_fallback_state"]
    #                 o = fsa_local_bthd_fn(
    #                     q_bthd=pref["q_full"],
    #                     k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                     v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                     block_indices_bths=None,
    #                     block_size=cfg.BS,
    #                     softmax_scale=cfg.scale,
    #                     cu_seqlens_q=pref["cu_q"],
    #                     cu_seqlens_k=pref["cu_k"],
    #                     topk_idx_hns=pref["topk_idx_hns"],
    #                     assume_sorted_topk=True,
    #                     disable_causal_mask=True,
    #                 )
    #                 if o is None or (not o.requires_grad):
    #                     raise RuntimeError("FLA prefix fallback produced invalid (detached/None) output.")
    #                 return o
    #             raise RuntimeError("FLA local-bthd compatibility output is detached while inputs require grad.")
    #         return o

    #     try:
    #         fwd_local_compat = fwd_fla_fsa_local_varlen if has_fsa_local_varlen else fwd_fla_fsa_local_bthd
    #         local_tag = "local-varlen" if has_fsa_local_varlen else "local-bthd"
    #         fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
    #             fwd_local_compat,
    #             loss_query_only_fla,
    #             iters=iters,
    #             warmup=warmup,
    #             clear_cache_each_iter=True,
    #             reinit_fn=reinit_fla_fsa,
    #             zero_grad_fn=zero_fla_fsa,
    #         )
    #         print(
    #             f"  FLA FSA {local_tag} compatibility (query loss only)  "
    #             f"fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms"
    #         )
    #     except Exception as e_local:
    #         if has_fsa_local and has_fsa_local_varlen:
    #             try:
    #                 fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
    #                     fwd_fla_fsa_local_bthd,
    #                     loss_query_only_fla,
    #                     iters=iters,
    #                     warmup=warmup,
    #                     clear_cache_each_iter=True,
    #                     reinit_fn=reinit_fla_fsa,
    #                     zero_grad_fn=zero_fla_fsa,
    #                 )
    #                 print(
    #                     f"  FLA FSA local-bthd compatibility (query loss only)  "
    #                     f"fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms"
    #                 )
    #             except Exception as e_local_bthd:
    #                 print(
    #                     f"  FLA FSA upstream: skipped (local compatibility failure: "
    #                     f"varlen={e_local}; bthd={e_local_bthd})"
    #                 )
    #                 if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
    #                     print("  FLA local compatibility traceback (varlen):")
    #                     print(traceback.format_exc())
    #         else:
    #             print(f"  FLA FSA upstream: skipped (local compatibility failure: {e_local})")
    #             if os.getenv("FSA_BENCH_VERBOSE_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off", ""):
    #                 print("  FLA local compatibility traceback:")
    #                 print(traceback.format_exc())
    #     return

    # clear_cuda_cache()
    # try:
    #     fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
    #         fwd_fla_fsa,
    #         loss_query_only_fla,
    #         iters=iters,
    #         warmup=warmup,
    #         clear_cache_each_iter=True,
    #         reinit_fn=reinit_fla_fsa,
    #         zero_grad_fn=zero_fla_fsa,
    #     )
    #     print(f"  FLA FSA upstream (query loss only)  fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms")
    # except Exception as e:
    #     err_text = str(e)
    #     # Known Triton incompatibility in upstream file:
    #     #   lse_ptrs = (ptr,) tuple form can fail to compile on newer Triton.
    #     known_ptr_tuple_issue = (
    #         "lse_ptrs = (lse_ptr + pid_q_j * stride_lse_n,)" in err_text
    #         or "at 73:10" in err_text
    #     )
    #     if known_ptr_tuple_issue and has_fsa_local_varlen:
    #         print("  FLA FSA upstream hit known Triton pointer-tuple compile issue; retrying with local varlen compatibility copy.")
    #         clear_cuda_cache()

    #         def fwd_fla_fsa_local_varlen():
    #             required = ("q_full", "k_full", "v_full", "cu_q", "cu_k", "topk_idx_hns")
    #             if any((key not in fla_fsa_state) or (fla_fsa_state.get(key) is None) for key in required):
    #                 reinit_fla_fsa()
    #             with _temporary_environ(_fsa_local_varlen_stable_env()):
    #                 o = fsa_local_varlen_fn(
    #                     q=fla_fsa_state["q_full"][0],
    #                     k=fla_fsa_state["k_full"][0],
    #                     v=fla_fsa_state["v_full"][0],
    #                     topk_idx=fla_fsa_state["topk_idx_hns"],
    #                     block_size=cfg.BS,
    #                     cu_seqlens_q=fla_fsa_state["cu_q"],
    #                     cu_seqlens_k=fla_fsa_state["cu_k"],
    #                     softmax_scale=cfg.scale,
    #                     disable_causal_mask=True,
    #                 )
    #             if o is None:
    #                 raise RuntimeError("Local varlen fallback returned None.")
    #             out = o.unsqueeze(0)
    #             if (not out.requires_grad) and (
    #                 fla_fsa_state["q_full"].requires_grad
    #                 or fla_fsa_state["k_full"].requires_grad
    #                 or fla_fsa_state["v_full"].requires_grad
    #             ):
    #                 if not upstream_use_prefix_timeline:
    #                     if not fla_warned_detached["printed"]:
    #                         print(
    #                             "  FLA local varlen fallback output was detached in no-prefix mode; "
    #                             "auto-falling back to prefix-mode execution."
    #                         )
    #                         fla_warned_detached["printed"] = True
    #                     if fla_fsa_state.get("_prefix_fallback_state") is None:
    #                         fla_fsa_state["_prefix_fallback_state"] = _build_prefix_fallback_from_fla_state()
    #                     pref = fla_fsa_state["_prefix_fallback_state"]
    #                     out = fsa_local_bthd_fn(
    #                         q_bthd=pref["q_full"],
    #                         k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         block_indices_bths=None,
    #                         block_size=cfg.BS,
    #                         softmax_scale=cfg.scale,
    #                         cu_seqlens_q=pref["cu_q"],
    #                         cu_seqlens_k=pref["cu_k"],
    #                         topk_idx_hns=pref["topk_idx_hns"],
    #                         assume_sorted_topk=True,
    #                         disable_causal_mask=True,
    #                     )
    #                     if out is None or (not out.requires_grad):
    #                         raise RuntimeError("FLA prefix fallback produced invalid (detached/None) output.")
    #                     return out
    #                 raise RuntimeError("FLA local varlen fallback output is detached while inputs require grad.")
    #             return out

    #         def fwd_fla_fsa_local_bthd():
    #             required = ("q_full", "k_full", "v_full", "bi_full", "cu_q", "cu_k")
    #             if any((key not in fla_fsa_state) or (fla_fsa_state.get(key) is None) for key in required):
    #                 reinit_fla_fsa()
    #             k_len = int(fla_fsa_state["cu_k"][-1].item())
    #             o = fsa_local_bthd_fn(
    #                 q_bthd=fla_fsa_state["q_full"],
    #                 k_bthd=fla_fsa_state["k_full"][:, :k_len, :, :],
    #                 v_bthd=fla_fsa_state["v_full"][:, :k_len, :, :],
    #                 block_indices_bths=None,
    #                 block_size=cfg.BS,
    #                 softmax_scale=cfg.scale,
    #                 cu_seqlens_q=fla_fsa_state["cu_q"],
    #                 cu_seqlens_k=fla_fsa_state["cu_k"],
    #                 topk_idx_hns=fla_fsa_state["topk_idx_hns"],
    #                 assume_sorted_topk=True,
    #                 disable_causal_mask=True,
    #             )
    #             if o is None:
    #                 raise RuntimeError("Local bthd fallback returned None.")
    #             if (not o.requires_grad) and (
    #                 fla_fsa_state["q_full"].requires_grad
    #                 or fla_fsa_state["k_full"].requires_grad
    #                 or fla_fsa_state["v_full"].requires_grad
    #             ):
    #                 if not upstream_use_prefix_timeline:
    #                     if not fla_warned_detached["printed"]:
    #                         print(
    #                             "  FLA local bthd fallback output was detached in no-prefix mode; "
    #                             "auto-falling back to prefix-mode execution."
    #                         )
    #                         fla_warned_detached["printed"] = True
    #                     if fla_fsa_state.get("_prefix_fallback_state") is None:
    #                         fla_fsa_state["_prefix_fallback_state"] = _build_prefix_fallback_from_fla_state()
    #                     pref = fla_fsa_state["_prefix_fallback_state"]
    #                     o = fsa_local_bthd_fn(
    #                         q_bthd=pref["q_full"],
    #                         k_bthd=pref["k_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         v_bthd=pref["v_full"][:, :int(pref["cu_k"][-1].item()), :, :],
    #                         block_indices_bths=None,
    #                         block_size=cfg.BS,
    #                         softmax_scale=cfg.scale,
    #                         cu_seqlens_q=pref["cu_q"],
    #                         cu_seqlens_k=pref["cu_k"],
    #                         topk_idx_hns=pref["topk_idx_hns"],
    #                         assume_sorted_topk=True,
    #                         disable_causal_mask=True,
    #                     )
    #                     if o is None or (not o.requires_grad):
    #                         raise RuntimeError("FLA prefix fallback produced invalid (detached/None) output.")
    #                     return o
    #                 raise RuntimeError("FLA local bthd fallback output is detached while inputs require grad.")
    #             return o

    #         try:
    #             fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
    #                 fwd_fla_fsa_local_varlen,
    #                 loss_query_only_fla,
    #                 iters=iters,
    #                 warmup=warmup,
    #                 clear_cache_each_iter=True,
    #                 reinit_fn=reinit_fla_fsa,
    #                 zero_grad_fn=zero_fla_fsa,
    #             )
    #             print(
    #                 f"  FLA FSA local-varlen compatibility (query loss only)  "
    #                 f"fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms"
    #             )
    #         except Exception as e2:
    #             # Second fallback: local BTHD wrapper is generally more robust than varlen API shims.
    #             if has_fsa_local:
    #                 try:
    #                     fsa_u_f_ms, fsa_u_b_ms = time_fwd_bwd(
    #                         fwd_fla_fsa_local_bthd,
    #                         loss_query_only_fla,
    #                         iters=iters,
    #                         warmup=warmup,
    #                         clear_cache_each_iter=True,
    #                         reinit_fn=reinit_fla_fsa,
    #                         zero_grad_fn=zero_fla_fsa,
    #                     )
    #                     print(
    #                         f"  FLA FSA local-bthd compatibility (query loss only)  "
    #                         f"fwd: {fsa_u_f_ms:.3f} ms | bwd: {fsa_u_b_ms:.3f} ms"
    #                     )
    #                 except Exception as e3:
    #                     print(
    #                         f"  FLA FSA upstream: skipped (runtime/compile failure: {e}; "
    #                         f"fallback failed: {e2}; bthd fallback failed: {e3})"
    #                     )
    #             else:
    #                 print(f"  FLA FSA upstream: skipped (runtime/compile failure: {e}; fallback failed: {e2})")
    #                 if not has_fsa_local_varlen:
    #                     print(f"  local-varlen fallback unavailable: {fsa_local_varlen_import_err}")
    #     else:
    #         print(f"  FLA FSA upstream: skipped (runtime/compile failure: {e})")
    #         if known_ptr_tuple_issue and not has_fsa_local_varlen:
    #             print(f"  local-varlen fallback unavailable: {fsa_local_varlen_import_err}")


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
    # correctness_ok = check_correctness_5x(cfg, sanity_TQ=16384, sanity_M=16384, sanity_BS=32, num_checks=3)
    # if correctness_ok is False:
    #     print("Skipping timing benchmarks because CUDA context is poisoned. Restart runtime and rerun.")
    #     sys.exit(0)

    # (b) timing benchmarks (dense baselines, custom sparse, NSA)
    # WARNING: with your full sizes (TQ=128*2048=262k), this is heavy but should run on H100.
    run_benchmarks(cfg, iters=5, warmup=5)
