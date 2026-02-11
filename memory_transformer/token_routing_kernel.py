"""
Helpers for loading token-routing sparse attention kernels.

This module resolves the stable kernel variants from `kernels-final/`:
  - v1 -> kernel_v1.py
  - v2 -> kernel_v2.py (default)
  - v3 -> kernel_v3.py
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional


_KERNEL_FILES: Dict[str, str] = {
    "v1": "kernel_v1.py",
    "v2": "kernel_v2.py",
    "v3": "kernel_v3.py",
}

_KERNEL_FN_CACHE: Dict[str, Callable] = {}
_KERNEL_IMPORT_ERROR: Dict[str, Exception] = {}
_CACHE_LOCK = threading.Lock()


def normalize_kernel_version(version: Optional[str]) -> str:
    """Normalize user-provided kernel version to {'v1','v2','v3'}."""
    if version is None:
        return "v2"

    value = str(version).strip().lower()
    if value in {"1", "v1", "kernel_v1"}:
        return "v1"
    if value in {"2", "v2", "kernel_v2"}:
        return "v2"
    if value in {"3", "v3", "kernel_v3"}:
        return "v3"

    raise ValueError(
        f"Unsupported token_routing_kernel_version='{version}'. "
        "Expected one of: v1, v2, v3."
    )


def get_token_routing_kernel_fn(version: Optional[str]) -> Callable:
    """
    Load and return `FSA_topk_sparse_attention_bthd` for the requested version.

    Returns:
        Callable with signature matching kernels-final FSA wrapper.
    """
    ver = normalize_kernel_version(version)

    with _CACHE_LOCK:
        if ver in _KERNEL_FN_CACHE:
            return _KERNEL_FN_CACHE[ver]
        if ver in _KERNEL_IMPORT_ERROR:
            raise RuntimeError(
                f"Token-routing kernel '{ver}' failed to import previously."
            ) from _KERNEL_IMPORT_ERROR[ver]

    kernels_dir = Path(__file__).resolve().parents[1] / "kernels-final"
    kernel_file = kernels_dir / _KERNEL_FILES[ver]
    if not kernel_file.exists():
        err = FileNotFoundError(
            f"Token-routing kernel file not found: {kernel_file}"
        )
        with _CACHE_LOCK:
            _KERNEL_IMPORT_ERROR[ver] = err
        raise err

    module_name = f"_memory_token_routing_{ver}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(kernel_file))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to build module spec for: {kernel_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        fn = getattr(module, "FSA_topk_sparse_attention_bthd", None)
        if fn is None:
            raise AttributeError(
                f"{kernel_file} does not expose FSA_topk_sparse_attention_bthd"
            )
    except Exception as exc:
        with _CACHE_LOCK:
            _KERNEL_IMPORT_ERROR[ver] = exc
        raise

    with _CACHE_LOCK:
        _KERNEL_FN_CACHE[ver] = fn
    return fn

