"""Helpers for working with FAISS on optional GPU hardware."""

from __future__ import annotations

import importlib
from typing import Any, Optional


def _import_faiss() -> Optional[Any]:
    """Return the imported ``faiss`` module or ``None`` if unavailable."""
    try:
        return importlib.import_module("faiss")
    except ModuleNotFoundError:
        return None


def _gpu_runtime_available(faiss_module: Any) -> bool:
    required_attrs = ["index_cpu_to_all_gpus", "index_gpu_to_cpu", "StandardGpuResources"]
    if not all(hasattr(faiss_module, attr) for attr in required_attrs):
        return False
    get_num_gpus = getattr(faiss_module, "get_num_gpus", None)
    if get_num_gpus is None:
        return False
    try:
        return bool(get_num_gpus())
    except Exception:
        return False


def is_faiss_gpu_available() -> bool:
    """Return ``True`` if FAISS GPU bindings are importable and GPUs are visible."""
    faiss_module = _import_faiss()
    if faiss_module is None:
        return False
    return _gpu_runtime_available(faiss_module)


def clone_index_to_gpu(index: Any) -> Optional[Any]:
    """Clone a CPU index to GPU memory when supported."""
    faiss_module = _import_faiss()
    if faiss_module is None or not _gpu_runtime_available(faiss_module):
        return None
    try:
        return faiss_module.index_cpu_to_all_gpus(index)
    except Exception:
        return None


def clone_index_to_cpu(index: Any) -> Any:
    """Return a CPU copy of a FAISS index, even if the input lives on GPU."""
    faiss_module = _import_faiss()
    if faiss_module is None or not hasattr(faiss_module, "index_gpu_to_cpu"):
        return index
    try:
        return faiss_module.index_gpu_to_cpu(index)
    except Exception:
        return index
