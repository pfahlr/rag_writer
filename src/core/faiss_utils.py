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


def set_faiss_threads(num_threads: int) -> None:
    """Attempt to configure FAISS threading."""

    if num_threads <= 0:
        return

    faiss_module = _import_faiss()
    if faiss_module is None:
        return

    for attr in ("omp_set_num_threads", "set_num_threads"):
        setter = getattr(faiss_module, attr, None)
        if setter is None:
            continue
        try:
            setter(num_threads)
            return
        except Exception:
            continue


def ensure_cpu_index(index: Any) -> Any:
    """Return a CPU copy of ``index`` even if it already lives on GPU."""

    faiss_module = _import_faiss()
    if faiss_module is None:
        return index

    to_cpu = getattr(faiss_module, "index_gpu_to_cpu", None)
    if to_cpu is None:
        return index

    try:
        return to_cpu(index)
    except Exception:
        return index


def try_index_cpu_to_gpu(index: Any, device_id: Optional[int] = None) -> Optional[Any]:
    """Attempt to copy a CPU index to GPU memory."""

    faiss_module = _import_faiss()
    if faiss_module is None or not _gpu_runtime_available(faiss_module):
        return None

    if device_id is not None:
        to_gpu_device = getattr(faiss_module, "index_cpu_to_gpu", None)
        resources_cls = getattr(faiss_module, "StandardGpuResources", None)
        if callable(to_gpu_device) and resources_cls is not None:
            try:
                resources = resources_cls()
                return to_gpu_device(resources, device_id, index)
            except Exception:
                return None

    to_gpu = getattr(faiss_module, "index_cpu_to_all_gpus", None)
    if to_gpu is None:
        return None

    try:
        return to_gpu(index)
    except Exception:
        return None


def clone_index_to_gpu(index: Any) -> Optional[Any]:
    """Backward-compatible wrapper for ``try_index_cpu_to_gpu``."""

    return try_index_cpu_to_gpu(index)


def clone_index_to_cpu(index: Any) -> Any:
    """Backward-compatible wrapper for ``ensure_cpu_index``."""

    return ensure_cpu_index(index)
