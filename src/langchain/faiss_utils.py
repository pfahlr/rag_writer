"""LangChain-facing FAISS helpers exposing core utilities."""

from src.core.faiss_utils import (
    clone_index_to_cpu,
    clone_index_to_gpu,
    ensure_cpu_index,
    is_faiss_gpu_available,
    set_faiss_threads,
    try_index_cpu_to_gpu,
)

__all__ = [
    "clone_index_to_cpu",
    "clone_index_to_gpu",
    "ensure_cpu_index",
    "is_faiss_gpu_available",
    "set_faiss_threads",
    "try_index_cpu_to_gpu",
]

