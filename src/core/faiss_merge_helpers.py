from __future__ import annotations
import numpy as np
import faiss

from src.core.faiss_utils import ensure_cpu_index  # type: ignore


def _metric_of(index: faiss.Index) -> int:
    try:
        return int(getattr(index, "metric_type"))
    except Exception:
        if isinstance(index, faiss.IndexFlatIP):
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2


def _new_flat_like(acc_index: faiss.Index) -> faiss.Index:
    d = int(acc_index.d)
    mt = _metric_of(acc_index)
    if mt == faiss.METRIC_INNER_PRODUCT:
        try:
            return faiss.IndexFlatIP(d)
        except Exception:
            return faiss.IndexFlat(d, mt)
    else:
        try:
            return faiss.IndexFlatL2(d)
        except Exception:
            return faiss.IndexFlat(d, mt)


def _extract_flat_vectors(idx: faiss.Index) -> np.ndarray:
    """
    Return all vectors from a *flat* index as float32 [ntotal, d].
    Handles multiple FAISS Python variants.
    """
    nt = int(idx.ntotal)
    if nt == 0:
        return np.empty((0, int(idx.d)), dtype="float32")

    # Fast path: IndexFlat exposes a contiguous xb buffer
    xb = getattr(idx, "xb", None)
    if xb is not None:
        arr = faiss.vector_to_array(xb)  # 1-D float32
        return arr.reshape(nt, int(idx.d))

    # Next best: reconstruct_n (newer FAISS)
    rec_n = getattr(idx, "reconstruct_n", None)
    if callable(rec_n):
        vecs = rec_n(0, nt)  # returns np.ndarray [nt, d]
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)
        return vecs

    # Fallback: per-id reconstruct (slow but safe)
    rec1 = getattr(idx, "reconstruct", None)
    if callable(rec1):
        d = int(idx.d)
        out = np.empty((nt, d), dtype="float32")
        for i in range(nt):
            out[i] = rec1(i)
        return out

    raise TypeError(f"Cannot extract vectors from index type {type(idx)}")


def _is_flat(idx: faiss.Index) -> bool:
    return isinstance(idx, (faiss.IndexFlat, faiss.IndexFlatIP, faiss.IndexFlatL2))


def merge_faiss_vectorstores_cpu(acc_vs, shard_vs):
    """
    Robust, CPU-only merge for LangChain FAISS vectorstores that:
      - Assumes *FLAT* indexes (IndexFlat / IndexFlatIP / IndexFlatL2).
      - Avoids IndexShards and raw index.merge_from for maximum compatibility.
      - Preserves docstore & index_to_docstore_id.
    """
    # Ensure CPU indices
    acc_vs.index = ensure_cpu_index(acc_vs.index)
    shard_vs.index = ensure_cpu_index(shard_vs.index)

    if not (_is_flat(acc_vs.index) and _is_flat(shard_vs.index)):
        raise TypeError(
            f"Only flat indexes are supported by this merge helper; got "
            f"{type(acc_vs.index)} and {type(shard_vs.index)}"
        )

    # Build an empty flat index with the same metric/dim as accumulator
    merged = _new_flat_like(acc_vs.index)

    # Extract vectors and add
    acc_vecs = _extract_flat_vectors(acc_vs.index)
    if acc_vecs.size:
        merged.add(acc_vecs)

    shard_vecs = _extract_flat_vectors(shard_vs.index)
    if shard_vecs.size:
        merged.add(shard_vecs)

    # Swap merged FAISS index into accumulator
    acc_vs.index = merged

    # Merge LangChain bookkeeping
    acc_vs.docstore._dict.update(shard_vs.docstore._dict)
    acc_vs.index_to_docstore_id.extend(shard_vs.index_to_docstore_id)

    return acc_vs
