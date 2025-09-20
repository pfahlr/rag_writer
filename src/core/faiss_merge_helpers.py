from __future__ import annotations
import numpy as np
import faiss

from src.core.faiss_utils import ensure_cpu_index  # type: ignore


# ----------------------------- FAISS helpers -----------------------------

def _metric_of(index: faiss.Index) -> int:
    try:
        return int(getattr(index, "metric_type"))
    except Exception:
        if isinstance(index, faiss.IndexFlatIP):
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2


def _new_flat_like(acc_index: faiss.Index) -> faiss.Index:
    """
    Create an empty FLAT index with same dim + metric as acc_index.
    Supports IndexFlat, IndexFlatIP, IndexFlatL2.
    """
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
    d = int(idx.d)
    if nt == 0:
        return np.empty((0, d), dtype="float32")

    xb = getattr(idx, "xb", None)
    if xb is not None:
        arr = faiss.vector_to_array(xb)  # 1-D float32
        return arr.reshape(nt, d)

    rec_n = getattr(idx, "reconstruct_n", None)
    if callable(rec_n):
        vecs = rec_n(0, nt)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)
        return vecs

    rec1 = getattr(idx, "reconstruct", None)
    if callable(rec1):
        out = np.empty((nt, d), dtype="float32")
        for i in range(nt):
            out[i] = rec1(i)
        return out

    raise TypeError(f"Cannot extract vectors from index type {type(idx)}")


def _is_flat(idx: faiss.Index) -> bool:
    return isinstance(idx, (faiss.IndexFlat, faiss.IndexFlatIP, faiss.IndexFlatL2))


# ----------------------- LangChain mapping merge ------------------------

def _idmap_len(idmap) -> int:
    """Length of index_to_docstore_id whether list or dict."""
    if isinstance(idmap, list):
        return len(idmap)
    if isinstance(idmap, dict):
        return len(idmap)  # LC uses contiguous 0..n-1 keys
    raise TypeError(f"Unexpected idmap type: {type(idmap)}")


def _idmap_as_list(idmap) -> list[str]:
    """Return idmap (list or dict) as an ordered list by index key."""
    if isinstance(idmap, list):
        return idmap
    if isinstance(idmap, dict):
        # keys are 0..n-1; sort to be explicit
        return [id_ for _, id_ in sorted(idmap.items())]
    raise TypeError(f"Unexpected idmap type: {type(idmap)}")


def _merge_idmaps_inplace(acc_idmap, shard_idmap):
    """
    Merge shard_idmap into acc_idmap with the correct offset, handling all
    combinations of list/dict containers without reindexing errors.
    """
    acc_len = _idmap_len(acc_idmap)

    # list <- list
    if isinstance(acc_idmap, list) and isinstance(shard_idmap, list):
        acc_idmap.extend(shard_idmap)
        return

    # list <- dict
    if isinstance(acc_idmap, list) and isinstance(shard_idmap, dict):
        shard_list = _idmap_as_list(shard_idmap)
        acc_idmap.extend(shard_list)
        return

    # dict <- list
    if isinstance(acc_idmap, dict) and isinstance(shard_idmap, list):
        for i, doc_id in enumerate(shard_idmap):
            acc_idmap[acc_len + i] = doc_id
        return

    # dict <- dict
    if isinstance(acc_idmap, dict) and isinstance(shard_idmap, dict):
        for i, doc_id in shard_idmap.items():
            acc_idmap[acc_len + int(i)] = doc_id
        return

    raise TypeError(f"Unexpected idmap types: {type(acc_idmap)} <- {type(shard_idmap)}")


# ----------------------------- Public API -------------------------------

def merge_faiss_vectorstores_cpu(acc_vs, shard_vs):
    """
    Robust, CPU-only merge for LangChain FAISS vectorstores that:
      - Assumes *FLAT* indexes (IndexFlat / IndexFlatIP / IndexFlatL2).
      - Avoids IndexShards and raw index.merge_from for maximum compatibility.
      - Preserves docstore & index_to_docstore_id across LC versions (list/dict).
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
    # 1) docstore dicts
    acc_vs.docstore._dict.update(shard_vs.docstore._dict)

    # 2) index_to_docstore_id (list or dict depending on LC version)
    _merge_idmaps_inplace(acc_vs.index_to_docstore_id, shard_vs.index_to_docstore_id)

    return acc_vs
