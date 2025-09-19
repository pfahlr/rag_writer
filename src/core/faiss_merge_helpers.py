from __future__ import annotations
import faiss

# Use your existing helper
from src.core.faiss_utils import ensure_cpu_index  # type: ignore


def _metric_of(index: faiss.Index) -> int:
    """Return faiss.METRIC_* for the given index, defaulting sensibly."""
    try:
        return int(getattr(index, "metric_type"))
    except Exception:
        # Heuristics for older types
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

    # Prefer explicit classes when possible
    if mt == faiss.METRIC_INNER_PRODUCT:
        # Some builds expose IndexFlatIP; otherwise IndexFlat(d, METRIC_*)
        try:
            return faiss.IndexFlatIP(d)
        except Exception:
            return faiss.IndexFlat(d, mt)
    else:  # L2 (default)
        try:
            return faiss.IndexFlatL2(d)
        except Exception:
            return faiss.IndexFlat(d, mt)


def merge_faiss_vectorstores_cpu(acc_vs, shard_vs):
    """
    Robust, CPU-only merge for LangChain FAISS vectorstores that does NOT rely on
    raw index .merge_from (often missing in Python for IndexFlat*).

    - Works for IndexFlat / IndexFlatIP / IndexFlatL2 shards.
    - Preserves docstore and index_to_docstore_id without re-embedding.
    - Uses faiss.IndexShards + faiss.merge_index_shards for the raw FAISS merge.
    """
    # 1) Ensure CPU indexes
    acc_vs.index = ensure_cpu_index(acc_vs.index)
    shard_vs.index = ensure_cpu_index(shard_vs.index)

    # 2) Merge the raw FAISS indexes on CPU using IndexShards
    merged = _new_flat_like(acc_vs.index)
    d = int(acc_vs.index.d)
    shards = faiss.IndexShards(d, threaded=True, successive_ids=True)
    shards.add_shard(acc_vs.index)
    shards.add_shard(shard_vs.index)
    faiss.merge_index_shards(merged, shards)

    # 3) Swap merged index back and stitch LangChain bookkeeping
    acc_vs.index = merged
    # Merge docstores + mappings (LangChain internals)
    acc_vs.docstore._dict.update(shard_vs.docstore._dict)
    acc_vs.index_to_docstore_id.extend(shard_vs.index_to_docstore_id)

    return acc_vs

