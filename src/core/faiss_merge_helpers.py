from __future__ import annotations

import faiss

# We depend on your existing helpers in src/core/faiss_utils.py.
# If you prefer to inline these later, that's fine — this keeps changes minimal.
from src.core.faiss_utils import ensure_cpu_index  # type: ignore


def _new_flat_like(acc_index: faiss.Index) -> faiss.Index:
    """Create an empty flat index (IP or L2) matching accumulator metric."""
    d = acc_index.d
    if isinstance(acc_index, faiss.IndexFlatIP):
        return faiss.IndexFlatIP(d)
    if isinstance(acc_index, faiss.IndexFlatL2):
        return faiss.IndexFlatL2(d)
    # Fallback: try to clone an empty index of same type (rare path).
    # Many non-flat indexes won’t support this; we explicitly target Flat here.
    try:
        # This creates an empty index with the same parameters when supported.
        return faiss.clone_index(acc_index, faiss.IO_FLAG_MMAP)
    except Exception as e:
        raise TypeError(f"Unsupported index type for robust merge: {type(acc_index)}") from e


def merge_faiss_vectorstores_cpu(acc_vs, shard_vs):
    """
    Robust, CPU-only merge for LangChain FAISS vectorstores that does NOT rely on
    raw index .merge_from (often missing in Python for IndexFlat*).

    - Works for IndexFlatIP/L2 shards (most common for LangChain FAISS).
    - Preserves docstore and index_to_docstore_id without re-embedding.
    - Uses faiss.IndexShards + faiss.merge_index_shards for the raw FAISS merge.

    Parameters
    ----------
    acc_vs : langchain_community.vectorstores.FAISS
        Accumulator vectorstore (will be modified in-place and returned).
    shard_vs : langchain_community.vectorstores.FAISS
        Shard vectorstore to merge into accumulator.

    Returns
    -------
    langchain_community.vectorstores.FAISS
        The accumulator after merge (same object, mutated).
    """
    # 1) Ensure CPU indexes
    acc_vs.index = ensure_cpu_index(acc_vs.index)
    shard_vs.index = ensure_cpu_index(shard_vs.index)

    # 2) Merge the raw FAISS indexes on CPU using IndexShards
    merged = _new_flat_like(acc_vs.index)
    d = acc_vs.index.d
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

