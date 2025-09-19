# tests/test_faiss_merge_gpu_safe.py
import os
import math
import shutil
import random
import string
import faiss
import pytest

# LangChain vectorstore
from langchain_community.vectorstores import FAISS

# Try to import your helpers; adjust import path if needed
# e.g., from src.langchain.faiss_utils import ensure_cpu_index, try_index_cpu_to_gpu, is_faiss_gpu_available
try:
    from src.langchain.faiss_utils import (
        ensure_cpu_index,
        try_index_cpu_to_gpu,
        is_faiss_gpu_available,
    )
except Exception:
    # Fallback shims so test still runs; remove once your module is in place.
    def is_faiss_gpu_available() -> bool:
        try:
            return faiss.get_num_gpus() > 0
        except Exception:
            return False

    def ensure_cpu_index(index):
        try:
            if hasattr(faiss, "GpuIndex") and isinstance(index, faiss.GpuIndex):
                return faiss.index_gpu_to_cpu(index)
        except Exception:
            pass
        return index

    def try_index_cpu_to_gpu(index, device_id: int = 0):
        try:
            if not is_faiss_gpu_available():
                return None
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, device_id, index)
        except Exception:
            return None


# ---- Dummy, deterministic embedder (no network, no GPU needed) ----
class DummyEmbeddings:
    """Deterministic, fast, no-network embeddings for tests."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, text: str):
        # Simple stable hash → vector
        rnd = random.Random(hash(text) & 0xFFFFFFFF)
        v = [rnd.uniform(-1, 1) for _ in range(self.dim)]
        # L2 normalize to mimic real models
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def rand_text(n=8):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


@pytest.fixture(scope="module")
def corpus():
    # small synthetic corpus
    random.seed(1337)
    texts = [f"{i} " + rand_text(10) for i in range(120)]
    metas = [{"id": i} for i in range(len(texts))]
    return texts, metas


def test_faiss_threads_respected(monkeypatch):
    # Pretend we want 6 threads; verify FAISS reports it
    faiss.omp_set_num_threads(6)
    assert faiss.omp_get_max_threads() == 6


def test_cpu_only_merge_and_portable_save(tmp_path, corpus):
    texts, metas = corpus
    emb = DummyEmbeddings(dim=24)

    # Build two shards (CPU)
    shard_size = 70
    shard1_texts, shard1_metas = texts[:shard_size], metas[:shard_size]
    shard2_texts, shard2_metas = texts[shard_size:], metas[shard_size:]

    vs1 = FAISS.from_texts(shard1_texts, emb, metadatas=shard1_metas)
    vs2 = FAISS.from_texts(shard2_texts, emb, metadatas=shard2_metas)

    # Ensure CPU indexes (safety)
    vs1.index = ensure_cpu_index(vs1.index)
    vs2.index = ensure_cpu_index(vs2.index)

    # Merge (CPU → CPU)
    vs1.merge_from(vs2)

    # Total docs accounted for
    # LangChain stores docstore separately; check doc count via docstore._dict
    total_docs = len(vs1.docstore._dict)  # internal but stable in tests
    assert total_docs == len(texts)

    # Save to disk (CPU format)
    out_dir = tmp_path / "faiss_out"
    vs1.save_local(str(out_dir))

    # Reload on "different" environment (still CPU in test)
    vs_loaded = FAISS.load_local(str(out_dir), embeddings=emb, allow_dangerous_deserialization=True)

    # Quick semantic sanity: search should return something
    q = texts[5].split()[1]  # a token
    results = vs_loaded.similarity_search(q, k=3)
    assert len(results) == 3
    # Returned docs have metadata with 'id'
    assert all("id" in r.metadata for r in results)


@pytest.mark.parametrize("device_id", [0])
def test_optional_gpu_move_after_save(tmp_path, corpus, device_id):
    texts, metas = corpus
    emb = DummyEmbeddings(dim=16)

    vs = FAISS.from_texts(texts, emb, metadatas=metas)
    vs.index = ensure_cpu_index(vs.index)

    # Save CPU index
    out_dir = tmp_path / "faiss_out_gpu"
    vs.save_local(str(out_dir))

    # Try moving to GPU (if available). Should not affect the saved files.
    gpu_idx = try_index_cpu_to_gpu(vs.index, device_id=device_id)
    if is_faiss_gpu_available() and gpu_idx is not None:
        # We successfully got a GPU index in memory
        assert hasattr(faiss, "GpuIndex") and isinstance(gpu_idx, faiss.GpuIndex)
        # But on disk, still only CPU artifacts exist
        assert (out_dir / "index.faiss").exists()
        # Ensure we can still load it as CPU later
        vs_cpu = FAISS.load_local(str(out_dir), embeddings=emb, allow_dangerous_deserialization=True)
        assert not (hasattr(faiss, "GpuIndex") and isinstance(vs_cpu.index, faiss.GpuIndex))
    else:
        # No GPU available or faiss without GPU support: move should be a no-op
        assert gpu_idx is None


def test_shard_pipeline_like_yours(tmp_path, corpus):
    """Emulate your shard → save → load → merge → save flow (CPU only)."""
    texts, metas = corpus
    emb = DummyEmbeddings(dim=20)

    base = tmp_path / "storage" / "faiss_textiles__dummy"
    shards_dir = base / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_size = 40
    shard_paths = []
    for i, start in enumerate(range(0, len(texts), shard_size)):
        sp = shards_dir / f"shard_{i:03d}"
        sp.mkdir(parents=True, exist_ok=True)
        t = texts[start : start + shard_size]
        m = metas[start : start + shard_size]
        vs = FAISS.from_texts(t, emb, metadatas=m)
        vs.index = ensure_cpu_index(vs.index)  # enforce CPU for shard save
        vs.save_local(str(sp))
        shard_paths.append(sp)

    # Merge shards (CPU)
    accumulator = None
    for sp in shard_paths:
        vs = FAISS.load_local(str(sp), embeddings=emb, allow_dangerous_deserialization=True)
        vs.index = ensure_cpu_index(vs.index)
        if accumulator is None:
            accumulator = vs
        else:
            accumulator.index = ensure_cpu_index(accumulator.index)
            accumulator.merge_from(vs)

    # Save final (CPU)
    base.mkdir(parents=True, exist_ok=True)
    accumulator.index = ensure_cpu_index(accumulator.index)
    accumulator.save_local(str(base))

    # Verify final load works and has all docs
    vs_final = FAISS.load_local(str(base), embeddings=emb, allow_dangerous_deserialization=True)
    total_docs = len(vs_final.docstore._dict)
    assert total_docs == len(texts)

    # Clean up shards like your code path (optional)
    shutil.rmtree(shards_dir, ignore_errors=True)
