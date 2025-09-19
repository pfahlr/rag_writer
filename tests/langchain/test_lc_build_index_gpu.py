from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.langchain import lc_build_index


class DummyIndex:
    def __init__(self, name: str):
        self.name = name
        self.merged = []

    def merge_from(self, other: "DummyIndex") -> None:
        self.merged.append(other.name)


class DummyVectorStore:
    saved_paths: list[Path] = []
    shard_counter = 0

    def __init__(self, name: str):
        self.name = name
        self.index = DummyIndex(name)

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Any, metadatas: list[dict]) -> "DummyVectorStore":
        inst = cls(f"shard-{cls.shard_counter}")
        cls.shard_counter += 1
        return inst

    def save_local(self, folder_path: str) -> None:
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(path)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Any,
        allow_dangerous_deserialization: bool = True,
    ) -> "DummyVectorStore":
        # Each shard is reloaded with a new instance whose name is derived from the folder
        name = Path(folder_path).name
        return cls(name)

    def merge_from(self, other: "DummyVectorStore") -> None:
        self.index.merge_from(other.index)


class DummyEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def _make_docs(count: int) -> list[Document]:
    return [Document(page_content=f"doc {i}", metadata={"i": i}) for i in range(count)]


def test_build_promotes_indexes_to_gpu(monkeypatch, tmp_path):
    docs = _make_docs(4)

    gpu_indices = []
    cpu_indices = []
    DummyVectorStore.saved_paths.clear()
    DummyVectorStore.shard_counter = 0

    monkeypatch.setattr(lc_build_index, "HuggingFaceEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(lc_build_index, "FAISS", DummyVectorStore)
    monkeypatch.setattr(lc_build_index.faiss_utils, "is_faiss_gpu_available", lambda: True)

    def fake_clone_gpu(index: DummyIndex) -> DummyIndex:
        gpu_index = DummyIndex(f"gpu-{index.name}")
        gpu_indices.append(gpu_index.name)
        return gpu_index

    def fake_clone_cpu(index: DummyIndex) -> DummyIndex:
        cpu_indices.append(index.name)
        return DummyIndex(f"cpu-{index.name}")

    monkeypatch.setattr(lc_build_index.faiss_utils, "clone_index_to_gpu", fake_clone_gpu)
    monkeypatch.setattr(lc_build_index.faiss_utils, "clone_index_to_cpu", fake_clone_cpu)
    monkeypatch.chdir(tmp_path)

    lc_build_index.build_faiss_for_models(
        docs,
        key="demo",
        embedding_models=["model"],
        shard_size=2,
        resume=False,
        keep_shards=True,
    )

    # Two shards are merged, so GPU promotion should have been attempted at least twice
    assert len(gpu_indices) >= 2
    # Before saving, the GPU index should be converted back to CPU once
    assert cpu_indices and all(name.startswith("gpu-") for name in cpu_indices)


def test_build_skips_gpu_when_unavailable(monkeypatch, tmp_path):
    docs = _make_docs(2)

    DummyVectorStore.saved_paths.clear()
    DummyVectorStore.shard_counter = 0
    monkeypatch.setattr(lc_build_index, "HuggingFaceEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(lc_build_index, "FAISS", DummyVectorStore)
    monkeypatch.setattr(lc_build_index.faiss_utils, "is_faiss_gpu_available", lambda: False)

    def fail_clone(_: DummyIndex) -> DummyIndex:  # pragma: no cover - should not be hit
        raise AssertionError("GPU clone should not be attempted when unavailable")

    monkeypatch.setattr(lc_build_index.faiss_utils, "clone_index_to_gpu", fail_clone)
    monkeypatch.setattr(lc_build_index.faiss_utils, "clone_index_to_cpu", fail_clone)
    monkeypatch.chdir(tmp_path)

    lc_build_index.build_faiss_for_models(
        docs,
        key="demo",
        embedding_models=["model"],
        shard_size=2,
        resume=False,
        keep_shards=True,
    )
