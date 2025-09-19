from types import SimpleNamespace

from src.core import retriever


class DummyVectorStore:
    def __init__(self) -> None:
        self.index = SimpleNamespace(name="cpu-index")


class DummyFAISS:
    @staticmethod
    def load_local(path: str, embeddings, allow_dangerous_deserialization: bool = True):
        return DummyVectorStore()


class DummyEmbeddings:
    pass


def test_retriever_promotes_index_to_gpu(monkeypatch, tmp_path):
    factory = retriever.RetrieverFactory(root_dir=tmp_path)

    monkeypatch.setattr(retriever, "FAISS", DummyFAISS)
    monkeypatch.setattr(retriever.faiss_utils, "is_faiss_gpu_available", lambda: True)

    gpu_calls = []

    def fake_clone_gpu(index):
        gpu_calls.append(index.name)
        return SimpleNamespace(name=f"gpu-{index.name}")

    monkeypatch.setattr(retriever.faiss_utils, "clone_index_to_gpu", fake_clone_gpu)
    embeddings = DummyEmbeddings()

    vectorstore = factory._load_faiss_index(tmp_path, embeddings)
    assert vectorstore.index.name == "gpu-cpu-index"
    assert gpu_calls == ["cpu-index"]


def test_retriever_stays_on_cpu_without_gpu(monkeypatch, tmp_path):
    factory = retriever.RetrieverFactory(root_dir=tmp_path)

    monkeypatch.setattr(retriever, "FAISS", DummyFAISS)
    monkeypatch.setattr(retriever.faiss_utils, "is_faiss_gpu_available", lambda: False)

    def fail_clone(index):  # pragma: no cover - should never run
        raise AssertionError("GPU clone should not execute when unavailable")

    monkeypatch.setattr(retriever.faiss_utils, "clone_index_to_gpu", fail_clone)
    embeddings = DummyEmbeddings()

    vectorstore = factory._load_faiss_index(tmp_path, embeddings)
    assert vectorstore.index.name == "cpu-index"
