import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def lc_ask_module(monkeypatch):
    """Import lc_ask with optional LangChain dependencies stubbed as needed."""

    def ensure_stub(module_name: str, attrs: dict[str, object]) -> None:
        if module_name in sys.modules:
            return

        mod = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, module_name, mod)

    class _Document:  # pragma: no cover - stub for import-time use only
        ...

    class _StubFAISS:  # pragma: no cover - stub for import-time use only
        @staticmethod
        def load_local(*_args, **_kwargs):
            raise NotImplementedError

    class _HuggingFaceEmbeddings:  # pragma: no cover - stub for import-time use only
        def __init__(self, *_args, **_kwargs):  # noqa: D401 - stub
            raise NotImplementedError

    class _ChatOpenAI:  # pragma: no cover - stub for import-time use only
        ...

    class _RetrievalQA:  # pragma: no cover - stub for import-time use only
        ...

    ensure_stub("langchain_core.documents", {"Document": _Document})
    ensure_stub("langchain_community.vectorstores", {"FAISS": _StubFAISS})
    ensure_stub(
        "langchain_community.embeddings",
        {"HuggingFaceEmbeddings": _HuggingFaceEmbeddings},
    )
    ensure_stub("langchain_openai", {"ChatOpenAI": _ChatOpenAI})
    ensure_stub("langchain.chains", {"RetrievalQA": _RetrievalQA})

    return importlib.import_module("src.langchain.lc_ask")


class _DummyClient:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _DummyEmbedder:
    def __init__(self, dim: int) -> None:
        self.client = _DummyClient(dim)


class _DummyIndex:
    def __init__(self, dim: int) -> None:
        self.d = dim


class _DummyVectorStore:
    def __init__(self, dim: int) -> None:
        self.index = _DummyIndex(dim)


def test_validate_index_embedding_dimension_mismatch(lc_ask_module) -> None:
    embedder = _DummyEmbedder(dim=384)
    vectorstore = _DummyVectorStore(dim=768)

    with pytest.raises(SystemExit) as excinfo:
        lc_ask_module._validate_index_embedding_compatibility(
            embedder, vectorstore, Path("/tmp/index")
        )

    assert "Embedding dimension mismatch" in str(excinfo.value)


def test_validate_index_embedding_dimension_match(lc_ask_module) -> None:
    embedder = _DummyEmbedder(dim=768)
    vectorstore = _DummyVectorStore(dim=768)

    lc_ask_module._validate_index_embedding_compatibility(
        embedder, vectorstore, Path("/tmp/index")
    )
