import sys
import types
from pathlib import Path

import pytest

# Provide lightweight stubs for optional LangChain dependencies so lc_ask can import
if "langchain_core.documents" not in sys.modules:
    mod = types.ModuleType("langchain_core.documents")

    class Document:  # pragma: no cover - stub for import-time use only
        ...

    mod.Document = Document
    sys.modules["langchain_core.documents"] = mod

if "langchain_community.vectorstores" not in sys.modules:
    mod = types.ModuleType("langchain_community.vectorstores")

    class _StubFAISS:  # pragma: no cover - stub for import-time use only
        @staticmethod
        def load_local(*_args, **_kwargs):
            raise NotImplementedError

    mod.FAISS = _StubFAISS
    sys.modules["langchain_community.vectorstores"] = mod

if "langchain_community.embeddings" not in sys.modules:
    mod = types.ModuleType("langchain_community.embeddings")

    class HuggingFaceEmbeddings:  # pragma: no cover - stub for import-time use only
        def __init__(self, *args, **kwargs):  # noqa: D401 - stub
            raise NotImplementedError

    mod.HuggingFaceEmbeddings = HuggingFaceEmbeddings
    sys.modules["langchain_community.embeddings"] = mod

if "langchain_openai" not in sys.modules:
    mod = types.ModuleType("langchain_openai")

    class ChatOpenAI:  # pragma: no cover - stub for import-time use only
        ...

    mod.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = mod

if "langchain.chains" not in sys.modules:
    mod = types.ModuleType("langchain.chains")

    class RetrievalQA:  # pragma: no cover - stub for import-time use only
        ...

    mod.RetrievalQA = RetrievalQA
    sys.modules["langchain.chains"] = mod

from src.langchain import lc_ask


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


def test_validate_index_embedding_dimension_mismatch() -> None:
    embedder = _DummyEmbedder(dim=384)
    vectorstore = _DummyVectorStore(dim=768)

    with pytest.raises(SystemExit) as excinfo:
        lc_ask._validate_index_embedding_compatibility(
            embedder, vectorstore, Path("/tmp/index")
        )

    assert "Embedding dimension mismatch" in str(excinfo.value)


def test_validate_index_embedding_dimension_match() -> None:
    embedder = _DummyEmbedder(dim=768)
    vectorstore = _DummyVectorStore(dim=768)

    lc_ask._validate_index_embedding_compatibility(
        embedder, vectorstore, Path("/tmp/index")
    )
