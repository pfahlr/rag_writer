from pathlib import Path

import pytest

pytest.importorskip(
    "langchain_core.documents",
    reason="LangChain core document classes are required for lc_ask import",
)
pytest.importorskip(
    "langchain_community.vectorstores",
    reason="LangChain community vectorstores are required for lc_ask import",
)
pytest.importorskip(
    "langchain_community.embeddings",
    reason="LangChain community embeddings are required for lc_ask import",
)
pytest.importorskip(
    "langchain_openai", reason="LangChain OpenAI client is required for lc_ask import"
)
pytest.importorskip(
    "langchain.chains", reason="LangChain retrieval chains are required for lc_ask import"
)

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
