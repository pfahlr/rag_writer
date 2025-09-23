from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


# Provide lightweight stand-ins for optional LangChain dependencies so lc_ask can import.
if "langchain_core.documents" not in sys.modules:
    langchain_core = ModuleType("langchain_core")
    langchain_core.documents = ModuleType("langchain_core.documents")

    class _DummyDocument:
        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    langchain_core.documents.Document = _DummyDocument
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.documents"] = langchain_core.documents

if "langchain_community" not in sys.modules:
    langchain_community = ModuleType("langchain_community")
    langchain_community.__path__ = []  # mark as package
    sys.modules["langchain_community"] = langchain_community

if "langchain_community.vectorstores" not in sys.modules:
    vectorstores = ModuleType("langchain_community.vectorstores")
    vectorstores.FAISS = SimpleNamespace()
    sys.modules["langchain_community.vectorstores"] = vectorstores

if "langchain_community.retrievers" not in sys.modules:
    retrievers = ModuleType("langchain_community.retrievers")

    class _DummyRetriever:
        @classmethod
        def from_documents(cls, docs):
            return cls()

    retrievers.BM25Retriever = _DummyRetriever
    retrievers.EnsembleRetriever = _DummyRetriever
    retrievers.ContextualCompressionRetriever = _DummyRetriever
    retrievers.ParentDocumentRetriever = _DummyRetriever
    sys.modules["langchain_community.retrievers"] = retrievers

if "langchain_community.document_transformers" not in sys.modules:
    transformers = ModuleType("langchain_community.document_transformers")
    transformers.CrossEncoderReranker = SimpleNamespace
    sys.modules["langchain_community.document_transformers"] = transformers

if "langchain_community.cross_encoders" not in sys.modules:
    cross_encoders = ModuleType("langchain_community.cross_encoders")
    cross_encoders.HuggingFaceCrossEncoder = SimpleNamespace
    sys.modules["langchain_community.cross_encoders"] = cross_encoders

if "langchain_huggingface" not in sys.modules:
    langchain_huggingface = ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = object
    sys.modules["langchain_huggingface"] = langchain_huggingface

if "langchain_openai" not in sys.modules:
    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    sys.modules["langchain_openai"] = langchain_openai

if "langchain" not in sys.modules:
    langchain = ModuleType("langchain")
    langchain.__path__ = []
    sys.modules["langchain"] = langchain

if "langchain.chains" not in sys.modules:
    chains = ModuleType("langchain.chains")
    chains.RetrievalQA = SimpleNamespace
    sys.modules["langchain.chains"] = chains

if "langchain.retrievers" not in sys.modules:
    langchain_retrievers = ModuleType("langchain.retrievers")
    langchain_retrievers.EnsembleRetriever = SimpleNamespace
    langchain_retrievers.ContextualCompressionRetriever = SimpleNamespace
    langchain_retrievers.ParentDocumentRetriever = SimpleNamespace
    sys.modules["langchain.retrievers"] = langchain_retrievers

from src.langchain import lc_ask


@pytest.mark.parametrize(
    "chunks_dir,index_dir",
    [
        (Path("/tmp/chunks"), Path("/tmp/storage")),
        (Path("relative/chunks"), Path("relative/storage")),
    ],
)
def test_resolve_paths_align_with_build_defaults(chunks_dir: Path, index_dir: Path) -> None:
    chunk_path, base_dir, repacked_dir = lc_ask._resolve_paths(
        key="demo",
        embed_model="BAAI/bge-small-en-v1.5",
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )

    assert chunk_path == Path(chunks_dir) / "lc_chunks_demo.jsonl"
    expected_base = Path(index_dir) / "faiss_demo__BAAI-bge-small-en-v1.5"
    assert base_dir == expected_base
    assert repacked_dir == expected_base.parent / f"{expected_base.name}_repacked"
