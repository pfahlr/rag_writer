import pytest
import importlib
import json
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    
def _install_dummy_langchain_modules(monkeypatch):
    docs_mod = types.ModuleType("langchain_core.documents")

    class DummyDocument:  # pragma: no cover - trivial shim
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    docs_mod.Document = DummyDocument
    monkeypatch.setitem(sys.modules, "langchain_core.documents", docs_mod)

    vect_mod = types.ModuleType("langchain_community.vectorstores")

    class DummyFAISS:  # pragma: no cover - trivial shim
        @staticmethod
        def load_local(*args, **kwargs):
            raise AssertionError("should be patched in tests")

    vect_mod.FAISS = DummyFAISS
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", vect_mod)

    hf_mod = types.ModuleType("langchain_huggingface")

    class DummyEmb:  # pragma: no cover - trivial shim
        pass

    hf_mod.HuggingFaceEmbeddings = DummyEmb
    monkeypatch.setitem(sys.modules, "langchain_huggingface", hf_mod)

    lc_comm_emb = types.ModuleType("langchain_community.embeddings")
    lc_comm_emb.HuggingFaceEmbeddings = DummyEmb
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", lc_comm_emb)

    openai_mod = types.ModuleType("langchain_openai")

    class DummyChatOpenAI:  # pragma: no cover - trivial shim
        def __init__(self, **kwargs):
            self.model_name = "dummy"
            self.temperature = kwargs.get("temperature", 0)

    openai_mod.ChatOpenAI = DummyChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

    chains_mod = types.ModuleType("langchain.chains")

    class DummyRetrievalQA:  # pragma: no cover - trivial shim
        @classmethod
        def from_chain_type(cls, *args, **kwargs):
            raise AssertionError("should be patched in tests")

    chains_mod.RetrievalQA = DummyRetrievalQA
    monkeypatch.setitem(sys.modules, "langchain.chains", chains_mod)


def test_lc_ask_supports_question_flag(monkeypatch, tmp_path):
    _install_dummy_langchain_modules(monkeypatch)
    lc_ask = importlib.import_module("src.langchain.lc_ask")
    key = "test"
    question = "Define neuroplasticity in one sentence."

    monkeypatch.chdir(tmp_path)

    chunks_path = tmp_path / "data_processed"
    chunks_path.mkdir(exist_ok=True)
    chunks_file = chunks_path / f"lc_chunks_{key}.jsonl"
    chunks_file.write_text(json.dumps({"text": "Sample", "metadata": {}}) + "\n", encoding="utf-8")

    emb_safe = "BAAI-bge-small-en-v1.5"
    faiss_dir = (tmp_path / "storage") / f"faiss_{key}__{emb_safe}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "index.faiss").write_text("", encoding="utf-8")

    class DummyEmbeddings:
        pass

    class DummyVectorStore:
        pass

    captured = {}

    class DummyChain:
        def invoke(self, payload):
            captured["payload"] = payload
            return {"result": "ok", "source_documents": []}

    class DummyLLM:
        model_name = "dummy"
        temperature = 0

    monkeypatch.setattr(lc_ask, "HuggingFaceEmbeddings", lambda model_name: DummyEmbeddings())
    monkeypatch.setattr(lc_ask.FAISS, "load_local", lambda *args, **kwargs: DummyVectorStore())
    monkeypatch.setattr(lc_ask, "make_retriever", lambda **kwargs: object())
    monkeypatch.setattr(
        lc_ask,
        "RetrievalQA",
        SimpleNamespace(from_chain_type=lambda *args, **kwargs: DummyChain()),
    )
    monkeypatch.setattr(lc_ask, "ChatOpenAI", lambda **kwargs: DummyLLM())

    monkeypatch.setenv("TRACE_QID", "test-qid")

    monkeypatch.setattr(
        lc_ask.sys,
        "argv",
        [
            "lc_ask.py",
            "--key",
            key,
            "--question",
            question,
            "--chunks-dir",
            str(chunks_path),
            "--index-dir",
            str(tmp_path / "storage"),
        ],
    )

    lc_ask.main()

    assert captured["payload"]["query"] == question


def test_lc_ask_falls_back_to_vectorstore_docs_for_faiss(monkeypatch, tmp_path):
    _install_dummy_langchain_modules(monkeypatch)
    lc_ask = importlib.import_module("src.langchain.lc_ask")

    key = "missing-chunks"
    question = "How do neurons adapt?"

    monkeypatch.chdir(tmp_path)

    chunks_dir = tmp_path / "data_processed"
    chunks_dir.mkdir(exist_ok=True)

    emb_safe = "BAAI-bge-small-en-v1.5"
    faiss_dir = (tmp_path / "storage") / f"faiss_{key}__{emb_safe}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "index.faiss").write_text("", encoding="utf-8")

    class DummyEmbeddings:
        pass

    class DummyVectorStore:
        pass

    docs_seen: dict[str, list] = {}

    monkeypatch.setattr(lc_ask, "HuggingFaceEmbeddings", lambda model_name: DummyEmbeddings())
    monkeypatch.setattr(lc_ask.FAISS, "load_local", lambda *args, **kwargs: DummyVectorStore())
    monkeypatch.setattr(
        lc_ask,
        "_extract_docs_from_vectorstore",
        lambda vectorstore: [lc_ask.Document(page_content="vector", metadata={})],
    )

    def fake_make_retriever(**kwargs):
        docs_seen["docs"] = kwargs.get("docs")
        return object()

    class DummyChain:
        def invoke(self, payload):
            return {"result": "ok", "source_documents": []}

    class DummyLLM:
        model_name = "dummy"
        temperature = 0

    monkeypatch.setattr(lc_ask, "make_retriever", fake_make_retriever)
    monkeypatch.setattr(
        lc_ask,
        "RetrievalQA",
        SimpleNamespace(from_chain_type=lambda *args, **kwargs: DummyChain()),
    )
    monkeypatch.setattr(lc_ask, "ChatOpenAI", lambda **kwargs: DummyLLM())

    monkeypatch.setenv("TRACE_QID", "test-qid")

    monkeypatch.setattr(
        lc_ask.sys,
        "argv",
        [
            "lc_ask.py",
            "--key",
            key,
            "--question",
            question,
            "--chunks-dir",
            str(chunks_dir),
            "--index-dir",
            str(tmp_path / "storage"),
            "--mode",
            "faiss",
        ],
    )

    lc_ask.main()

    assert len(docs_seen["docs"]) == 1
    assert docs_seen["docs"][0].page_content == "vector"
    assert getattr(docs_seen["docs"][0], "metadata", {}) == {}


def test_lc_ask_accepts_custom_directories(monkeypatch, tmp_path):
    _install_dummy_langchain_modules(monkeypatch)
    lc_ask = importlib.import_module("src.langchain.lc_ask")

    key = "custom/index"
    question = "What is neuroplasticity?"

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    emb_safe = "BAAI-bge-small-en-v1.5"
    safe_key = "custom-index"
    chunk_file = chunks_dir / f"lc_chunks_{safe_key}.jsonl"
    chunk_file.write_text(
        json.dumps({"text": "Sample", "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    faiss_dir = index_dir / f"faiss_{safe_key}__{emb_safe}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "index.faiss").write_text("", encoding="utf-8")

    class DummyEmbeddings:
        pass

    class DummyVectorStore:
        pass

    chunk_call = {}
    faiss_call = {}

    monkeypatch.setattr(lc_ask, "HuggingFaceEmbeddings", lambda model_name: DummyEmbeddings())

    def fake_load_chunks(path):
        chunk_call["path"] = Path(path)
        return [
            lc_ask.Document(page_content="Sample", metadata={})
        ]

    def fake_load_local(path, *args, **kwargs):
        faiss_call["path"] = Path(path)
        return DummyVectorStore()

    monkeypatch.setattr(lc_ask, "_load_chunks_jsonl", fake_load_chunks)
    monkeypatch.setattr(lc_ask.FAISS, "load_local", fake_load_local)
    monkeypatch.setattr(lc_ask, "make_retriever", lambda **kwargs: object())

    class DummyChain:
        def invoke(self, payload):
            return {"result": "ok", "source_documents": []}

    class DummyLLM:
        model_name = "dummy"
        temperature = 0

    monkeypatch.setattr(
        lc_ask,
        "RetrievalQA",
        SimpleNamespace(from_chain_type=lambda *args, **kwargs: DummyChain()),
    )
    monkeypatch.setattr(lc_ask, "ChatOpenAI", lambda **kwargs: DummyLLM())

    monkeypatch.setenv("TRACE_QID", "test-qid")
    monkeypatch.setattr(
        lc_ask.sys,
        "argv",
        [
            "lc_ask.py",
            "--key",
            key,
            "--question",
            question,
            "--chunks-dir",
            str(chunks_dir),
            "--index-dir",
            str(index_dir),
        ],
    )

    lc_ask.main()

    assert chunk_call["path"] == chunk_file
    assert faiss_call["path"] == faiss_dir


def test_lc_ask_accepts_explicit_index_directory(monkeypatch, tmp_path):
    _install_dummy_langchain_modules(monkeypatch)
    lc_ask = importlib.import_module("src.langchain.lc_ask")

    question = "What is neuroplasticity?"
    embed_model = "BAAI/bge-small-en-v1.5"
    embed_safe = "BAAI-bge-small-en-v1.5"
    safe_key = "papers"

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    index_dir = tmp_path / "index"
    faiss_dir = index_dir / f"faiss_{safe_key}__{embed_safe}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "index.faiss").write_text("", encoding="utf-8")
    chunk_file = faiss_dir / f"lc_chunks_{safe_key}.jsonl"
    chunk_file.write_text(
        json.dumps({"text": "Sample", "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    class DummyEmbeddings:
        pass

    class DummyVectorStore:
        pass

    chunk_call: dict[str, Path] = {}
    faiss_call: dict[str, Path] = {}

    monkeypatch.setattr(
        lc_ask,
        "HuggingFaceEmbeddings",
        lambda model_name: DummyEmbeddings(),
    )

    def fake_load_chunks(path):
        chunk_call["path"] = Path(path)
        return [lc_ask.Document(page_content="Sample", metadata={})]

    def fake_load_local(path, *args, **kwargs):
        faiss_call["path"] = Path(path)
        return DummyVectorStore()

    monkeypatch.setattr(lc_ask, "_load_chunks_jsonl", fake_load_chunks)
    monkeypatch.setattr(lc_ask.FAISS, "load_local", fake_load_local)
    monkeypatch.setattr(lc_ask, "make_retriever", lambda **kwargs: object())

    class DummyChain:
        def invoke(self, payload):
            return {"result": "ok", "source_documents": []}

    class DummyLLM:
        model_name = "dummy"
        temperature = 0

    monkeypatch.setattr(
        lc_ask,
        "RetrievalQA",
        SimpleNamespace(from_chain_type=lambda *args, **kwargs: DummyChain()),
    )
    monkeypatch.setattr(lc_ask, "ChatOpenAI", lambda **kwargs: DummyLLM())

    monkeypatch.setenv("TRACE_QID", "test-qid")
    monkeypatch.setattr(
        lc_ask.sys,
        "argv",
        [
            "lc_ask.py",
            "--index",
            str(faiss_dir),
            "--question",
            question,
            "--chunks-dir",
            str(chunks_dir),
            "--embed-model",
            embed_model,
        ],
    )

    lc_ask.main()

    assert chunk_call["path"] == chunk_file
    assert faiss_call["path"] == faiss_dir


