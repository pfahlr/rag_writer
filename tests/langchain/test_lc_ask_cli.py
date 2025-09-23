import importlib
import json
import sys
import types
from pathlib import Path


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

    chunks_path = Path("data_processed")
    chunks_path.mkdir(exist_ok=True)
    chunks_file = chunks_path / f"lc_chunks_{key}.jsonl"
    chunks_file.write_text(json.dumps({"text": "Sample", "metadata": {}}) + "\n", encoding="utf-8")

    emb_safe = "BAAI-bge-small-en-v1.5"
    faiss_dir = Path("storage") / f"faiss_{key}__{emb_safe}"
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
        lc_ask.RetrievalQA,
        "from_chain_type",
        lambda *args, **kwargs: DummyChain(),
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
        ],
    )

    lc_ask.main()

    assert captured["payload"]["query"] == question
