import sys
from langchain_core.documents import Document
from src.langchain import lc_build_index


def test_lc_build_index_accepts_resume_value(monkeypatch):
    docs = [Document(page_content="test", metadata={})]
    monkeypatch.setattr(lc_build_index, "load_pdfs", lambda: docs)
    monkeypatch.setattr(lc_build_index, "write_chunks_jsonl", lambda chunks, out: None)

    called = {}

    def fake_build(chunks, key, embedding_models, shard_size, resume, keep_shards):
        called["resume"] = resume

    monkeypatch.setattr(lc_build_index, "build_faiss_for_models", fake_build)

    monkeypatch.setattr(sys, "argv", ["lc_build_index.py", "foo", "--resume", "1"])

    lc_build_index.main()

    assert called["resume"] is True
