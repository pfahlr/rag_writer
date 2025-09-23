import sys
import types
from pathlib import Path

faiss_stub = types.ModuleType("faiss")
sys.modules.setdefault("faiss", faiss_stub)

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


def test_lc_build_index_respects_custom_output_dirs(monkeypatch, tmp_path):
    docs = [Document(page_content="test", metadata={})]
    monkeypatch.setattr(lc_build_index, "load_pdfs", lambda: docs)

    recorded = {}

    def fake_write(chunks, out_path):
        recorded["chunks_out"] = out_path

    def fake_splitter(*args, **kwargs):  # pragma: no cover - helper
        class _DummySplitter:
            def split_documents(self, docs):
                return docs

        return _DummySplitter()

    def fake_build(chunks, key, embedding_models, shard_size, resume, keep_shards, **_):
        recorded["index_dir"] = lc_build_index.INDEX_DIR
        recorded["key"] = key
        recorded["embedding_models"] = embedding_models
        return None

    monkeypatch.setattr(lc_build_index, "write_chunks_jsonl", fake_write)
    monkeypatch.setattr(lc_build_index, "RecursiveCharacterTextSplitter", fake_splitter)
    monkeypatch.setattr(lc_build_index, "build_faiss_for_models", fake_build)

    key = "custom/index"
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lc_build_index.py",
            key,
            "--input-dir",
            str(input_dir),
            "--chunks-dir",
            str(chunks_dir),
            "--index-dir",
            str(index_dir),
            "--shard-size",
            "10",
        ],
    )

    lc_build_index.main()

    safe_key = lc_build_index._fs_safe(key)
    assert recorded["key"] == key
    assert recorded["embedding_models"]
    assert recorded["chunks_out"] == Path(chunks_dir) / f"lc_chunks_{safe_key}.jsonl"
    assert Path(recorded["index_dir"]) == Path(index_dir)
