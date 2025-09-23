"""Regression tests for the RAG verification harness helpers."""

from __future__ import annotations

from pathlib import Path

import run_rag_verification as harness


def test_convert_markdown_to_pdfs(tmp_path: Path) -> None:
    """Markdown corpora should be rendered to PDFs in a target directory."""

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "note-one.md").write_text("# Heading\n\nBody text", encoding="utf-8")
    (corpus_dir / "note-two.md").write_text("Just some text", encoding="utf-8")

    pdf_dir = tmp_path / "pdfs"
    pdf_paths = harness.convert_markdown_to_pdfs(corpus_dir, pdf_dir)

    assert sorted(p.name for p in pdf_paths) == ["note-one.pdf", "note-two.pdf"]
    for pdf_path in pdf_paths:
        data = pdf_path.read_bytes()
        assert data.startswith(b"%PDF")


def test_build_faiss_index_uses_expected_cli_flags(monkeypatch, tmp_path: Path) -> None:
    """Ensure the FAISS builder is invoked with supported CLI options."""

    invoked: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
        invoked["cmd"] = cmd
        invoked["env"] = env or {}

    monkeypatch.setattr(harness, "_run_subprocess", fake_run)

    pdf_dir = tmp_path / "pdfs"
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "storage"

    harness.build_faiss_index(
        key="neuroplasticity",
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        resume=False,
        keep_shards=False,
    )

    cmd = invoked["cmd"]
    assert cmd[0].endswith("python") or cmd[0].endswith("python3") or cmd[0].endswith("pytest")
    assert cmd[1:] == [
        "src/langchain/lc_build_index.py",
        "neuroplasticity",
        "--input-dir",
        str(pdf_dir),
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
    ]


def test_query_clis_match_expected_interfaces(monkeypatch, tmp_path: Path) -> None:
    """The harness must call lc_ask and multi_agent with real CLI signatures."""

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
        calls.append(cmd)

    monkeypatch.setattr(harness, "_run_subprocess", fake_run)

    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "storage"

    harness.run_lc_ask(
        question="What is neuroplasticity?",
        key="neuroplasticity",
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embed_model="BAAI/bge-small-en-v1.5",
    )
    harness.run_multi_agent(
        question="Summarize the findings",
        key="neuroplasticity",
    )

    ask_cmd, agent_cmd = calls

    assert ask_cmd[1:] == [
        "src/langchain/lc_ask.py",
        "What is neuroplasticity?",
        "--key",
        "neuroplasticity",
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
        "--embed-model",
        "BAAI/bge-small-en-v1.5",
    ]

    assert agent_cmd[1:] == [
        "src/cli/multi_agent.py",
        "ask",
        "Summarize the findings",
        "--key",
        "neuroplasticity",
    ]
