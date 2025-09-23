"""Tests for helper utilities in ``run_rag_verification``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from run_rag_verification import (
    Question,
    build_builder_command,
    build_question_invocation,
    prepare_pdf_corpus,
)


def test_prepare_pdf_corpus_creates_pdfs(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "md"
    markdown_dir.mkdir()
    (markdown_dir / "sample.md").write_text("hello world\nsecond line", encoding="utf-8")

    output_dir = tmp_path / "pdf"
    pdf_paths = prepare_pdf_corpus(markdown_dir, output_dir)

    assert len(pdf_paths) == 1
    pdf_path = pdf_paths[0]
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_prepare_pdf_corpus_requires_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdf"
    with pytest.raises(FileNotFoundError):
        prepare_pdf_corpus(tmp_path, output_dir)


def test_build_builder_command_targets_pdf_and_chunks(tmp_path: Path) -> None:
    builder = Path("src/langchain/lc_build_index.py").resolve()
    pdf_dir = tmp_path / "pdf"
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"

    command = build_builder_command(
        builder=builder,
        index_key="neuro",
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )

    assert command[:2] == [sys.executable, str(builder)]
    assert "--input-dir" in command
    assert str(pdf_dir) in command
    assert "--chunks-dir" in command
    assert str(chunks_dir) in command
    assert "--index-dir" in command
    assert str(index_dir) in command


def test_build_question_invocation_for_asker_uses_key_and_index(tmp_path: Path) -> None:
    asker = Path("src/langchain/lc_ask.py").resolve()
    question = Question(
        qid="q1",
        qtype="single",
        prompt="What is neuroplasticity?",
        gold_docs=[],
        answer="",
    )
    command, route = build_question_invocation(
        question=question,
        index_dir=tmp_path / "index",
        chunks_dir=tmp_path / "chunks",
        asker=asker,
        multi=None,
        topk=None,
        index_key="neuro",
        embed_model="BAAI/bge-small-en-v1.5",
    )

    assert route == "asker"
    assert "--key" in command
    assert "--index-dir" in command
    assert "--chunks-dir" in command
    assert "--embed-model" in command


def test_build_question_invocation_for_multi_agent_includes_subcommand(tmp_path: Path) -> None:
    asker = Path("src/langchain/lc_ask.py").resolve()
    multi = Path("src/cli/multi_agent.py").resolve()
    question = Question(
        qid="q2",
        qtype="multiturn",
        prompt="Summarize the latest findings",
        gold_docs=[],
        answer="",
    )

    command, route = build_question_invocation(
        question=question,
        index_dir=tmp_path / "index",
        chunks_dir=tmp_path / "chunks",
        asker=asker,
        multi=multi,
        topk=None,
        index_key="neuro",
        embed_model="BAAI/bge-small-en-v1.5",
    )

    assert route == "multi"
    assert command[2] == "ask"
    assert "--key" in command
    assert "--index-dir" in command
