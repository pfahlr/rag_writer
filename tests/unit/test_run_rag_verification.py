"""Tests for helper utilities in ``run_rag_verification``."""

from __future__ import annotations

from pathlib import Path

import pytest

from run_rag_verification import prepare_pdf_corpus


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
