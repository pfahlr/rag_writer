"""Tests for helper utilities in ``run_rag_verification``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import run_rag_verification


def test_prepare_pdf_corpus_creates_pdfs(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "md"
    markdown_dir.mkdir()
    (markdown_dir / "sample.md").write_text("hello world\nsecond line", encoding="utf-8")

    output_dir = tmp_path / "pdf"
    pdf_paths = run_rag_verification.prepare_pdf_corpus(markdown_dir, output_dir)

    assert len(pdf_paths) == 1
    pdf_path = pdf_paths[0]
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_prepare_pdf_corpus_requires_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdf"
    with pytest.raises(FileNotFoundError):
        run_rag_verification.prepare_pdf_corpus(tmp_path, output_dir)


def test_run_lc_ask_query_includes_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(command, *, verbose: bool, cwd: Path | None):
        captured["command"] = list(command)
        captured["verbose"] = verbose
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(run_rag_verification, "run_command", fake_run_command)

    script_path = tmp_path / "lc_ask.py"
    result = run_rag_verification.run_lc_ask_query(
        script_path,
        "What is neuroplasticity?",
        index_key="study",
        index_dir=tmp_path,
        verbose=True,
        cwd=tmp_path,
    )

    assert isinstance(result, subprocess.CompletedProcess)
    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1] == str(script_path)
    assert command[2] == "What is neuroplasticity?"
    assert "--key" in command
    assert "--index" in command
    index_position = command.index("--index")
    assert command[index_position + 1] == str(tmp_path)
    assert captured["verbose"] is True
    assert captured["cwd"] == tmp_path


def test_run_multi_agent_query_uses_module_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(command, *, verbose: bool, cwd: Path | None):
        captured["command"] = list(command)
        captured["verbose"] = verbose
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(run_rag_verification, "run_command", fake_run_command)

    result = run_rag_verification.run_multi_agent_query(
        "Clarify the prior answer",
        index_key="study",
        index_dir=tmp_path,
        verbose=False,
        cwd=tmp_path,
    )

    assert isinstance(result, subprocess.CompletedProcess)
    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "src.cli.multi_agent"]
    assert "--key" in command
    assert "--index" in command
    index_position = command.index("--index")
    assert command[index_position + 1] == str(tmp_path)
    assert captured["verbose"] is False
    assert captured["cwd"] == tmp_path
