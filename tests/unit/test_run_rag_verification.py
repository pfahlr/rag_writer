"""Tests for the ``run_rag_verification`` harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import run_rag_verification as harness


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Return a fake repository root for command invocations."""

    return tmp_path


def test_build_index_invokes_lc_build_index_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """Ensure ``build_index`` calls the LangChain script with the new CLI."""

    recorded: dict[str, object] = {}

    def fake_run_command(*command_args, **command_kwargs):  # type: ignore[no-untyped-def]
        recorded["command"] = list(command_args[0])
        recorded["verbose"] = command_kwargs.get("verbose")
        recorded["cwd"] = command_kwargs.get("cwd")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(harness, "run_command", fake_run_command)

    script_path = repo_root / "src" / "langchain" / "lc_build_index.py"
    corpus_dir = repo_root / "eval" / "docs"
    chunks_dir = repo_root / "data_processed"
    index_dir = repo_root / "storage"
    key = "verification_tmp"

    harness.build_index(
        script_path,
        corpus_dir,
        key=key,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        verbose=True,
        cwd=repo_root,
    )

    expected_command = [
        sys.executable,
        str(script_path),
        key,
        "--input-dir",
        str(corpus_dir),
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
    ]

    assert recorded["command"] == expected_command
    assert recorded["verbose"] is True
    assert recorded["cwd"] == repo_root


def test_run_lc_ask_invokes_cli_with_question_and_key(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """``run_lc_ask`` should forward the question and key to the script."""

    captured: dict[str, object] = {}

    def fake_run_command(*command_args, **command_kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = list(command_args[0])
        captured["cwd"] = command_kwargs.get("cwd")

        class Result:
            returncode = 0
            stderr = ""
            stdout = "{}"

        return Result()

    monkeypatch.setattr(harness, "run_command", fake_run_command)

    script_path = repo_root / "src" / "langchain" / "lc_ask.py"
    question = "What is neuroplasticity?"
    key = "verification_tmp"

    harness.run_lc_ask(
        script_path,
        key=key,
        question=question,
        verbose=False,
        cwd=repo_root,
    )

    expected = [sys.executable, str(script_path), question, "--key", key]
    assert captured["command"] == expected
    assert captured["cwd"] == repo_root


def test_run_multi_agent_invokes_cli_directly(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """``run_multi_agent`` should execute the Typer CLI with question and key."""

    captured: dict[str, object] = {}

    def fake_run_command(*command_args, **command_kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = list(command_args[0])
        captured["cwd"] = command_kwargs.get("cwd")

        class Result:
            returncode = 0
            stderr = ""
            stdout = "Agent output"

        return Result()

    monkeypatch.setattr(harness, "run_command", fake_run_command)

    script_path = repo_root / "src" / "cli" / "multi_agent.py"
    key = "verification_tmp"
    question = "Summarise the findings"

    harness.run_multi_agent(
        script_path,
        key=key,
        question=question,
        verbose=True,
        cwd=repo_root,
    )

    expected = [sys.executable, str(script_path), question, "--key", key]
    assert captured["command"] == expected
    assert captured["cwd"] == repo_root


def test_cleanup_index_outputs_removes_generated_files(tmp_path: Path) -> None:
    """Only the verification artefacts should be removed."""

    chunks_dir = tmp_path / "data_processed"
    chunks_dir.mkdir()
    index_dir = tmp_path / "storage"
    index_dir.mkdir()

    key = "verification_tmp"
    chunk_file = chunks_dir / f"lc_chunks_{key}.jsonl"
    chunk_file.write_text("{}", encoding="utf-8")

    created_index = index_dir / f"faiss_{key}__model"
    created_index.mkdir()
    (created_index / "index.faiss").write_text("dummy", encoding="utf-8")

    repacked = index_dir / f"faiss_{key}__model_repacked"
    repacked.mkdir()
    (repacked / "index.faiss").write_text("dummy", encoding="utf-8")

    untouched = index_dir / "faiss_other__model"
    untouched.mkdir()

    harness.cleanup_index_outputs(key, chunks_dir, index_dir)

    assert not chunk_file.exists()
    assert not created_index.exists()
    assert not repacked.exists()
    assert untouched.exists()

