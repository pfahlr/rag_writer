#!/usr/bin/env python3
"""Harness script to verify RAG/MCP stack against gold corpus."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import List, Sequence

import yaml


SUPPORTED_DIRECT_TYPES = {"direct", "synthesis", "conflict", "freshness"}
SUPPORTED_MULTI_TYPES = {"ambiguous", "multiturn"}
CITATION_PATTERN = re.compile(r"NP\d{2}")
SUMMARY_LINE_TEMPLATE = "{status:<5}  {qid:<4} {note}"
DEFAULT_INDEX_KEY = "verification_tmp"


class CommandError(RuntimeError):
    """Raised when a subprocess invocation fails."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verification against the neuroplasticity gold corpus."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo commands before executing them.",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Do not delete generated index artefacts after verification completes.",
    )
    return parser.parse_args(argv)


def resolve_script(repo_root: Path, *candidates: str) -> Path:
    """Resolve a script path by checking a sequence of candidate relative paths."""

    for candidate in candidates:
        script_path = repo_root / candidate
        if script_path.exists():
            return script_path
    raise FileNotFoundError(
        f"Unable to locate script. Checked: {', '.join(str(repo_root / c) for c in candidates)}"
    )


def run_command(
    command: Sequence[str], *, verbose: bool = False, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute a command, returning the completed process."""

    if verbose:
        print("$", " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    return result


def load_questions(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Iterable):
        raise ValueError(f"questions.yaml must contain a list, found {type(data)!r}")
    questions: List[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each question entry must be a mapping")
        questions.append(entry)
    return questions


def normalize_expected_docs(entry: dict) -> List[str]:
    for key in ("gold_docs", "gold_doc", "gold_documents", "gold"):
        if key in entry and entry[key] is not None:
            value = entry[key]
            if isinstance(value, str):
                return [value.strip()]
            if isinstance(value, Iterable):
                docs: List[str] = []
                for item in value:
                    if isinstance(item, str):
                        docs.append(item.strip())
                return docs
    return []


def normalize_followups(entry: dict) -> List[str]:
    followups: List[str] = []
    for key in ("clarify", "followups"):
        if key not in entry or entry[key] is None:
            continue
        value = entry[key]
        if isinstance(value, str):
            if value.strip():
                followups.append(value.strip())
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str) and item.strip():
                    followups.append(item.strip())
    return followups


def extract_citations(text: str) -> List[str]:
    return sorted(set(CITATION_PATTERN.findall(text)))


def build_index(
    script_path: Path,
    corpus_docs: Path,
    *,
    key: str,
    chunks_dir: Path,
    index_dir: Path,
    verbose: bool,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        str(script_path),
        key,
        "--input-dir",
        str(corpus_docs),
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
    ]
    result = run_command(command, verbose=verbose, cwd=cwd)
    if result.returncode != 0:
        raise CommandError(
            f"Index build failed with code {result.returncode}: {result.stderr.strip()}"
        )


def run_lc_ask(
    script_path: Path,
    *,
    key: str,
    question: str,
    verbose: bool,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script_path),
        question,
        "--key",
        key,
    ]
    return run_command(command, verbose=verbose, cwd=cwd)


def run_multi_agent(
    script_path: Path,
    *,
    key: str,
    question: str,
    verbose: bool,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script_path),
        question,
        "--key",
        key,
    ]
    return run_command(command, verbose=verbose, cwd=cwd)


def cleanup_index_outputs(key: str, chunks_dir: Path, index_dir: Path) -> None:
    chunk_file = chunks_dir / f"lc_chunks_{key}.jsonl"
    if chunk_file.exists():
        chunk_file.unlink()

    for candidate in index_dir.glob(f"faiss_{key}__*"):
        if candidate.exists():
            shutil.rmtree(candidate, ignore_errors=True)


def evaluate_answer(
    qid: str,
    stdout: str,
    expected_docs: Sequence[str],
    *,
    question_type: str,
) -> tuple[bool, str]:
    citations = extract_citations(stdout)
    citations_set = set(citations)
    missing_docs = [doc for doc in expected_docs if doc not in citations_set]
    pass_check = not missing_docs
    notes: List[str] = []

    if expected_docs:
        if pass_check:
            notes.append("cites " + ", ".join(expected_docs))
        else:
            notes.append(
                "missing citations for " + ", ".join(missing_docs)
            )

    if question_type == "conflict" and qid.upper() == "Q11":
        required = {"NP07", "NP15"}
        has_required = required.issubset(citations_set)
        has_range = "60" in stdout and "80" in stdout
        if not has_required or not has_range:
            pass_check = False
            if not has_required:
                notes.append("requires NP07 and NP15 citations")
            if not has_range:
                notes.append("missing numeric range 60-80")
    if question_type == "freshness" and qid.upper() == "Q12":
        phrase_present = "not convincingly demonstrated" in stdout.lower()
        cites_np05 = "NP05" in citations_set
        if not (phrase_present and cites_np05):
            pass_check = False
            if not phrase_present:
                notes.append("missing 'not convincingly demonstrated'")
            if not cites_np05:
                notes.append("missing NP05 citation")

    if not notes:
        if citations:
            notes.append("cites " + ", ".join(citations))
        else:
            notes.append("no citations detected")

    return pass_check, "; ".join(notes)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parent
    cwd = repo_root
    chunks_dir = repo_root / "data_processed"
    index_dir = repo_root / "storage"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_key = DEFAULT_INDEX_KEY

    try:
        lc_build = resolve_script(
            repo_root,
            "lc_build_index.py",
            "src/langchain/lc_build_index.py",
        )
        lc_ask = resolve_script(
            repo_root,
            "lc_ask.py",
            "src/langchain/lc_ask.py",
        )
        multi_agent = resolve_script(
            repo_root,
            "multi_agent.py",
            "src/cli/multi_agent.py",
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    corpus_dir = repo_root / "eval/verification" / "rag_gold_corpus_neuroplasticity" / "docs"
    questions_path = repo_root / "eval/verification" / "rag_gold_corpus_neuroplasticity" / "questions.yaml"

    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1
    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        return 1

    cleanup_index_outputs(index_key, chunks_dir, index_dir)
    cleanup_requested = not args.keep_index

    try:
        build_index(
            lc_build,
            corpus_dir,
            key=index_key,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            verbose=args.verbose,
            cwd=cwd,
        )
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        if cleanup_requested:
            cleanup_index_outputs(index_key, chunks_dir, index_dir)
        return 1

    try:
        questions = load_questions(questions_path)
        log_records: List[dict] = []
        summary_lines: List[str] = []
        all_passed = True

        for entry in questions:
            qid = str(entry.get("id") or entry.get("qid") or "?").strip() or "?"
            question_text = (
                str(entry.get("question") or entry.get("query") or "").strip()
            )
            question_type = str(entry.get("type") or "").strip().lower()
            expected_docs = [doc.upper() for doc in normalize_expected_docs(entry)]
            followups = normalize_followups(entry)

            if not question_text:
                summary_lines.append(
                    SUMMARY_LINE_TEMPLATE.format(
                        status="FAIL", qid=qid, note="missing question text"
                    )
                )
                log_records.append(
                    {
                        "qid": qid,
                        "query": question_text,
                        "stdout": "",
                        "pass": False,
                        "notes": "missing question text",
                    }
                )
                all_passed = False
                continue

            stdout_segments: List[str] = []
            final_stdout = ""

            if question_type in SUPPORTED_DIRECT_TYPES:
                result = run_lc_ask(
                    lc_ask,
                    key=index_key,
                    question=question_text,
                    verbose=args.verbose,
                    cwd=cwd,
                )
                if result.returncode != 0:
                    note = f"lc_ask.py failed: {result.stderr.strip()}"
                    summary_lines.append(
                        SUMMARY_LINE_TEMPLATE.format(
                            status="FAIL", qid=qid, note=note
                        )
                    )
                    log_records.append(
                        {
                            "qid": qid,
                            "query": question_text,
                            "stdout": result.stdout,
                            "pass": False,
                            "notes": note,
                        }
                    )
                    all_passed = False
                    continue
                final_stdout = result.stdout
                stdout_segments.append(result.stdout)
            elif question_type in SUPPORTED_MULTI_TYPES:
                current_query = question_text
                outputs: List[str] = []
                result = run_multi_agent(
                    multi_agent,
                    key=index_key,
                    question=current_query,
                    verbose=args.verbose,
                    cwd=cwd,
                )
                if result.returncode != 0:
                    note = f"multi_agent.py failed: {result.stderr.strip()}"
                    summary_lines.append(
                        SUMMARY_LINE_TEMPLATE.format(
                            status="FAIL", qid=qid, note=note
                        )
                    )
                    log_records.append(
                        {
                            "qid": qid,
                            "query": current_query,
                            "stdout": result.stdout,
                            "pass": False,
                            "notes": note,
                        }
                    )
                    all_passed = False
                    continue
                outputs.append(result.stdout)

                for follow in followups:
                    current_query = f"{current_query}\n{follow}"
                    follow_result = run_multi_agent(
                        multi_agent,
                        key=index_key,
                        question=current_query,
                        verbose=args.verbose,
                        cwd=cwd,
                    )
                    if follow_result.returncode != 0:
                        note = (
                            "multi_agent.py follow-up failed: "
                            f"{follow_result.stderr.strip()}"
                        )
                        summary_lines.append(
                            SUMMARY_LINE_TEMPLATE.format(
                                status="FAIL", qid=qid, note=note
                            )
                        )
                        log_records.append(
                            {
                                "qid": qid,
                                "query": current_query,
                                "stdout": follow_result.stdout,
                                "pass": False,
                                "notes": note,
                            }
                        )
                        all_passed = False
                        break
                    outputs.append(follow_result.stdout)
                else:
                    final_stdout = outputs[-1] if outputs else ""
                    stdout_segments.extend(outputs)

                if not final_stdout:
                    continue
            else:
                note = f"unsupported question type '{question_type}'"
                summary_lines.append(
                    SUMMARY_LINE_TEMPLATE.format(status="FAIL", qid=qid, note=note)
                )
                log_records.append(
                    {
                        "qid": qid,
                        "query": question_text,
                        "stdout": "",
                        "pass": False,
                        "notes": note,
                    }
                )
                all_passed = False
                continue

            passed, note = evaluate_answer(
                qid,
                final_stdout,
                expected_docs,
                question_type=question_type,
            )
            status = "PASS" if passed else "FAIL"
            summary_lines.append(
                SUMMARY_LINE_TEMPLATE.format(status=status, qid=qid, note=note)
            )
            log_records.append(
                {
                    "qid": qid,
                    "query": question_text,
                    "stdout": "\n---\n".join(stdout_segments),
                    "pass": passed,
                    "notes": note,
                }
            )
            all_passed = all_passed and passed

        logs_dir = repo_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_path = logs_dir / "rag_verification.jsonl"
        with logs_path.open("w", encoding="utf-8") as log_file:
            for record in log_records:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        for line in summary_lines:
            print(line)

        exit_code = 0 if all_passed else 1
    finally:
        if cleanup_requested:
            cleanup_index_outputs(index_key, chunks_dir, index_dir)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
