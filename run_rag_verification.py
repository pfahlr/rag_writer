#!/usr/bin/env python3
"""Verification harness that shells out to the RAG CLI stack."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import yaml


SUPPORTED_TYPES = {
    "direct",
    "ambiguous",
    "synthesis",
    "conflict",
    "freshness",
    "multiturn",
}

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass(slots=True)
class Question:
    """Validated question entry from the YAML manifest."""

    qid: str
    qtype: str
    prompt: str
    gold_docs: list[str]
    clarify: str | None = None
    note: str | None = None
    followups: list[str] = field(default_factory=list)
    answer: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG verification harness against a corpus",
    )
    parser.add_argument("--verbose", action="store_true", help="Echo subprocesses")
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Retain the built index directory after verification",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Directory containing source documents for the index",
    )
    parser.add_argument(
        "--workdir",
        default=".rag_tmp",
        help="Working directory for intermediate artifacts",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions.yaml in meta+questions format",
    )
    parser.add_argument(
        "--builder",
        default="lc_build_index.py",
        help="Path to the index builder CLI script",
    )
    parser.add_argument(
        "--asker",
        default="lc_ask.py",
        help="Path to the QA CLI script",
    )
    parser.add_argument(
        "--multi",
        help="Optional multi-turn CLI script (defaults to multi_agent.py if present)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="Optional retrieval depth to forward to the asker if supported",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout (seconds) for builder/asker subprocesses",
    )
    parser.add_argument(
        "--require-answers",
        action="store_true",
        help="Fail if any question is missing a gold answer",
    )
    parser.add_argument(
        "--save-outputs",
        help="Directory to store per-question model outputs (defaults to workdir/outputs)",
    )
    parser.add_argument(
        "--jsonl-out",
        default="verification_results.jsonl",
        help="Path to write JSONL summary results",
    )
    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Always exit 0 even if graded questions fail",
    )
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Abort immediately when a subprocess exits with a non-zero status",
    )
    return parser.parse_args(list(argv) if argv is not None else sys.argv[1:])


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def norm_text(text: str) -> str:
    cleaned = re.sub(r"[\W_]+", " ", text.casefold())
    return re.sub(r"\s+", " ", cleaned).strip()


def f1_score(reference: str, candidate: str) -> float:
    ref_tokens = set(norm_text(reference).split())
    cand_tokens = set(norm_text(candidate).split())
    if not ref_tokens or not cand_tokens:
        return 0.0
    overlap = ref_tokens & cand_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(cand_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exactish(reference: str, candidate: str) -> bool:
    ref_norm = norm_text(reference)
    cand_norm = norm_text(candidate)
    if not ref_norm or not cand_norm:
        return False
    if ref_norm == cand_norm:
        return True
    return ref_norm in cand_norm or cand_norm in ref_norm


def resolve_script(script: str | None, *, default: str | None = None) -> Path | None:
    """Resolve a CLI script path relative to common repository roots."""

    if script:
        candidate = Path(script)
        if candidate.is_file():
            return candidate.resolve()

    repo_root = Path(__file__).resolve().parent
    names = [script] if script else []
    if default and default not in names:
        names.append(default)
    candidates: list[Path] = []
    for name in names:
        if not name:
            continue
        path = Path(name)
        if path.is_file():
            candidates.append(path.resolve())
            continue
        for base in (repo_root, repo_root / "src", repo_root / "src" / "langchain", repo_root / "tools", repo_root / "scripts"):
            candidate = base / name
            if candidate.is_file():
                candidates.append(candidate.resolve())
    if candidates:
        return candidates[0]
    return None


def _load_yaml(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, dict):
        questions = data.get("questions")
        if not isinstance(questions, list):
            raise ValueError("questions.yaml must include a 'questions' list")
        return questions
    if isinstance(data, list):
        return data
    raise ValueError("questions.yaml must be either a list or a mapping with 'questions'")


def load_questions(path: Path, *, require_answers: bool) -> list[Question]:
    raw_entries = _load_yaml(path)
    questions: list[Question] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each question entry must be a mapping")
        missing = {"id", "type", "question"} - set(entry)
        if missing:
            raise ValueError(f"Question entry missing required keys: {sorted(missing)}")
        qid = str(entry["id"])
        qtype = str(entry["type"]).strip().lower()
        if qtype not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported question type '{qtype}' for id={qid}")
        prompt = str(entry["question"])
        gold_doc = entry.get("gold_doc")
        gold_docs = entry.get("gold_docs")
        if bool(gold_doc) == bool(gold_docs):
            raise ValueError(
                f"Question {qid} must include exactly one of 'gold_doc' or 'gold_docs'",
            )
        docs: list[str] = []
        if gold_doc:
            docs = [str(gold_doc)]
        else:
            if not isinstance(gold_docs, Iterable):
                raise ValueError(f"Question {qid} gold_docs must be an iterable of strings")
            docs = [str(item) for item in gold_docs if isinstance(item, str)]
            if not docs:
                raise ValueError(f"Question {qid} gold_docs must contain at least one string")
        clarify = entry.get("clarify")
        note = entry.get("note")
        followups_raw = entry.get("followups")
        followups: list[str] = []
        if followups_raw:
            if isinstance(followups_raw, Iterable):
                followups = [str(item) for item in followups_raw if isinstance(item, str)]
            else:
                raise ValueError(f"Question {qid} followups must be a list of strings")
        answer = entry.get("answer")
        if require_answers and not isinstance(answer, str):
            raise ValueError(f"Question {qid} is missing a gold answer")
        questions.append(
            Question(
                qid=qid,
                qtype=qtype,
                prompt=prompt,
                gold_docs=docs,
                clarify=str(clarify) if isinstance(clarify, str) else None,
                note=str(note) if isinstance(note, str) else None,
                followups=followups,
                answer=str(answer) if isinstance(answer, str) else None,
            )
        )
    return questions


@lru_cache(maxsize=None)
def script_help_text(script: Path) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def determine_flag(script: Path, candidates: Sequence[str]) -> str | None:
    help_text = script_help_text(script)
    for flag in candidates:
        if flag and flag in help_text:
            return flag
    return None


def determine_builder_flags(builder: Path) -> tuple[str, str]:
    corpus_flag = os.getenv("RAG_VERIFICATION_BUILDER_CORPUS_FLAG")
    out_flag = os.getenv("RAG_VERIFICATION_BUILDER_OUT_FLAG")
    if corpus_flag and out_flag:
        return corpus_flag, out_flag
    help_text = script_help_text(builder)
    corpus_candidates = [
        corpus_flag,
        "--corpus",
        "--docs",
        "--source",
        "--input",
        "--path",
    ]
    out_candidates = [
        out_flag,
        "--out",
        "--output",
        "--index",
        "--dest",
    ]
    for cand in corpus_candidates:
        if cand and cand in help_text:
            corpus_flag = cand
            break
    else:
        corpus_flag = corpus_flag or "--corpus"
    for cand in out_candidates:
        if cand and cand in help_text:
            out_flag = cand
            break
    else:
        out_flag = out_flag or "--out"
    return corpus_flag, out_flag


def write_log(path: Path, stdout: str, stderr: str, *, command: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if command:
            handle.write("$ " + " ".join(shlex.quote(p) for p in command) + "\n\n")
        if stdout:
            handle.write("[stdout]\n")
            handle.write(stdout)
            if not stdout.endswith("\n"):
                handle.write("\n")
            handle.write("\n")
        if stderr:
            handle.write("[stderr]\n")
            handle.write(stderr)
            if not stderr.endswith("\n"):
                handle.write("\n")


def run_builder(
    *,
    builder: Path,
    corpus_dir: Path,
    index_dir: Path,
    logs_dir: Path,
    timeout: float,
    verbose: bool,
) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_flag, out_flag = determine_builder_flags(builder)
    command = [
        sys.executable,
        str(builder),
        corpus_flag,
        str(corpus_dir),
        out_flag,
        str(index_dir),
    ]
    if verbose:
        print("$", " ".join(shlex.quote(part) for part in command))
    log_path = logs_dir / "build_index.log"
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = strip_ansi(exc.stdout or "")
        stderr = strip_ansi(exc.stderr or "")
        write_log(log_path, stdout, stderr, command=command)
        raise RuntimeError(
            f"Builder timed out after {timeout} seconds. See {log_path}",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stdout = strip_ansi(exc.stdout or "")
        stderr = strip_ansi(exc.stderr or "")
        write_log(log_path, stdout, stderr, command=command)
        raise RuntimeError(
            f"Builder exited with status {exc.returncode}. See {log_path}",
        ) from exc
    else:
        stdout = strip_ansi(proc.stdout or "")
        stderr = strip_ansi(proc.stderr or "")
        write_log(log_path, stdout, stderr, command=command)


def extract_model_answer(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        for key in ("answer", "result", "output"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return text


def build_question_command(
    *,
    question: Question,
    index_dir: Path,
    asker: Path,
    multi: Path | None,
    topk: int | None,
) -> tuple[list[str], str]:
    use_multi = bool(
        multi
        and (question.qtype == "multiturn" or question.clarify or question.followups)
    )
    script = multi if use_multi else asker
    if script is None:
        raise RuntimeError("No asker script available")
    command = [sys.executable, str(script)]
    route = "multi" if use_multi else "asker"
    if not use_multi:
        index_flag = determine_flag(asker, ["--index", "--key", "--collection"]) or "--index"
        command.extend([index_flag, str(index_dir)])
        command.extend(["--question", question.prompt])
        docs_flag = determine_flag(asker, ["--docs", "--doc", "--gold" ])
        if docs_flag:
            command.extend([docs_flag, ",".join(question.gold_docs)])
        if topk is not None:
            topk_flag = determine_flag(asker, ["--topk", "--k", "--limit"])
            if topk_flag:
                command.extend([topk_flag, str(topk)])
    else:
        command.extend(["--question", question.prompt])
        if question.clarify:
            clarify_flag = determine_flag(multi, ["--clarify", "--followup", "--context"])
            if clarify_flag:
                command.extend([clarify_flag, question.clarify])
        if question.followups:
            follow_flag = determine_flag(multi, ["--followups", "--followup", "--steps"])
            if follow_flag:
                command.extend([follow_flag, json.dumps(question.followups)])
    return command, route


def run_question(
    *,
    question: Question,
    index_dir: Path,
    asker: Path,
    multi: Path | None,
    topk: int | None,
    timeout: float,
    outputs_dir: Path,
    logs_dir: Path,
    strict: bool,
    verbose: bool,
) -> tuple[str, str, int]:
    command, route = build_question_command(
        question=question,
        index_dir=index_dir,
        asker=asker,
        multi=multi,
        topk=topk,
    )
    if verbose:
        print("$", " ".join(shlex.quote(part) for part in command))
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = strip_ansi(proc.stdout or "")
        stderr = strip_ansi(proc.stderr or "")
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = strip_ansi(exc.stdout or "")
        stderr = strip_ansi(exc.stderr or "")
        returncode = -1
    output_path = outputs_dir / f"{question.qid}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stdout, encoding="utf-8")
    err_path = logs_dir / f"{question.qid}.err"
    write_log(err_path, "", stderr, command=command)
    if strict and returncode != 0:
        raise RuntimeError(
            f"Command for question {question.qid} failed (exit={returncode}). See {err_path}",
        )
    return stdout, route, returncode


def score_answer(gold: str | None, model_answer: str) -> tuple[bool | None, float | None]:
    if gold is None:
        return None, None
    if not model_answer.strip():
        return False, 0.0
    if exactish(gold, model_answer):
        return True, 1.0
    score = f1_score(gold, model_answer)
    return (score >= 0.6, score)


def summary_counts(records: list[dict]) -> dict[str, int]:
    total = len(records)
    graded = sum(1 for rec in records if rec.get("pass") is not None)
    passed = sum(1 for rec in records if rec.get("pass") is True)
    failed = sum(1 for rec in records if rec.get("pass") is False)
    skipped = total - graded
    return {
        "total": total,
        "graded": graded,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def print_summary(counts: dict[str, int]) -> None:
    print("\nSummary")
    print("=======")
    print(f"Total questions:        {counts['total']}")
    print(f"Graded (with answers): {counts['graded']}")
    print(f"Passed:                 {counts['passed']}")
    print(f"Failed:                 {counts['failed']}")
    print(f"Skipped (no answer):    {counts['skipped']}")


def ensure_outputs_dir(workdir: Path, save_outputs: str | None) -> Path:
    if save_outputs:
        path = Path(save_outputs)
    else:
        path = workdir / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_dir = Path(args.corpus).resolve()
    questions_path = Path(args.questions).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    index_dir = workdir / "index_dir"
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = ensure_outputs_dir(workdir, args.save_outputs)

    builder_path = resolve_script(args.builder, default="lc_build_index.py")
    if builder_path is None:
        raise SystemExit(f"Unable to locate builder script: {args.builder}")
    asker_path = resolve_script(args.asker, default="lc_ask.py")
    if asker_path is None:
        raise SystemExit(f"Unable to locate asker script: {args.asker}")
    multi_path = resolve_script(args.multi, default="multi_agent.py")

    try:
        questions = load_questions(questions_path, require_answers=args.require_answers)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to load questions: {exc}") from exc

    try:
        run_builder(
            builder=builder_path,
            corpus_dir=corpus_dir,
            index_dir=index_dir,
            logs_dir=logs_dir,
            timeout=args.timeout,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    results: list[dict] = []
    for question in questions:
        try:
            stdout, route, returncode = run_question(
                question=question,
                index_dir=index_dir,
                asker=asker_path,
                multi=multi_path,
                topk=args.topk,
                timeout=args.timeout,
                outputs_dir=outputs_dir,
                logs_dir=logs_dir,
                strict=args.strict_errors,
                verbose=args.verbose,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            if args.strict_errors:
                raise SystemExit(1) from exc
            stdout = ""
            route = "asker"
            returncode = 1
        model_answer = extract_model_answer(stdout)
        passed, score = score_answer(question.answer, model_answer)
        record: dict[str, object] = {
            "qid": question.qid,
            "type": question.qtype,
            "question": question.prompt,
            "gold_answer": question.answer,
            "model_answer": model_answer,
            "pass": passed,
            "score": score,
            "route": route,
        }
        if question.answer is None:
            record["note"] = "no gold answer"
        elif returncode != 0:
            record["note"] = f"command exit code {returncode}"
        results.append(record)

    jsonl_path = Path(args.jsonl_out)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in results:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = summary_counts(results)
    print_summary(counts)
    should_fail = counts["failed"] > 0 and not args.no_fail_on_error
    if not args.keep_index:
        pass  # index retention currently default behaviour
    return 1 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())

