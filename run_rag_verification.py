#!/usr/bin/env python3
"""Verification harness that shells out to the RAG CLI stack."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import yaml

from src.langchain.trace import TRACE_PREFIX, redact as trace_redact


SUPPORTED_TYPES = {
    "direct",
    "ambiguous",
    "synthesis",
    "conflict",
    "freshness",
    "multiturn",
}

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf_stream(markdown_text: str) -> bytes:
    lines = markdown_text.replace("\r", "").split("\n")
    if not lines:
        lines = [""]

    content_lines = ["BT", "/F1 12 Tf", "72 720 Td"]
    first = True
    for line in lines:
        escaped = _pdf_escape(line)
        if first:
            first = False
        else:
            content_lines.append("T*")
        content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")

    stream = "\n".join(content_lines).encode("utf-8")

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: list[int] = []

    def write_object(object_id: int, payload: bytes) -> None:
        offsets.append(buffer.tell())
        buffer.write(f"{object_id} 0 obj\n".encode("ascii"))
        buffer.write(payload)
        buffer.write(b"\nendobj\n")

    write_object(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    write_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    write_object(
        3,
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
    )
    write_object(
        4,
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    )
    write_object(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_offset = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))

    buffer.write(
        (
            b"trailer\n<< /Size "
            + str(len(offsets) + 1).encode("ascii")
            + b" /Root 1 0 R >>\nstartxref\n"
        )
    )
    buffer.write(str(xref_offset).encode("ascii") + b"\n%%EOF\n")
    return buffer.getvalue()


def prepare_pdf_corpus(markdown_dir: Path, output_dir: Path) -> list[Path]:
    """Render Markdown files into a simple PDF corpus for verification runs."""

    markdown_dir = Path(markdown_dir)
    output_dir = Path(output_dir)

    if not markdown_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    markdown_files = sorted(
        p for p in markdown_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"
    )
    if not markdown_files:
        raise FileNotFoundError("No markdown files found to convert into PDFs")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for md_path in markdown_files:
        text = md_path.read_text(encoding="utf-8")
        pdf_bytes = _build_pdf_stream(text)
        pdf_path = output_dir / f"{md_path.stem}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        generated.append(pdf_path)

    return generated


class TraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        directory: Path,
        console: Console | None,
        redact: bool,
        include_context: bool,
    ) -> None:
        self.enabled = enabled
        self.directory = directory
        self.console = console
        self.redact = redact
        self.include_context = include_context
        self.events: list[dict] = []
        self.events_by_qid: dict[str, list[dict]] = {}
        self._file_handles: dict[str, object] = {}
        self.master_path = directory / "all.ndjson"
        directory.mkdir(parents=True, exist_ok=True)
        if enabled:
            self._master_handle = self.master_path.open("w", encoding="utf-8")
        else:
            self._master_handle = None
            self.master_path.touch()

    def close(self) -> None:
        if self._master_handle:
            self._master_handle.close()
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._file_handles.clear()

    def handle_event(self, event: dict, *, default_qid: str | None = None) -> None:
        if not self.enabled:
            return
        payload = dict(event)
        qid = payload.get("qid") or default_qid or "general"
        payload["qid"] = qid
        if self.redact:
            payload = trace_redact(payload)
        line = json.dumps(payload, ensure_ascii=False)
        if self._master_handle:
            self._master_handle.write(line + "\n")
            self._master_handle.flush()
        file_handle = self._file_handles.get(qid)
        if file_handle is None:
            path = self.directory / f"{qid}.ndjson"
            file_handle = path.open("a", encoding="utf-8")
            self._file_handles[qid] = file_handle
        file_handle.write(line + "\n")
        file_handle.flush()
        self.events.append(payload)
        self.events_by_qid.setdefault(qid, []).append(payload)
        if self.console:
            self._render_event(payload)

    def start_question(self, question: Question) -> None:
        if not self.enabled or not self.console:
            return
        title = f"{question.qid} — {question.prompt}"
        self.console.print(
            Panel(Text(title), title="Question", expand=False, border_style="cyan")
        )

    def _render_event(self, event: dict) -> None:
        qid = event.get("qid", "general")
        event_type = event.get("type", "")
        name = event.get("name", "")
        detail = event.get("detail") or {}
        metrics = event.get("metrics") or {}
        lines: list[str] = []
        if event_type == "llm.prompt" and isinstance(detail, dict):
            messages = detail.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = (msg.get("content", "") or "")[:500]
                    lines.append(f"{role}: {content}")
        elif event_type == "llm.completion" and isinstance(detail, dict):
            content = (detail.get("content", "") or "")[:500]
            lines.append(content)
        elif event_type == "vector.query" and isinstance(detail, dict):
            query_text = (detail.get("query_text", "") or "")[:500]
            backend = detail.get("backend")
            lines.append(
                f"backend={backend}, top_k={detail.get('top_k')}\n{query_text}"
            )
        elif event_type == "vector.results" and isinstance(detail, dict):
            hits = detail.get("hits", [])
            table = Table(show_header=True, header_style="bold", expand=False)
            table.add_column("Rank", justify="right")
            table.add_column("Doc ID")
            table.add_column("Score")
            for hit in hits[:5]:
                if isinstance(hit, dict):
                    table.add_row(
                        str(hit.get("rank") or ""),
                        str(hit.get("doc_id") or ""),
                        str(hit.get("score") or ""),
                    )
                    if self.include_context:
                        snippet = hit.get("text") or hit.get("snippet") or ""
                        if snippet:
                            lines.append(snippet[:400])
            if hits:
                self.console.print(table)
        elif isinstance(detail, dict) and detail:
            lines.append(json.dumps(detail, ensure_ascii=False)[:500])
        text_lines = "\n".join(lines)
        metrics_line = ""
        if metrics:
            metrics_line = " | " + ", ".join(
                f"{key}={value}" for key, value in metrics.items() if value is not None
            )
        body = Text(text_lines or name or event_type)
        title = f"{qid} — {event_type}{metrics_line}"
        self.console.print(Panel(body, title=title, expand=False))

    def write_markdown(self, path: Path) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for qid, events in self.events_by_qid.items():
                handle.write(f"## {qid}\n\n")
                for event in events:
                    title = f"### {event.get('type', '')} — {event.get('name', '')}\n"
                    handle.write(title)
                    handle.write("```json\n")
                    handle.write(json.dumps(event, ensure_ascii=False, indent=2))
                    handle.write("\n```\n\n")


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
    parser.set_defaults(trace=True)
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
        "--index-key",
        default="verification",
        help="Storage key used when building and querying the FAISS index",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name forwarded to the asker CLI",
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
    parser.add_argument(
        "--trace",
        dest="trace",
        action="store_true",
        help="Enable live TRACE rendering and transcripts (default)",
    )
    parser.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="Disable TRACE rendering and transcript capture",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Show retrieved text snippets in the live view",
    )
    parser.add_argument(
        "--transcript-out",
        help="Directory for NDJSON trace transcripts (default: <workdir>/transcripts)",
    )
    parser.add_argument(
        "--transcript-md",
        help="Optional Markdown transcript output path",
    )
    parser.add_argument(
        "--redact",
        default="true",
        help="Control redaction of secrets in transcripts (true/false)",
    )
    return parser.parse_args(list(argv) if argv is not None else sys.argv[1:])


def parse_bool(value: str, *, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "default"}:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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


def resolve_script(
    explicit: Optional[Path | str],
    default: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve a CLI script path relative to common repository roots."""

    repo_root = (repo_root or Path(__file__).resolve().parent).resolve()
    search_roots = [
        repo_root,
        repo_root / "src",
        repo_root / "src" / "langchain",
        repo_root / "tools",
        repo_root / "scripts",
        repo_root / "src" / "cli",
    ]

    names: list[str] = []
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.is_file():
            return explicit_path.resolve()
        names.append(
            explicit_path.name if explicit_path.is_absolute() else str(explicit_path)
        )
    if default and default not in names:
        names.append(default)

    for name in names:
        candidate = Path(name)
        if candidate.is_file():
            return candidate.resolve()
        for root in search_roots:
            path = (root / name).resolve()
            if path.is_file():
                return path

    raise FileNotFoundError(
        f"Could not resolve script {default!r} under {search_roots}"
    )


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
    raise ValueError(
        "questions.yaml must be either a list or a mapping with 'questions'"
    )


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
                raise ValueError(
                    f"Question {qid} gold_docs must be an iterable of strings"
                )
            docs = [str(item) for item in gold_docs if isinstance(item, str)]
            if not docs:
                raise ValueError(
                    f"Question {qid} gold_docs must contain at least one string"
                )
        clarify = entry.get("clarify")
        note = entry.get("note")
        followups_raw = entry.get("followups")
        followups: list[str] = []
        if followups_raw:
            if isinstance(followups_raw, Iterable):
                followups = [
                    str(item) for item in followups_raw if isinstance(item, str)
                ]
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


_FLAG_PATTERN = re.compile(r"(?<![\w-])(-{1,2}[A-Za-z0-9][A-Za-z0-9-]*)(?![\w-])")


def _advertised_flags(help_text: str) -> set[str]:
    return {match.group(1) for match in _FLAG_PATTERN.finditer(help_text)}


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
    advertised = _advertised_flags(help_text)
    for flag in candidates:
        if flag and flag in advertised:
            return flag
    return None


def _script_supports_flag(
    script: Path, flag_candidates: Sequence[str]
) -> Optional[str]:
    """Return the first flag advertised by the script's help output."""

    help_text = script_help_text(script)
    advertised = _advertised_flags(help_text)
    for flag in flag_candidates:
        if flag and flag in advertised:
            return flag
    return None


def build_question_command(
    script: Path, prompt: str, base_args: List[str]
) -> List[str]:
    """
    Build argv for CLIs that may or may not support a --question flag.
    If the script advertises a question flag, use it; otherwise pass the prompt positionally.
    """

    argv: List[str] = [sys.executable, str(script), *base_args]
    flag = _script_supports_flag(script, ["--question", "-q"])
    if flag:
        argv.extend([flag, prompt])
    else:
        argv.append(prompt)
    return argv


def build_builder_command(
    *,
    builder: Path,
    index_key: str,
    pdf_dir: Path,
    chunks_dir: Path,
    index_dir: Path,
) -> list[str]:
    """Construct the lc_build_index command for the verification run."""

    return [
        sys.executable,
        str(builder),
        index_key,
        "--input-dir",
        str(pdf_dir),
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
    ]


def write_log(
    path: Path, stdout: str, stderr: str, *, command: Sequence[str] | None = None
) -> None:
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


def run_traceable_subprocess(
    command: Sequence[str],
    *,
    env: dict[str, str] | None,
    timeout: float,
    recorder: TraceRecorder | None,
    default_qid: str | None,
    verbose: bool,
    console: Console | None,
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _read_stdout() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stdout_lines.append(line)
            if verbose and console:
                console.print(line.rstrip("\n"))

    def _read_stderr() -> None:
        if proc.stderr is None:
            return
        for raw in proc.stderr:
            stripped = raw.rstrip("\n")
            if stripped.startswith(TRACE_PREFIX):
                payload = stripped[len(TRACE_PREFIX) :].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    stderr_lines.append(raw)
                    if verbose and console:
                        console.print(stripped, style="dim")
                    continue
                if recorder:
                    recorder.handle_event(event, default_qid=default_qid)
            else:
                stderr_lines.append(raw)
                if verbose and console:
                    console.print(stripped, style="dim")

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout_thread.join()
        stderr_thread.join()
        raise subprocess.TimeoutExpired(
            command, timeout, output="".join(stdout_lines), stderr="".join(stderr_lines)
        ) from exc

    stdout_thread.join()
    stderr_thread.join()

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)
    return proc.returncode, stdout_text, stderr_text


def run_builder(
    *,
    builder: Path,
    pdf_dir: Path,
    chunks_dir: Path,
    index_dir: Path,
    index_key: str,
    logs_dir: Path,
    timeout: float,
    verbose: bool,
    trace: bool,
    recorder: TraceRecorder | None,
    console: Console | None,
) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    command = build_builder_command(
        builder=builder,
        index_key=index_key,
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )
    if verbose:
        print("$", " ".join(shlex.quote(part) for part in command))
    log_path = logs_dir / "build_index.log"
    env = os.environ.copy()
    if trace:
        env["RAG_TRACE"] = "1"
    try:
        returncode, stdout_raw, stderr_raw = run_traceable_subprocess(
            command,
            env=env,
            timeout=timeout,
            recorder=recorder if trace else None,
            default_qid=None,
            verbose=verbose,
            console=console,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = strip_ansi(getattr(exc, "output", "") or "")
        stderr = strip_ansi(getattr(exc, "stderr", "") or "")
        write_log(log_path, stdout, stderr, command=command)
        raise RuntimeError(
            f"Builder timed out after {timeout} seconds. See {log_path}",
        ) from exc
    stdout = strip_ansi(stdout_raw)
    stderr = strip_ansi(stderr_raw)
    write_log(log_path, stdout, stderr, command=command)
    if returncode != 0:
        raise RuntimeError(
            f"Builder exited with status {returncode}. See {log_path}",
        )


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


def build_question_invocation(
    *,
    question: Question,
    index_dir: Path,
    chunks_dir: Path,
    asker: Path,
    multi: Path | None,
    topk: int | None,
    index_key: str,
    embed_model: str,
) -> tuple[list[str], str]:
    use_multi = bool(
        multi
        and (question.qtype == "multiturn" or question.clarify or question.followups)
    )
    script = multi if use_multi else asker
    if script is None:
        raise RuntimeError("No asker script available")
    route = "multi" if use_multi else "asker"
    if not use_multi:
        base_args: List[str] = []
        key_flag = determine_flag(asker, ["--key"]) or "--key"
        if key_flag:
            base_args.extend([key_flag, index_key])
        index_flag = determine_flag(asker, ["--index-dir", "--index"]) or "--index-dir"
        if index_flag:
            base_args.extend([index_flag, str(index_dir)])
        chunks_flag = determine_flag(asker, ["--chunks-dir"]) or "--chunks-dir"
        if chunks_flag:
            base_args.extend([chunks_flag, str(chunks_dir)])
        embed_flag = determine_flag(asker, ["--embed-model"]) or "--embed-model"
        if embed_flag:
            base_args.extend([embed_flag, embed_model])
        command = build_question_command(asker, question.prompt, base_args)
        docs_flag = determine_flag(asker, ["--docs", "--doc", "--gold"])
        if docs_flag:
            command.extend([docs_flag, ",".join(question.gold_docs)])
        if topk is not None:
            topk_flag = determine_flag(asker, ["--topk", "--k", "--limit"])
            if topk_flag:
                command.extend([topk_flag, str(topk)])
    else:
        base_args: list[str] = []
        key_flag = determine_flag(multi, ["--key", "-k"]) or "--key"
        if key_flag:
            base_args.extend([key_flag, index_key])
        index_flag = determine_flag(multi, ["--index-dir", "--index"]) or "--index-dir"
        if index_flag:
            base_args.extend([index_flag, str(index_dir)])
        command = build_question_command(script, question.prompt, base_args)
        if question.clarify:
            clarify_flag = determine_flag(
                multi, ["--clarify", "--followup", "--context"]
            )
            if clarify_flag:
                command.extend([clarify_flag, question.clarify])
        if question.followups:
            follow_flag = determine_flag(
                multi, ["--followups", "--followup", "--steps"]
            )
            if follow_flag:
                command.extend([follow_flag, json.dumps(question.followups)])
    return command, route


def run_question(
    *,
    question: Question,
    index_dir: Path,
    chunks_dir: Path,
    asker: Path,
    multi: Path | None,
    topk: int | None,
    index_key: str,
    embed_model: str,
    timeout: float,
    outputs_dir: Path,
    logs_dir: Path,
    strict: bool,
    verbose: bool,
    trace: bool,
    recorder: TraceRecorder | None,
    console: Console | None,
) -> tuple[str, str, int]:
    command, route = build_question_invocation(
        question=question,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        asker=asker,
        multi=multi,
        topk=topk,
        index_key=index_key,
        embed_model=embed_model,
    )
    if verbose:
        print("$", " ".join(shlex.quote(part) for part in command))
    env = os.environ.copy()
    if trace:
        env["RAG_TRACE"] = "1"
        env["TRACE_QID"] = question.qid
    try:
        returncode, stdout_raw, stderr_raw = run_traceable_subprocess(
            command,
            env=env,
            timeout=timeout,
            recorder=recorder if trace else None,
            default_qid=question.qid,
            verbose=verbose,
            console=console,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = strip_ansi(getattr(exc, "output", "") or "")
        stderr = strip_ansi(getattr(exc, "stderr", "") or "")
        returncode = -1
    else:
        stdout = strip_ansi(stdout_raw)
        stderr = strip_ansi(stderr_raw)
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


def score_answer(
    gold: str | None, model_answer: str
) -> tuple[bool | None, float | None]:
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
    chunks_dir = workdir / "chunks"
    pdf_dir = workdir / "pdf_corpus"
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = ensure_outputs_dir(workdir, args.save_outputs)
    console = Console(highlight=False)
    redact = parse_bool(args.redact, default=True)
    trace_dir = (
        Path(args.transcript_out) if args.transcript_out else workdir / "transcripts"
    )
    recorder = TraceRecorder(
        enabled=args.trace,
        directory=trace_dir,
        console=console if args.trace else None,
        redact=redact,
        include_context=args.include_context,
    )

    should_fail = False
    repo_root = Path(__file__).resolve().parent
    try:
        try:
            builder_path = resolve_script(
                args.builder, "lc_build_index.py", repo_root=repo_root
            )
            asker_path = resolve_script(args.asker, "lc_ask.py", repo_root=repo_root)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        try:
            multi_path = resolve_script(
                args.multi, "multi_agent.py", repo_root=repo_root
            )
        except FileNotFoundError:
            multi_path = None

        try:
            questions = load_questions(
                questions_path, require_answers=args.require_answers
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to load questions: {exc}") from exc

        try:
            if pdf_dir.exists():
                shutil.rmtree(pdf_dir)
            prepare_pdf_corpus(corpus_dir, pdf_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        try:
            run_builder(
                builder=builder_path,
                pdf_dir=pdf_dir,
                chunks_dir=chunks_dir,
                index_dir=index_dir,
                index_key=args.index_key,
                logs_dir=logs_dir,
                timeout=args.timeout,
                verbose=args.verbose,
                trace=args.trace,
                recorder=recorder if args.trace else None,
                console=console,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        results: list[dict] = []
        for question in questions:
            if args.trace:
                recorder.start_question(question)
            try:
                stdout, route, returncode = run_question(
                    question=question,
                    index_dir=index_dir,
                    chunks_dir=chunks_dir,
                    asker=asker_path,
                    multi=multi_path,
                    topk=args.topk,
                    index_key=args.index_key,
                    embed_model=args.embed_model,
                    timeout=args.timeout,
                    outputs_dir=outputs_dir,
                    logs_dir=logs_dir,
                    strict=args.strict_errors,
                    verbose=args.verbose,
                    trace=args.trace,
                    recorder=recorder if args.trace else None,
                    console=console,
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
        if args.transcript_md:
            recorder.write_markdown(Path(args.transcript_md))
        should_fail = counts["failed"] > 0 and not args.no_fail_on_error
    finally:
        recorder.close()
    if not args.keep_index:
        pass  # index retention currently default behaviour
    return 1 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
