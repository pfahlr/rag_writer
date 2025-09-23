"""Utility harness for rebuilding RAG indices and running verification checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", "")
    )


def _render_pdf_stream(text: str) -> bytes:
    lines = text.split("\n")
    if not lines:
        lines = [""]

    contents: list[str] = ["BT", "/F1 11 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = _pdf_escape(line)
        if index == 0:
            contents.append(f"({escaped}) Tj")
        else:
            contents.append("T*")
            contents.append(f"({escaped}) Tj")
    contents.append("ET")
    return "\n".join(contents).encode("utf-8")


def _build_pdf_bytes(stream: bytes) -> bytes:
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objects: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        (4, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"),
        (5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]

    parts: list[bytes] = [header]
    offsets: list[int] = []
    current = len(header)

    for number, body in objects:
        obj_bytes = f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        parts.append(obj_bytes)
        offsets.append(current)
        current += len(obj_bytes)

    body_bytes = b"".join(parts)
    xref_offset = len(body_bytes)
    size = len(objects) + 1

    xref_lines = ["0000000000 65535 f \n"] + [
        f"{offset:010d} 00000 n \n" for offset in offsets
    ]
    xref = ("xref\n0 {size}\n".format(size=size) + "".join(xref_lines)).encode("ascii")
    trailer = (
        "trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".format(
            size=size, xref=xref_offset
        ).encode("ascii")
    )

    return body_bytes + xref + trailer


def convert_markdown_to_pdfs(markdown_dir: Path, pdf_dir: Path) -> list[Path]:
    markdown_dir = Path(markdown_dir)
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths: list[Path] = []
    for md_path in sorted(markdown_dir.rglob("*.md")):
        relative = md_path.relative_to(markdown_dir)
        target = pdf_dir / relative.with_suffix(".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = md_path.read_text(encoding="utf-8")
        stream = _render_pdf_stream(text)
        pdf_bytes = _build_pdf_bytes(stream)
        target.write_bytes(pdf_bytes)
        pdf_paths.append(target)

    return pdf_paths


def _run_subprocess(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(list(cmd), check=True, text=True, env=env)


def build_faiss_index(
    *,
    key: str,
    pdf_dir: Path,
    chunks_dir: Path,
    index_dir: Path,
    resume: bool = False,
    keep_shards: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    pdf_dir = Path(pdf_dir)
    chunks_dir = Path(chunks_dir)
    index_dir = Path(index_dir)

    cmd: list[str] = [
        sys.executable,
        "src/langchain/lc_build_index.py",
        key,
        "--input-dir",
        str(pdf_dir),
        "--chunks-dir",
        str(chunks_dir),
        "--index-dir",
        str(index_dir),
    ]
    if resume:
        cmd.append("--resume")
    if keep_shards:
        cmd.append("--keep-shards")

    _run_subprocess(cmd, env=env)


def run_lc_ask(
    *,
    question: str,
    key: str,
    chunks_dir: Path,
    index_dir: Path,
    embed_model: str,
    env: dict[str, str] | None = None,
) -> None:
    cmd: list[str] = [
        sys.executable,
        "src/langchain/lc_ask.py",
        question,
        "--key",
        key,
        "--chunks-dir",
        str(Path(chunks_dir)),
        "--index-dir",
        str(Path(index_dir)),
        "--embed-model",
        embed_model,
    ]
    _run_subprocess(cmd, env=env)


def run_multi_agent(
    *,
    question: str,
    key: str,
    env: dict[str, str] | None = None,
    mcp: str | None = None,
) -> None:
    cmd: list[str] = [
        sys.executable,
        "src/cli/multi_agent.py",
        "ask",
        question,
        "--key",
        key,
    ]
    if mcp:
        cmd.extend(["--mcp", mcp])
    _run_subprocess(cmd, env=env)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--sandbox-dir", type=Path, default=Path(".rag_verification"))
    parser.add_argument("--key", default="verification")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--skip-multi-agent", action="store_true")
    parser.add_argument("--mcp")
    return parser.parse_args(argv)


def _load_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = list(args.question)
    if args.questions_file and args.questions_file.exists():
        data = args.questions_file.read_text(encoding="utf-8").splitlines()
        questions.extend(q for q in data if q.strip())
    return questions


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    sandbox = Path(args.sandbox_dir)
    pdf_dir = sandbox / "pdfs"
    chunks_dir = sandbox / "chunks"
    index_dir = sandbox / "indices"

    convert_markdown_to_pdfs(args.corpus_dir, pdf_dir)
    build_faiss_index(
        key=args.key,
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        resume=args.resume,
        keep_shards=args.keep_shards,
    )

    questions = _load_questions(args)
    for question in questions:
        run_lc_ask(
            question=question,
            key=args.key,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            embed_model=args.embed_model,
        )
        if not args.skip_multi_agent:
            run_multi_agent(
                question=question,
                key=args.key,
                mcp=args.mcp,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
