"""Integration tests for the run_rag_verification CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_builder_script(path: Path) -> None:
    path.write_text(
        """
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = {}
    for file in sorted(corpus_dir.glob("*.md")):
        docs[file.name] = file.read_text(encoding="utf-8")

    (out_dir / "index.json").write_text(json.dumps(docs), encoding="utf-8")


if __name__ == "__main__":
    main()
"""
    )


def _write_asker_script(path: Path) -> None:
    path.write_text(
        """
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--docs")
    parser.add_argument("--topk", type=int)
    args = parser.parse_args()

    data = json.loads((Path(args.index) / "index.json").read_text(encoding="utf-8"))
    question = args.question.lower()

    if "beta" in question or "follow" in question:
        print("Beta follows alpha in the Greek alphabet.")
    elif "alpha" in question:
        print("Alpha is the first letter of the Greek alphabet.")
    else:
        print("No answer available.")


if __name__ == "__main__":
    main()
"""
    )


def test_run_rag_verification_cli(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text(
        "Alpha document. Alpha is the first letter of the Greek alphabet.",
        encoding="utf-8",
    )
    (corpus_dir / "doc2.md").write_text(
        "Beta document. Beta follows alpha in the Greek alphabet.",
        encoding="utf-8",
    )

    questions = {
        "meta": {"title": "Sample"},
        "questions": [
            {
                "id": "q1",
                "type": "direct",
                "question": "What is alpha?",
                "gold_doc": "doc1.md",
                "answer": "Alpha is the first letter of the Greek alphabet.",
            },
            {
                "id": "q2",
                "type": "direct",
                "question": "Which letter follows alpha?",
                "gold_doc": "doc2.md",
                "answer": "Beta follows alpha in the Greek alphabet.",
            },
        ],
    }

    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(yaml.safe_dump(questions), encoding="utf-8")

    builder_script = tmp_path / "builder.py"
    asker_script = tmp_path / "asker.py"
    _write_builder_script(builder_script)
    _write_asker_script(asker_script)

    workdir = tmp_path / "work"
    jsonl_out = tmp_path / "results.jsonl"
    outputs_dir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            "run_rag_verification.py",
            "--corpus",
            str(corpus_dir),
            "--workdir",
            str(workdir),
            "--questions",
            str(questions_path),
            "--builder",
            str(builder_script),
            "--asker",
            str(asker_script),
            "--jsonl-out",
            str(jsonl_out),
            "--save-outputs",
            str(outputs_dir),
            "--require-answers",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    index_dir = workdir / "index_dir"
    assert index_dir.exists()

    for q in questions["questions"]:
        out_file = outputs_dir / f"{q['id']}.txt"
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8").strip()

    lines = jsonl_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(questions["questions"])
    for line in lines:
        record = json.loads(line)
        assert record["qid"] in {"q1", "q2"}
        assert record["pass"] is True
        assert isinstance(record["score"], float)

