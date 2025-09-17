from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest


@pytest.fixture
def manifest_module(monkeypatch):
    import importlib

    mod = importlib.import_module("src.research.manifest_complete_process")
    # ensure clean state for optional pdfio
    monkeypatch.setattr(mod, "HAVE_PDFIO", False)
    monkeypatch.setattr(mod, "pdfio", SimpleNamespace(write_pdf_metadata=None), raising=False)
    return mod


def test_parse_aria2_summary_lines_filters_and_normalises(manifest_module, tmp_path):
    lines = """
ff786a|ERR |       0B/s|../research/out/data/pdfs_inbox/DL-00142-e68f4f.pdf
aa11bb|OK  |    120KiB/s|../research/out/data/pdfs_inbox/DL-00143-a1b2c3.pdf
cc22dd|FAILED|       0B/s|/tmp/DL-00144-d4e5f6.pdf
"""
    path = tmp_path / "aria2.log"
    path.write_text(lines.strip())

    results = manifest_module.parse_aria2_summary_lines(path)
    assert results == [
        ("ERR", "DL-00142-e68f4f.pdf"),
        ("OK", "DL-00143-a1b2c3.pdf"),
        ("FAILED", "DL-00144-d4e5f6.pdf"),
    ]


def test_build_retry_batch_from_manifest_matches_entries(monkeypatch, manifest_module):
    entries: List[Dict[str, Any]] = [
        {
            "temp_filename": "DL-00142-e68f4f.pdf",
            "pdf_url": "https://example.com/file1.pdf",
            "download_status": "failed",
            "failure_reason": "ERR",
        },
        {
            "temp_filename": "DL-00144-d4e5f6.pdf",
            "pdf_url": "",
        },
    ]

    lines, matched, missing = manifest_module.build_retry_batch_from_manifest(
        entries,
        {"DL-00142-e68f4f.pdf", "DL-00144-d4e5f6.pdf", "DL-00145-unknown.pdf"},
        Path("/downloads"),
    )

    assert matched == 1
    assert missing == ["DL-00144-d4e5f6.pdf", "DL-00145-unknown.pdf"]
    assert lines == [
        "https://example.com/file1.pdf",
        "  out=DL-00142-e68f4f.pdf",
        "  dir=/downloads",
    ]
    assert entries[0]["download_status"] == "retry"
    assert "failure_reason" not in entries[0]


def test_safe_enrich_crossref_retries_then_warn(monkeypatch, capsys, manifest_module):
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def fake_enrich(**kwargs):
        calls.append((kwargs.get("doi"), kwargs))
        raise RuntimeError("boom")

    monkeypatch.setattr(manifest_module.time, "sleep", lambda *_: None)

    result = manifest_module.safe_enrich_crossref(
        fake_enrich,
        doi="10.1234/example",
        timeout=0.1,
        tries=2,
        backoff=0,
        ua="Agent/1.0",
    )

    captured = capsys.readouterr()
    assert result == {}
    assert "Crossref enrichment failed after 2 attempts" in captured.out
    assert len(calls) == 2
    # ensure User-Agent header provided on each attempt
    assert all(call[1]["headers"]["User-Agent"] == "Agent/1.0" for call in calls)


def test_write_pdf_metadata_converts_str_to_path(monkeypatch, tmp_path, manifest_module):
    called = {}

    class DummyPdfIo:
        @staticmethod
        def write_pdf_metadata(pdf_path, **kwargs):
            called["type"] = type(pdf_path)
            raise RuntimeError("force fallback")

    monkeypatch.setattr(manifest_module, "HAVE_PDFIO", True)
    monkeypatch.setattr(manifest_module, "pdfio", DummyPdfIo)

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%")

    manifest_module.write_pdf_metadata(str(pdf_file), {"title": "Test"})

    assert issubclass(called["type"], Path)
