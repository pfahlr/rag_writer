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


def _make_candidate(manifest_module, tmp_path, name: str, **kwargs):
    path = tmp_path / name
    path.write_bytes(b"")
    return manifest_module.PdfCandidate(path=path, **kwargs)


def test_match_entry_prefers_temp_filename_over_other_heuristics(manifest_module, tmp_path):
    cand1 = _make_candidate(
        manifest_module,
        tmp_path,
        "expected.pdf",
        metadata_doi="10.1234/example",
    )
    cand2 = _make_candidate(
        manifest_module,
        tmp_path,
        "other.pdf",
        metadata_doi="10.1234/example",
    )

    index = manifest_module.build_inbox_index_from_candidates([cand1, cand2])
    entry = {"temp_filename": "expected.pdf", "doi": "10.1234/example"}

    claimed = set()
    matched = manifest_module.match_entry_to_candidate(entry, index, claimed)

    assert matched is cand1
    assert cand1.path in claimed


def test_match_entry_matches_by_metadata_doi_when_filename_mismatch(manifest_module, tmp_path):
    cand1 = _make_candidate(
        manifest_module,
        tmp_path,
        "random.pdf",
        metadata_doi="10.5555/alpha",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand1])
    entry = {"temp_filename": "missing.pdf", "doi": "10.5555/ALPHA"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand1


def test_match_entry_falls_back_to_content_title_and_respects_claimed(manifest_module, tmp_path):
    cand1 = _make_candidate(
        manifest_module,
        tmp_path,
        "mystery.pdf",
        content_title="Understanding Transformers",
    )
    cand2 = _make_candidate(
        manifest_module,
        tmp_path,
        "spare.pdf",
        metadata_title="Understanding Transformers",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand1, cand2])

    entry1 = {"temp_filename": "nope.pdf", "title": "Understanding Transformers"}
    claimed: set = set()

    # metadata title match should win before content title
    first = manifest_module.match_entry_to_candidate(entry1, index, claimed)
    assert first is cand2

    entry2 = {"title": "Understanding Transformers"}
    second = manifest_module.match_entry_to_candidate(entry2, index, claimed)
    assert second is cand1
    assert cand1.path in claimed

    entry3 = {"title": "Understanding Transformers"}
    third = manifest_module.match_entry_to_candidate(entry3, index, claimed)
    assert third is None


def test_match_entry_uses_doi_substring_in_filename(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "paper-101234abc123.pdf",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1234/ABC-123"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_uses_doi_substring_in_content_snippet(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "unmatched.pdf",
        content_compact="xx101234bbccdd",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1234/bb-cc"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_uses_doi_suffix_digits(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "15391523.2025.2478424.pdf",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1080/15391523.2025.2478424"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_handles_journal_suffix_token(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "bjet.13411.pdf",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1111/BJET.13411"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_matches_extensionless_basename(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "jcal.13009",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1111/JCAL.13009"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_matches_short_suffix_digits(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "jee.20503",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"doi": "10.1002/jee.20503"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_uses_pdf_url_basename(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "447499335_oa.pdf",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {"pdf_url": "https://research.monash.edu/files/447499584/447499335_oa.pdf"}

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_match_entry_uses_doi_suffix_in_url_path(manifest_module, tmp_path):
    cand = _make_candidate(
        manifest_module,
        tmp_path,
        "17439884.2021.1876725",
    )
    index = manifest_module.build_inbox_index_from_candidates([cand])
    entry = {
        "doi": "10.1080/17439884.2021.1876725",
        "pdf_url": "https://www.tandfonline.com/doi/pdf/10.1080/17439884.2021.1876725",
    }

    matched = manifest_module.match_entry_to_candidate(entry, index, set())

    assert matched is cand


def test_scan_inbox_includes_extensionless_pdfs(monkeypatch, tmp_path, manifest_module):
    target = tmp_path / "15391523.2025.2478424"
    target.write_bytes(b"%PDF-1.4\n%stub")

    captured: Dict[str, bool] = {"called": False}

    def fake_build(path):
        captured["called"] = True
        return manifest_module.PdfCandidate(path=path)

    monkeypatch.setattr(manifest_module, "build_candidate_from_pdf", fake_build)
    monkeypatch.setattr(manifest_module.magic, "from_file", lambda *_args, **_kwargs: "application/pdf")

    index = manifest_module.scan_inbox_for_candidates(tmp_path)

    assert captured["called"] is True
    assert any(cand.basename == target.name for cand in index.candidates)
