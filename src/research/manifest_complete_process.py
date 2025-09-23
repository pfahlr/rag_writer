#!/usr/bin/env python3
"""
manifest_complete_process.py
Two-phase PDF pipeline without Selenium.

USAGE
------
# 1) Prepare download scripts and update manifest with temp filenames:
python manifest_complete_process.py preprocess \
  --manifest data/manifest.json \
  --downloader aria2c \
  --script-out data/downloads.aria2c.txt \
  --downloads-dir data/pdfs_inbox \
  [--probe-size]

# (OR for JDownloader)
python manifest_complete_process.py preprocess \
  --manifest data/manifest.json \
  --downloader jdownloader \
  --script-out data/downloads.crawljob \
  --downloads-dir data/pdfs_inbox \
  [--probe-size]

# 2) After the user runs the downloader and places PDFs into --inbox,
#    complete enrichment, embedding, and canonical rename:
python manifest_complete_process.py process \
  --manifest data/manifest.json \
  --inbox data/pdfs_inbox \
  --output data/pdfs_final

NOTES
-----
- preprocess:
  - Adds: download_id, temp_filename, expected_size (if --probe-size), download_status="pending"
  - Emits: downloads_map.json (next to script), plus aria2c.txt OR .crawljob

- process:
  - Validates PDFs, enriches (Crossref/GB/arXiv if your clients are available),
    embeds metadata (via pdf_io if available; falls back to pypdf),
    renames to [title-slug]-[year].pdf, updates manifest with final fields.

Dependencies:
  pip install requests pypdf python-magic
  # libmagic: Fedora: sudo dnf install file-libs ; Debian/Ubuntu: sudo apt-get install libmagic1
"""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import requests
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.pretty import pprint
from rich.table import Table
from rich.panel import Panel
from rich.live import Live


# --- MIME + PDF text ---
try:
    import magic  # type: ignore
    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

    class _MagicFallback:
        def from_file(self, *_args, **_kwargs):
            raise RuntimeError("python-magic is not installed")

    magic = _MagicFallback()  # type: ignore

try:
    from pypdf import PdfReader, PdfWriter
except Exception as e:
    raise SystemExit("pypdf is required. Install with: pip install pypdf\n" + str(e))

# --- enrichment clients (optional but expected in your tree) ---
HAVE_ARXIV = False
HAVE_PDFIO = False
try:
    from clients.arxiv_client import enrich_via_arxiv
    HAVE_ARXIV = True
except Exception:
    print("arxiv metadata module missing!")

try:
    from clients.crossref_client import enrich_via_crossref
except Exception:
    def enrich_via_crossref(**kwargs):
        print("crossref metadata module missing!")
        return {}
try:
    from clients.google_books_client import enrich_via_google_books
except Exception:
    def enrich_via_google_books(*args, **kwargs):
        print("google books metadata module missing!")
        return {}

try:
    from functions import pdf_io as pdfio
    HAVE_PDFIO = True
except Exception:
    print("PDFIO Not Installed!")



DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ISBN_LABELED_RE = re.compile(r"\bISBN(?:-1[03])?:?\s*([0-9Xx][0-9Xx\s\-]{8,})\b")
ISBN_CHUNK_RE = re.compile(r"\b[0-9Xx][0-9Xx\s\-]{9,18}[0-9Xx]\b")


# ----------------------- utils -----------------------
def load_manifest(path: Path) -> Tuple[List[Dict], bool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "entries" in raw:
        return raw["entries"], True
    if isinstance(raw, list):
        return raw, False
    raise SystemExit("Manifest must be a list of entries or an object with 'entries'.")


def save_manifest(path: Path, entries: List[Dict], wrapped: bool) -> None:
    out = {"entries": entries} if wrapped else entries
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "untitled"


def extract_year(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    m = re.search(r"(?:19|20|21)\d{2}", date_str)
    return m.group(0) if m else None


def propose_filename(title: Optional[str], date_str: Optional[str]) -> str:
    title_slug = slugify(title or "untitled")
    year = extract_year(date_str) or "undated"
    return f"{title_slug}-{year}.pdf"


def parse_doi_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = DOI_RE.search(url)
    if not m:
        return None
    doi = m.group(0).strip().rstrip(".,;)")
    doi = doi.replace("\\", "/")
    doi = re.sub(r"\s+", "", doi)
    return doi.lower()


def parse_isbn_from_text(text: str) -> Optional[str]:
    for rx in (ISBN_LABELED_RE, ISBN_CHUNK_RE):
        for m in rx.finditer(text or ""):
            cand = m.group(1) if m.groups() else m.group(0)
            digits = re.sub(r"(?i)isbn(?:-1[03])?:?", "", cand)
            digits = re.sub(r"[\s\-]", "", digits).strip()
            if len(digits) == 10 or len(digits) == 13:
                return digits.upper()
    return None


def normalize_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    match = DOI_RE.search(str(raw))
    if not match:
        return None
    doi = match.group(0).strip()
    doi = doi.rstrip(".,;)")
    doi = doi.replace("\\", "/")
    doi = re.sub(r"\s+", "", doi)
    return doi.lower()


def parse_doi_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = DOI_RE.search(text)
    if not match:
        return None
    return normalize_doi(match.group(0))


def strip_pdf_extension(name: str) -> str:
    lname = name.lower()
    return lname[:-4] if lname.endswith(".pdf") else lname


def compact_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^0-9a-z]", "", value.lower())


def normalize_isbn(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    digits = re.sub(r"[^0-9Xx]", "", str(raw)).upper()
    if len(digits) not in (10, 13):
        return None
    return digits


@dataclass
class PdfCandidate:
    path: Path
    metadata_doi: Optional[str] = None
    metadata_isbn: Optional[str] = None
    metadata_title: Optional[str] = None
    filename_doi: Optional[str] = None
    content_doi: Optional[str] = None
    content_title: Optional[str] = None
    filename_title_slug: Optional[str] = None
    metadata_title_slug: Optional[str] = None
    content_title_slug: Optional[str] = None
    basename: str = field(init=False)
    filename_compact: str = field(init=False)
    content_compact: Optional[str] = None
    basename_lower: str = field(init=False)
    basename_without_ext: str = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.basename = self.path.name
        stem_lower = self.path.stem.lower()
        self.filename_compact = re.sub(r"[^0-9a-z]", "", stem_lower)
        self.basename_lower = self.basename.lower()
        self.basename_without_ext = strip_pdf_extension(self.basename_lower)

        self.metadata_doi = normalize_doi(self.metadata_doi)
        self.content_doi = normalize_doi(self.content_doi)
        self.filename_doi = normalize_doi(self.filename_doi) if self.filename_doi else parse_doi_from_text(self.basename)

        self.metadata_isbn = normalize_isbn(self.metadata_isbn)

        self.metadata_title = (self.metadata_title or None)
        if self.metadata_title:
            self.metadata_title = self.metadata_title.strip() or None
        if self.metadata_title and not self.metadata_title_slug:
            self.metadata_title_slug = slugify(self.metadata_title)

        self.content_title = (self.content_title or None)
        if self.content_title:
            self.content_title = self.content_title.strip() or None
        if self.content_title and not self.content_title_slug:
            self.content_title_slug = slugify(self.content_title)

        if not self.filename_title_slug:
            self.filename_title_slug = slugify(self.path.stem)


@dataclass
class InboxIndex:
    candidates: List[PdfCandidate]
    by_basename: Dict[str, PdfCandidate]
    by_metadata_doi: Dict[str, List[PdfCandidate]]
    by_metadata_isbn: Dict[str, List[PdfCandidate]]
    by_metadata_title_slug: Dict[str, List[PdfCandidate]]
    by_filename_doi: Dict[str, List[PdfCandidate]]
    by_content_doi: Dict[str, List[PdfCandidate]]
    by_content_title_slug: Dict[str, List[PdfCandidate]]
    by_filename_title_slug: Dict[str, List[PdfCandidate]]


def build_inbox_index_from_candidates(candidates: List[PdfCandidate]) -> InboxIndex:
    by_basename: Dict[str, PdfCandidate] = {}
    by_metadata_doi: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_metadata_isbn: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_metadata_title_slug: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_filename_doi: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_content_doi: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_content_title_slug: Dict[str, List[PdfCandidate]] = defaultdict(list)
    by_filename_title_slug: Dict[str, List[PdfCandidate]] = defaultdict(list)

    for cand in candidates:
        by_basename[cand.basename] = cand
        if cand.metadata_doi:
            by_metadata_doi[cand.metadata_doi].append(cand)
        if cand.metadata_isbn:
            by_metadata_isbn[cand.metadata_isbn].append(cand)
        if cand.metadata_title_slug:
            by_metadata_title_slug[cand.metadata_title_slug].append(cand)
        if cand.filename_doi:
            by_filename_doi[cand.filename_doi].append(cand)
        if cand.content_doi:
            by_content_doi[cand.content_doi].append(cand)
        if cand.content_title_slug:
            by_content_title_slug[cand.content_title_slug].append(cand)
        if cand.filename_title_slug:
            by_filename_title_slug[cand.filename_title_slug].append(cand)

    return InboxIndex(
        candidates=list(candidates),
        by_basename=by_basename,
        by_metadata_doi={k: list(v) for k, v in by_metadata_doi.items()},
        by_metadata_isbn={k: list(v) for k, v in by_metadata_isbn.items()},
        by_metadata_title_slug={k: list(v) for k, v in by_metadata_title_slug.items()},
        by_filename_doi={k: list(v) for k, v in by_filename_doi.items()},
        by_content_doi={k: list(v) for k, v in by_content_doi.items()},
        by_content_title_slug={k: list(v) for k, v in by_content_title_slug.items()},
        by_filename_title_slug={k: list(v) for k, v in by_filename_title_slug.items()},
    )


def build_candidate_from_pdf(path: Path) -> PdfCandidate:
    metadata_doi: Optional[str] = None
    metadata_isbn: Optional[str] = None
    metadata_title: Optional[str] = None
    content_doi: Optional[str] = None
    content_title: Optional[str] = None
    content_compact: Optional[str] = None

    try:
        reader = PdfReader(str(path))
    except Exception:
        return PdfCandidate(path=path)

    raw_metadata = {}
    try:
        raw_metadata = reader.metadata or {}
    except Exception:
        raw_metadata = {}

    meta_values: List[str] = []
    for key, value in getattr(raw_metadata, "items", lambda: [])():
        if isinstance(value, str):
            meta_values.append(value)
        if not metadata_title and isinstance(value, str) and key in {"/Title", "Title", "title"}:
            metadata_title = value.strip() or None

    if meta_values:
        merged = " ".join(meta_values)
        metadata_doi = parse_doi_from_text(merged)
        metadata_isbn = parse_isbn_from_text(merged) or metadata_isbn

    if not metadata_title:
        try:
            maybe_title = getattr(reader.metadata, "title", None)
            if isinstance(maybe_title, str) and maybe_title.strip():
                metadata_title = maybe_title.strip()
        except Exception:
            pass

    text_chunks: List[str] = []
    for page_index in range(min(5, len(reader.pages))):
        try:
            text_chunks.append(reader.pages[page_index].extract_text() or "")
        except Exception:
            continue

    text_blob = "\n".join(text_chunks)
    if text_blob:
        content_doi = parse_doi_from_text(text_blob)
        if not metadata_isbn:
            metadata_isbn = parse_isbn_from_text(text_blob) or metadata_isbn
        lowered = text_blob.lower()
        content_compact = re.sub(r"[^0-9a-z]", "", lowered)
        if content_compact:
            content_compact = content_compact[:10000]

    try:
        content_title = guess_title(reader)
    except Exception:
        content_title = None

    return PdfCandidate(
        path=path,
        metadata_doi=metadata_doi,
        metadata_isbn=metadata_isbn,
        metadata_title=metadata_title,
        filename_doi=parse_doi_from_text(path.name),
        content_doi=content_doi,
        content_title=content_title,
        content_compact=content_compact,
    )


def scan_inbox_for_candidates(inbox_dir: Path) -> InboxIndex:
    candidates: List[PdfCandidate] = []
    for path in sorted(inbox_dir.glob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        is_pdf = suffix == ".pdf"

        if not is_pdf and hasattr(magic, "from_file"):
            try:
                mime = magic.from_file(str(path), mime=True)
                is_pdf = bool(mime) and str(mime).lower().startswith("application/pdf")
            except Exception:
                is_pdf = False

        if not is_pdf:
            continue

        candidates.append(build_candidate_from_pdf(path))
    return build_inbox_index_from_candidates(candidates)


def match_entry_to_candidate(entry: Dict, index: InboxIndex, claimed: Set[Path]) -> Optional[PdfCandidate]:
    def claim(candidate: Optional[PdfCandidate]) -> Optional[PdfCandidate]:
        if candidate and candidate.path not in claimed:
            claimed.add(candidate.path)
            return candidate
        return None

    def claim_from_list(candidates: List[PdfCandidate]) -> Optional[PdfCandidate]:
        for cand in candidates:
            if cand.path not in claimed:
                claimed.add(cand.path)
                return cand
        return None

    def claim_where(predicate: Callable[[PdfCandidate], bool]) -> Optional[PdfCandidate]:
        for cand in index.candidates:
            if cand.path in claimed:
                continue
            if predicate(cand):
                claimed.add(cand.path)
                return cand
        return None

    temp_name = (entry.get("temp_filename") or "").strip()
    if temp_name:
        candidate = claim(index.by_basename.get(temp_name))
        if candidate:
            return candidate

    entry_doi = normalize_doi(entry.get("doi"))
    entry_isbn = normalize_isbn(entry.get("isbn"))
    title_value = (entry.get("title") or "").strip()
    entry_title_slug = slugify(title_value) if title_value else None
    entry_doi_token = compact_token(entry_doi)
    entry_doi_suffix = entry_doi.split("/", 1)[1] if entry_doi and "/" in entry_doi else entry_doi
    entry_doi_suffix_compact = compact_token(entry_doi_suffix)
    url_basenames: Set[str] = set()
    url_basename_compact: Set[str] = set()
    for key in ("pdf_url", "scholar_url"):
        parsed = entry.get(key)
        if not parsed:
            continue
        try:
            path = urlparse(parsed).path
        except Exception:
            path = ""
        if not path:
            continue
        segments = [seg.lower() for seg in path.split("/") if seg]
        if not segments:
            continue
        for seg in segments:
            url_basenames.add(seg)
            url_basenames.add(strip_pdf_extension(seg))
            comp = compact_token(seg)
            if comp:
                url_basename_compact.add(comp)
        if len(segments) >= 2:
            for i in range(len(segments) - 1):
                combo = f"{segments[i]}.{segments[i + 1]}"
                url_basenames.add(combo)
                url_basenames.add(strip_pdf_extension(combo))
                comp = compact_token(combo)
                if comp:
                    url_basename_compact.add(comp)
        joined = ".".join(segments)
        url_basenames.add(joined)
        url_basenames.add(strip_pdf_extension(joined))
        comp = compact_token(joined)
        if comp:
            url_basename_compact.add(comp)

    if entry_doi:
        candidate = claim_from_list(index.by_metadata_doi.get(entry_doi, []))
        if candidate:
            return candidate

    if entry_isbn:
        candidate = claim_from_list(index.by_metadata_isbn.get(entry_isbn, []))
        if candidate:
            return candidate

    if entry_title_slug:
        candidate = claim_from_list(index.by_metadata_title_slug.get(entry_title_slug, []))
        if candidate:
            return candidate

    if entry_doi:
        candidate = claim_from_list(index.by_filename_doi.get(entry_doi, []))
        if candidate:
            return candidate

        candidate = claim_from_list(index.by_content_doi.get(entry_doi, []))
        if candidate:
            return candidate

    if entry_doi_suffix:
        candidate = claim_where(
            lambda cand: (
                cand.basename_lower == entry_doi_suffix
                or cand.basename_without_ext == entry_doi_suffix
                or (
                    entry_doi_suffix_compact
                    and cand.filename_compact == entry_doi_suffix_compact
                )
            )
        )
        if candidate:
            return candidate

    if url_basenames:
        candidate = claim_where(
            lambda cand: cand.basename_lower in url_basenames or cand.basename_without_ext in url_basenames
        )
        if candidate:
            return candidate

    if url_basename_compact:
        candidate = claim_where(
            lambda cand: any(
                comp and (
                    comp == cand.filename_compact
                    or comp == cand.basename_without_ext
                    or comp in cand.filename_compact
                    or cand.filename_compact in comp
                )
                for comp in url_basename_compact
            )
        )
        if candidate:
            return candidate

    token_candidates = [entry_doi_token, entry_doi_suffix_compact]
    for token in token_candidates:
        if not token or len(token) < 6:
            continue

        candidate = claim_where(
            lambda cand: bool(
                cand.filename_compact
                and len(cand.filename_compact) >= 6
                and (
                    token in cand.filename_compact
                    or cand.filename_compact in token
                )
            )
        )
        if candidate:
            return candidate

        candidate = claim_where(
            lambda cand: bool(
                cand.content_compact
                and len(cand.content_compact) >= 6
                and (
                    token in cand.content_compact
                    or cand.content_compact in token
                )
            )
        )
        if candidate:
            return candidate

    if entry_title_slug:
        candidate = claim_from_list(index.by_content_title_slug.get(entry_title_slug, []))
        if candidate:
            return candidate

        candidate = claim_from_list(index.by_filename_title_slug.get(entry_title_slug, []))
        if candidate:
            return candidate

    return None


def guess_title(reader: PdfReader) -> Optional[str]:
    try:
        t = (reader.metadata.title or "").strip() if reader.metadata else ""
        if t and t.lower() not in {"", "untitled", "unknown"}:
            return t
    except Exception:
        pass
    # crude first-page heuristic
    try:
        for i in range(min(3, len(reader.pages))):
            txt = (reader.pages[i].extract_text() or "")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for ln in lines[:15]:
                ln_clean = re.sub(r"\s+", " ", ln)
                if 5 <= len(ln_clean) <= 120:
                    low = ln_clean.lower()
                    if any(b in low for b in ("doi:", "doi.org/", "http", "www.", "issn", "abstract", "copyright", "introduction")):
                        continue
                    if re.search(r"\b[A-Z]\.\s*[A-Z]\.", ln_clean):
                        continue
                    return ln_clean
    except Exception:
        pass
    return None


def is_arxiv_source(url: Optional[str]) -> bool:
    return bool(url) and ("arxiv.org" in url.lower())


def write_pdf_metadata(pdf_path: Path, meta: Dict) -> None:
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)

    if HAVE_PDFIO and hasattr(pdfio, "write_pdf_metadata"):
        authors = meta.get("authors") or []
        authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors or "")
        doi = meta.get("doi") or ""
        if doi:
            meta["doi_url"] = f"https://doi.org/{doi}"
        core = {
            "/Title": meta.get("title") or "",
            "/Author": authors_str,
            "/Subject": json.dumps(meta, ensure_ascii=False),
            "/Keywords": ", ".join(filter(None, [meta.get("doi"), meta.get("isbn"), meta.get("publication")])),
        }
        dc = {
            "title": meta.get("title") or "",
            "creator": authors if isinstance(authors, list) else ([authors_str] if authors_str else []),
            "date": meta.get("date") or "",
            "identifier": next(
                (str(val) for val in (meta.get("doi"), meta.get("isbn"), meta.get("arXivID")) if val),
                "",
            ),
        }
        prism = {
            "doi": meta.get("doi") or "",
            "isbn": meta.get("isbn") or "",
            "publicationName": meta.get("publication") or "",
            "publicationDate": meta.get("date") or "",
            "url": meta.get("pdf_url") or "",
        }
        dcterms = {"issued": meta.get("date")}
        try:
            pdfio.write_pdf_metadata(
                pdf_path,
                core=core,
                dc=dc,
                prism=prism,
                dcterms=dcterms,
                full_meta_for_subject=meta,
            )
            return
        except Exception as e:
            print(f"[⚠️] pdf_io.write_pdf_metadata failed, falling back to pypdf: {e}")

    # Fallback: pypdf Info dictionary
    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        authors = meta.get("authors") or []
        authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors or "")
        info = {
            "/Title": meta.get("title") or "",
            "/Author": authors_str,
            "/Subject": json.dumps(meta, ensure_ascii=False),
        }
        writer.add_metadata(info)
        with open(pdf_path, "wb") as f:
            writer.write(f)
    except Exception as e:
        print(f"[⚠️] pypdf metadata write failed: {e}")


# ----------------------- retry helpers -----------------------
def parse_aria2_summary_lines(path: Path) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 4:
            continue
        status = parts[1]
        target = parts[3]
        if not status or not target:
            continue
        basename = Path(target).name
        if not basename:
            continue
        results.append((status.upper(), basename))
    return results


def build_retry_batch_from_manifest(
    entries: List[Dict], failed_basenames: Set[str], downloads_dir: Path
) -> Tuple[List[str], int, List[str]]:
    by_temp = {}
    for entry in entries:
        temp = (entry.get("temp_filename") or "").strip()
        if temp:
            by_temp[temp] = entry

    lines: List[str] = []
    matched = 0
    missing: List[str] = []

    seen: Set[str] = set()
    for base in sorted(failed_basenames):
        if base in seen:
            continue
        seen.add(base)
        entry = by_temp.get(base)
        if entry and (entry.get("pdf_url") or "").strip():
            url = entry["pdf_url"].strip()
            lines.extend([url, f"  out={base}", f"  dir={str(downloads_dir)}"])
            matched += 1
            entry["download_status"] = "retry"
            entry.pop("failure_reason", None)
        else:
            missing.append(base)

    return lines, matched, missing


def safe_enrich_crossref(
    enrich_fn,
    doi: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    timeout: float = 25.0,
    tries: int = 4,
    backoff: float = 1.5,
    ua: str = "Content-Expanse/1.0 (mailto:pfahlr@gmail.com)",
):
    last: Optional[Exception] = None
    max_tries = max(1, tries)
    for attempt in range(max_tries):
        try:
            return enrich_fn(
                doi=doi,
                title=title,
                author=author,
                headers={"User-Agent": ua},
                timeout=timeout,
            )
        except Exception as exc:
            last = exc
            time.sleep(backoff * (2 ** attempt))
    print(f"[⚠️] Crossref enrichment failed after {tries} attempts: {last}")
    return {}


def retry_from_aria2_log(
    manifest_path: Path,
    aria2_log: Path,
    script_out: Path,
    downloads_dir: Path,
    update_manifest: bool = True,
) -> None:
    entries, wrapped = load_manifest(manifest_path)
    rows = parse_aria2_summary_lines(aria2_log)
    fail_status = {"ERR", "ERROR", "NG", "FAILED"}
    failed_basenames = {basename for status, basename in rows if status in fail_status}

    total_failed = len(failed_basenames)
    if not failed_basenames:
        print("[retry] No failures found.")
        return

    lines, matched, missing = build_retry_batch_from_manifest(entries, failed_basenames, downloads_dir)

    if not lines:
        print("[retry] No failed items matched manifest.")
        if missing:
            print("[retry] Unmatched:")
            for base in missing:
                print(f"  - {base}")
        return

    script_out.parent.mkdir(parents=True, exist_ok=True)
    script_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if update_manifest:
        save_manifest(manifest_path, entries, wrapped)

    print(f"[retry] Total failed in log: {total_failed}")
    print(f"[retry] Matched: {matched} | Unmatched: {len(missing)}")
    if missing:
        print("[retry] Unmatched:")
        for base in missing:
            print(f"  - {base}")
    print(f"[retry] Wrote: {script_out}")


# ----------------------- prune-downloads -----------------------
def prune_downloads(manifest_path: Path, inbox_dir: Path):
    data, wrapped = load_manifest(manifest_path)
    
    display_table = Table.grid()
    progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    job = progress.add_task("[red] entries processed", total=len(data))
    display_table.add_row(
        Panel.fit(progress, title="Pruning HTML downloads", border_style="purple", padding=(1,1)),
        #Panel.fit(progress.console, title="Log", border_style="blue", padding=(1,1))
    )
    with Live(display_table, refresh_per_second=30) as live:
        progress.console.print(f"Read manifest: got {len(data)} entries...")
        progress.console.print(f"downloads stored in: {inbox_dir}")
        for i in range(0, len(data)):
            progress.update(job, completed=i)
            if 'temp_filename' in data[i] and data[i]['temp_filename'] != "":
                filename = data[i]['temp_filename']
                temp_filepath = Path(str(inbox_dir)+'/'+str(filename))
                if os.path.exists(temp_filepath):
                    mime_type = None
                    if HAVE_MAGIC:
                        try:
                            mime_type = magic.from_file(temp_filepath, mime=True)
                            progress.console.print(f"[🧿] {temp_filepath} is type: {mime_type}")
                        except Exception as e:
                            progress.console.print(f"[⛔] ERROR:{e}")
                            continue

                    if mime_type is not None and mime_type != "application/pdf":
                        try:
                            progress.console.print(f"[🚫] unlinking {filename}")
                            temp_filepath.unlink()
                            progress.console.print(f"[⚠️] setting 'download_status' in manifest to 'E01_bad_mimetype'")
                            data[i]['download_status'] = 'E01_bad_mimetype'
                        except Exception as e:
                            progress.console.print(f"[⛔] ERROR:{e}")
                else:
                    progress.console.print(f"[{i}] checking {temp_filepath} does not exist")
        progress.update(job, completed=(i+1))
        progress.console.print(f"✅ DONE! 🎆...🎇...🎉...🌋...🎊...🌠")

    save_manifest(manifest_path, data, True)


# ----------------------- preprocess -----------------------
def preprocess(
    manifest_path: Path,
    script_out: Path,
    downloads_dir: Path,
    probe_size: bool,
) -> None:
    entries, wrapped = load_manifest(manifest_path)
    downloads_dir.mkdir(parents=True, exist_ok=True)


    # Ensure stable download_id/temp_filename for any entries that already have them
    def ensure_id(i: int, e: Dict) -> str:
        if e.get("download_id"):
            return e["download_id"]
        # try to make it reasonably stable: index + short uuid
        did = f"DL-{i:05d}-{uuid.uuid4().hex[:6]}"
        e["download_id"] = did
        return did

    lines_aria2: List[str] = []
    crawls: List[str] = []
    dlmap: List[Dict] = []

    sess = requests.Session() if probe_size else None
    entries_count = len(entries)
    # Progress bar output setup
    display_table = Table.grid()
    progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    job = progress.add_task("[pink] entries processed", total=entries_count)
    display_table.add_row(
        Panel.fit(progress, title="Creating Download Manifest", border_style="blue", padding=(1,1))
    )

    with Live(display_table, refresh_per_second=30) as live:

        for i, e in enumerate(entries):
            progress.update(job, completed=i)

            url = (e.get("pdf_url") or "").strip()
            if not url:
                progress.console.print(f"[⛔] no pdf_url found for record: {i}")
                continue
            if str(e.get("download_status") or "").lower() == "done":
                progress.console.print(f"[💓] pdf already processed: {i}")
                continue

            did = ensure_id(i, e)
            # temp filename we expect the downloader to save
            temp_filename = f"{did}.pdf"
            progress.console.print(f"[✅] setting temp_filename = {temp_filename} for record {i}")
            e["temp_filename"] = temp_filename
            e.setdefault("download_status", "pending")
            
            expected_size = None
            if probe_size:
                try:
                    progress.console.print(f"[❔] requesting headers of download target")
                    r = sess.head(url, allow_redirects=True, timeout=20)
                    if "content-length" in r.headers:
                        expected_size = int(r.headers["content-length"])
                        progress.console.print(f"[🧿] got content-length {r.headers['content-length']}")
                except Exception:
                    expected_size = None

            dlmap.append({
                "id": did,
                "url": url,
                "temp_filename": temp_filename,
                "expected_size": expected_size,
            })

            # Emit downloader-specific batch
            #if downloader == "aria2c":

            # aria2c input file format: URL newline + indented "out="
            # Also set dir= to downloads_dir so the user can run aria2c -i file
            progress.console.print(f"[✏️] adding fields for aria2c")
            lines_aria2.append(url)
            lines_aria2.append(f"  out={temp_filename}")
            lines_aria2.append(f"  dir={str(downloads_dir)}")
            #elif downloader == "jdownloader":
            
            # JDownloader .crawljob: key=value per job, blank line between jobs
            # Minimal keys: text, downloadFolder, filename, enabled, autoStart
            progress.console.print(f"[✏️] adding fields for jdownloader")
            crawls.append(
                "\n".join([
                    f"text={url}",
                    f"downloadFolder={str(downloads_dir)}",
                    f"filename={temp_filename}",
                    "enabled=true",
                    "autoStart=false",
                    "autoConfirm=false",
                    "forcedStart=false",
                    "extractAfterDownload=false",
                    "overwritePackagizerEnabled=true",
                ])
            )
            #else:
            #    raise SystemExit("--downloader must be one of: aria2c, jdownloader")

        # Write script/batch
        script_out.parent.mkdir(parents=True, exist_ok=True)
        #if downloader == "aria2c":
        counter_tmp = 1
        aria2c_script = Path(str(script_out)+'/aria2c.txt')
        
        while aria2c_script.exists():
            aria2c_script = Path(str(script_out)+'/aria2c.txt')
            aria2c_script = Path(str(aria2c_script)+'.'+str(counter_tmp))
            counter_tmp = counter_tmp + 1
        aria2c_script.write_text("\n".join(lines_aria2) + "\n", encoding="utf-8")
        progress.console.print(f"[☑️] Wrote aria2c batch → {aria2c_script}")
        progress.console.print(f"Run example: aria2c -i '{script_out}' --check-certificate=false")
        
        #else:
        counter_tmp = 1
        jdownloader_script = Path(str(script_out)+'/.crawljob')
        while jdownloader_script.exists():
            jdownloader_script = Path(str(script_out)+'/.crawljob')
            jdownloader_script = Path(str(jdownloader_script)+'.'+str(counter_tmp))
            counter_tmp = counter_tmp + 1        
        jdownloader_script.write_text("\n\n".join(crawls) + "\n", encoding="utf-8")
        progress.console.print(f"[☑️] Wrote JDownloader .crawljob → {jdownloader_script}")
        progress.console.print("Import in JDownloader: LinkGrabber menu → Add New Links → Load crawljob file")

        # Write downloads_map.json next to script
        map_path = script_out.with_suffix(".downloads_map.json")
        map_path.write_text(json.dumps(dlmap, ensure_ascii=False, indent=2), encoding="utf-8")
        progress.console.print(f"[☑️] Wrote downloads_map.json → {map_path}")

        # Save updated manifest
        save_manifest(manifest_path, entries, wrapped)
        progress.console.print(f"[☑️] Updated manifest with temp filenames and status=pending → {manifest_path}")
        progress.console.print(f"[✅] DONE! 🎆...🎇...🎉...🌋...🎊...🌠")


# ----------------------- process -----------------------
def process_pipeline(
    manifest_path: Path,
    inbox_dir: Path,
    output_dir: Path,
    *,
    crossref_timeout: float = 25.0,
    crossref_tries: int = 4,
    crossref_backoff: float = 1.5,
    crossref_ua: str = "Content-Expanse/1.0 (mailto:none@example.com)",
) -> None:


    data, wrapped = load_manifest(manifest_path)

    entries_count = len(data)

    output_dir.mkdir(parents=True, exist_ok=True)

    success_log = output_dir / f"process-success.{int(time.time())}.jsonl"
    fail_log = output_dir / f"process-fail.{int(time.time())}.jsonl"

    succ_fp = success_log.open("a", encoding="utf-8")
    fail_fp = fail_log.open("a", encoding="utf-8")

    processed = 0

    # Progress bar output setup
    display_table = Table.grid()
    progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    job = progress.add_task("[blue] entries processed", total=entries_count)
    display_table.add_row(
        Panel.fit(progress, title="📇 Completing Metadata and Writing Ready-for-Indexing PDFs ", border_style="green", padding=(1,1))
    )
    index = scan_inbox_for_candidates(inbox_dir)
    claimed: Set[Path] = set()

    with Live(display_table, refresh_per_second=30) as live:
        try:
            for entry_idx in range(entries_count):
                progress.update(job, completed=entry_idx)
                entry = data[entry_idx]

                url = (entry.get("pdf_url") or "").strip()
                if not url:
                    continue

                if str(entry.get("download_status") or "").lower() == "done":
                    continue

                manifest_temp_name = entry.get("temp_filename")

                candidate = match_entry_to_candidate(entry, index, claimed)
                if not candidate:
                    label = manifest_temp_name or entry.get("title") or entry.get("doi") or f"entry_{entry_idx}"
                    progress.console.print(f"[SKIP] no downloaded PDF matched manifest entry {label}")
                    continue

                src = candidate.path
                if not src.exists():
                    progress.console.print(f"[⚠️] matched file missing on disk: {src}")
                    entry.setdefault("download_status", "pending")
                    continue

                temp_name = candidate.basename
                entry["matched_filename"] = temp_name

                # MIME verify
                mime = None
                if HAVE_MAGIC:
                    try:
                        mime = magic.from_file(str(src), mime=True)
                    except Exception as ex:
                        progress.console.print(f"[⚠️] MIME check failed for {temp_name}: {ex}")

                if mime is not None and mime != "application/pdf":
                    progress.console.print(f"[❌] MIME type for {temp_name}: {mime} - deleting")
                    src.unlink()
                    entry["download_status"] = "failed"
                    entry["failure_reason"] = f"non_pdf_mime:{mime}"
                    fail_fp.write(json.dumps({"entry": entry, "reason": entry["failure_reason"]}, ensure_ascii=False) + "\n")
                    continue

                # Open and extract skim text
                try:
                    reader = PdfReader(str(src))
                except Exception as ex:
                    progress.console.print(f"[⚠️] failed to open downloaded file: {src} - Exception:{ex}")
                    entry["download_status"] = "failed"
                    entry["failure_reason"] = f"pdf_open_error:{ex}"
                    fail_fp.write(json.dumps({"entry": entry, "reason": entry["failure_reason"]}, ensure_ascii=False) + "\n")
                    continue

                meta: Dict = {
                    "title": entry.get("title"),
                    "authors": entry.get("authors"),
                    "date": entry.get("date"),
                    "publication": "",
                    "arxivid":"",
                    "doi": entry.get("doi"),
                    "isbn": entry.get("isbn"),
                    "pdf_url": url,
                    "scholar_url": entry.get("scholar_url"),
                    "original_filename": temp_name,
                    "source": "manifest",
                }

                if candidate.metadata_doi and not meta.get("doi"):
                    entry["doi"] = meta["doi"] = candidate.metadata_doi
                if candidate.content_doi and not meta.get("doi"):
                    entry["doi"] = meta["doi"] = candidate.content_doi
                if candidate.metadata_isbn and not meta.get("isbn"):
                    entry["isbn"] = meta["isbn"] = candidate.metadata_isbn
                if candidate.metadata_title and not meta.get("title"):
                    entry["title"] = meta["title"] = candidate.metadata_title
                if candidate.content_title and not meta.get("title"):
                    entry["title"] = meta["title"] = candidate.content_title

                arxivid_regex = re.compile(r"(?:\d{4}\.\d{4,5})")
                match = False
                if "arxiv.org" in meta['pdf_url']:
                    match = arxivid_regex.search(meta['pdf_url'])
                if not match and "arxiv.org" in meta['scholar_url']:
                    match = arxivid_regex.search(meta['scholar_url'])
                if match:
                    meta['arxivid']=match.group(0)
                    entry['arxivid']=match.group(0)
                    progress.console.print(f"[🔖] updated arXivID field {entry['arxivid']}")

                # DOI from URLs fallback
                if not meta.get("doi"):
                    for u in (entry.get("pdf_url"), entry.get("scholar_url")):
                        cand = parse_doi_from_url(u or "")
                        if cand:
                            progress.console.print(f"[🔖] updated doi field from url pattern match {cand}")
                            entry['doi'] = meta["doi"] = cand
                            break

                # First pages text for ISBN/title fallback
                text = ""
                try:
                    for page_index in range(min(5, len(reader.pages))):
                        text += "\n" + (reader.pages[page_index].extract_text() or "")
                except Exception:
                    pass

                if not meta.get("isbn"):
                    cand_isbn = parse_isbn_from_text(text)
                    if cand_isbn:
                        entry['isbn'] = meta["isbn"] = cand_isbn
                        progress.console.print(f"[🔖] updated isbn field from body text match {cand_isbn}")

                if not meta.get("title"):
                    entry['title'] = meta["title"] = guess_title(reader)
                    progress.console.print(f"[🔖] updated title field from content match guess_title() function {meta['title']}")

                # Enrichment
                try:
                    if is_arxiv_source(entry.get("pdf_url")) or is_arxiv_source(entry.get("scholar_url")):
                        if HAVE_ARXIV:
                            am = enrich_via_arxiv(entry.get("pdf_url") or entry.get("scholar_url"))
                            if am:
                                for k, v in am.items():
                                    if v and not meta.get(k):
                                        meta[k] = v
                except Exception as ex:
                    progress.console.print(f"[⚠️] arXiv enrichment failed: {ex}")

                cr = safe_enrich_crossref(
                    enrich_via_crossref,
                    doi=meta.get("doi"),
                    title=meta.get("title"),
                    author=(
                        meta.get("authors")[0]
                        if isinstance(meta.get("authors"), list) and meta.get("authors")
                        else ""
                    ),
                    timeout=crossref_timeout,
                    tries=crossref_tries,
                    backoff=crossref_backoff,
                    ua=crossref_ua,
                )
                if cr:
                    for k, v in cr.items():
                        if v and not meta.get(k):
                            meta[k] = v

                if meta.get("isbn"):
                    try:
                        gb = enrich_via_google_books(meta["isbn"])
                        if gb:
                            for k, v in gb.items():
                                if v and not meta.get(k):
                                    meta[k] = v
                    except Exception as ex:
                        progress.console.print(f"[⚠️] Google Books enrichment failed: {ex}")

                # Final filename and move
                final_name = propose_filename(meta.get("title"), meta.get("date"))
                final_path = output_dir / final_name
                if final_path.exists():
                    final_path = output_dir / f"{final_path.stem}--{int(time.time())}{final_path.suffix}"

                try:
                    src.replace(final_path)
                except Exception as ex:
                    entry["download_status"] = "failed"
                    entry["failure_reason"] = f"move_error:{ex}"
                    progress.console.print(f"move_error:{ex}")
                    fail_fp.write(json.dumps({"entry": entry, "reason": entry["failure_reason"]}, ensure_ascii=False) + "\n")
                    data[entry_idx] = entry
                    continue

                # Embed metadata, write sidecar, update entry
                try:
                    write_pdf_metadata(final_path, meta)
                except Exception as ex:
                    progress.console.print(f"[⚠️] metadata embed failed: {ex}")

                sidecar = final_path.with_suffix(".meta.json")
                sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                entry["local_path"] = str(final_path)
                entry["final_filename"] = final_path.name
                entry["file_sha256"] = sha256(final_path)
                entry["download_status"] = "done"
                entry.pop("failure_reason", None)

                succ_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
                processed += 1
                if manifest_temp_name and manifest_temp_name != temp_name:
                    display_label = f"{manifest_temp_name} ({temp_name})"
                else:
                    display_label = temp_name
                progress.console.print(f"[✅] {display_label} → {final_path.name}")

        finally:
            progress.update(job, completed=entries_count)
            succ_fp.close()
            fail_fp.close()
            save_manifest(manifest_path, data, wrapped=True)
            progress.console.print(f"[⛴️] Updated manifest → {manifest_path}")
            progress.console.print(f"[🛫] Success log: {success_log}")
            progress.console.print(f"[🚀] Fail log   : {fail_log}")
            progress.console.print(f"[🛸] Completed {processed} entries.")
            progress.console.print(f"[✅] DONE! 🎆...🎇...🎉...🌋...🎊...🌠")



# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Two-phase PDF pipeline (preprocess → process) without Selenium.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Generate download script (aria2c/jdownloader) and update manifest.")
    sp.add_argument("--manifest", required=True, type=Path)
    sp.add_argument("--script-out", required=True, type=Path, help="Path to write aria2c.txt or .crawljob")
    sp.add_argument("--downloads-dir", required=True, type=Path, help="Directory where the downloader will place PDFs.")
    sp.add_argument("--probe-size", action="store_true", help="HEAD each URL to record expected_size.")

    sp2 = sub.add_parser("process", help="Verify/enrich/rename PDFs downloaded to --inbox and update manifest.")
    sp2.add_argument("--manifest", required=True, type=Path)
    sp2.add_argument("--inbox", required=True, type=Path, help="Folder containing downloaded temp PDFs (from preprocess).")
    sp2.add_argument("--output", required=True, type=Path, help="Final folder for renamed PDFs + sidecars + logs.")
    sp2.add_argument("--crossref-timeout", type=float, default=25.0)
    sp2.add_argument("--crossref-tries", type=int, default=4)
    sp2.add_argument("--crossref-backoff", type=float, default=1.5)
    sp2.add_argument("--crossref-ua", type=str, default="Content-Expanse/1.0 (mailto:pfahlr@gmail.com)")

    pd = sub.add_parser("prune-downloads", help="Prune non-pdfs from downloaded files.")
    pd.add_argument("--manifest", required=True, type=Path)
    pd.add_argument("--inbox", required=True, type=Path, help="Folder containing downloaded temp PDFs (from preprocess).")

    sr = sub.add_parser("retry", help="Parse aria2 log; regenerate aria2 batch for failed items.")
    sr.add_argument("--manifest", required=True, type=Path)
    sr.add_argument("--aria2-log", required=True, type=Path)
    sr.add_argument("--script-out", required=True, type=Path)
    sr.add_argument("--downloads-dir", required=True, type=Path)
    sr.add_argument("--no-manifest-update", action="store_true")

    args = ap.parse_args()

    if args.cmd == "preprocess":
        preprocess(args.manifest, args.script_out, args.downloads_dir, args.probe_size)
    elif args.cmd == "process":
        process_pipeline(
            args.manifest,
            args.inbox,
            args.output,
            crossref_timeout=args.crossref_timeout,
            crossref_tries=args.crossref_tries,
            crossref_backoff=args.crossref_backoff,
            crossref_ua=args.crossref_ua,
        )
    elif args.cmd == "prune-downloads":
        prune_downloads(
            args.manifest,
            args.inbox
        )
    # args.cmd == 'retry'
    else:
        retry_from_aria2_log(
            args.manifest,
            args.aria2_log,
            args.script_out,
            args.downloads_dir,
            update_manifest=not args.no_manifest_update,
        )


if __name__ == "__main__":
    main()
