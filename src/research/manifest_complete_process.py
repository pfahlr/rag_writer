#!/usr/bin/env python3
"""
---
Headless-download PDFs from a manifest, verify/enrich metadata, rename, and persist results.
---
Rick Pfahl <pfahlr@gmail.com>
September 2025
---

Usage:
  python fetch_and_enrich_pdfs.py --manifest /path/to/manifest.json --output /path/to/output_dir

Manifest entry example (list of objects):
{
  "title": "An integrated model of consumers' intention to buy second-hand clothing",
  "authors": ["KY Koay", "CW Cheah", "HS Lom"],
  "date": "2022",
  "publication": "International Journal of Retail & …",
  "doi": "10.1108/IJRDM-10-2021-0470",
  "isbn": "",
  "pdf_url": "https://example.com/file.pdf",
  "scholar_url": "https://publisher/page"
}

What this script does for each entry:
  1) Download the file at pdf_url (headless Chrome) into a per-entry temp dir.
  2) Verify with python-magic that mimetype == application/pdf; otherwise delete + skip.
  3) Prefer manifest metadata; fill missing fields by scanning the PDF (title/doi/isbn).
  4) Enrich/verify via Crossref (for DOI) and Google Books (for ISBN).
  5) Rename file per your rules, avoiding collisions (append --<unix_ts> if needed).
  6) Save sidecar metadata JSON and append to output/metadata.jsonl.
      Successes → --output/download-success.manifest.<unixtime>.jsonl
  7) Failures  → --output/download-failed.manifest.<unixtime>.jsonl (and per-entry sidecars)
  8) Embed core, DC, and PRISM metadata into the PDF via src/research/functions/pdf_io.py,
     - stash a json.dumps(meta) into PDF Subject.
     - Crossref/GB enrichment moved into src/research/clients/* with improved logic.
     - arXiv URLs auto-enriched via your arxiv client if present.
  9) Filenames: [title-slug]-[year].pdf (DOI/ISBN go into metadata instead).

Usage:
  python fetch_and_enrich_pdfs.py --manifest /path/to/manifest.json --output /path/to/output_dir

Dependencies:
  pip install selenium webdriver-manager python-magic-bin pypdf requests
  # Linux libmagic (Fedora): sudo dnf install file-libs
  # Linux libmagic (Deb/Ubuntu): sudo apt-get install libmagic1
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# --- selenium / driver ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAVE_WDM = True
except Exception:
    HAVE_WDM = False

# --- MIME + PDF text ---
try:
    import magic  # python-magic
except Exception as e:
    sys.exit("python-magic is required. Try: pip install python-magic-bin\n" + str(e))

try:
    from pypdf import PdfReader, PdfWriter
except Exception as e:
    sys.exit("pypdf is required. Install with: pip install pypdf\n" + str(e))

# --- enrichment clients (your new/updated modules) ---
# Crossref + Google Books live in src/research/clients; arXiv client is optional
try:
    from src.research.clients.crossref_client import (
        fetch_crossref_by_doi,
        search_crossref,
        enrich_via_crossref,
    )
except Exception as e:
    sys.exit("Missing src/research/clients/crossref_client.py with the required functions.\n" + str(e))

try:
    from src.research.clients.google_books_client import enrich_via_google_books
except Exception as e:
    sys.exit("Missing src/research/clients/google_books_client.py.\n" + str(e))

try:
    from src.research.clients.arxiv_client import enrich_via_arxiv  # your existing client
    HAVE_ARXIV = True
except Exception:
    HAVE_ARXIV = False

# PDF XMP/core metadata writer
try:
    from src.research.functions import pdf_io as pdfio
    HAVE_PDFIO = True
except Exception:
    HAVE_PDFIO = False


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ISBN_LABELED_RE = re.compile(r"\bISBN(?:-1[03])?:?\s*([0-9Xx][0-9Xx\s\-]{8,})\b")
ISBN_CHUNK_RE = re.compile(r"\b[0-9Xx][0-9Xx\s\-]{9,18}[0-9Xx]\b")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "untitled"


def parse_doi_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = DOI_RE.search(url)
    if m:
        doi = m.group(0)
        doi = doi.strip().rstrip(".,;)")
        doi = doi.replace("\\", "/")
        doi = re.sub(r"\s+", "", doi)
        return doi.lower()
    return None


def isbn10_is_valid(isbn10: str) -> bool:
    if len(isbn10) != 10:
        return False
    s = 0
    for i, ch in enumerate(isbn10):
        v = 10 if ch.upper() == "X" else (int(ch) if ch.isdigit() else None)
        if v is None:
            return False
        s += (10 - i) * v
    return s % 11 == 0


def isbn13_is_valid(isbn13: str) -> bool:
    if len(isbn13) != 13 or not isbn13.isdigit():
        return False
    s = sum((int(d) * (1 if i % 2 == 0 else 3)) for i, d in enumerate(isbn13))
    return s % 10 == 0


def normalize_isbn(raw: str) -> Optional[str]:
    if not raw:
        return None
    digits = re.sub(r"(?i)isbn(?:-1[03])?:?", "", raw)
    digits = re.sub(r"[\s\-]", "", digits).strip()
    if len(digits) == 10:
        return digits.upper() if isbn10_is_valid(digits.upper()) else None
    if len(digits) == 13:
        return digits if isbn13_is_valid(digits) else None
    return None


def parse_isbn_from_text(text: str) -> Optional[str]:
    for rx in (ISBN_LABELED_RE, ISBN_CHUNK_RE):
        for m in rx.finditer(text or ""):
            cand = normalize_isbn(m.group(1) if m.groups() else m.group(0))
            if cand:
                return cand
    return None


def guess_title(reader: PdfReader) -> Optional[str]:
    try:
        t = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        t = ""
    if t and t.lower() not in {"", "untitled", "unknown"}:
        return t

    try:
        for i in range(min(3, len(reader.pages))):
            txt = (reader.pages[i].extract_text() or "")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for ln in lines[:15]:
                ln_clean = re.sub(r"\s+", " ", ln)
                if 5 <= len(ln_clean) <= 120:
                    low = ln_clean.lower()
                    if any(b in low for b in ("doi:", "doi.org/", "http", "www.", "issn", "abstract", "introduction", "copyright")):
                        continue
                    if re.search(r"\b[A-Z]\.\s*[A-Z]\.", ln_clean):
                        continue
                    return ln_clean
    except Exception:
        pass
    return None


def extract_year(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    m = re.search(r"(?:19|20|21)\d{2}", date_str)
    return m.group(0) if m else None


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem, ext = path.stem, path.suffix
    while True:
        cand = path.with_name(f"{stem}--{int(time.time())}{ext}")
        if not cand.exists():
            return cand
        time.sleep(1)


def propose_filename(title: Optional[str], date_str: Optional[str]) -> str:
    title_slug = slugify(title or "untitled")
    year = extract_year(date_str) or "undated"
    return f"{title_slug}-{year}.pdf"


def build_driver(base_download_dir: Path) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(base_download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    })
    service = Service(ChromeDriverManager().install()) if HAVE_WDM else Service()
    driver = webdriver.Chrome(service=service, options=opts)
    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": str(base_download_dir)
        })
    except Exception:
        pass
    return driver


def wait_for_download(dir_path: Path, before: set, timeout_s: int = 180) -> Optional[Path]:
    start = time.time()
    while time.time() - start < timeout_s:
        time.sleep(0.5)
        current = set(os.listdir(dir_path))
        new = [Path(dir_path, f) for f in (current - before) if not f.endswith(".crdownload")]
        if new:
            new.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return new[0]
        # in-progress still?
        if any(str(p).endswith(".crdownload") for p in Path(dir_path).glob("*.crdownload")):
            continue
    return None


def write_pdf_metadata(pdf_path: Path, meta: Dict) -> None:
    """
    Embed metadata into the PDF. Prefer your pdf_io helpers if available; else fall back to core info with pypdf.
    - Core: Title, Author(s), Subject=json.dumps(meta), (optionally) ModDate
    - DC/PRISM: doi/isbn placed via pdf_io if available
    """
    if HAVE_PDFIO and hasattr(pdfio, "write_pdf_metadata"):
        # Expected flexible signature:
        # write_pdf_metadata(pdf_path, core:dict, dc:dict=None, prism:dict=None)
        authors = meta.get("authors") or []
        authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors or "")

        filename = os.path.basename(pdf_path)

        doi = meta.get("doi") or ""

        if doi != "":
           meta['doi_url'] = f"https://doi.org/{doi}"

        core = {
            "Title": meta.get("title") or "",
            "Author": authors_str,
            "Subject": json.dumps(meta, ensure_ascii=False),
            "Keywords": ", ".join(filter(None, [meta.get("doi"), meta.get("isbn"), meta.get("publication")]))
        }
        dc = {
            "title": meta.get("title") or "",
            "creator": authors if isinstance(authors, list) else [authors_str] if authors_str else [],
            "date": meta.get("date") or "",
            "identifier": list(filter(None, [meta.get("doi_url"), meta.get("isbn")]))
        }
        prism = {
            "doi": meta.get("doi") or "",
            "isbn": meta.get("isbn") or "",
            "publicationName": meta.get("publication") or "",
            "publicationDate": meta.get("date") or "",
            "url": meta.get("pdf_url") or "",
        }

        dcterms={"issued": meta.get("date")}

        full_meta_for_subject = meta

        try:
            pdfio.write_pdf_metadata(str(filename), core=core, dc=dc, prism=prism, dcterms=dcterms, full_meta_for_subject=full_meta_for_subject)
            return
        except Exception as e:
            print(f"[WARN] pdf_io.write_pdf_metadata failed, falling back to pypdf: {e}")

    # Fallback: basic Info dictionary via pypdf
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
        print(f"[WARN] pypdf metadata write failed: {e}")


def prefer_manifest(meta: Dict) -> Dict:
    """Trim empty strings to None for cleaner merging."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, str):
            out[k] = v.strip() or None
        else:
            out[k] = v if v else None
    return out


def is_arxiv_source(url: str) -> bool:
    return bool(url) and ("arxiv.org" in url.lower())


def process_entry(entry: Dict, out_dir: Path, driver: webdriver.Chrome,
                  successes_fp, failures_fp, ts: int) -> None:
    # Prepare a per-entry temp dir so we can detect the new file cleanly
    temp_dir = out_dir / f"_tmp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = (entry.get("pdf_url") or "").strip()
    scholar_url = (entry.get("scholar_url") or "").strip()
    if not pdf_url:
        fail_payload = {**entry, "failure_reason": "missing_pdf_url", "stage": "pre-download"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[SKIP] No pdf_url in manifest.")
        return

    print(f"[FETCH] {pdf_url}")
    before = set(os.listdir(temp_dir))
    try:
        driver.get(pdf_url)
    except WebDriverException as e:
        fail_payload = {**entry, "failure_reason": f"selenium_navigation_error: {e}", "stage": "navigate"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    dl_path = wait_for_download(temp_dir, before, timeout_s=180)
    if not dl_path or not dl_path.exists():
        fail_payload = {**entry, "failure_reason": "download_timeout_or_missing_file", "stage": "download"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Verify it's a PDF
    try:
        mime = magic.from_file(str(dl_path), mime=True)
    except Exception as e:
        mime = None
        print(f"[ERROR] MIME check failed: {e}")

    if mime != "application/pdf":
        print(f"[DELETE] Not a PDF (mime={mime}): {dl_path.name}")
        try:
            dl_path.unlink()
        except Exception:
            pass
        fail_payload = {**entry, "failure_reason": f"non_pdf_mime:{mime}", "stage": "verify"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Open PDF for text/title fallback
    try:
        reader = PdfReader(str(dl_path))
    except Exception as e:
        print(f"[ERROR] Failed to open downloaded PDF: {e}")
        try:
            dl_path.unlink()
        except Exception:
            pass
        fail_payload = {**entry, "failure_reason": f"pdf_open_error:{e}", "stage": "open"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Build starting metadata (manifest is preferred)
    meta: Dict = prefer_manifest({
        "title": entry.get("title"),
        "authors": entry.get("authors"),
        "date": entry.get("date"),
        "publication": entry.get("publication"),
        "doi": entry.get("doi"),
        "isbn": entry.get("isbn"),
        "pdf_url": pdf_url,
        "scholar_url": scholar_url,
        "original_filename": dl_path.name,
        "source": "manifest",
    })

    # Try DOI from URLs if missing
    if not meta.get("doi"):
        for u in (pdf_url, scholar_url):
            cand = parse_doi_from_url(u or "")
            if cand:
                meta["doi"] = cand
                break

    # Extract minimal text for DOI/ISBN/title fallback
    text = ""
    try:
        for i in range(min(5, len(reader.pages))):
            text += "\n" + (reader.pages[i].extract_text() or "")
    except Exception:
        pass

    if not meta.get("isbn"):
        cand_isbn = parse_isbn_from_text(text)
        if cand_isbn:
            meta["isbn"] = cand_isbn
    if not meta.get("title"):
        meta["title"] = guess_title(reader)

    # Enrichment strategy:
    # - arXiv URL? Prefer arXiv enrichment.
    # - Else Crossref (DOI first; else title/author search).
    # - If ISBN present (or still missing title), also try Google Books.
    if is_arxiv_source(pdf_url) or is_arxiv_source(scholar_url):
        if HAVE_ARXIV:
            try:
                meta = {**meta, **{k: v for k, v in (enrich_via_arxiv(pdf_url or scholar_url) or {}).items() if v}}
            except Exception as e:
                print(f"[WARN] arXiv enrichment failed: {e}")

    # Crossref (uses your new clients)
    try:
        meta = {**meta, **{k: v for k, v in (enrich_via_crossref(doi=meta.get("doi"), title=meta.get("title"),
                                                                 author=(meta.get("authors")[0] if isinstance(meta.get("authors"), list) and meta.get("authors") else "")) or {}).items() if v}}
    except Exception as e:
        print(f"[WARN] Crossref enrichment failed: {e}")

    # Google Books (ISBN)
    if meta.get("isbn"):
        try:
            gb = enrich_via_google_books(meta["isbn"])
            if gb:
                for k, v in gb.items():
                    if not meta.get(k) and v:
                        meta[k] = v
        except Exception as e:
            print(f"[WARN] Google Books enrichment failed: {e}")

    # Final file name [title-slug]-[year].pdf
    final_name = propose_filename(meta.get("title"), meta.get("date"))
    final_path = ensure_unique(out_dir / final_name)

    # Move to output
    try:
        shutil.move(str(dl_path), str(final_path))
    except Exception as e:
        print(f"[ERROR] Failed to move file to output: {e}")
        try:
            dl_path.unlink()
        except Exception:
            pass
        fail_payload = {**entry, "failure_reason": f"move_error:{e}", "stage": "finalize"}
        failures_fp.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
        sidecar = out_dir / f"failed--{uuid.uuid4().hex}.meta.json"
        sidecar.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Clean temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Merge filename/path
    meta["final_filename"] = final_path.name
    meta["saved_path"] = str(final_path)

    # Embed metadata (core + DC/PRISM) and write sidecar
    write_pdf_metadata(final_path, meta)
    sidecar = final_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append to success manifest (JSONL)
    successes_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[OK] {meta.get('original_filename')} → {meta.get('final_filename')}")


def main():
    ap = argparse.ArgumentParser(description="Download PDFs from manifest, verify/enrich, rename, and persist metadata.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.json (array of entries).")
    ap.add_argument("--output", required=True, help="Directory to save PDFs and manifests.")
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"Failed to read manifest: {e}")
    if not isinstance(entries, list):
        sys.exit("Manifest must be a JSON array of entry objects.")

    ts = int(time.time())
    success_manifest_path = out_dir / f"download-success.manifest.{ts}.jsonl"
    failed_manifest_path = out_dir / f"download-failed.manifest.{ts}.jsonl"

    # Driver base temp dir (per-entry temp dirs will be under out_dir)
    session_tmp = out_dir / f"_session_{uuid.uuid4().hex}"
    session_tmp.mkdir(parents=True, exist_ok=True)
    driver = build_driver(session_tmp)

    processed = 0
    try:
        with success_manifest_path.open("a", encoding="utf-8") as succ_fp, \
             failed_manifest_path.open("a", encoding="utf-8") as fail_fp:
            for entry in entries:
                process_entry(entry, out_dir, driver, succ_fp, fail_fp, ts)
                processed += 1
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        shutil.rmtree(session_tmp, ignore_errors=True)

    print(f"\nProcessed {processed} entries.")
    print(f"Success manifest: {success_manifest_path}")
    print(f"Failed manifest : {failed_manifest_path}")


if __name__ == "__main__":
    main()
