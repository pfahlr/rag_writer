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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# --- MIME + PDF text ---
try:
    import magic  # python-magic
except Exception as e:
    raise SystemExit("python-magic is required. Try: pip install python-magic\n" + str(e))

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
    pass

try:
    from clients.crossref_client import enrich_via_crossref
except Exception:
    def enrich_via_crossref(**kwargs):
        return {}
try:
    from clients.google_books_client import enrich_via_google_books
except Exception:
    def enrich_via_google_books(*args, **kwargs):
        return {}

try:
    from functions import pdf_io as pdfio
    HAVE_PDFIO = True
except Exception:
    pass


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
            "identifier": list(filter(None, [meta.get("doi_url"), meta.get("isbn")])),
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
            # we want it to also write the standard fields
            # this isn;'t working yet
           # return
        except Exception as e:
            print(f"[WARN] pdf_io.write_pdf_metadata failed, falling back to pypdf: {e}")

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
        print(f"[WARN] pypdf metadata write failed: {e}")


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
    print(f"[WARN] Crossref enrichment failed after {tries} attempts: {last}")
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


# ----------------------- preprocess -----------------------
# ----------------------- preprocess -----------------------
def preprocess(
    manifest_path: Path,
    downloader: str,
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

    for i, e in enumerate(entries):
        url = (e.get("pdf_url") or "").strip()
        if not url:
            continue

        did = ensure_id(i, e)
        # temp filename we expect the downloader to save
        temp_filename = f"{did}.pdf"
        e["temp_filename"] = temp_filename
        e.setdefault("download_status", "pending")

        expected_size = None
        if probe_size:
            try:
                r = sess.head(url, allow_redirects=True, timeout=20)
                if "content-length" in r.headers:
                    expected_size = int(r.headers["content-length"])
            except Exception:
                expected_size = None

        dlmap.append({
            "id": did,
            "url": url,
            "temp_filename": temp_filename,
            "expected_size": expected_size,
        })

        # Emit downloader-specific batch
        if downloader == "aria2c":
            # aria2c input file format: URL newline + indented "out="
            # Also set dir= to downloads_dir so the user can run aria2c -i file
            lines_aria2.append(url)
            lines_aria2.append(f"  out={temp_filename}")
            lines_aria2.append(f"  dir={str(downloads_dir)}")
        elif downloader == "jdownloader":
            # JDownloader .crawljob: key=value per job, blank line between jobs
            # Minimal keys: text, downloadFolder, filename, enabled, autoStart
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
        else:
            raise SystemExit("--downloader must be one of: aria2c, jdownloader")

    # Write script/batch
    script_out.parent.mkdir(parents=True, exist_ok=True)
    if downloader == "aria2c":
        script_out.write_text("\n".join(lines_aria2) + "\n", encoding="utf-8")
        print(f"[preprocess] Wrote aria2c batch → {script_out}")
        print(f"Run example: aria2c -i '{script_out}' --check-certificate=false")
    else:
        script_out.write_text("\n\n".join(crawls) + "\n", encoding="utf-8")
        print(f"[preprocess] Wrote JDownloader .crawljob → {script_out}")
        print("Import in JDownloader: LinkGrabber menu → Add New Links → Load crawljob file")

    # Write downloads_map.json next to script
    map_path = script_out.with_suffix(".downloads_map.json")
    map_path.write_text(json.dumps(dlmap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preprocess] Wrote downloads_map.json → {map_path}")

    # Save updated manifest
    save_manifest(manifest_path, entries, wrapped)
    print(f"[preprocess] Updated manifest with temp filenames and status=pending → {manifest_path}")


# ----------------------- process -----------------------
def process_pipeline(
    manifest_path: Path,
    inbox_dir: Path,
    output_dir: Path,
    *,
    crossref_timeout: float = 25.0,
    crossref_tries: int = 4,
    crossref_backoff: float = 1.5,
    crossref_ua: str = "Content-Expanse/1.0 (mailto:pfahlr@gmail.com)",
) -> None:
    entries, wrapped = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_log = output_dir / f"process-success.{int(time.time())}.jsonl"
    fail_log = output_dir / f"process-fail.{int(time.time())}.jsonl"

    succ_fp = success_log.open("a", encoding="utf-8")
    fail_fp = fail_log.open("a", encoding="utf-8")

    processed = 0

    try:
        for e in entries:
            url = (e.get("pdf_url") or "").strip()
            temp_name = e.get("temp_filename")
            if not url or not temp_name:
                continue

            src = inbox_dir / temp_name
            if not src.exists():
                # leave pending; user may not have downloaded yet
                continue

            # MIME verify
            try:
                mime = magic.from_file(str(src), mime=True)
            except Exception as ex:
                mime = None
                print(f"[WARN] MIME check failed for {temp_name}: {ex}")
            if mime != "application/pdf":
                e["download_status"] = "failed"
                e["failure_reason"] = f"non_pdf_mime:{mime}"
                fail_fp.write(json.dumps({"entry": e, "reason": e["failure_reason"]}, ensure_ascii=False) + "\n")
                continue

            # Open and extract skim text
            try:
                reader = PdfReader(str(src))
            except Exception as ex:
                e["download_status"] = "failed"
                e["failure_reason"] = f"pdf_open_error:{ex}"
                fail_fp.write(json.dumps({"entry": e, "reason": e["failure_reason"]}, ensure_ascii=False) + "\n")
                continue

            meta: Dict = {
                "title": e.get("title"),
                "authors": e.get("authors"),
                "date": e.get("date"),
                "publication": e.get("publication"),
                "doi": e.get("doi"),
                "isbn": e.get("isbn"),
                "pdf_url": url,
                "scholar_url": e.get("scholar_url"),
                "original_filename": temp_name,
                "source": "manifest",
            }

            # DOI from URLs fallback
            if not meta.get("doi"):
                for u in (e.get("pdf_url"), e.get("scholar_url")):
                    cand = parse_doi_from_url(u or "")
                    if cand:
                        meta["doi"] = cand
                        break

            # First pages text for ISBN/title fallback
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

            # Enrichment
            try:
                if is_arxiv_source(e.get("pdf_url")) or is_arxiv_source(e.get("scholar_url")):
                    if HAVE_ARXIV:
                        am = enrich_via_arxiv(e.get("pdf_url") or e.get("scholar_url"))
                        if am:
                            for k, v in am.items():
                                if v and not meta.get(k):
                                    meta[k] = v
            except Exception as ex:
                print(f"[WARN] arXiv enrichment failed: {ex}")

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
                    print(f"[WARN] Google Books enrichment failed: {ex}")

            # Final filename and move
            final_name = propose_filename(meta.get("title"), meta.get("date"))
            final_path = output_dir / final_name
            if final_path.exists():
                final_path = output_dir / f"{final_path.stem}--{int(time.time())}{final_path.suffix}"

            try:
                src.replace(final_path)
            except Exception as ex:
                e["download_status"] = "failed"
                e["failure_reason"] = f"move_error:{ex}"
                fail_fp.write(json.dumps({"entry": e, "reason": e["failure_reason"]}, ensure_ascii=False) + "\n")
                continue

            # Embed metadata, write sidecar, update entry
            try:
                write_pdf_metadata(final_path, meta)
            except Exception as ex:
                print(f"[WARN] metadata embed failed: {ex}")

            sidecar = final_path.with_suffix(".meta.json")
            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            e["local_path"] = str(final_path)
            e["final_filename"] = final_path.name
            e["file_sha256"] = sha256(final_path)
            e["download_status"] = "done"
            e.pop("failure_reason", None)

            succ_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
            processed += 1
            print(f"[OK] {temp_name} → {final_path.name}")

    finally:
        succ_fp.close()
        fail_fp.close()
        save_manifest(manifest_path, entries, wrapped)
        print(f"[process] Updated manifest → {manifest_path}")
        print(f"[process] Success log: {success_log}")
        print(f"[process] Fail log   : {fail_log}")
        print(f"[process] Completed {processed} entries.")


# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Two-phase PDF pipeline (preprocess → process) without Selenium.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Generate download script (aria2c/jdownloader) and update manifest.")
    sp.add_argument("--manifest", required=True, type=Path)
    sp.add_argument("--downloader", required=True, choices=["aria2c", "jdownloader"])
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

    sr = sub.add_parser("retry", help="Parse aria2 log; regenerate aria2 batch for failed items.")
    sr.add_argument("--manifest", required=True, type=Path)
    sr.add_argument("--aria2-log", required=True, type=Path)
    sr.add_argument("--script-out", required=True, type=Path)
    sr.add_argument("--downloads-dir", required=True, type=Path)
    sr.add_argument("--no-manifest-update", action="store_true")

    args = ap.parse_args()

    if args.cmd == "preprocess":
        preprocess(args.manifest, args.downloader, args.script_out, args.downloads_dir, args.probe_size)
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
