#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import typer

from functions.manifest import file_checksum, load_manifest, save_manifest
from functions.pdf_io import write_pdf_info, write_pdf_xmp
from clients.crossref_client import fetch_crossref_by_doi, search_crossref
from clients.isbn_openlibrary import fetch_openlibrary_by_isbn

from clients.arxiv_client import fetch_arxiv_by_doi
from pypdf import PdfReader


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]")
ISBN_RE = re.compile(r"\b(?:97[89][- ]?)?[0-9][- 0-9]{9,}[0-9Xx]\b")


def _slugify(s: str, maxlen: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if len(s) > maxlen else s


def _detect_ids(text: str) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    m = DOI_RE.search(text)
    if m:
        ids["doi"] = m.group(0).lower()
    m = ISBN_RE.search(text)
    if m:
        # normalize hyphens/spaces
        ids["isbn"] = re.sub(r"[- ]", "", m.group(0))
    return ids


def _default_manifest_path() -> Path:
    return Path("research/out/tmp/manifest.json")

 
def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)

def _extract_text_first_pages(pdf_path: Path, max_pages: int = 5) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text = []
        for i in range(min(max_pages, len(doc))):
            text.append(doc[i].get_text())
        return "\n".join(text)
    except Exception:
        return ""


def _gather_pdf_info(pdf_path: Path) -> Tuple[Dict[str, Any], bool]:
    """Gather initial metadata for a PDF file.

    Returns a tuple of (metadata, record_found) where record_found indicates
    whether a remote lookup (Crossref/OpenLibrary) succeeded.
    """

    text = pdf_path.name + "\n" + _extract_text_first_pages(pdf_path)
    ids = _detect_ids(text)
    meta: Dict[str, Any] = {
        "title": pdf_path.stem,
        "authors": [],
        "publication": "",
        "date": "",
        "year": None,
    }

    # Try to read existing PDF metadata for title/author hints
    try:
        reader = PdfReader(str(pdf_path))
        info = reader.metadata or {}
        if info.get("/Title"):
            meta["title"] = info.get("/Title")
        if info.get("/Author"):
            meta["authors"] = [a.strip() for a in re.split(r",|;| and ", info.get("/Author")) if a.strip()]
    except Exception:
        pass

    meta.update(ids)
    record_found = False
    if ids.get("doi"):
        data = fetch_crossref_by_doi(ids["doi"]) or fetch_arxiv_by_doi(ids["doi"])
        if data:
            meta.update({k: v for k, v in data.items() if v})
            record_found = True
    if ids.get("isbn") and not record_found:

        data = fetch_openlibrary_by_isbn(ids["isbn"])
        if data:
            meta.update({k: v for k, v in data.items() if v})
            record_found = True
    if not record_found:
        query_title = meta.get("title") or pdf_path.stem
        query_author = meta.get("authors", [""])[0] if meta.get("authors") else ""
        data = search_crossref(query_title, query_author)
        if data:
            meta.update({k: v for k, v in data.items() if v})
            record_found = True
    return meta, record_found

def main(
    dir: Path = typer.Option(Path("data_raw"), "--dir", help="Directory to scan for PDFs"),
    glob: str = typer.Option("**/*.pdf", "--glob", help="Glob pattern for PDFs"),
    write: bool = typer.Option(False, "--write", help="Write manifest entries and update PDF metadata (Info+XMP)"),
    manifest: Path = typer.Option(_default_manifest_path(), "--manifest", help="Manifest JSON path"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip files already present in manifest as processed"),
    rename: str = typer.Option("yes", "--rename", help="Rename files to slugified title and year [yes|no]"),
    allow_delete: bool = typer.Option(False, "--allow-delete", help="Enable [R] remove option to delete files"),
    rescan: bool = typer.Option(False, "--rescan", help="Ignore cached results and re-query remote APIs"),
):
    """Scan PDFs for DOI/ISBN, fetch metadata, and record to manifest (v1)."""
    pdfs: List[Path] = sorted(Path(dir).glob(glob))
    data = load_manifest(manifest)
    entries: List[Dict[str, Any]] = data.get("entries", [])
    known_ids = {e.get("id"): e for e in entries if e.get("id")}

    for pdf in pdfs:
        if not pdf.is_file():
            continue
        checksum = file_checksum(pdf)
        if (
            skip_existing
            and not rescan
            and checksum in known_ids
            and known_ids[checksum].get("processed")
        ):
            typer.echo(f"[skip] {pdf}")
            continue

        meta, found = _gather_pdf_info(pdf)
        entry = {
            "id": checksum,
            "filename": _relpath(pdf, manifest.parent),
            "processed": False,
            **meta,
        }

        while True:
            typer.echo(f"\nFile: {pdf}")
            typer.echo(json.dumps(entry, ensure_ascii=False, indent=2))
            if found:
                prompt = "[W]rite/[D]OI/[I]SBN/[S]kip/[R]emove"
            else:
                prompt = "No metadata found. [D]OI/[I]SBN/[S]kip/[R]emove"
            choice = typer.prompt(prompt).strip().lower()

            if choice == "w" and found:
                if write:
                    title = entry.get("title") or Path(entry["filename"]).stem
                    year = entry.get("year")
                    authors = entry.get("authors") or []
                    publication = entry.get("publication") or ""
                    doi = entry.get("doi") or ""
                    isbn = entry.get("isbn") or ""

                    base_slug = _slugify(title)
                    if year:
                        base_slug = f"{base_slug}_{year}"
                    src_path = pdf.resolve()
                    dest_dir = src_path.parent
                    dest_name = (
                        f"{base_slug}.pdf" if rename.lower() == "yes" else src_path.name
                    )
                    dest_path = dest_dir / dest_name

                    if dest_path.exists() and dest_path.resolve() != src_path:
                        n = 1
                        while True:
                            candidate = dest_dir / f"{base_slug}_{n}.pdf"
                            if not candidate.exists():
                                dest_path = candidate
                                break
                            n += 1

                    entry["filename"] = _relpath(dest_path, manifest.parent)
                    entry["retrieved_at"] = datetime.utcnow().isoformat()

                    replaced = False
                    for i, e in enumerate(entries):
                        if e.get("id") == checksum:
                            entries[i] = {**e, **entry, "processed": True}
                            replaced = True
                            break
                    if not replaced:
                        entry["processed"] = True
                        entries.append(entry)
                    data["entries"] = entries
                    save_manifest(manifest, data)
                    typer.echo(f"[write] manifest updated: {manifest}")

                    tmp_path = dest_path.with_suffix(".tmp.pdf")
                    index_meta = {
                        "title": title,
                        "authors": authors,
                        "publication": publication,
                        "date": entry.get("date"),
                        "year": year,
                        "doi": doi,
                        "isbn": isbn,
                    }
                    info_meta = {
                        "/Title": title,
                        "/Author": ", ".join(authors) if authors else "",
                        "/Subject": json.dumps(index_meta, ensure_ascii=False),
                        "/doi": doi,
                        "/isbn": isbn,
                    }
                    write_pdf_info(src_path, tmp_path, info_meta)
                    write_pdf_xmp(
                        tmp_path,
                        dc={"title": title, "creator": authors},
                        prism={"publicationName": publication, "doi": doi, "isbn": isbn},
                    )
                    tmp_path.replace(dest_path)
                    if dest_path != src_path:
                        try:
                            src_path.unlink()
                        except Exception:
                            pass
                    typer.echo(f"[write] updated PDF metadata: {dest_path}")
                else:
                    typer.echo("[dry-run] metadata would be written")
                break
            elif choice == "d":
                new_doi = typer.prompt("Enter DOI").strip()
                if new_doi:
                    entry["doi"] = new_doi.lower()

                    data = fetch_crossref_by_doi(entry["doi"]) or fetch_arxiv_by_doi(entry["doi"]) or {}

                    entry.update({k: v for k, v in data.items() if v})
                    found = bool(data)
                else:
                    found = False
            elif choice == "i":
                new_isbn = typer.prompt("Enter ISBN").strip()
                if new_isbn:
                    entry["isbn"] = re.sub(r"[- ]", "", new_isbn)
                    data = fetch_openlibrary_by_isbn(entry["isbn"]) or {}
                    entry.update({k: v for k, v in data.items() if v})
                    found = bool(data)
                else:
                    found = False
            elif choice == "s":
                if write and checksum not in known_ids:
                    entries.append(
                        {
                            "id": checksum,
                            "filename": _relpath(pdf, manifest.parent),
                            "processed": False,
                            "retrieved_at": datetime.utcnow().isoformat(),
                        }
                    )
                    data["entries"] = entries
                    save_manifest(manifest, data)
                    typer.echo(f"[skip] recorded in manifest: {manifest}")
                else:
                    typer.echo("[skip] no changes")
                break
            elif choice == "r":
                if allow_delete:
                    try:
                        pdf.unlink()
                        typer.echo(f"[remove] deleted {pdf}")
                    except Exception as exc:
                        typer.echo(f"[remove] failed to delete {pdf}: {exc}")
                    if write:
                        entries[:] = [e for e in entries if e.get("id") != checksum]
                        data["entries"] = entries
                        save_manifest(manifest, data)
                else:
                    typer.echo("[remove] deletion disabled (use --allow-delete)")
                break
            else:
                typer.echo("Invalid choice")


if __name__ == "__main__":
    typer.run(main)
