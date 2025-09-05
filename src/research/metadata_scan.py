#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import typer

from functions.manifest import file_checksum, load_manifest, save_manifest
from functions.pdf_io import write_pdf_info, write_pdf_xmp
from clients.crossref_client import fetch_crossref_by_doi
from clients.isbn_openlibrary import fetch_openlibrary_by_isbn

app = typer.Typer(add_completion=False)


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
    return Path("research/out/manifest.json")


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


def _gather_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    # Detect IDs from filename and the first few pages (to avoid works-cited noise)
    text = pdf_path.name + "\n" + _extract_text_first_pages(pdf_path)
    ids = _detect_ids(text)
    meta: Dict[str, Any] = {"title": pdf_path.stem, "authors": [], "publication": "", "date": "", "year": None}
    meta.update(ids)
    # Query metadata providers
    if ids.get("doi"):
        data = fetch_crossref_by_doi(ids["doi"]) or {}
        meta.update({k: v for k, v in data.items() if v})
    if ids.get("isbn") and not meta.get("title"):
        data = fetch_openlibrary_by_isbn(ids["isbn"]) or {}
        meta.update({k: v for k, v in data.items() if v})
    return meta


@app.command()
def scan(
    dir: Path = typer.Option(Path("data_raw"), "--dir", help="Directory to scan for PDFs"),
    glob: str = typer.Option("**/*.pdf", "--glob", help="Glob pattern for PDFs"),
    write: bool = typer.Option(False, "--write", help="Write manifest entries and update PDF metadata (Info+XMP)"),
    manifest: Path = typer.Option(_default_manifest_path(), "--manifest", help="Manifest JSON path"),
    rename: str = typer.Option("yes", "--rename", help="Rename files to slugified title and year [yes|no]"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip files already present in manifest as processed"),
):
    """Scan PDFs for DOI/ISBN, fetch metadata, and record to manifest (v1)."""
    pdfs: List[Path] = sorted(Path(dir).glob(glob))
    print(pdfs)
    data = load_manifest(manifest)
    entries: List[Dict[str, Any]] = data.get("entries", [])
    known_ids = {e.get("id"): e for e in entries if e.get("id")}

    for pdf in pdfs:
        if not pdf.is_file():
            continue
        checksum = file_checksum(pdf)
        if skip_existing and checksum in known_ids and known_ids[checksum].get("processed"):
            typer.echo(f"[skip] {pdf}")
            continue

        meta = _gather_pdf_info(pdf)
        entry = {
            "id": checksum,
            "filename": str(pdf),
            "processed": False,
            **meta,
        }
        typer.echo(json.dumps(entry, ensure_ascii=False, indent=2))

        if write:
            # Upsert in manifest
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

            # Write PDF Info + XMP and optionally rename
            title = entry.get("title") or Path(entry["filename"]).stem
            year = entry.get("year")
            authors = entry.get("authors") or []
            subject = entry.get("publication") or ""
            doi = entry.get("doi") or ""
            isbn = entry.get("isbn") or ""

            # build dest filename
            base_slug = _slugify(title)
            if year:
                base_slug = f"{base_slug}_{year}"
            dest_dir = Path(entry["filename"]).parent
            dest_name = f"{base_slug}.pdf" if rename.lower() == 'yes' else Path(entry["filename"]).name
            dest_path = dest_dir / dest_name

            # avoid collision
            if dest_path.exists() and dest_path.resolve() != Path(entry["filename"]).resolve():
                n = 1
                while True:
                    candidate = dest_dir / f"{base_slug}_{n}.pdf"
                    if not candidate.exists():
                        dest_path = candidate
                        break
                    n += 1

            src_path = Path(entry["filename"]).resolve()
            tmp_path = dest_path.with_suffix('.tmp.pdf')

            info_meta = {
                "/Title": title,
                "/Author": ", ".join(authors) if authors else "",
                "/Subject": subject,
                "/doi": doi,
                "/isbn": isbn,
            }
            # Write Info dict to temp, then XMP and finalize to dest
            write_pdf_info(src_path, tmp_path, info_meta)
            write_pdf_xmp(tmp_path, dc={"title": title, "creator": authors}, prism={"publicationName": subject, "doi": doi, "isbn": isbn})
            tmp_path.replace(dest_path)
            if dest_path != src_path:
                try:
                    src_path.unlink()
                except Exception:
                    pass
            typer.echo(f"[write] updated PDF metadata: {dest_path}")


def main():
    app()


if __name__ == "__main__":
    main()
