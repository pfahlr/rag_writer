#!/usr/bin/env python3
"""
Rename and clean PDFs based on detected Title / DOI / ISBN.

Usage:
  python rename_and_clean_pdfs.py --dir /path/to/dir [--write]

Defaults to dry-run unless --write is provided.

Requirements:
  pip install python-magic pypdf
  # On Linux you may need the libmagic system package as well.
"""

import argparse
import os
import re
import time
import unicodedata
from typing import Optional, Tuple

# python-magic (libmagic wrapper)
try:
    import magic  # type: ignore
except Exception as e:
    raise SystemExit("python-magic is required. Install with: pip install python-magic\n" + str(e))

# PDF reader
try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:
    raise SystemExit("pypdf is required. Install with: pip install pypdf\n" + str(e))


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

# Common “ISBN …” label form
ISBN_LABELED_RE = re.compile(r"\bISBN(?:-1[03])?:?\s*([0-9Xx][0-9Xx\s\-]{8,})\b")

# Fallback: grab any digit/X sequences that look like ISBN chunks (10 or 13 digits after cleanup)
ISBN_CHUNK_RE = re.compile(r"\b[0-9Xx][0-9Xx\s\-]{9,18}[0-9Xx]\b")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)  # keep letters/numbers/underscore/space/hyphen
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "untitled"


def normalize_doi(doi: str) -> str:
    doi = doi.strip().rstrip(".,;)")
    doi = doi.replace("\\", "/")
    doi = re.sub(r"\s+", "", doi)
    return doi.lower()


def isbn10_is_valid(isbn10: str) -> bool:
    # ISBN-10: weights 10..1; X==10
    if len(isbn10) != 10:
        return False
    total = 0
    for i, ch in enumerate(isbn10):
        if ch.upper() == "X":
            val = 10
        elif ch.isdigit():
            val = int(ch)
        else:
            return False
        weight = 10 - i
        total += weight * val
    return total % 11 == 0


def isbn13_is_valid(isbn13: str) -> bool:
    if len(isbn13) != 13 or not isbn13.isdigit():
        return False
    total = 0
    for i, ch in enumerate(isbn13):
        d = int(ch)
        total += d * (1 if i % 2 == 0 else 3)
    return total % 10 == 0


def normalize_isbn(raw: str) -> Optional[str]:
    digits = re.sub(r"[\s\-]", "", raw)
    # Remove prefixes like "ISBN" accidentally captured
    digits = re.sub(r"(?i)^isbn(?:-1[03])?:?", "", digits)
    digits = digits.strip()
    if len(digits) == 10:
        return digits if isbn10_is_valid(digits.upper()) else None
    if len(digits) == 13:
        return digits if isbn13_is_valid(digits) else None
    return None


def extract_doi(text: str) -> Optional[str]:
    m = DOI_RE.search(text)
    if not m:
        return None
    return normalize_doi(m.group(0))


def extract_isbn(text: str) -> Optional[str]:
    # Prefer explicit “ISBN …” labels
    for rx in (ISBN_LABELED_RE, ISBN_CHUNK_RE):
        for m in rx.finditer(text):
            cand = normalize_isbn(m.group(1) if m.groups() else m.group(0))
            if cand:
                return cand
    return None


def guess_title(reader: PdfReader) -> Optional[str]:
    # 1) PDF metadata
    meta_title = None
    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        meta_title = ""
    if meta_title and meta_title.lower() not in {"", "untitled", "unknown"}:
        return meta_title

    # 2) Heuristic from first page text
    try:
        first_pages = min(len(reader.pages), 3)
        for i in range(first_pages):
            txt = reader.pages[i].extract_text() or ""
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for ln in lines[:15]:  # only the first few lines
                ln_clean = re.sub(r"\s+", " ", ln)
                if 5 <= len(ln_clean) <= 120:
                    low = ln_clean.lower()
                    if any(bad in low for bad in ("doi:", "doi.org/", "http", "www.", "issn", "abstract", "introduction", "copyright")):
                        continue
                    # avoid lines that look like author lists (commas + initials)
                    if re.search(r"\b[A-Z]\.\s*[A-Z]\.", ln_clean):
                        continue
                    return ln_clean
    except Exception:
        pass
    return None


def ensure_unique_path(target_path: str) -> str:
    if not os.path.exists(target_path):
        return target_path
    stem, ext = os.path.splitext(target_path)
    while True:
        suffix = f"--{int(time.time())}"
        candidate = f"{stem}{suffix}{ext}"
        if not os.path.exists(candidate):
            return candidate
        time.sleep(1)


def propose_new_name(title: Optional[str], doi: Optional[str], isbn: Optional[str], fallback_stem: str) -> str:
    # Build according to user’s spec
    parts = []
    if title:
        parts.append(slugify(title))

    if isbn and not title:
        parts.append(f"isbn--{isbn}")
    elif isbn and title:
        parts.append(f"isbn--{isbn}")

    if doi:
        safe_doi = doi.replace("/", "--")
        parts.append(f"doi--{safe_doi}")

    if not parts:
        # No matches: use original filename (lower + spaces→dashes)
        stem = fallback_stem.lower().replace(" ", "-")
        stem = re.sub(r"-{2,}", "-", stem)
    else:
        # If we had doi+isbn with no title, ensure the order is isbn then doi
        if not title and isbn and doi:
            parts = [f"isbn--{isbn}", f"doi--{doi.replace('/', '--')}"]
        stem = "__".join(parts)

    return f"{stem}.pdf"


def process_file(path: str, write: bool) -> None:
    # 1) MIME type check via libmagic
    try:
        mime = magic.from_file(path, mime=True)  # e.g., 'application/pdf'
    except Exception as e:
        print(f"[ERROR] Could not detect MIME for: {path} ({e})")
        return

    if mime != "application/pdf":
        action = "DELETE" if write else "DRY-RUN delete"
        print(f"[{action}] Non-PDF detected ({mime}): {path}")
        if write:
            try:
                os.remove(path)
            except Exception as e:
                print(f"[ERROR] Failed to delete {path}: {e}")
        return

    # 2) Read PDF → detect title / DOI / ISBN
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"[ERROR] Failed to open PDF: {path} ({e})")
        return

    # Gather text for DOI/ISBN detection (first few pages)
    text = ""
    try:
        pages_to_scan = min(len(reader.pages), 5)
        for i in range(pages_to_scan):
            text += "\n" + (reader.pages[i].extract_text() or "")
    except Exception:
        pass

    doi = extract_doi((text or "") + " " + str(reader.metadata or ""))
    isbn = extract_isbn((text or ""))
    title = guess_title(reader)

    # Build target name
    dirpath, filename = os.path.split(path)
    orig_stem, _orig_ext = os.path.splitext(filename)
    new_name = propose_new_name(title, doi, isbn, orig_stem)
    target = os.path.join(dirpath, new_name)

    # No-op if name doesn’t change (but fix extension to .pdf if needed)
    if os.path.normcase(path) == os.path.normcase(target):
        print(f"[SKIP] Already named correctly: {path}")
        return

    target = ensure_unique_path(target)

    action = "RENAME" if write else "DRY-RUN rename"
    print(f"[{action}] {filename}  →  {os.path.basename(target)}")

    if write:
        try:
            os.rename(path, target)
        except FileExistsError:
            # Extremely unlikely due to ensure_unique_path, but guard anyway
            target = ensure_unique_path(target)
            os.rename(path, target)
        except Exception as e:
            print(f"[ERROR] Failed to rename {path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Clean directory: delete non-PDFs; rename PDFs by Title / DOI / ISBN.")
    ap.add_argument("--dir", required=True, help="Directory containing files to process (non-recursive).")
    ap.add_argument("--write", action="store_true", help="Apply changes. Omit for dry-run.")
    args = ap.parse_args()

    base = os.path.abspath(args.dir)
    if not os.path.isdir(base):
        raise SystemExit(f"--dir must be a directory: {base}")

    print(f"Scanning: {base}  (mode: {'WRITE' if args.write else 'DRY-RUN'})")
    entries = sorted(os.listdir(base))

    for name in entries:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue
        process_file(path, args.write)


if __name__ == "__main__":
    main()
