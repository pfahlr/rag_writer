# src/research/functions/pdf_io.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from pypdf import PdfReader, PdfWriter

try:
    import pikepdf  # optional; enables XMP writing
except Exception:  # pragma: no cover - optional dependency
    pikepdf = None


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _as_list(v: Union[str, Iterable[str], None]) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(x).strip() for x in v if str(x).strip()]

def _filter_nonempty(seq: Iterable[Optional[str]]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if x:
            s = str(x).strip()
            if s:
                out.append(s)
    return out

def _normalize_isbn_list(value: Union[str, List[str], Dict[str, str], None]) -> List[str]:
    """
    Accept str, list[str], or dicts and return a de-duped list of plausible ISBNs
    (only 10 or 13 chars after removing hyphens/spaces; uppercased).
    Supports values like 'urn:isbn:978...' or 'isbn:...' and normalizes them.
    """
    def _clean(x: str) -> str:
        x = str(x)
        x = re.sub(r'(?i)\b(?:urn:)?isbn:?', '', x)  # strip prefixes
        x = re.sub(r'[\s\-]', '', x)                 # drop spaces/hyphens
        return x.upper()

    isbns: List[str] = []
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        for v in value:
            if v:
                isbns.append(_clean(str(v)))
    elif isinstance(value, dict):
        for k in ("isbn", "isbn10", "isbn13"):
            if value.get(k):
                isbns.append(_clean(str(value[k])))
    else:
        isbns.append(_clean(str(value)))

    # keep only plausible 10/13 lengths and de-dup preserving order
    out: List[str] = []
    for i in isbns:
        if len(i) in (10, 13) and i not in out:
            out.append(i)
    return out

def _prefer_isbn13(isbns: List[str]) -> Optional[str]:
    for i in isbns:
        if len(i) == 13:
            return i
    return isbns[0] if isbns else None


# --------------------------------------------------------------------
# Backward-compatible APIs
# --------------------------------------------------------------------

def write_pdf_info(src_pdf: Path, dest_pdf: Path, metadata: Dict[str, str]) -> Path:
    """
    Copy src_pdf to dest_pdf and write classic PDF Info keys.
    Keys can be given without the leading slash; it will be added.
    """
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter(clone_from=str(src_pdf))
    meta = {}
    for k, v in metadata.items():
        if v is None:
            continue
        key = str(k)
        if not key.startswith("/"):
            key = "/" + key
        meta[key] = str(v)
    writer.add_metadata(meta)
    with dest_pdf.open("wb") as fp:
        writer.write(fp)
    return dest_pdf


def write_pdf_xmp(path: Path, dc: Dict[str, Union[str, List[str]]], prism: Dict[str, str]) -> None:
    """
    Legacy XMP writer for Dublin Core + Prism (minimal subset).
    Prefer write_pdf_metadata() for richer metadata.
    """
    if pikepdf is None:
        return
    with pikepdf.Pdf.open(str(path)) as pdf:
        with pdf.open_metadata() as meta:
            #meta.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
            #meta.register_namespace('prism', 'http://prismstandard.org/namespaces/basic/2.0/')

            # Dublin Core
            if dc.get('title'):
                meta['dc:title'] = str(dc['title'])
            creators = _as_list(dc.get('creator'))
            if creators:
                meta['dc:creator'] = creators
            if dc.get('description'):
                meta['dc:description'] = str(dc['description'])
            identifiers = _as_list(dc.get('identifier'))
            if identifiers:
                meta['dc:identifier'] = identifiers

            # Prism
            if prism.get('publicationName'):
                meta['prism:publicationName'] = str(prism['publicationName'])
            if prism.get('doi'):
                meta['prism:doi'] = str(prism['doi'])
            if prism.get('isbn'):
                meta['prism:isbn'] = str(prism['isbn'])
            if prism.get('issn'):
                meta['prism:issn'] = str(prism['issn'])
            if prism.get('url'):
                meta['prism:url'] = str(prism['url'])
            if prism.get('publicationDate'):
                meta['prism:publicationDate'] = str(prism['publicationDate'])

        pdf.save(str(path))


# --------------------------------------------------------------------
# Preferred high-level API
# --------------------------------------------------------------------

def write_pdf_metadata(
    path: Path,
    *,
    core: Optional[Dict[str, str]] = None,
    dc: Optional[Dict[str, Union[str, List[str]]]] = None,
    prism: Optional[Dict[str, str]] = None,
    dcterms: Optional[Dict[str, str]] = None,
    arxiv: Optional[Dict[str, str]] = None,
    full_meta_for_subject: Optional[dict] = None,
) -> None:
    """
    Write both classic PDF Info (Title/Author/Subject/Keywords) and XMP metadata.

    - core:    Title, Author, Subject, Keywords (classic Info dictionary).
               If full_meta_for_subject is provided, Subject will be set to json.dumps(metadata).
    - dc:      Dublin Core: title, creator[list], description, identifier[list], date.
               We ensure dc:identifier also includes 'urn:isbn:<...>' entries for any ISBNs found.
    - prism:   doi, isbn, issn, publicationName, publicationDate, url.
               We ensure prism:isbn is present (preferring ISBN-13) if any ISBNs are found.
    - dcterms: issued (published) / modified (updated).
    - arxiv:   arXiv-specific props under 'arxiv:' custom namespace:
               { id, primaryCategory, secondaryCategory, scheme, sourceUrl, pdfUrl, arxivDoi }

    If pikepdf isn't available, only classic Info is written (XMP skipped).
    """

    # -------------------------
    # Normalize/augment ISBNs
    # -------------------------
    isbns_all: List[str] = []

    # from PRISM
    if prism and prism.get("isbn"):
        isbns_all.extend(_normalize_isbn_list(prism.get("isbn")))

    # from DC identifiers (any form that looks like isbn)
    if dc and dc.get("identifier"):
        for ident in _as_list(dc.get("identifier")):
            if ident and re.search(r'(?i)(?:^|:)(?:urn:)?isbn:?\s*[\d\-xX]+', str(ident)):
                isbns_all.extend(_normalize_isbn_list(ident))

    # from full metadata blob (optional)
    if full_meta_for_subject and isinstance(full_meta_for_subject, dict) and "isbn" in full_meta_for_subject:
        isbns_all.extend(_normalize_isbn_list(full_meta_for_subject.get("isbn")))

    # de-dup normalized
    norm_isbns = []
    for i in _normalize_isbn_list(isbns_all):
        if i not in norm_isbns:
            norm_isbns.append(i)

    # ensure DC identifiers include urn:isbn entries
    if norm_isbns:
        dc = dc or {}
        dc_idents = _as_list(dc.get("identifier"))
        for i in norm_isbns:
            uri = f"urn:isbn:{i}"
            if uri not in dc_idents:
                dc_idents.append(uri)
        dc["identifier"] = dc_idents

        # ensure PRISM has a primary ISBN (prefer 13)
        prism = prism or {}
        if not prism.get("isbn"):
            primary = _prefer_isbn13(norm_isbns)
            if primary:
                prism["isbn"] = primary

        # add ISBN(s) into PDF Info keywords
        core = core or {}
        existing_kw = [k.strip() for k in (core.get("Keywords") or "").split(",") if k.strip()]
        existing_kw.extend([f"ISBN:{i}" for i in norm_isbns])
        # de-dup
        seen = set()
        dedup_kw: List[str] = []
        for k in existing_kw:
            if k not in seen:
                seen.add(k)
                dedup_kw.append(k)
        core["Keywords"] = ", ".join(dedup_kw)

    # -------------------------
    # Write classic PDF Info
    # -------------------------
    reader = PdfReader(str(path))
    pdf_length = len(reader.pages)

    #writer = PdfWriter()
    #   for page in reader.pages:
    #   writer.add_page(page)

    writer = PdfWriter(str(path))

    info: Dict[str, str] = {}
    # Copy caller-provided core fields
    if core:
        for k, v in core.items():
            if v is None:
                continue
            key = "/" + k if not str(k).startswith("/") else str(k)
            info[key] = str(v)

    # Always store the full JSON blob in /Subject if provided.
    if full_meta_for_subject is not None:
        info["/Subject"] = json.dumps(full_meta_for_subject, ensure_ascii=False)

    writer.add_metadata(info)

    tmp = path.with_suffix(".tmp.pdf")
    with tmp.open("wb") as fp:
        writer.write(fp)
    tmp.replace(path)

    # Created, Modified, Author, Title, Subject, Pages Count
    # -------------------------
    # Write XMP (if available)
    # -------------------------
    if pikepdf is None:
        return

    with pikepdf.Pdf.open(str(path), allow_overwriting_input=True) as pdf:
       # print("\n\n")
       # print("---------old style metadata--------")
        #copy the  deprecated documentinfo block into the new metadata
        with pdf.open_metadata() as meta:
            meta.load_from_docinfo(pdf.docinfo)
            # Namespaces
            #meta.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
            #meta.register_namespace("dcterms", "http://purl.org/dc/terms/")
            #meta.register_namespace("prism", "http://prismstandard.org/namespaces/basic/2.0/")
            #meta.register_namespace("arxiv", "http://arxiv.org/schemas/atom")

            # Dublin Core
            if dc:
                if dc.get("title"):
                    meta["dc:title"] = str(dc["title"])
                creators = _as_list(dc.get("creator"))
                if creators:
                    meta["dc:creator"] = creators
                if dc.get("description"):
                    meta["dc:description"] = str(dc["description"])
                identifiers =  dc.get("identifier")
                if identifiers:
                    meta["dc:identifier"] = identifiers
                if dc.get("date"):
                    meta["dc:date"] = str(dc["date"])

            # DCTERMS (issued / modified)
            if dcterms:
                if dcterms.get("issued"):
                    meta["dcterms:issued"] = str(dcterms["issued"])
                if dcterms.get("modified"):
                    meta["dcterms:modified"] = str(dcterms["modified"])

            # PRISM
            if prism:
                if prism.get("publicationName"):
                    meta["prism:publicationName"] = str(prism["publicationName"])
                if prism.get("doi"):
                    meta["prism:doi"] = str(prism["doi"])
                if prism.get("isbn"):
                    meta["prism:isbn"] = str(prism["isbn"])
                if prism.get("issn"):
                    meta["prism:issn"] = str(prism["issn"])
                if prism.get("url"):
                    meta["prism:url"] = str(prism["url"])
                if prism.get("publicationDate"):
                    meta["prism:publicationDate"] = str(prism["publicationDate"])

            # arXiv custom props
            if arxiv:
                if arxiv.get("id"):
                    meta["arxiv:id"] = str(arxiv["id"])
                if arxiv.get("primaryCategory"):
                    meta["arxiv:primaryCategory"] = str(arxiv["primaryCategory"])
                if arxiv.get("secondaryCategory"):
                    meta["arxiv:secondaryCategory"] = str(arxiv["secondaryCategory"])
                if arxiv.get("scheme"):
                    meta["arxiv:scheme"] = str(arxiv["scheme"])
                if arxiv.get("sourceUrl"):
                    meta["arxiv:sourceUrl"] = str(arxiv["sourceUrl"])
                if arxiv.get("pdfUrl"):
                    meta["arxiv:pdfUrl"] = str(arxiv["pdfUrl"])
                if arxiv.get("arxivDoi"):
                    meta["arxiv:doi"] = str(arxiv["arxivDoi"])

        pdf.save(str(path))


# --------------------------------------------------------------------
# Convenience: build payloads from arXiv metadata
# --------------------------------------------------------------------

def build_metadata_payload_from_arxiv(meta: Dict[str, object]) -> Dict[str, dict]:
    """
    Convert enrich_via_arxiv(...) output to the pieces expected by write_pdf_metadata().

    Expected keys in input:
      arxiv_doi, arxiv_id, published, updated, summary, title, authors,
      pdf_url, scholar_url, primary_category, secondary_category, scheme
    """
    title = str(meta.get("title") or "").strip()
    authors = _as_list(meta.get("authors"))
    desc = str(meta.get("summary") or "").strip()

    identifiers = _filter_nonempty([
        str(meta.get("arxiv_doi") or ""),
        f"arXiv:{meta.get('arxiv_id')}" if meta.get("arxiv_id") else None,
    ])

    # Core PDF Info
    core = {
        "Title": title,
        "Author": "; ".join(authors),
        # Subject will be populated from full_meta_for_subject by write_pdf_metadata()
        "Keywords": ", ".join(_filter_nonempty([
            str(meta.get("arxiv_doi") or ""),
            f"arXiv:{meta.get('arxiv_id')}" if meta.get("arxiv_id") else None,
            str(meta.get("primary_category") or ""),
        ])),
    }

    dc = {
        "title": title,
        "creator": authors,
        "description": desc,
        "identifier": identifiers,  # will be augmented with urn:isbn:* if any
        "date": str(meta.get("published") or "") or None,
    }

    dcterms = {
        "issued": str(meta.get("published") or "") or None,
        "modified": str(meta.get("updated") or "") or None,
    }

    prism = {
        "doi": str(meta.get("arxiv_doi") or "") or None,
        "publicationName": "arXiv",
        "publicationDate": str(meta.get("published") or "") or None,
        "url": str(meta.get("pdf_url") or meta.get("scholar_url") or "") or None,
        # prism:isbn will be filled automatically if available upstream
    }

    arxiv_ns = {
        "id": str(meta.get("arxiv_id") or "") or None,
        "primaryCategory": str(meta.get("primary_category") or "") or None,
        "secondaryCategory": str(meta.get("secondary_category") or "") or None,
        "scheme": str(meta.get("scheme") or "") or None,
        "sourceUrl": str(meta.get("scholar_url") or "") or None,
        "pdfUrl": str(meta.get("pdf_url") or "") or None,
        "arxivDoi": str(meta.get("arxiv_doi") or "") or None,
    }

    full_subject = dict(meta)  # dump everything into /Subject as JSON

    return {
        "core": core,
        "dc": dc,
        "prism": prism,
        "dcterms": dcterms,
        "arxiv": arxiv_ns,
        "full_subject": full_subject,
    }
