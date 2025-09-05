from __future__ import annotations

from pathlib import Path
from typing import Dict
from pypdf import PdfWriter
try:
    import pikepdf
except Exception:  # pragma: no cover - optional dependency
    pikepdf = None


def write_pdf_info(src_pdf: Path, dest_pdf: Path, metadata: Dict[str, str]) -> Path:
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter(clone_from=str(src_pdf))
    # Normalize metadata keys to PDF Info style (leading slash is added by pypdf)
    meta = {str(k): str(v) for k, v in metadata.items() if v is not None}
    writer.add_metadata(meta)
    with dest_pdf.open("wb") as fp:
        writer.write(fp)
    return dest_pdf


def write_pdf_xmp(path: Path, dc: Dict[str, str], prism: Dict[str, str]) -> None:
    """Write XMP (Dublin Core + Prism) metadata in-place using pikepdf.

    If pikepdf is not installed, this function silently returns.
    """
    if pikepdf is None:
        return
    with pikepdf.Pdf.open(str(path)) as pdf:
        with pdf.open_metadata() as meta:
            meta.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
            meta.register_namespace('prism', 'http://prismstandard.org/namespaces/basic/2.0/')
            # Dublin Core
            if title := dc.get('title'):
                meta['dc:title'] = title
            creators = dc.get('creator')
            if creators:
                if isinstance(creators, str):
                    creators = [creators]
                meta['dc:creator'] = creators
            if desc := dc.get('description'):
                meta['dc:description'] = desc
            # Prism
            if pub := prism.get('publicationName'):
                meta['prism:publicationName'] = pub
            if doi := prism.get('doi'):
                meta['prism:doi'] = doi
            if isbn := prism.get('isbn'):
                meta['prism:isbn'] = isbn
            if issn := prism.get('issn'):
                meta['prism:issn'] = issn
        pdf.save(str(path))

