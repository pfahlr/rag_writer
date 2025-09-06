from __future__ import annotations

import re
import requests
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET

ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(.+)", re.IGNORECASE)

def fetch_arxiv_by_doi(doi: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetch metadata from arXiv given a DOI like 10.48550/arXiv.XXXX."""
    m = ARXIV_DOI_RE.match(doi)
    if not m:
        return None
    arxiv_id = m.group(1)
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(r.text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        authors = [
            a.findtext("atom:name", default="", namespaces=ns).strip()
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", default="", namespaces=ns)
        date = published[:10] if published else ""
        year = int(published[:4]) if published else None
        doi_value = entry.findtext("arxiv:doi", namespaces=ns) or doi
        return {
            "title": title,
            "authors": authors,
            "publication": "arXiv",
            "date": date,
            "year": year,
            "doi": doi_value,
        }
    except Exception:
        return None
