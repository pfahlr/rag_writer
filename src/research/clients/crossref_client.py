from __future__ import annotations

import requests
from typing import Optional, Dict, Any


def fetch_crossref_by_doi(doi: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json().get("message", {})
        return {
            "title": (data.get("title") or [""])[0],
            "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in data.get("author", [])],
            "publication": data.get("container-title", [""])[0],
            "date": "-".join(map(str, (data.get("published-print") or data.get("published-online") or {}).get("date-parts", [[""]])[0])),
            "year": (data.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "doi": data.get("DOI"),
        }
    except Exception:
        return None

