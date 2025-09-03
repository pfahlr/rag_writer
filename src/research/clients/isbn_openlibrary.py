from __future__ import annotations

import requests
from typing import Optional, Dict, Any


def fetch_openlibrary_by_isbn(isbn: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    url = f"https://openlibrary.org/isbn/{isbn}.json"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        title = data.get("title", "")
        publishers = data.get("publishers") or []
        publish_date = data.get("publish_date", "")
        return {
            "title": title,
            "authors": [],
            "publication": ", ".join(publishers) if publishers else "",
            "date": publish_date,
            "year": _extract_year(publish_date),
            "isbn": isbn,
        }
    except Exception:
        return None


def _extract_year(s: str) -> Optional[int]:
    import re
    m = re.search(r"(20\d{2}|19\d{2})", s or "")
    return int(m.group(1)) if m else None

