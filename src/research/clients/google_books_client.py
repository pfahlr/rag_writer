#!/usr/bin/env python3
"""
Google Books client for ISBN enrichment.

expose:
  enrich_via_google_books(isbn: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]

Returns a subset:
  { "title", "authors", "publication", "date", "isbn" }
"""

from typing import Any, Dict, List, Optional
import requests

def enrich_via_google_books(isbn: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    if not isbn:
        return None
    url = "https://www.googleapis.com/books/v1/volumes"
    r = requests.get(url, params={"q": f"isbn:{isbn}"}, timeout=timeout)
    if not r.ok:
        return None
    items = r.json().get("items") or []
    if not items:
        return None
    info = items[0].get("volumeInfo", {})
    meta: Dict[str, Any] = {}
    if info.get("title"):
        meta["title"] = info["title"]
    if info.get("authors"):
        meta["authors"] = info["authors"]
    if info.get("publisher"):
        meta["publication"] = info["publisher"]
    if info.get("publishedDate"):
        meta["date"] = info["publishedDate"]
    meta["isbn"] = isbn
    return meta
