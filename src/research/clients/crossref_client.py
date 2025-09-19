#!/usr/bin/env python3
"""
Crossref client (fetch/search) with robust parsing.
Exposes:
  - fetch_crossref_by_doi(doi: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]
  - search_crossref(title: str, author: str = "", timeout: float = 10.0) -> Optional[Dict[str, Any]]
  - enrich_via_crossref(doi: Optional[str] = None, title: Optional[str] = None,
                        author: str = "", timeout: float = 10.0) -> Optional[Dict[str, Any]]
Return payloads include keys: title, authors[list], publication, date, doi
"""

from typing import Any, Dict, List, Optional
import requests

USER_AGENT = "pdf-fetcher/1.0 (+https://crossref.org)"

def _normalize_authors(auth: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for a in auth or []:
        lit = a.get("literal")
        if lit:
            out.append(lit)
            continue
        given, family = a.get("given"), a.get("family")
        name = " ".join([p for p in [given, family] if p]).strip()
        if name:
            out.append(name)
    return out

def _message_to_meta(msg: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if msg.get("title"):
        meta["title"] = msg["title"][0]
    if msg.get("author"):
        meta["authors"] = _normalize_authors(msg["author"])
    if msg.get("container-title"):
        meta["publication"] = msg["container-title"][0]
    # prefer 'issued', fallback to others
    for key in ("issued", "published-print", "published-online"):
        v = msg.get(key, {})
        parts = v.get("date-parts", [])
        if parts and parts[0]:
            meta["date"] = "-".join(str(x) for x in parts[0])
            break
    if msg.get("DOI"):
        meta["doi"] = msg["DOI"].lower()
    return meta

def fetch_crossref_by_doi(doi: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{requests.utils.quote(doi)}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if not r.ok:
        return None
    msg = r.json().get("message", {})
    return _message_to_meta(msg)

def search_crossref(title: str, author: str = "", timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    if not title:
        return None
    params = {"query.title": title, "rows": 1}
    if author:
        params["query.author"] = author

    r = requests.get("https://api.crossref.org/works", params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=timeout)

    if not r.ok:
        return None
    items = r.json().get("message", {}).get("items", [])
    if not items:
        return None
    return _message_to_meta(items[0])

def enrich_via_crossref(doi: Optional[str] = None, title: Optional[str] = None,
                        author: str = "", timeout: float = 10.0, headers={}) -> Optional[Dict[str, Any]]:
    """
    If DOI available: fetch by DOI.
    Else: search by title (+optional author).
    """
    if doi:
        meta = fetch_crossref_by_doi(doi, timeout=timeout)
        if meta:
            return meta
    if title:
        meta = search_crossref(title, author=author or "", timeout=timeout)
        if meta:
            return meta
    return None

