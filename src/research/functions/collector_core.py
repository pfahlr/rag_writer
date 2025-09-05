from __future__ import annotations

import json
import re
import requests
from pathlib import Path
from typing import List
import os
BASE_PATH = os.getenv('BASE_PATH', '../../')
print(BASE_PATH)
from src.globals import SRC_DIR, ROOT_DIR

try:
    from bs4 import BeautifulSoup  # type: ignore
    HAS_BEAUTIFULSOUP = True
except Exception:
    HAS_BEAUTIFULSOUP = False

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]")
_ISBN_RE = re.compile(r"\b(?:97[89][- ]?)?[0-9][- 0-9]{9,}[0-9Xx]\b")

from classes.article_metadata import ArticleMetadata

def _now_ts() -> str:
    return str(int(time.time()))

def _extract_pdf_links_from_html(html: str) -> List[str]:
    if not HAS_BS4:
        return []
    links: List[str] = []
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".pdf"):
            links.append(href)
    return sorted(list(dict.fromkeys(links)))

def _extract_pdf_links_from_xml(xml: str) -> List[str]:
    if not HAS_BS4:
        return []
    links: List[str] = []
    soup = BeautifulSoup(xml or "", "html.parser")
    for tag in soup.find_all(["pdf", "a"]):
        if tag.name == "pdf":
            val = (tag.text or "").strip()
            if val:
                links.append(val)
        elif tag.name == "a":
            href = (tag.get("href") or "").strip()
            if href.lower().endswith(".pdf"):
                links.append(href)
    return sorted(list(dict.fromkeys(links)))

def _load_manifest_links() -> List[str]:
    if not MANIFEST.exists():
        return []
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        entries = data.get("entries") if isinstance(data, dict) else data
        out: List[str] = []
        for e in entries or []:
            url = (e.get("pdf_url") or "").strip()
            if url:
                out.append(url)
        return sorted(list(dict.fromkeys(out)))
    except Exception:
        return []


def parse_google_scholar_html(html: str) -> List[ArticleMetadata]:
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    articles: List[ArticleMetadata] = []
    # Simple heuristic: each result block often contains a title link
    blocks = soup.find_all('div', class_=re.compile(r'gs_r|gs_ri|gs_scl|gs_or'))
    for div in blocks:
        a = div.find('a')
        title = (a.text or '').strip() if a else ''
        scholar_url = (a.get('href') or '').strip() if a else ''
        # Try to find a direct PDF link
        pdf_url = ''
        for link in div.find_all('a'):
            href = (link.get('href') or '').strip()
            if href.lower().endswith('.pdf'):
                pdf_url = href
                break
        # Basic DOI/ISBN detection from text
        text = div.get_text(" ", strip=True)
        doi = _extract_doi(text)
        isbn = _extract_isbn(text)
        articles.append(ArticleMetadata(title=title, scholar_url=scholar_url, pdf_url=pdf_url, doi=doi, isbn=isbn))
    return articles

def parse_xml_markup(xml_content: str) -> List[ArticleMetadata]:
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(xml_content, 'html.parser')
    out: List[ArticleMetadata] = []
    for i, entry in enumerate(soup.find_all(['article', 'book'])):
        def get_field(name: str) -> str:
            val = (entry.get(name) or '').strip()
            if not val:
                child = entry.find(name)
                if child:
                    val = child.get_text(strip=True)
            return val
        title = get_field('title') or get_field('publication') or f"Entry {i+1}"
        doi = get_field('doi')
        pdf = get_field('pdf')
        if pdf and not (pdf.startswith('http://') or pdf.startswith('https://') or pdf.startswith('file://')):
            pdf = f"file://{Path(pdf).resolve()}"
        date = get_field('date')
        authors = [a.strip() for a in get_field('author').split(',') if a.strip()] if get_field('author') else []
        publication = get_field('publication')
        scholar_url = get_field('scholar_url')
        isbn = get_field('isbn')
        out.append(ArticleMetadata(title=title, doi=doi, pdf_url=pdf, date=date, authors=authors, publication=publication, scholar_url=scholar_url, isbn=isbn))
    return out

def _extract_doi(text: str) -> str:
    m = _DOI_RE.search(text or '')
    return (m.group(0).lower() if m else '')

def _extract_isbn(text: str) -> str:
    m = _ISBN_RE.search(text or '')
    return re.sub(r"[- ]", "", m.group(0)) if m else ''
