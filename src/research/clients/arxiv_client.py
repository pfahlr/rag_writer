#!/usr/bin/env python3
"""
ARXIV Client for ARXIV_ID enrichment

expose:
  enrich_via_arxiv(isbn: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]

Returns a subset:
  ( arxiv_doi, arxiv_id, published, updated, summary, title, authors, pdf_url, scholar_url, primary_category, secondary_category, scheme )
"""
from __future__ import annotations

import re
import requests
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET

ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(.+)", re.IGNORECASE)

ARXIV_ID = pattern = re.compile(r'\b(?P<yymm>\d{4})\.(?P<number>\d{4,5})(?:v(?P<version>\d+))?\b')

def fetch_arxiv_by_str_with_arxiv_id(input: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetch metadata from arXiv given a DOI like 10.48550/arXiv.XXXX."""
    m = ARXIV_DOI_RE.match(input)

    # ARXIV_ID is pretty much their own sort of DOI, and we may as well treat it as another variation that can be used in place of it.
    m2 = ARXIV_ID.match(input)

    if not m and not m2:
        return None

    arxiv_id = m2.group()
    arxiv_doi = m.group()

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            # request failed exit
            return None

       # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
        }

        xml_data = r.content

        # Parse the XML data
        tree = ET.ElementTree(ET.fromstring(xml_data))
        root = tree.getroot()


        # Use the namespace in the find call
        entry = root.find('atom:entry', ns)

        # To see the children tags and attributes:
        published = entry.find('atom:published', ns).text
        updated = entry.find('atom:updated', ns).text
        summary = entry.find('atom:summary', ns).text
        title = entry.find('atom:title', ns).text

        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None:
                authors.append(name_elem.text)

        pdf_url = None
        scholar_url = None
        for item in entry.findall('atom:link', ns):
            if ('title' in item.attrib and item.attrib['title'] == 'pdf') or ('type' in item.attrib and item.attrib['type'] == 'application/pdf'):
                pdf_url = item.attrib['href']
            elif 'type' in item.attrib and item.attrib['type'] == 'text/html':
                scholar_url = item.attrib['href']

        primary_category_elem = entry.find('arxiv:primary_category', ns)
        primary_category = primary_category_elem.attrib['term'] if primary_category_elem is not None else None

        secondary_category = None
        scheme = None
        for item in entry.findall('atom:category', ns):
            if item.attrib['term'] == primary_category:
                continue
            secondary_category = item.attrib['term']
            scheme = item.attrib['scheme']
            break  # Only take the first secondary category

        data = (
            'arxiv_doi': arxiv_doi,
            'arxiv_id': arxix_id,
            'published': published,
            'updated': updated,
            'summary': summary,
            'title': title,
            'authors': authors,
            'pdf_url': pdf_url,
            'scholar_url': scholar_url,
            'primary_category': primary_category,
            'secondary_category': secondary_category,
            'scheme': scheme
            )

        return data

# facade functions
def fetch_arxiv_by_doi(input: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    return fetch_arxiv_by_str_with_arxiv_id(input, timeout)

def enrich_via_arxiv(url_or_id: str) -> dict:
    return fetch_arxiv_by_str_with_arxiv_id(url_or_id)


