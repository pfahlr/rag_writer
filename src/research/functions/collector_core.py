"""Core parsing helpers for the research collector."""

from __future__ import annotations

import json
import re
import time
import requests
from pathlib import Path
from typing import List
from textual.color import Color
from functions.filelogger import _fllog

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text

headers={
  'sec-ch-ua':'"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
  'Content-Type': "application/json; charset=utf-8",
  'Accept-Language': "en-US,en;q=0.9",
  'Accept-Ranges': "bytes",
  'sec-ch-ua-platform': "Linux",
  'sec-fetch-dest': "empty",
  'Priority': "u=4, i",
  'Dnt': "1",
  'sec-ch-ua-mobile': '?0',
  'Accept': "*/*",
  'Accept-Encoding': "gzip, deflate, br, zstd",
  'sec-fetch-mode': "cors",
  'sec-fetch-site': "same-origin",
  'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
}
console = Console()

try:  # BeautifulSoup is optional at runtime
    from bs4 import BeautifulSoup  # type: ignore
    HAS_BEAUTIFULSOUP = True
except Exception:  # pragma: no cover - optional dependency
    HAS_BEAUTIFULSOUP = False

from classes.article_metadata import ArticleMetadata

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]")
_ISBN_RE = re.compile(r"\b(?:97[89][- ]?)?[0-9][- 0-9]{9,}[0-9Xx]\b")
OUT_DIR = Path("../../research/out")

def _now_ts() -> str:
    """Return the current timestamp as a string."""
    return str(int(time.time()))


def _extract_pdf_links_from_html(html: str) -> List[str]:
    """Return a deduplicated list of PDF links found in HTML."""
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(html or "", "html.parser")
    links = [
        (a.get("href") or "").strip()
        for a in soup.find_all("a")
        if (a.get("href") or "").strip().lower().endswith(".pdf")
    ]
    # Deduplicate while preserving order
    return list(dict.fromkeys(links))


def _extract_pdf_links_from_xml(xml: str) -> List[str]:
    """Return a deduplicated list of PDF links found in XML-like markup."""
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(xml or "", "html.parser")
    links: List[str] = []
    for tag in soup.find_all(["pdf", "a"]):
        if tag.name == "pdf":
            val = (tag.text or "").strip()
            if val:
                links.append(val)
        else:  # anchor tag
            href = (tag.get("href") or "").strip()
            if href.lower().endswith(".pdf"):
                links.append(href)
    return list(dict.fromkeys(links))


def _load_manifest_links(manifest: Path | None = None) -> List[str]:
    """Load existing PDF links from a manifest file."""
    manifest = manifest or Path("../research/out/manifest.json")
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    links: List[str] = []
    for e in entries or []:
        url = (e.get("pdf_url") or "").strip()
        if url:
            links.append(url)
    return list(dict.fromkeys(links))

def to_dict(article) -> Dict:
    """Convert to dictionary for JSON serialization."""
    data = asdict(article)
    # Convert authors list to comma-separated string for display
    data['authors_str'] = ', '.join(article.authors) if article.authors else ''
    return data

def slugify_title(title, date) -> str:
    """Create a slugified filename from title and year."""
    if not title:
        return f"untitled_{int(time.time())}"

    # Extract year from date if available
    year = ""
    if date:
        # Try to extract 4-digit year
        year_match = re.search(r'\b(20\d{2})\b', date)
        if year_match:
            year = f"_{year_match.group(1)}"

    # Slugify title
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[\s_-]+', '_', slug)
    slug = slug.strip('_')

    # Limit length and add year
    if len(slug) > 50:
        slug = slug[:50].rstrip('_')

    return f"{slug}{year}.pdf"

def parse_xml_markup_simple(xml_content: str) -> List[ArticleMetadata]:
    """Parse simplified XML-like markup into article metadata objects."""
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(xml_content, "html.parser")
    out: List[ArticleMetadata] = []
    for i, entry in enumerate(soup.find_all(["article", "book"])):
        def get_field(name: str) -> str:
            val = (entry.get(name) or "").strip()
            if not val:
                child = entry.find(name)
                if child:
                    val = child.get_text(strip=True)
            return val

        title = get_field("title") or get_field("publication") or f"Entry {i+1}"
        doi = get_field("doi")
        pdf = get_field("pdf")
        if pdf and not pdf.startswith(("http://", "https://", "file://")):
            pdf = f"file://{Path(pdf).resolve()}"
        date = get_field("date")
        author_field = get_field("author")
        authors = [a.strip() for a in author_field.split(",")] if author_field else []
        publication = get_field("publication")
        scholar_url = get_field("scholar_url")
        isbn = get_field("isbn")
        out.append(
            ArticleMetadata(
                title=title,
                doi=doi,
                pdf_url=pdf,
                date=date,
                authors=authors,
                publication=publication,
                scholar_url=scholar_url,
                isbn=isbn,
            )
        )
    return out

def parse_google_scholar_html_simple(html: str) -> List[ArticleMetadata]:
    """Parse Google Scholar result HTML into article metadata objects."""
    if not HAS_BEAUTIFULSOUP:
        return []
    soup = BeautifulSoup(html, "html.parser")
    articles: List[ArticleMetadata] = []
    blocks = soup.find_all("div", class_=re.compile(r"gs_r|gs_ri|gs_scl|gs_or"))
    for div in blocks:
        a = div.find("a")
        title = (a.text or "").strip() if a else ""
        scholar_url = (a.get("href") or "").strip() if a else ""
        pdf_url = ""
        for link in div.find_all("a"):
            href = (link.get("href") or "").strip()
            if href.lower().endswith(".pdf"):
                pdf_url = href
                break
        text = div.get_text(" ", strip=True)
        doi = _extract_doi(text)
        isbn = _extract_isbn(text)
        articles.append(
            ArticleMetadata(
                title=title,
                scholar_url=scholar_url,
                pdf_url=pdf_url,
                doi=doi,
                isbn=isbn,
            )
        )
    return articles

def parse_xml_markup(xml_content: str) -> List[ArticleMetadata]:
    """Parse proprietary XML markup for articles and books."""
    if not HAS_BEAUTIFULSOUP:
        console.print("[red]Error: BeautifulSoup required for XML parsing. Install with: pip install beautifulsoup4[/red]")
        sys.exit(1)

    soup = BeautifulSoup(xml_content, 'html')  # Use html parser for better handling of malformed XML
    articles = []

    # Find all article and book tags
    entries = soup.find_all(['article', 'book'])

    for i, entry in enumerate(entries):
        article = ArticleMetadata()

        # Helper function to get field value from attr or child tag
        def get_field(field_name):
            value = entry.get(field_name, '').strip()
            if not value:
                child = entry.find(field_name)
                if child:
                    value = child.get_text(strip=True)
            return value

        # Extract title (use provided title, or publication, or default)
        article.title = get_field('title') or get_field('publication') or f"Entry {i+1} from XML"

        # Extract fields
        article.doi = get_field('doi')
        pdf_value = get_field('pdf')
        if pdf_value:
            if pdf_value.startswith(('http://', 'https://')):
                article.pdf_url = pdf_value
            elif pdf_value.startswith('file://'):
                # Already a file:// URL, use as is
                article.pdf_url = pdf_value
            else:
                # Local path, convert to file:// URL
                article.pdf_url = f"file://{Path(pdf_value).resolve()}"
        article.date = get_field('date')
        author_value = get_field('author')
        if author_value:
            article.authors = [a.strip() for a in author_value.split(',') if a.strip()]
        article.publication = get_field('publication')
        article.scholar_url = get_field('scholar_url')
        article.isbn = get_field('isbn')

        articles.append(article)
        console.print(f"[dim]Parsed XML entry: {article.title[:50]}...[/dim]")

    console.print(f"[green]Parsed {len(articles)} entries from XML[/green]")
    return articles

def parse_google_scholar_html(html: str) -> List[ArticleMetadata]:
    """Parse Google Scholar HTML to extract article metadata."""
    if not HAS_BEAUTIFULSOUP:
        console.print("[red]Error: BeautifulSoup required. Install with: pip install beautifulsoup4[/red]")
        sys.exit(1)

    console.print(f"[dim]HTML length: {len(html)} characters[/dim]")
    soup = BeautifulSoup(html, 'html.parser')
    console.print(f"[dim]Soup title: {soup.title}[/dim]")

    articles = []
    seen_titles = set()
    
    # Find all article entries (Google Scholar uses various class patterns)
    article_divs = soup.find_all('div', class_=re.compile(r'gs_r|gs_ri|gs_scl|gs_or'))
    
    for div in article_divs:
        article = ArticleMetadata()

        # Extract title and scholar URL
        title_elem = div.find('h3', class_=re.compile(r'gs_rt')) or div.find('h3')
        if title_elem:
            title_link = title_elem.find('a')
            if title_link:
                article.title = title_link.get_text(strip=True)
                article.scholar_url = title_link.get('href', '')

                if article.scholar_url and not article.scholar_url.startswith('http'):
                    article.scholar_url = urljoin('https://scholar.google.com', article.scholar_url)

                _fllog(f"Extracted title: {article.title}")
                _fllog(f"Scholar URL: {article.scholar_url}")

                # Try to extract DOI from scholar_url if not already found
                if not article.doi:
                    doi_match = re.search(r'10\.\d+/.+?(?=/|$|\?)', article.scholar_url)
                    if doi_match:
                        article.doi = doi_match.group(0)
                        _fllog(f"Extracted DOI from scholar_url: {article.doi}")

                # Check if title link is a PDF or construct PDF URL for arxiv
                if 'arxiv.org' in article.scholar_url and '/abs/' in article.scholar_url:
                    article.pdf_url = article.scholar_url.replace('/abs/', '/pdf/')
                    article.pdf_source_url = article.pdf_url
                    _fllog(f"Constructed PDF URL from arxiv abs: {article.pdf_url}")
                elif 'pdf' in article.scholar_url.lower():
                    article.pdf_url = article.scholar_url
                    article.pdf_source_url = article.pdf_url
                    _fllog(f"Found PDF URL in title link: {article.pdf_url}")

        # Extract authors, publication, and date from gs_a div
        authors_div = div.find('div', class_=re.compile(r'gs_a'))
        if authors_div:
            authors_text = authors_div.get_text(strip=True)
            console.print(f"[dim]Parsing authors text: {authors_text}[/dim]")

            # Handle format: "Author1, Author2 - Journal Name, Year - Publisher"
            parts = authors_text.split('-')
            if len(parts) >= 1:
                # item[0] remains in authors field
                authors_part = parts[0].strip()
                article.authors = [a.strip() for a in authors_part.split(',') if a.strip()]
                _fllog(f"Extracted authors: {article.authors}")

            if len(parts) >= 2:
                # item[1] split on ','
                pub_info = parts[1].strip()
                pub_parts = pub_info.split(',')

                if len(pub_parts) >= 1:
                    # split_item[0] goes in publication
                    article.publication = pub_parts[0].strip()
                    
                if len(pub_parts) >= 2:
                    # split_item[1] goes in date if date is empty
                    date_part = pub_parts[1].strip()
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_part)
                    if year_match and not article.date:  # Only set if date is empty
                        article.date = year_match.group(0)
                        _fllog(f"Extracted date: {article.date}")
        
        # Extract DOI from various sources
        doi_links = div.find_all('a', href=re.compile(r'doi\.org|doi:'))
        for link in doi_links:
            href = link.get('href', '')
            doi_match = re.search(r'(?:doi\.org/|doi:)(.+)', href)
            if doi_match:
                article.doi = unquote(doi_match.group(1).strip())
                _fllog(f"Extracted DOI from link: {article.doi}")
                break

        # If not found in links, try to find DOI in text content of this article div
        if not article.doi:
            doi_pattern = re.compile(r'10\.\d+/.+?(?=\s|$|[^\w/.-])')
            div_text = div.get_text()
            doi_match = doi_pattern.search(div_text)
            if doi_match:
                article.doi = doi_match.group(0).strip()
                _fllog(f"Extracted DOI from text: {article.doi}")
        
        # Look for PDF links in multiple locations
        pdf_links = div.find_all('a', href=re.compile(r'\.pdf'))
        for link in pdf_links:
            href = link.get('href', '')
            if href and not href.startswith('javascript:'):
                article.pdf_url = urljoin('https://scholar.google.com', href) if not href.startswith('http') else href
                article.source_pdf_url = article.pdf_url
                _fllog(f"Found PDF URL in div regex: {article.pdf_url}")
                break

        # Also check for PDF links in the article's data attributes or other elements
        if not article.pdf_url:
            all_links = div.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                if '.pdf' in href.lower() and not href.startswith('javascript:'):
                    article.pdf_url = urljoin('https://scholar.google.com', href) if not href.startswith('http') else href
                    article.pdf_source_url = article.pdf_url
                    _fllog(f"Found PDF URL in all_links: {article.pdf_url}")
                    break

        pdf_link_div = div.find(class_="gs_or_ggsm")
        if pdf_link_div:
            real_pdf_link = pdf_link_div.find('a')
            real_pdf_link_href = real_pdf_link.get('href','')
            if real_pdf_link_href and real_pdf_link_href.startswith('http'):
                article.pdf_url = real_pdf_link_href

        # Look for PDF links in gs_or_ggsm divs (child elements)
#        if not article.pdf_url:
#            next_div = div.find('div', class_=re.compile(r'gs_or_ggsm|gs_or|gs_.*'))
#            if next_div:
#                _fllog(f"Found gs_or_ggsm div for article: {article.title}")
#                pdf_links = next_div.find_all('a', href=True)
#                _fllog("why isn't it reaching here?3")
#                for link in pdf_links:
#                    href = link.get('href', '')
#                    if href and not href.startswith('javascript:'):
#                        article.pdf_url = urljoin('https://scholar.google.com', href) if not href.startswith('http') else href
#                        article.pdf_source_url = article.pdf_url
#                        _fllog(f"Found PDF URL in gs_or_ggsm: {article.pdf_url}")

        # Only add if we have at least a title or DOI
        if article.title or article.doi:
            # Avoid duplicates
            if article.title and article.title not in seen_titles:
                seen_titles.add(article.title)
                articles.append(article)
                console.print(f"[dim]Added article: {article.title[:50]}...[/dim]")
            elif not article.title and article.doi:
                # For articles without title, use DOI to check uniqueness
                if article.doi not in seen_titles:
                    seen_titles.add(article.doi)
                    articles.append(article)
                    console.print(f"[dim]Added article (DOI): {article.doi}...[/dim]")
    
    console.print(f"[green]Parsed {len(articles)} articles from HTML[/green]")

    # Debug: Print what we found
    for i, article in enumerate(articles):
        console.print(f"[dim]Article {i+1}: {article.title[:50]}...[/dim]")
        console.print(f"[dim]  Authors: {article.authors}[/dim]")
        console.print(f"[dim]  Date: {article.date}[/dim]")
        console.print(f"[dim]  Publication: {article.publication}[/dim]")
        console.print(f"[dim]  DOI: {article.doi}[/dim]")
        console.print(f"[dim]  ISBN: {article.isbn}[/dim]")
        console.print(f"[dim]  PDF: {article.pdf_url}[/dim]")

    return articles

def complete_article_fields(article: ArticleMetadata):
    """Complete missing fields by parsing the article's Google Scholar page."""
    
    _fllog(f"Completing fields for article: {article['title']}")

    if not article['scholar_url'] or article['scholar_url'].endswith('.pdf'):
        _fllog("No scholar URL or it's already a PDF")
        return

    try:
        _fllog(f"Fetching HTML from: {article['scholar_url']}")
        response = requests.get(article['scholar_url'], headers=headers, timeout=30)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')

        # Extract title if not set
        if not article['title']:
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                article['title'] = title_elem.get_text(strip=True)
                _fllog(f"Extracted title: {article['title']}")

        # Extract authors (update even if already set) - DISABLED
        # Look for author information (Google Scholar specific)
        # author_divs = soup.find_all('div', class_=re.compile(r'gs_a|author'))
        # for div in author_divs:
        #     authors_text = div.get_text(strip=True)
        #     if authors_text:
        #         article.authors = [a.strip() for a in authors_text.split(',') if a.strip()]
        #         _fllog(f"Extracted authors: {article.authors}")
        #         break

        # Extract date (update even if already set)
        # Look for date patterns
        date_patterns = soup.find_all(text=re.compile(r'\b(19|20)\d{2}\b'))
        for text in date_patterns:
            year_match = re.search(r'\b(19|20)\d{2}\b', text)
            if year_match:
                article['date'] = year_match.group(0)
                _fllog(f"Extracted date: {article['date']}")
                break

        # Extract DOI if not set
        if not article['doi']:
            # First check if DOI is in the scholar_url itself
            doi_match = re.search(r'(?:doi\.org/|doi:)(.+?)(?=&|$)', article['scholar_url'])
            if doi_match:
                article['doi'] = unquote(doi_match.group(1).strip())
                _fllog(f"Extracted DOI from scholar_url: {article['doi']}")
            else:
                # Try links with doi in href
                doi_links = soup.find_all('a', href=re.compile(r'doi\.org|doi:'))
                for link in doi_links:
                    href = link.get('href', '')
                    doi_match = re.search(r'(?:doi\.org/|doi:)(.+)', href)
                    if doi_match:
                        article['doi'] = unquote(doi_match.group(1).strip())
                        _fllog(f"Extracted DOI from link: {article['doi']}")
                        break

                # If not found in links, try to find DOI in text content
                if not article['doi']:
                    doi_pattern = re.compile(r'10\.\d+/.+?(?=\s|$|[^\w/.-])')
                    text_content = soup.get_text()
                    doi_match = doi_pattern.search(text_content)
                    if doi_match:
                        article['doi'] = doi_match.group(0).strip()
                        _fllog(f"Extracted DOI from text: {article['doi']}")

        # Extract ISBN if not set
        if not article['isbn']:
            # Try to find ISBN in text content
            isbn_pattern = re.compile(r'\b(?:ISBN(?:-1[03])?:?\s*)?(?=[0-9X]{10}|(?=(?:[0-9]+[-\s]){3})[-\s0-9X]{13}|97[89][0-9]{10}|(?=(?:[0-9]+[-\s]){4})[-\s0-9]{17})(?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9X]\b')
            text_content = soup.get_text()
            isbn_match = isbn_pattern.search(text_content)
            if isbn_match:
                article['isbn'] = isbn_match.group(0).strip()
                _fllog(f"Extracted ISBN from text: {article['isbn']}")
        # Extract publication (update even if already set) - DISABLED
        # Look for publication info
        # pub_divs = soup.find_all('div', class_=re.compile(r'gs_pub|publication'))
        # for div in pub_divs:
        #     pub_text = div.get_text(strip=True)
        #     if pub_text:
        #         article.publication = pub_text
        #         _fllog(f"Extracted publication: {article.publication}")
        #         break
        _fllog("Fields completion finished")


    except Exception as e:
        _fllog(f"Error completing fields: {e}")
    
    return article

def download_pdf(article: ArticleMetadata) -> bool:
    print('download_pdf function')
    """Download PDF for an article."""
    if not article['pdf_url']:
        console.print("[yellow]No PDF URL available[/yellow]")
        return False

    try:
        console.print(f"[dim]Downloading PDF from: {article['pdf_url']}[/dim]")

        response = requests.get(article['pdf_url'], headers=headers, timeout=30)
        response.raise_for_status()

        # Generate filename
        filename = article.slugify_title()
        filepath = OUT_DIR / filename
    
        metadata = {
            'title':article['title'],
            'date':article['date'],
            'publication':article['publication'],
            'authors':article['authors'],
            'doi':article['doi'],
            'web_url':article['scholar_url'],
            'pdf_source_url':article['pdf_url'],
            'metadata_created_by':'RAG content generation suite developed by Rick Pfahl 2025'         
        }

        self.notify("Downloading PDF...", severity="information")
        filename = slugify_title(article['title'],article['date'])
        save_pdf(article['pdf_url'], filename, metadata, './out', tmp_path="/tmp")


        # Save PDF
        with open(filepath, 'wb') as f:
            f.write(response.content)

        article['pdf_filename'] = filename
        article['downloaded'] = True

        console.print(f"[green]Downloaded: {filename}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Error downloading PDF: {e}[/red]")
        return False

def _extract_doi(text: str) -> str:
    """Extract a DOI from free text."""
    m = _DOI_RE.search(text or "")
    return m.group(0).lower() if m else ""


def _extract_isbn(text: str) -> str:
    """Extract an ISBN from free text."""
    m = _ISBN_RE.search(text or "")
    return re.sub(r"[- ]", "", m.group(0)) if m else ""

