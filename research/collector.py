#!/usr/bin/env python3
"""
Research Collector - Extract and manage academic sources from Google Scholar

This script processes Google Scholar search results (URL or saved HTML) to:
1. Extract article metadata (title, authors, date, publication, DOI)
2. Find PDF download links
3. Provide an interactive terminal form for editing metadata
4. Download PDFs with automatic slugified naming
5. Store metadata in PDF files and maintain a manifest.json

Usage:
    python research/collector.py --url "https://scholar.google.com/scholar?q=..."
    python research/collector.py --file "path/to/saved_search.html"
"""

import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 



import json
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urljoin, quote, unquote
from dataclasses import dataclass, asdict
from datetime import datetime
from pdfwriter import save_pdf
from textual.app import App, ComposeResult
from rich.pretty import pprint
from filelogger import _fllog



import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
import subprocess
import shlex
from slugify import slugify

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Input, Button, Header, Footer, Label, Static
    from textual.containers import Vertical, Horizontal, Container
    from textual.binding import Binding
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

from textual.widgets import Link
import webbrowser

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

@dataclass
class ArticleMetadata:
    """Metadata for an academic article."""
    title: str = ""
    authors: List[str] = None
    date: str = ""
    publication: str = ""
    doi: str = ""
    pdf_url: str = ""
    pdf_source_url: str = ""
    scholar_url: str = ""
    pdf_filename: str = ""
    downloaded: bool = False

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert authors list to comma-separated string for display
        data['authors_str'] = ', '.join(self.authors) if self.authors else ''
        return data

    def slugify_title(self) -> str:
        """Create a slugified filename from title and year."""
        if not self.title:
            return f"untitled_{int(time.time())}"

        # Extract year from date if available
        year = ""
        if self.date:
            # Try to extract 4-digit year
            year_match = re.search(r'\b(20\d{2})\b', self.date)
            if year_match:
                year = f"_{year_match.group(1)}"

        # Slugify title
        slug = re.sub(r'[^\w\s-]', '', self.title.lower())
        slug = re.sub(r'[\s_-]+', '_', slug)
        slug = slug.strip('_')

        # Limit length and add year
        if len(slug) > 50:
            slug = slug[:50].rstrip('_')

        return f"{slug}{year}.pdf"

class ArticleFormApp(App):

    CSS_PATH = "styles.tcss"

    """Simple Textual app for editing article metadata."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save", "Save"),
        ("d", "download", "Download PDF"),
        ("ctrl+right", "action_next", "Next Article"),
        ("ctrl+left", "action_previous", "Previous Article"),
    ]

    def __init__(self, collector, article_index):
        super().__init__()
        self.collector = collector
        self.article_index = article_index
        self.article = collector.articles[article_index]

    def compose(self) -> ComposeResult:

        """Create the form layout."""
        yield Header(show_clock=True)

        with Vertical(classes="form-container"):
            # Title
            with Vertical(classes="heading-bar"):
                title_text = self.article.title[:60] + "..." if len(self.article.title) > 60 else self.article.title
                yield Label(f"Research Collector - Article {self.article_index + 1}/{len(self.collector.articles)}",
                    classes="form-title")
                # Article title display
                yield Label(f"Title: {title_text}", classes="status-text")

            # Two-column layout for form fields
            with Horizontal(classes="horizontal topfour"):
                # Left column
                with Vertical(classes="field-container"):
                    yield Label("Date", classes="field-label")
                    date_input = Input(
                        placeholder="2023",
                        id="date_input",
                        classes="form-field form-field-date"
                    )
                    date_input.value = self.article.date or ''
                    yield date_input

                    yield Label("DOI", classes="field-label")
                    doi_input = Input(
                        placeholder="10.1000/example",
                        id="doi_input",
                        classes="form-field form-field-doi"
                    )
                    doi_input.value = self.article.doi or ''
                    yield doi_input

                # Right column
                with Vertical(classes="field-container"):
                    yield Label("Authors", classes="field-label")
                    authors_input = Input(
                        placeholder="Author1, Author2, Author3",
                        id="authors_input",
                        classes="form-field form-field-author"
                    )
                    authors_input.value = ', '.join(self.article.authors) if self.article.authors else ''
                    yield authors_input

                    yield Label("Publication", classes="field-label")
                    pub_input = Input(
                        placeholder="Journal Name",
                        id="publication_input",
                        classes="form-field form-field-publication"
                    )
                    pub_input.value = self.article.publication or ''
                    yield pub_input

            # PDF URL row
            with Horizontal(classes="pdf-url-container"):
                with Vertical():
                    yield Label("PDF URL", classes="field-label")
                    pdf_input = Input(
                        placeholder="https://example.com/paper.pdf",
                        id="pdf_input",
                        classes="form-field form-field-file"
                    )
                    pdf_input.value = self.article.pdf_url or ''
                    yield pdf_input
                if not self.article.downloaded and self.article.pdf_url:
                    yield Button("Download", variant="primary", id="download_button", classes="button download-button")

            with Horizontal(classes="pdf-source-url-container"):
                with Vertical():
                    yield Label("PDF SOURCE URL", classes="field-label")
                    pdf_source_url = Input(
                        placeholder="https://example.com/paper.pdf",
                        id="pdf_source_url",
                        classes="form-field form-field-file-source"
                    )
                    pdf_source_url.value = self.article.pdf_source_url or ''
                    yield pdf_source_url

            with Horizontal(classes="web-url-container"):
                with Vertical():
                    yield Label("Scholar (web) URL", classes="field-label")
                    scholar_url = Input(
                        placeholder="ARTICLE WEB URL",
                        id="scholar_url",
                        classes="form-field form-field-url"
                    )
                    scholar_url.value = self.article.scholar_url
                    yield scholar_url
                    with Horizontal():
                        yield Button("Open Page", variant="primary", id="open_page_button", classes="button open-page-button")
                        yield Button("Complete Fields", variant="primary", id="complete_fields_button", classes="button complete-fields-button")
            #with Vertical(classes="form-bottom-container"):
            #    with Horizontal(classes="status-text-container"):
            #        pass

            # Action buttons
            with Horizontal():
                yield Button("Previous", variant="default", id="prev_button", classes="button prev-button")
                yield Button("Save", variant="primary", id="save_button", classes="button save-button")
                yield Button("Next", variant="default", id="next_button", classes="button next-button")
                yield Button("Quit", variant="error", id="quit_button", classes="button quit-button")

        yield Footer()

    def on_button_pressed(self, event):
        """Handle button presses."""
        _fllog("button pressed")
        _fllog(event.button.id)
        if event.button.id == "download_button":
            print('download button pressed')
            self.action_download()
        elif event.button.id == "save_button":
            self.action_save()
        elif event.button.id == "prev_button":
            self.action_previous()
        elif event.button.id == "next_button":
            self.action_next()
        elif event.button.id == "quit_button":
            self.action_quit()
        elif event.button.id == "open_page_button":
            webbrowser.open(self.article.scholar_url)
        elif event.button.id == "complete_fields_button":
            self.action_complete_fields()

    def action_complete_fields(self):
        _fllog("Completing fields button pressed")
        self.collector.complete_article_fields(self.article)
        # Update form inputs with new values
        authors_input = self.query_one("#authors_input", Input)
        authors_input.value = ', '.join(self.article.authors) if self.article.authors else ''
        authors_input.refresh()
        date_input = self.query_one("#date_input", Input)
        date_input.value = self.article.date or ''
        date_input.refresh()
        pub_input = self.query_one("#publication_input", Input)
        pub_input.value = self.article.publication or ''
        pub_input.refresh()
        doi_input = self.query_one("#doi_input", Input)
        doi_input.value = self.article.doi or ''
        doi_input.refresh()
        pdf_input = self.query_one("#pdf_input", Input)
        pdf_input.value = self.article.pdf_url or ''
        pdf_input.refresh()
        pdf_source_url_input = self.query_one("#pdf_source_url", Input)
        pdf_source_url_input.value = self.article.pdf_source_url or ''
        pdf_source_url_input.refresh()
        scholar_url_input = self.query_one("#scholar_url", Input)
        scholar_url_input.value = self.article.scholar_url
        scholar_url_input.refresh()
        self.notify("Fields completed", severity="success")
        _fllog("Fields completion done")

    def action_download(self):
        """Download PDF action."""
        _fllog('download action: BUTTON')
        if not self.article.downloaded and self.article.pdf_url:
            authors_input = self.query_one("#authors_input", Input)
            date_input = self.query_one("#date_input", Input)
            pub_input = self.query_one("#publication_input", Input)
            doi_input = self.query_one("#doi_input", Input)
            pdf_input = self.query_one("#pdf_input", Input)

            meta_metadata = {
             #'title':self.article.title,
             'date':self.article.date,
             'publication':self.article.publication,
             'authors':self.article.authors,
             'doi':self.article.doi,
             'web_url':self.article.scholar_url,
             'pdf_source_url':self.article.pdf_source_url,
            }
            metadata = {
                '/CreationDate':self.article.date,
                '/Publication':self.article.publication,
                '/Author':self.article.authors,
                '/DOI':self.article.doi,
                '/WebUrl': self.article.scholar_url,
                '/Metadata': json.dumps(meta_metadata),
            }
            self.notify("Downloading PDF...", severity="information")
            filename = self.article.slugify_title()
            _fllog(filename)
            _fllog(self.article.pdf_url)
            _fllog(filename)
            _fllog(ROOT_DIR)
            stored_filepath = save_pdf(self.article.pdf_url, filename, metadata, '/srv/IOMEGA_EXTERNAL/rag_writer/research/out', tmp_path='/srv/IOMEGA_EXTERNAL/rag_writer/research/tmp')
            if stored_filepath is not None:
              self.article.pdf_url = "file://"+stored_filepath
              # Update form inputs
              pdf_input = self.query_one("#pdf_input", Input)
              pdf_input.value = self.article.pdf_url
              pdf_input.refresh()
              pdf_source_url_input = self.query_one("#pdf_source_url", Input)
              pdf_source_url_input.value = self.article.pdf_source_url or ''
              pdf_source_url_input.refresh()
            else:
              _fllog("opening url"+str(self.article.pdf_url))
              webbrowser.open(self.article.pdf_url)

    def action_save(self):
        """Save changes action."""
        # Update article data from form inputs
        authors_input = self.query_one("#authors_input", Input)
        date_input = self.query_one("#date_input", Input)
        pub_input = self.query_one("#publication_input", Input)
        doi_input = self.query_one("#doi_input", Input)
        pdf_input = self.query_one("#pdf_input", Input)

        self.article.authors = [a.strip() for a in authors_input.value.split(',') if a.strip()]
        self.article.date = date_input.value
        self.article.publication = pub_input.value
        self.article.doi = doi_input.value
        self.article.pdf_url = pdf_input.value
        self.article.pdf_source_url = pdf_input.value

        self.collector.save_manifest()
        self.notify("Changes saved!", severity="success")

    def action_next(self):

        if self.article_index < len(self.collector.articles) - 1:
            self.collector.current_index += 1
            self.exit(True)  # Signal to redisplay form
        else:
            self.notify("Already at last article", severity="warning")

    def action_previous(self):
        """Previous article action."""
        if self.article_index > 0:
            self.collector.current_index -= 1
            self.exit(True)  # Signal to redisplay form
        else:
            self.notify("Already at first article", severity="warning")

    def action_quit(self):
        """Quit action."""
        self.collector.save_manifest()
        self.exit(False)

class ResearchCollector:
    """Main research collector class."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("research/out")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.output_dir / "manifest.json"
        self.articles: List[ArticleMetadata] = []
        self.current_index = 0

        # Load existing manifest if it exists
        self.load_manifest()

    def load_manifest(self):
        """Load existing manifest file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # Remove computed fields that shouldn't be passed to constructor
                        item_copy = item.copy()
                        item_copy.pop('authors_str', None)
                        article = ArticleMetadata(**item_copy)
                        self.articles.append(article)
                console.print(f"[green]Loaded {len(self.articles)} articles from manifest[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load manifest: {e}[/yellow]")

    def save_manifest(self):
        """Save current articles to manifest file."""
        try:
            data = [article.to_dict() for article in self.articles]
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved {len(self.articles)} articles to manifest[/green]")
        except Exception as e:
            console.print(f"[red]Error saving manifest: {e}[/red]")

    def fetch_html_from_url(self, url: str) -> str:
        """Fetch HTML content from Google Scholar URL."""
        console.print(f"[dim]Fetching HTML from: {url}[/dim]")

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            console.print(f"[red]Error fetching URL: {e}[/red]")
            sys.exit(1)

    def load_html_from_file(self, file_path: str) -> str:
        """Load HTML content from saved file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            sys.exit(1)

    def parse_google_scholar_html(self, html: str) -> List[ArticleMetadata]:
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

            # Look for PDF links in gs_or_ggsm divs (child elements)
            if not article.pdf_url:
                next_div = div.find('div', class_=re.compile(r'gs_or_ggsm|gs_or|gs_.*'))
                if next_div:
                    _fllog(f"Found gs_or_ggsm div for article: {article.title}")
                    pdf_links = next_div.find_all('a', href=True)
                    for link in pdf_links:
                        href = link.get('href', '')
                        if href and not href.startswith('javascript:'):
                            article.pdf_url = urljoin('https://scholar.google.com', href) if not href.startswith('http') else href
                            article.pdf_source_url = article.pdf_url
                            _fllog(f"Found PDF URL in gs_or_ggsm: {article.pdf_url}")
                            break

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
            console.print(f"[dim]  PDF: {article.pdf_url}[/dim]")

        return articles

    def complete_article_fields(self, article: ArticleMetadata):
        """Complete missing fields by parsing the article's Google Scholar page."""
        _fllog(f"Completing fields for article: {article.title}")

        if not article.scholar_url or article.scholar_url.endswith('.pdf'):
            _fllog("No scholar URL or it's already a PDF")
            return

        try:
            _fllog(f"Fetching HTML from: {article.scholar_url}")
            response = requests.get(article.scholar_url, headers=headers, timeout=30)
            response.raise_for_status()
            html = response.text
            _fllog(f"Fetched HTML length: {len(html)}")

            soup = BeautifulSoup(html, 'html.parser')

            # Extract title if not set
            if not article.title:
                title_elem = soup.find('h1') or soup.find('title')
                if title_elem:
                    article.title = title_elem.get_text(strip=True)
                    _fllog(f"Extracted title: {article.title}")

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
                    article.date = year_match.group(0)
                    _fllog(f"Extracted date: {article.date}")
                    break

            # Extract DOI if not set
            if not article.doi:
                # First check if DOI is in the scholar_url itself
                doi_match = re.search(r'(?:doi\.org/|doi:)(.+?)(?=&|$)', article.scholar_url)
                if doi_match:
                    article.doi = unquote(doi_match.group(1).strip())
                    _fllog(f"Extracted DOI from scholar_url: {article.doi}")
                else:
                    # Try links with doi in href
                    doi_links = soup.find_all('a', href=re.compile(r'doi\.org|doi:'))
                    for link in doi_links:
                        href = link.get('href', '')
                        doi_match = re.search(r'(?:doi\.org/|doi:)(.+)', href)
                        if doi_match:
                            article.doi = unquote(doi_match.group(1).strip())
                            _fllog(f"Extracted DOI from link: {article.doi}")
                            break

                    # If not found in links, try to find DOI in text content
                    if not article.doi:
                        doi_pattern = re.compile(r'10\.\d+/.+?(?=\s|$|[^\w/.-])')
                        text_content = soup.get_text()
                        doi_match = doi_pattern.search(text_content)
                        if doi_match:
                            article.doi = doi_match.group(0).strip()
                            _fllog(f"Extracted DOI from text: {article.doi}")

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

    def download_pdf(self, article: ArticleMetadata) -> bool:
        print('download_pdf function')
        """Download PDF for an article."""
        if not article.pdf_url:
            console.print("[yellow]No PDF URL available[/yellow]")
            return False

        try:
            console.print(f"[dim]Downloading PDF from: {article.pdf_url}[/dim]")

            response = requests.get(article.pdf_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Generate filename
            filename = article.slugify_title()
            filepath = self.output_dir / filename
            authors_input = self.query_one("#authors_input", Input)
            date_input = self.query_one("#date_input", Input)
            pub_input = self.query_one("#publication_input", Input)
            doi_input = self.query_one("#doi_input", Input)
            pdf_input = self.query_one("#pdf_input", Input)

            metadata = {
             'title':self.article.title,
             'date':self.article.date,
             'publication':self.article.publication,
             'authors':self.article.authors,
             'doi':self.article.doi,
             'web_url':self.article.scholar_url,
             'pdf_source_url':self.article.pdf_url,
             'authors_v2': authors_input,
             'date_v2': date_input,
             'pub_v2': pub_input,
             'doi_v2':doi_input,
             'pdf_link_v2':pdf_input,
             'metadata_created_by':'RAG content generation suite developed by Rick Pfahl 2025'         
            }

            self.notify("Downloading PDF...", severity="information")
            filename = self.article.slugify_title()
            save_pdf(self.article.pdf_url, filename, metadata, './out', tmp_path="/tmp")


            # Save PDF
            with open(filepath, 'wb') as f:
                f.write(response.content)

            article.pdf_filename = filename
            article.downloaded = True

            console.print(f"[green]Downloaded: {filename}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error downloading PDF: {e}[/red]")
            return False

    def display_article_form(self, article: ArticleMetadata) -> bool:
        """Display interactive form for editing article metadata using Textual."""
        if not HAS_TEXTUAL:
            console.print("[red]Error: Textual required for interactive form. Install with: pip install textual[/red]")
            return False

        try:
            app = ArticleFormApp(self, self.current_index)
            result = app.run()
            return result if result is not None else False
        except Exception as e:
            console.print(f"[red]Error running Textual app: {e}[/red]")
            console.print("[yellow]Falling back to console-based interface...[/yellow]")
            return self.display_fallback_form(article)

    def display_fallback_form(self, article: ArticleMetadata) -> bool:
        """Fallback console-based form when Textual is not available."""
        console.clear()

        # Display header
        console.print()
        console.print(Panel.fit(
            f"[bold blue]📚 Research Collector (Console Mode)[/bold blue]\n"
            f"[dim]Article {self.current_index + 1} of {len(self.articles)}[/dim]",
            border_style="blue"
        ))
        console.print()

        # Display article information in a clean table
        table = Table(title=f"[bold]{article.title[:60]}{'...' if len(article.title) > 60 else ''}[/bold]")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Title
        table.add_row("📖 Title", article.title or "[dim]Not extracted[/dim]")

        # Authors
        authors_str = ', '.join(article.authors) if article.authors else "[dim]Not extracted[/dim]"
        table.add_row("👥 Authors", authors_str)

        # Date
        table.add_row("📅 Date", article.date or "[dim]Not extracted[/dim]")

        # Publication
        table.add_row("🏛️ Publication", article.publication or "[dim]Not extracted[/dim]")

        # DOI
        table.add_row("🔗 DOI", article.doi or "[dim]Not extracted[/dim]")

        # PDF Status
        if article.downloaded:
            pdf_status = f"[green]✓ Downloaded: {article.pdf_filename}[/green]"
        elif article.pdf_url:
            pdf_status = f"[yellow]📄 Available (press 'd' to download)[/yellow]"
        else:
            pdf_status = "[red]❌ No PDF found[/red]"
        table.add_row("📑 PDF", pdf_status)

        # Scholar URL
        if article.scholar_url:
            # Escape square brackets in URL for Rich markup
            safe_url = article.scholar_url.replace("[", "\\[")
            safe_url = safe_url.replace("]", "\\]")
            scholar_display = f"[blue][link={safe_url}]View on Google Scholar[/link][/blue]"
        else:
            scholar_display = "[dim]No link available[/dim]"
        table.add_row("🔍 Google Scholar", scholar_display)

        console.print(table)
        console.print()

        # Display action buttons
        console.print("[bold]Available Actions:[/bold]")
        actions_table = Table(show_header=False, box=None)
        actions_table.add_column("", style="bold yellow")
        actions_table.add_column("", style="white")

        if not article.downloaded and article.pdf_url:
            actions_table.add_row("📥 (d)", "Download PDF")
        actions_table.add_row("💾 (s)", "Save changes")
        actions_table.add_row("⬅️  (p)", "Previous article")
        actions_table.add_row("➡️  (n)", "Next article")
        actions_table.add_row("❌ (q)", "Quit and save")
        actions_table.add_row("📝 [field name]", "Edit field (title, authors, date, publication, doi)")

        console.print(actions_table)
        console.print()

        # Handle user input
        while True:
            try:
                cmd = Prompt.ask("[bold green]What would you like to do?[/bold green]").strip().lower()

                if cmd == 'd' and not article.downloaded and article.pdf_url:
                    console.print("[yellow]Downloading PDF...[/yellow]")
                    if self.download_pdf(article):
                        console.print("[green]✅ PDF downloaded successfully![/green]")
                        time.sleep(2)
                        return True  # Redisplay form
                    else:
                        console.print("[red]❌ PDF download failed[/red]")
                        time.sleep(2)

                elif cmd == 's':
                    console.print("[green]Saving changes...[/green]")
                    self.save_manifest()
                    console.print("[green]✅ Changes saved![/green]")
                    time.sleep(2)
                    return True  # Redisplay form

                elif cmd == 'n':
                    if self.current_index < len(self.articles) - 1:
                        self.current_index += 1
                        return True
                    else:
                        console.print("[yellow]⚠️ Already at last article[/yellow]")
                        time.sleep(1)

                elif cmd == 'p':
                    if self.current_index > 0:
                        self.current_index -= 1
                        return True
                    else:
                        console.print("[yellow]⚠️ Already at first article[/yellow]")
                        time.sleep(1)

                elif cmd == 'q':
                    console.print("[yellow]Saving and exiting...[/yellow]")
                    self.save_manifest()
                    return False

                elif cmd in ['title', 'authors', 'date', 'publication', 'doi']:
                    current_value = getattr(article, cmd)
                    if cmd == 'authors' and current_value:
                        current_value = ', '.join(current_value)

                    new_value = Prompt.ask(f"Current {cmd}: {current_value}\nNew {cmd}")

                    if cmd == 'authors':
                        # Handle authors as comma-separated list
                        article.authors = [a.strip() for a in new_value.split(',') if a.strip()]
                    else:
                        setattr(article, cmd, new_value)

                    console.print(f"[green]✅ {cmd.capitalize()} updated![/green]")
                    time.sleep(1)
                    return True  # Redisplay form

                else:
                    console.print("[yellow]❓ Unknown command. Try: d, s, n, p, q, or a field name[/yellow]")

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️ Interrupted. Saving and exiting...[/yellow]")
                self.save_manifest()
                return False

    def run_interactive_session(self):
        """Run the interactive article editing session."""
        if not self.articles:
            console.print("[yellow]No articles to process[/yellow]")
            return

        while self.current_index < len(self.articles):
            article = self.articles[self.current_index]
            if not self.display_article_form(article):
                break

        console.print("[green]Session complete![/green]")

def main(
    url: str = typer.Option(None, "--url", "-u", help="Google Scholar search URL"),
    file: str = typer.Option(None, "--file", "-f", help="Path to saved HTML file"),
    output_dir: str = typer.Option("research/out", "--output", "-o", help="Output directory"),
    show_only: bool = typer.Option(False, "--show-only", help="Show parsed articles without interactive form")
):
    """Research Collector - Extract and manage academic sources from Google Scholar."""

    # Initialize collector
    collector = ResearchCollector(Path(output_dir))

    # If no URL or file provided, just load from manifest and run interactive session
    if not url and not file:
        if not collector.articles:
            console.print("[yellow]No articles found in manifest. Provide --url or --file to add new articles.[/yellow]")
            return
        console.print(f"[green]Loaded {len(collector.articles)} articles from manifest[/green]")
    else:
        # Process new content (URL or file)
        if url:
            html = collector.fetch_html_from_url(url)
            source_name = "URL"
        else:
            # Check if file has already been processed
            file_path = Path(file)
            processed_path = file_path.parent / f"{file_path.stem}_processed{file_path.suffix}"

            if processed_path.exists():
                console.print(f"[yellow]File {file} has already been processed. Loading from manifest instead.[/yellow]")
                if not collector.articles:
                    console.print("[yellow]No articles found in manifest. The processed file may have been moved or deleted.[/yellow]")
                    return
            else:
                html = collector.load_html_from_file(file)
                source_name = file_path.name

                # Parse articles
                new_articles = collector.parse_google_scholar_html(html)

                if not new_articles:
                    console.print("[yellow]No articles found in the provided content[/yellow]")
                    sys.exit(1)

                console.print(f"[green]Found {len(new_articles)} articles in {source_name}[/green]")

                # Add new articles to existing ones
                added_count = 0
                for article in new_articles:
                    # Check if article already exists (by DOI or title)
                    exists = False
                    for existing in collector.articles:
                        if (article.doi and existing.doi == article.doi) or \
                           (article.title and existing.title == article.title):
                            exists = True
                            break

                    if not exists:
                        collector.articles.append(article)
                        added_count += 1

                console.print(f"[green]Added {added_count} new articles[/green]")

                # Rename the processed file to disconnect association
                if file_path.exists():
                    try:
                        file_path.rename(processed_path)
                        console.print(f"[green]Renamed {file_path.name} to {processed_path.name}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not rename file: {e}[/yellow]")

    console.print(f"[green]Total articles: {len(collector.articles)}[/green]")

    # Run interactive session or just show parsed data
    if show_only:
        # Just show parsed articles without interactive form
        if collector.articles:
            console.print(f"\n[green]📚 Found {len(collector.articles)} articles:[/green]")
            for i, article in enumerate(collector.articles, 1):
                console.print(f"\n[bold]{i}. {article.title}[/bold]")
                console.print(f"   👥 Authors: {', '.join(article.authors) if article.authors else 'None'}")
                console.print(f"   📅 Date: {article.date or 'Unknown'}")
                console.print(f"   🏛️ Publication: {article.publication or 'Unknown'}")
                console.print(f"   🔗 DOI: {article.doi or 'None'}")
                console.print(f"   📄 PDF: {'Available' if article.pdf_url else 'Not found'}")
                if article.scholar_url:
                    console.print(f"   🔍 Scholar: {article.scholar_url}")
        else:
            console.print("[yellow]No articles found[/yellow]")
        return  # Exit early, don't run interactive session
    else:
        # Run interactive session
        collector.run_interactive_session()


if __name__ == "__main__":
    typer.run(main)