#!/usr/bin/env python3
"""Command line interface for collecting research article metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .classes.research_collector import ResearchCollector
from .functions.collector_core import (
    parse_google_scholar_html,
    parse_xml_markup,
)

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    url: Optional[str] = typer.Option(None, help="Google Scholar search URL"),
    file: Optional[str] = typer.Option(None, help="HTML file with Google Scholar results"),
    xml: Optional[str] = typer.Option(None, help="XML markup describing articles"),
    manifest: Path = typer.Option(Path("research/out/manifest.json"), help="Path to manifest file"),
) -> None:
    """Parse article listings and update the manifest file."""

    collector = ResearchCollector(manifest.parent, manifest_file=manifest)

    if url:
        html = collector.fetch_html_from_url(url)
        articles = parse_google_scholar_html(html)
    elif file:
        html = collector.load_html_from_file(file)
        articles = parse_google_scholar_html(html)
    elif xml:
        xml_text = Path(xml).read_text(encoding="utf-8")
        articles = parse_xml_markup(xml_text)
    else:
        typer.echo("Provide --url, --file or --xml")
        raise typer.Exit(code=1)

    collector.add_articles(articles)

    table = Table(title="Collected Articles")
    table.add_column("Title")
    table.add_column("DOI")
    table.add_column("PDF")
    for art in articles:
        table.add_row(art.title, art.doi or "-", art.pdf_url or "-")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    app()

