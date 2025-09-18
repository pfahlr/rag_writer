#!/usr/bin/env python3
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.getenv('BASE_PATH','./src')))


#from __future__ import annotations
import json
import re
import time

from typing import Any, Dict, List, Optional

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from bs4 import BeautifulSoup  # type: ignore
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Label, Input, TextArea
from textual.containers import Vertical, Horizontal
from rich.console import Console

console = Console()

from functions.pdfwriter import save_pdf
from functions.filelogger import _fllog
from functions.collector_core import parse_google_scholar_html, parse_xml_markup, complete_article_fields, _load_manifest_links, _extract_doi, _extract_isbn, _now_ts, _extract_pdf_links_from_html, _extract_pdf_links_from_xml

from classes.research_collector import ResearchCollector

import argparse
import webbrowser
import requests

MANIFEST = Path("../research/out/manifest.json")
OUT_DIR = Path("../research/out")

class CollectorUI(App):
    CSS_PATH = "styles.tcss"

    def __init__(self, *,
                    file: Optional[str] = None,
                    xml: Optional[str] = None,
                    skip_existing: bool = False,
                    allow_delete: bool = True,
                    rescan: bool = False,
                    depth: int = 0,
                    jobs: int = 0, 
                    article_index: int =0,
                    manifest = MANIFEST) -> None:
        super().__init__()
        self.manifest = manifest
        self.mode_label: Label | None = None
        self.import_text: TextArea | None = None
        self.links_text: TextArea | None = None
        self.current_links: List[str] = []
        # Edit state
        self.articles: List[Dict[str, Any]] = []
        self.current_index: int = 0
        # Options
        self.opt_file = file
        self.opt_xml = xml
        self.opt_skip_existing = skip_existing
        self.opt_allow_delete = allow_delete
        self.opt_rescan = rescan
        self.opt_depth = depth
        self.opt_jobs = jobs
        self.article_index = article_index

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="app-container ui-section ui-section-vertical"):
            # Top navigation buttons
            with Horizontal(id="main-menu-buttons-panel", classes="ui-section ui-section-horizontal"):
                yield Button("Edit", id="btn_edit", classes="button main-menu-btn btn-edit")
                yield Button("Import", id="btn_import", classes="button main-menu-btn btn-import")
                yield Button("Links", id="btn_links", classes="button main-menu-btn btn-links")

            # Mode/status label
            self.mode_label = Label("Initializing...", id="mode_label", classes="field-label")
            yield self.mode_label

            # Edit Panel
            with Vertical(id="panel_edit", classes="form-edit-container  main-app-section-container form-container ui-section ui-section-vertical"):
                # Basic fields
                with Horizontal(classes="ui-section ui-section-horizontal"):
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("Title", classes="field-label")
                        yield Input(placeholder="Title", id="title_input", classes="form-field")
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("Authors", classes="field-label")
                        yield Input(placeholder="Author1, Author2", id="authors_input", classes="form-field")

                with Horizontal(classes="ui-section ui-section-horizontal"):
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("Date", classes="field-label")
                        yield Input(placeholder="2023", id="date_input", classes="form-field")
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("Publication", classes="field-label")
                        yield Input(placeholder="Journal Name", id="pub_input", classes="form-field")

                with Horizontal(classes="ui-section ui-section-horizontal"):
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("DOI", classes="field-label")
                        yield Input(placeholder="10.1000/example", id="doi_input", classes="form-field")
                    with Vertical(classes="ui-section ui-section-vertical"):
                        yield Label("ISBN", classes="field-label")
                        yield Input(placeholder="978-0-123456-78-9", id="isbn_input", classes="form-field")

                # PDF URL row + actions
                yield Label("PDF URL", classes="field-label")
                yield Input(placeholder="https://example.com/paper.pdf", id="pdf_input", classes="form-field")
                with Horizontal(classes="ui-section ui-section-horizontal"):
                    yield Button("Download", id="btn_download", variant="primary")
                    yield Button("Open PDF", id="btn_open_pdf")

                # Scholar URL row + actions
                yield Label("Scholar (web) URL", classes="field-label")
                yield Input(placeholder="ARTICLE WEB URL", id="scholar_input", classes="form-field")
                with Horizontal(classes="article-buttons-panel ui-section ui-section-horizontal"):
                    yield Button("Previous", id="btn_prev", classes="button article-btn")
                    yield Button("Open Page", id="btn_open_url", classes="button article-btn")
                    yield Button("Complete Fields", id="btn_complete", classes="button article-btn")
                    yield Button("Save", id="btn_save", classes="button article-btn btn-save")
                    yield Button("Delete", id="btn_delete", variant="error", classes="button article-btn btn-delete")
                    yield Button("Next", id="btn_next", classes="button article-btn btn-next")

            with Vertical(classes="form-import-container main-app-section-container form-container ui-section ui-section-vertical"):
                # Import Panel
                self.import_text = TextArea()
                self.import_text.display = True
                yield self.import_text
                with Horizontal(id="import_buttons", classes="import-buttons-panel ui-section ui-section-horizontal"):
                    yield Button("Import HTML", id="btn_import_html", classes="button import-btn btn-import-html")
                    yield Button("Import XML", id="btn_import_xml", classes="button import-btn btn-import-xml")
            
            with Vertical(classes="form-links-container main-app-section-container form-container ui-section ui-section-vertical"):
                # Links Panel
                self.links_text = TextArea()
                self.links_text.display = True
                yield self.links_text
                with Horizontal(id="links_buttons", classes="links-buttons-panel ui-section ui-section-horizontal"):
                    yield Button("Save Links", id="btn_save_links", variant="primary", classes="button")
                    yield Button("Back", id="btn_back_from_links", classes="button")

                # Navigation row
        with Horizontal(classes="footer-buttons-panel ui-section ui-section-horizontal"):
            yield Button("Quit", id="btn_quit", classes="button nav-btn btn-quit")

        yield Footer()



    def _merge_articles_lists(self, existing, new):
        delete_idx = []
        for i in range(0, len(new)):
            for existing_item in existing:
                if new[i]['pdf_url'] == existing_item['pdf_url']:
                    delete_idx.append(i)
                    break
                if new[i]['scholar_url'] == existing_item['scholar_url']:
                    delete_idx.append(i)
                    break

        delete_idx.sort(reverse=True)
        for index in delete_idx:
            del new[index]

        final = existing + new
        return final

    def _dedup_article_list(self, articles):
        temp = []
        for a in articles:
            bDup = False
            for b in temp:
                if (a.doi == b.doi) or (a.scholar_url == b.scholar_url):
                    bDup = True
                    break
            if not bDup:
                temp.append(a)
        return temp

    def _html_parser(self, html):
        arts = parse_google_scholar_html(html)

        arts = self._dedup_article_list(arts)

        articles = [
            {
                "title": a.title,
                "authors": a.authors or [],
                "date": a.date,
                "publication": a.publication,
                "doi": a.doi,
                "isbn": a.isbn,
                "pdf_url": a.pdf_url,
                "scholar_url": a.scholar_url,
            }
            for a in arts
        ]
        return articles

    def _xml_parser(self, xml):
        arts = parse_xml_markup(xml)
        articles = [
            {
                "title": a.title,
                "authors": a.authors or [],
                "date": a.date,
                "publication": a.publication,
                "doi": a.doi,
                "isbn": a.isbn,
                "pdf_url": a.pdf_url,
                "scholar_url": a.scholar_url,
            }
            for a in arts
        ]
        return articles

    def on_mount(self) -> None:
        # Load from provided sources or manifest
        if self.opt_file and Path(self.opt_file).exists():
            try:
                html = Path(self.opt_file).read_text(encoding="utf-8")
                new_articles = self._html_parser(html)
                #self.articles += new_articles
                self.articles = self._merge_articles_lists(self.articles, new_articles)
                self._save_current_article(update_only=True)

            except Exception:
                self.articles += []
            
        elif self.opt_xml and Path(self.opt_xml).exists():
            try:
                xml = Path(self.opt_xml).read_text(encoding="utf-8")
                self.articles += self._xml_parser(xml)
                self._save_current_article(update_only=True)
               
            except Exception:
                self.articles += []
        else:
            self._load_articles()
        if self.articles:
            self._refresh_edit_form()
            self._show_edit()
        else:
            self._show_import()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn_edit":
            self._show_edit()
        elif bid == "btn_import":
            self._show_import()
        elif bid == "btn_links":
            self._show_links()
        elif bid == "btn_prev":
            self._nav_prev()
        elif bid == "btn_next":
            self._nav_next()
        elif bid == "btn_save":
            self._save_current_article()
        elif bid == "btn_delete":
            self._delete_current_article()
        elif bid == "btn_download":
            self._download_pdf_current()
        elif bid == "btn_open_url":
            self._open_url_current()
        elif bid == "btn_open_pdf":
            self._open_pdf_current()
        elif bid == "btn_complete":
            self._complete_fields_current()
        elif bid == "btn_quit":
            self.exit()
        elif bid == "btn_import_html":
            self._handle_import_html()
        elif bid == "btn_import_xml":
            self._handle_import_xml()
        elif bid == "btn_back_from_import":
            self._show_edit()
        elif bid == "btn_back_to_import":
            self._show_import()
        elif bid == "btn_back_from_links":
            self._show_edit()
        elif bid == "btn_save_links":
            self._save_links_file()

    # ============ Panels ============
    def _toggle(self, *, edit: bool, imp: bool, links: bool) -> None:
        self.query_one("#panel_edit").display = edit
        if self.import_text:
            self.import_text.display = imp
            self.import_text.classes="textarea textarea-import"
        self.query_one("#import_buttons").display = imp
        if self.links_text:
            self.links_text.display = links
            self.links_text.classes="textarea textarea-links"
        self.query_one("#links_buttons").display = links

    def _show_edit(self) -> None:
        if self.mode_label:
            self.mode_label.update(self._edit_header_text())
        self.query(".main-app-section-container").remove_class("active-form-panel")
        self.query(".form-edit-container").add_class("active-form-panel")
        self.query(".main-menu-btn").remove_class("active")
        self.query(".btn-edit").add_class("active")
        self._toggle(edit=True, imp=False, links=False)

    def _show_import(self) -> None:
        if self.mode_label:
            self.mode_label.update("Import Mode: paste HTML/XML, then click a button")
        self.query(".main-app-section-container").remove_class("active-form-panel")
        self.query(".form-import-container").add_class("active-form-panel")
        self.query(".main-menu-btn").remove_class("btn-active")
        self.query(".btn-import").add_class("btn-active")
        self._toggle(edit=False, imp=True, links=False)

    def _show_links(self) -> None:
        if self.mode_label:
            self.mode_label.update("Links Mode: current known PDF links")
        # merge session links with manifest links
        self.query(".main-app-section-container").remove_class("active-form-panel")
        self.query(".form-links-container").add_class("active-form-panel")        
        self.query(".main-menu-btn").remove_class("btn-active")
        self.query(".btn-links").add_class("btn-active")
        combined = sorted(list(dict.fromkeys((self.current_links or []) + _load_manifest_links())))
        if self.links_text:
            self.links_text.load_text("\n".join(combined))
        self._toggle(edit=False, imp=False, links=True)

    # ============ Import handlers ============
    def _handle_import_html(self) -> None:
        text = self.import_text.document.text if self.import_text else ""
        #self.articles += self._html_parser(text)
        new_articles = self._html_parser(text)
        self.articles = self. _merge_articles_lists(self.articles, new_articles)
        self.import_text.clear()
        ts = _now_ts()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{ts}_processed.html").write_text(text, encoding="utf-8")
        self._show_links()

    def _handle_import_xml(self) -> None:
        text = self.import_text.document.text if self.import_text else ""
        self.articles += self._xml_parser(text)
        self.import_text.clear()
        ts = _now_ts()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{ts}_processed.xml").write_text(text, encoding="utf-8")
        self._show_links()

    def _save_links_file(self) -> None:
        text = self.links_text.document.text if self.links_text else ""
        ts = _now_ts()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"download_links_{ts}.txt"
        path.write_text(text, encoding="utf-8")
        if self.mode_label:
            self.mode_label.update(f"Saved links to {path}")

    # ============ Edit helpers ============
    def _edit_header_text(self) -> str:
        total = len(self.articles)
        idx = (self.current_index + 1) if total else 0
        return f"Edit Mode — Article {idx}/{total}"

    def _load_articles(self) -> None:
        self.articles = []
        if self.manifest.exists():
            try:
                data = json.loads(self.manifest.read_text(encoding="utf-8"))
                entries = data.get("entries") if isinstance(data, dict) else data
                for e in entries or []:
                    if self.opt_skip_existing and e.get("processed"):
                        continue
                    self.articles.append({
                        "title": e.get("title", ""),
                        "authors": e.get("authors", []),
                        "date": e.get("date", ""),
                        "publication": e.get("publication", ""),
                        "doi": e.get("doi", ""),
                        "isbn": e.get("isbn", ""),
                        "pdf_url": e.get("pdf_url", ""),
                        "scholar_url": e.get("scholar_url", ""),
                    })
            except Exception:
                pass

    def _refresh_edit_form(self) -> None:
        if self.mode_label:
            self.mode_label.update(self._edit_header_text())
        if not self.articles:
            return
        art = self.articles[self.current_index]
        _fllog(json.dumps(art))
        self._set_input("#title_input", art.get("title", ""))
        self._set_input("#authors_input", ", ".join(art.get("authors", [])))
        self._set_input("#date_input", art.get("date", ""))
        self._set_input("#pub_input", art.get("publication", ""))
        self._set_input("#doi_input", art.get("doi", ""))
        self._set_input("#isbn_input", art.get("isbn", ""))
        self._set_input("#pdf_input", art.get("pdf_url", ""))
        self._set_input("#scholar_input", art.get("scholar_url", ""))

    def _set_input(self, selector: str, value: str) -> None:
        try:
            w = self.query_one(selector, Input)
            w.value = value
        except Exception:
            pass

    def _collect_form(self) -> Dict[str, Any]:
        def gv(sel: str) -> str:
            try:
                return self.query_one(sel, Input).value or ""
            except Exception:
                return ""
        authors = [a.strip() for a in gv("#authors_input").split(',') if a.strip()]
        return {
            "title": gv("#title_input"),
            "authors": authors,
            "date": gv("#date_input"),
            "publication": gv("#pub_input"),
            "doi": gv("#doi_input"),
            "isbn": gv("#isbn_input"),
            "pdf_url": gv("#pdf_input"),
            "scholar_url": gv("#scholar_input"),
        }
        

    def _nav_prev(self) -> None:
        if not self.articles:
            return
        self._save_current_article(update_only=True)
        self.current_index = (self.current_index - 1) % len(self.articles)
        self._refresh_edit_form()

    def _nav_next(self) -> None:
        if not self.articles:
            return
        self._save_current_article(update_only=True)
        self.current_index = (self.current_index + 1) % len(self.articles)
        self._refresh_edit_form()

    def _save_current_article(self, update_only: bool = False) -> None:
        if not self.articles:
            return
        self.articles[self.current_index] = self._collect_form()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": self.articles}
        self.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.mode_label and not update_only:
            self.mode_label.update("Saved current article to manifest")

    def _delete_current_article(self) -> None:
        _fllog(self.current_index)
        if not self.articles:
            return
        _fllog(self.current_index)
        _fllog("attempting to delete:")
        _fllog(json.dumps(self.articles[self.current_index]))
        del self.articles[self.current_index]
    
        if self.current_index >= len(self.articles):
            self.current_index = max(0, len(self.articles) - 1)
        _fllog("now current index points to:")
        _fllog(json.dumps(self.articles[self.current_index]))
        _fllog("and remaining list of articles is:")
        _fllog(json.dumps(self.articles))
        self._refresh_edit_form()

    def _download_pdf_current(self) -> None:
        if not self.articles:
            return
        art = self._collect_form()
        url = (art.get("pdf_url") or "").strip()
        if not url:
            if self.mode_label:
                self.mode_label.update("No PDF URL set for this article")
            return
        title = art.get("title") or "untitled"
        m = re.search(r"(20\d{2}|19\d{2})", art.get("date") or "")
        year = f"_{m.group(1)}" if m else ""
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        filename = f"{slug}{year}.pdf"
        meta = {
            '/Title': title,
            '/Author': ", ".join(art.get('authors') or []),
            '/Subject': art.get('publication') or "",
        }
        dest = save_pdf(url, filename, meta, str(OUT_DIR), tmp_path=str(OUT_DIR / 'tmp'))
        if self.mode_label:
            self.mode_label.update(f"Downloaded: {dest or 'failed'}")
        if self.mode_label and not update_only:
            self.mode_label.update("Saved current article to manifest")

    def _open_url_current(self) -> None:
        if not self.articles:
            return
        art = self._collect_form()
        u = (art.get("scholar_url") or "").strip()
        if u:
            webbrowser.open(u)

    def _open_pdf_current(self) -> None:
        if not self.articles:
            return
        art = self._collect_form()
        u = (art.get("pdf_url") or "").strip()
        if u:
            webbrowser.open(u)

    def _complete_fields_current(self) -> None:
        self.articles[self.current_index] = complete_article_fields(self.articles[self.current_index] )
        self._save_current_article()
#        if not self.articles:
#            return
#        art = self._collect_form()
#        u = (art.get("scholar_url") or "").strip()
#        if not u:
#            return
#        try:
#            r = requests.get(u, timeout=15)
#            html = r.text
#            # Simple DOI detection and title heuristic
#            m = re.search(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]", html)
#            if m:
#                self._set_input("#doi_input", m.group(0))
#            # Title heuristic
#            tm = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
#            if tm and not (self.query_one("#title_input", Input).value or "").strip():
#                title = re.sub(r"\s+", " ", tm.group(1)).strip()
#                self._set_input("#title_input", title)
#        except Exception:
#            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Collector UI")
    ap.add_argument("--file", help="HTML file with Google Scholar results")
    ap.add_argument("--manifest", help="Manifest of existing data, defaults to ../research/out/manifest.json")
    ap.add_argument("--xml", help="XML-like markup file with entries")
    ap.add_argument("--skip-existing", action="store_true", help="Skip entries with processed=true in manifest")
    ap.add_argument("--allow-delete", action="store_true", help="Enable delete actions")
    ap.add_argument("--rescan", action="store_true", help="Ignore cached results (reserved)")
    ap.add_argument("--depth", type=int, default=0, help="Recursion depth (reserved)")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel lookups (reserved)")
    args = ap.parse_args()

    app = CollectorUI(file=args.file,
                      xml=args.xml,
                      skip_existing=args.skip_existing,
                      allow_delete=args.allow_delete,
                      rescan=args.rescan,
                      depth=args.depth,
                      jobs=args.jobs,
                      manifest=Path(args.manifest))
    app.run()


if __name__ == "__main__":
    main()
