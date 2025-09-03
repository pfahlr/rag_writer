from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ArticleMetadata:
    title: str = ""
    authors: List[str] = None
    date: str = ""
    publication: str = ""
    doi: str = ""
    isbn: str = ""
    pdf_url: str = ""
    pdf_source_url: str = ""
    scholar_url: str = ""
    pdf_filename: str = ""
    downloaded: bool = False

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['authors_str'] = ', '.join(self.authors) if self.authors else ''
        return d

