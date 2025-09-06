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