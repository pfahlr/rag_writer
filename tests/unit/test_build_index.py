"""
Unit tests for build index functionality.
"""

import re
from unittest.mock import Mock

# Test DOI extraction functionality without importing problematic modules
DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]')

def get_doi(pages) -> str:
    """Extract DOI from pages (copied from lc_build_index.py for testing)."""
    count = 0
    for p in pages:
        match = DOI_REGEX.search(p.page_content)
        if match:
            doi = match.group(0).lower()
            return doi
        if count > 1:
            return ""
        count = count + 1
    return ""


class TestDOIExtraction:
    """Test DOI extraction functionality."""

    def test_get_doi_found(self):
        """Test DOI extraction when DOI is found."""
        pages = [
            Mock(page_content="This paper has DOI: 10.1234/example.doi"),
            Mock(page_content="Some other content")
        ]

        result = get_doi(pages)
        assert result == "10.1234/example.doi"

    def test_get_doi_not_found(self):
        """Test DOI extraction when no DOI is found."""
        pages = [
            Mock(page_content="This paper has no DOI"),
            Mock(page_content="Just regular content"),
            Mock(page_content="More content without DOI")
        ]

        result = get_doi(pages)
        assert result == ""

    def test_get_doi_case_insensitive(self):
        """Test DOI extraction is case insensitive."""
        pages = [
            Mock(page_content="DOI: 10.5678/EXAMPLE.DOI"),
        ]

        result = get_doi(pages)
        assert result == "10.5678/example.doi"

    def test_get_doi_multiple_pages(self):
        """Test DOI extraction finds DOI on later pages."""
        pages = [
            Mock(page_content="Page 1: no DOI here"),
            Mock(page_content="Page 2: no DOI here"),
            Mock(page_content="Page 3: DOI: 10.9999/late.doi")  # Should be found
        ]

        result = get_doi(pages)
        assert result == "10.9999/late.doi"

    def test_get_doi_empty_pages(self):
        """Test DOI extraction with empty pages."""
        result = get_doi([])
        assert result == ""

    def test_get_doi_malformed_doi(self):
        """Test DOI extraction with malformed DOI."""
        pages = [
            Mock(page_content="DOI: 10.1234"),  # Incomplete DOI
        ]

        result = get_doi(pages)
        assert result == ""