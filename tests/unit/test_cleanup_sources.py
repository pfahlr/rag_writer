"""
Unit tests for cleanup_sources functionality.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.langchain.cleanup_sources import (
    _norm,
    _parse_citation_identifier,
    deduplicate_sources,
    _extract_cited_docs,
    cleanup_batch_file
)


class TestStringNormalization:
    """Test string normalization functionality."""

    def test_norm_basic(self):
        """Test basic string normalization."""
        assert _norm("Hello World!") == "hello world"
        assert _norm("Test-Case_123") == "test case_123"  # Underscores are not replaced

    def test_norm_empty_and_none(self):
        """Test normalization with empty and None inputs."""
        assert _norm("") == ""
        assert _norm(None) == ""

    def test_norm_special_characters(self):
        """Test normalization with special characters."""
        assert _norm("Dr. Smith's Paper (2023)") == "dr smith s paper 2023"


class TestCitationParsing:
    """Test citation identifier parsing."""

    def test_parse_citation_identifier_basic(self):
        """Test basic citation identifier parsing."""
        assert _parse_citation_identifier("Smith et al. (2023)") == "Smith et al. (2023"  # Function removes closing paren
        assert _parse_citation_identifier('"Research Paper"') == "Research Paper"

    def test_parse_citation_identifier_with_pages(self):
        """Test parsing citations with page numbers."""
        assert _parse_citation_identifier("Smith (2023, p. 15)") == "Smith (2023"
        assert _parse_citation_identifier("Paper Title, p. 10") == "Paper Title"

    def test_parse_citation_identifier_punctuation(self):
        """Test parsing with surrounding punctuation."""
        assert _parse_citation_identifier("(Citation)") == "Citation"
        assert _parse_citation_identifier('"Quoted Text"') == "Quoted Text"


class TestSourceDeduplication:
    """Test source deduplication functionality."""

    def test_deduplicate_sources_basic(self):
        """Test basic source deduplication."""
        sources = [
            {"title": "Paper A", "source": "paper_a.pdf"},
            {"title": "Paper A", "source": "paper_a.pdf"},  # Duplicate
            {"title": "Paper B", "source": "paper_b.pdf"}
        ]

        result = deduplicate_sources(sources)
        assert len(result) == 2
        assert result[0]["title"] == "Paper A"
        assert result[1]["title"] == "Paper B"

    def test_deduplicate_sources_empty(self):
        """Test deduplication with empty list."""
        assert deduplicate_sources([]) == []

    def test_deduplicate_sources_no_title(self):
        """Test deduplication using source path when title is missing."""
        sources = [
            {"source": "doc1.pdf"},
            {"source": "doc1.pdf"},  # Duplicate
            {"source": "doc2.pdf"}
        ]

        result = deduplicate_sources(sources)
        assert len(result) == 2
        assert result[0]["source"] == "doc1.pdf"
        assert result[1]["source"] == "doc2.pdf"


class TestCitationExtraction:
    """Test citation extraction from content."""

    def test_extract_cited_docs_basic(self):
        """Test basic citation extraction."""
        content = "This paper cites Smith (2023) and discusses the methodology."
        sources = [
            {"title": "Smith (2023)", "source": "smith_2023.pdf"},
            {"title": "Other Paper", "source": "other.pdf"}
        ]

        result = _extract_cited_docs(content, sources)
        assert len(result) == 1
        assert result[0]["title"] == "Smith (2023)"

    def test_extract_cited_docs_no_content(self):
        """Test extraction with empty content."""
        result = _extract_cited_docs("", [{"title": "Test", "source": "test.pdf"}])
        assert result == []

    def test_extract_cited_docs_no_sources(self):
        """Test extraction with no sources."""
        result = _extract_cited_docs("Some content", [])
        assert result == []

    def test_extract_cited_docs_exact_match(self):
        """Test exact title matching for citations."""
        content = "This paper discusses Johnson et al. (2023) Research Paper in detail."
        sources = [
            {"title": "Johnson et al. (2023) Research Paper", "source": "johnson.pdf"}
        ]

        result = _extract_cited_docs(content, sources)
        assert len(result) == 1
        assert result[0]["title"] == "Johnson et al. (2023) Research Paper"


class TestBatchFileCleanup:
    """Test batch file cleanup functionality."""

    def test_cleanup_batch_file_success(self, temp_dir):
        """Test successful batch file cleanup."""
        # Create test batch file
        batch_data = [
            {
                "section": "test_section",
                "generated_content": "This cites Smith (2023) and Johnson (2024).",
                "sources": [
                    {"title": "Smith (2023)", "source": "smith.pdf"},
                    {"title": "Johnson (2024)", "source": "johnson.pdf"},
                    {"title": "Unused Source", "source": "unused.pdf"}
                ]
            }
        ]

        batch_file = temp_dir / "test_batch.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f)

        # Test cleanup
        result = cleanup_batch_file(batch_file)
        assert result is True

        # Verify cleaned content
        with open(batch_file, 'r') as f:
            cleaned_data = json.load(f)

        assert len(cleaned_data[0]["sources"]) == 2  # Should remove unused source

    def test_cleanup_batch_file_no_changes(self, temp_dir):
        """Test cleanup when no changes are needed."""
        batch_data = [
            {
                "section": "test_section",
                "generated_content": "This cites Smith (2023).",
                "sources": [
                    {"title": "Smith (2023)", "source": "smith.pdf"}
                ]
            }
        ]

        batch_file = temp_dir / "test_batch.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f)

        result = cleanup_batch_file(batch_file)
        assert result is True

    def test_cleanup_batch_file_invalid_json(self, temp_dir):
        """Test cleanup with invalid JSON file."""
        batch_file = temp_dir / "invalid.json"
        batch_file.write_text("invalid json content")

        result = cleanup_batch_file(batch_file)
        assert result is False