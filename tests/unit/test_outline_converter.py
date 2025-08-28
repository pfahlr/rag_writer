"""
Unit tests for outline converter functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from langchain.lc_outline_converter import (
    parse_text_outline,
    parse_markdown_outline,
    parse_json_outline,
    detect_outline_format,
    load_outline_file,
    generate_hierarchical_id,
    generate_markdown_hierarchical_id,
    generate_book_structure,
    generate_job_file,
    OutlineSection,
    BookMetadata
)


class TestTextOutlineParsing:
    """Test text outline parsing functionality."""

    def test_simple_text_outline(self, sample_text_outline):
        """Test parsing of simple text outline with indentation."""
        sections, metadata = parse_text_outline(sample_text_outline)

        assert len(sections) == 6  # 2 chapters + 3 sections + 1 subsection
        assert metadata.title == "Professional Development Handbook for Primary School Educators"

        # Check hierarchical structure
        chapter_1 = next(s for s in sections if s.id == "1")
        assert chapter_1.title == "Foundations of Modern Education"
        assert chapter_1.level == 1

        section_1a = next(s for s in sections if s.id == "1A")
        assert section_1a.title == "Learning Theories"
        assert section_1a.level == 2
        assert section_1a.parent_id == "1"

        subsection_1a1 = next(s for s in sections if s.id == "1A1")
        assert subsection_1a1.title == "Constructivist Learning Theory"
        assert subsection_1a1.level == 3
        assert subsection_1a1.parent_id == "1A"

    def test_hierarchical_id_generation(self):
        """Test hierarchical ID generation for text outlines."""
        level_stack = []

        # Test level 1 (chapter)
        id1 = generate_hierarchical_id("1", 1, level_stack)
        assert id1 == "1"
        level_stack.append((1, "1"))

        # Test level 2 (section)
        id2 = generate_hierarchical_id("A", 2, level_stack)
        assert id2 == "1A"
        level_stack.append((2, "1A"))

        # Test level 3 (subsection)
        id3 = generate_hierarchical_id("1", 3, level_stack)
        assert id3 == "1A1"
        level_stack.append((3, "1A1"))

        # Test alphanumeric ID
        id4 = generate_hierarchical_id("1A1", 3, level_stack)
        assert id4 == "1A1"

    def test_complex_indentation(self):
        """Test parsing with complex indentation patterns."""
        complex_outline = """Book Title

1. Chapter One
  1A. Section A
    1A1. Subsection 1
    1A2. Subsection 2
  1B. Section B
    1B1. Subsection 1

2. Chapter Two
  2A. Section A
"""

        sections, metadata = parse_text_outline(complex_outline)

        assert len(sections) == 8  # 2 chapters + 2 sections + 4 subsections
        assert metadata.title == "Book Title"

        # Verify all IDs are correct
        expected_ids = ["1", "1A", "1A1", "1A2", "1B", "1B1", "2", "2A"]
        actual_ids = [s.id for s in sections]
        assert set(actual_ids) == set(expected_ids)


class TestMarkdownOutlineParsing:
    """Test markdown outline parsing functionality."""

    def test_markdown_headers(self, sample_markdown_outline):
        """Test parsing of markdown headers."""
        sections, metadata = parse_markdown_outline(sample_markdown_outline)

        assert len(sections) == 6  # 2 chapters + 2 sections + 2 subsections
        assert metadata.title == "Professional Development Handbook for Primary School Educators"

        # Check chapter parsing
        chapter_1 = next(s for s in sections if s.id == "1")
        assert chapter_1.title == "Foundations of Modern Education"
        assert chapter_1.level == 2  # Markdown H2 becomes level 2

        # Check section parsing
        section_1a = next(s for s in sections if s.id == "1A")
        assert section_1a.title == "Learning Theories"
        assert section_1a.level == 3  # Markdown H3 becomes level 3

    def test_markdown_hierarchical_id_generation(self):
        """Test hierarchical ID generation for markdown."""
        level_stack = []

        # Test H2 (chapter level)
        id1 = generate_markdown_hierarchical_id(2, level_stack)
        assert id1 == "1"
        level_stack.append((2, "1"))

        # Test H3 (section level)
        id2 = generate_markdown_hierarchical_id(3, level_stack)
        assert id2 == "1A"
        level_stack.append((3, "1A"))

        # Test H4 (subsection level)
        id3 = generate_markdown_hierarchical_id(4, level_stack)
        assert id3 == "1A1"
        level_stack.append((4, "1A1"))

        # Test second chapter
        id4 = generate_markdown_hierarchical_id(2, level_stack)
        assert id4 == "2"


class TestJSONOutlineParsing:
    """Test JSON outline parsing functionality."""

    def test_json_outline_parsing(self, sample_json_outline):
        """Test parsing of JSON outline format."""
        sections, metadata = parse_json_outline(json.dumps(sample_json_outline))

        assert len(sections) == 4  # 1 chapter + 1 section + 2 subsections
        assert metadata.title == "Professional Development Handbook for Primary School Educators"
        assert metadata.topic == "Education"
        assert metadata.target_audience == "Primary School Teachers"

        # Check chapter
        chapter = next(s for s in sections if s.id == "1")
        assert chapter.title == "Foundations of Modern Education"
        assert chapter.level == 2

        # Check section
        section = next(s for s in sections if s.id == "1A")
        assert section.title == "Learning Theories"
        assert section.level == 3
        assert section.parent_id == "1"

        # Check subsections
        subsections = [s for s in sections if s.level == 4]
        assert len(subsections) == 2
        assert all(s.parent_id == "1A" for s in subsections)


class TestFormatDetection:
    """Test outline format detection."""

    def test_detect_text_format(self, sample_text_outline):
        """Test detection of text format."""
        assert detect_outline_format(sample_text_outline) == "text"

    def test_detect_markdown_format(self, sample_markdown_outline):
        """Test detection of markdown format."""
        assert detect_outline_format(sample_markdown_outline) == "markdown"

    def test_detect_json_format(self, sample_json_outline):
        """Test detection of JSON format."""
        json_str = json.dumps(sample_json_outline)
        assert detect_outline_format(json_str) == "json"

    def test_detect_malformed_json(self):
        """Test handling of malformed JSON."""
        malformed = '{"title": "Test", "incomplete": }'
        assert detect_outline_format(malformed) == "text"  # Falls back to text


class TestBookStructureGeneration:
    """Test book structure generation."""

    def test_generate_book_structure(self, sample_outline_sections, sample_book_metadata, temp_dir):
        """Test generation of book structure JSON."""
        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            book_structure = generate_book_structure(sample_outline_sections, sample_book_metadata)

            assert book_structure["title"] == sample_book_metadata.title
            assert len(book_structure["sections"]) == 4  # All sections included

            # Check section structure
            section_1a1 = next(s for s in book_structure["sections"] if s["subsection_id"] == "1A1")
            assert section_1a1["title"] == "Constructivist Learning Theory"
            assert section_1a1["job_file"] == "data_jobs/1A1.jsonl"
            assert "batch_params" in section_1a1
            assert "merge_params" in section_1a1

    def test_book_structure_dependencies(self, sample_outline_sections, sample_book_metadata, temp_dir):
        """Test dependency generation in book structure."""
        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            book_structure = generate_book_structure(sample_outline_sections, sample_book_metadata)

            # Find sections with dependencies
            section_1a1 = next(s for s in book_structure["sections"] if s["subsection_id"] == "1A1")
            assert "1A" in section_1a1["dependencies"]

            section_1a = next(s for s in book_structure["sections"] if s["subsection_id"] == "1A")
            assert section_1a["dependencies"] == []  # No dependencies for level 2


class TestJobFileGeneration:
    """Test job file generation."""

    def test_generate_job_file(self, sample_outline_sections, sample_book_metadata, temp_dir, mock_console):
        """Test generation of job files."""
        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            section = sample_outline_sections[2]  # 1A1 subsection
            job_file = generate_job_file(section, sample_book_metadata, sample_outline_sections)

            assert job_file.exists()
            assert job_file.name == "1A1.jsonl"

            # Check job file content
            with open(job_file, 'r') as f:
                jobs = [json.loads(line) for line in f if line.strip()]

            assert len(jobs) == 4  # 4 standard jobs generated
            assert all('task' in job for job in jobs)
            assert all('instruction' in job for job in jobs)
            assert all('context' in job for job in jobs)

    def test_job_file_context(self, sample_outline_sections, sample_book_metadata, temp_dir, mock_console):
        """Test hierarchical context in job files."""
        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            section = sample_outline_sections[2]  # 1A1 subsection
            generate_job_file(section, sample_book_metadata, sample_outline_sections)

            with open(temp_dir / "data_jobs" / "1A1.jsonl", 'r') as f:
                job = json.loads(f.readline())

            context = job["context"]
            assert context["book_title"] == sample_book_metadata.title
            assert "Chapter 1" in context["hierarchy"]
            assert "Section A" in context["hierarchy"]
            assert context["subsection_id"] == "1A1"


class TestErrorHandling:
    """Test error handling in outline converter."""

    def test_missing_file(self, temp_dir):
        """Test handling of missing input files."""
        missing_file = temp_dir / "missing.txt"

        with pytest.raises(SystemExit):
            load_outline_file(missing_file)

    def test_empty_outline(self, temp_dir):
        """Test handling of empty outline files."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        sections, metadata = load_outline_file(empty_file)
        assert sections == []
        assert metadata.title == "Converted from Text Outline"

    def test_malformed_json(self, temp_dir):
        """Test handling of malformed JSON outlines."""
        malformed_file = temp_dir / "malformed.json"
        malformed_file.write_text('{"title": "Test", "incomplete": }')

        sections, metadata = load_outline_file(malformed_file)
        assert sections == []
        assert "Error parsing outline" in metadata.title


class TestIntegration:
    """Integration tests for outline converter."""

    def test_full_conversion_pipeline(self, sample_text_outline, temp_dir, mock_console):
        """Test the complete outline conversion pipeline."""
        # Create input file
        input_file = temp_dir / "test_outline.txt"
        input_file.write_text(sample_text_outline)

        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            # Load and parse outline
            sections, metadata = load_outline_file(input_file)

            assert len(sections) == 6
            assert metadata.title == "Professional Development Handbook for Primary School Educators"

            # Generate book structure
            book_structure = generate_book_structure(sections, metadata)

            assert len(book_structure["sections"]) == 6  # All sections included

            # Generate job files
            generated_jobs = 0
            for section in sections:
                if section.level >= 2:  # Generate jobs for sections and subsections
                    generate_job_file(section, metadata, sections)
                    generated_jobs += 1

            assert generated_jobs == 4  # 2 sections + 2 subsections

            # Verify job files exist
            job_files = list((temp_dir / "data_jobs").glob("*.jsonl"))
            assert len(job_files) == 4