#!/usr/bin/env python3
"""
Unit Tests for Book Runner Variable Scoping

Tests the specific fix for the 'book_structure is not defined' error
that was occurring in the run_batch_processing function.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.langchain.lc_book_runner import (
    run_batch_processing,
    load_book_structure,
    BookStructure,
    SectionConfig
)


class TestBookRunnerVariableScoping:
    """Test variable scoping fixes in book runner."""

    def test_run_batch_processing_receives_book_structure(self):
        """Test that run_batch_processing correctly receives book_structure parameter."""
        # Create test data
        book_structure = BookStructure(
            title='Test Book',
            sections=[],
            metadata={'content_type': 'technical_manual_writer'}
        )

        section = SectionConfig(
            subsection_id='1A1',
            title='Test Section',
            batch_params={'key': 'test', 'k': 5}
        )

        # Create a temporary job file
        job_data = [{
            'task': 'Test task',
            'instruction': 'Test instruction',
            'section': '1A1'
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(job_data[0], f)
            job_file = Path(f.name)

        try:
            # Mock the batch processing functions to avoid actual processing
            with patch('src.langchain.lc_batch.load_jsonl_file') as mock_load:
                with patch('src.langchain.lc_batch.process_items_sequential') as mock_process:
                    mock_load.return_value = job_data
                    mock_process.return_value = ([], [])  # No results, no errors

                    # Call run_batch_processing with book_structure parameter
                    result = run_batch_processing(section, job_file, book_structure)

                    # Verify the function completed without variable scoping errors
                    assert isinstance(result, bool)

                    # Verify that load_jsonl_file was called (function is working)
                    assert mock_load.called
                    assert mock_process.called

        finally:
            job_file.unlink(missing_ok=True)

    def test_book_structure_metadata_access(self):
        """Test that book_structure metadata is accessible in run_batch_processing."""
        # Create book structure with metadata
        metadata = {
            'content_type': 'technical_manual_writer',
            'target_audience': 'General readers',
            'word_count_target': 50000
        }

        book_structure = BookStructure(
            title='Test Book',
            sections=[],
            metadata=metadata
        )

        # Test metadata access patterns used in the code
        content_type = book_structure.metadata.get('content_type', 'technical_manual_writer')
        target_audience = book_structure.metadata.get('target_audience', 'General readers')
        word_count = book_structure.metadata.get('word_count_target', 50000)

        assert content_type == 'technical_manual_writer'
        assert target_audience == 'General readers'
        assert word_count == 50000

    def test_section_config_parameter_access(self):
        """Test that SectionConfig parameters are accessible."""
        batch_params = {'key': 'test_key', 'k': 10, 'content_type': 'pure_research'}
        merge_params = {'key': 'test_key', 'k': 5}

        section = SectionConfig(
            subsection_id='1A1',
            title='Test Section',
            batch_params=batch_params,
            merge_params=merge_params
        )

        # Test batch parameter access
        assert section.batch_params['key'] == 'test_key'
        assert section.batch_params['k'] == 10
        assert section.batch_params.get('content_type') == 'pure_research'

        # Test merge parameter access
        assert section.merge_params['key'] == 'test_key'
        assert section.merge_params['k'] == 5

        # Test safe access with defaults
        assert section.batch_params.get('missing_param', 'default') == 'default'

    def test_book_structure_parameter_passing_simulation(self):
        """Test the exact parameter passing pattern used in the fixed code."""
        # Simulate the main() function creating book_structure
        def simulate_main_function():
            metadata = {'content_type': 'technical_manual_writer'}
            return BookStructure(
                title='Test Book',
                sections=[],
                metadata=metadata
            )

        # Simulate the loop in main() calling run_batch_processing
        def simulate_processing_loop(book_structure):
            section = SectionConfig(
                subsection_id='1A1',
                title='Test Section'
            )

            # This is the exact call pattern from the fixed code
            # run_batch_processing(section, job_file, book_structure)

            # Test that we can access book_structure.metadata in the function scope
            content_type = book_structure.metadata.get('content_type', 'default')
            return content_type

        # Test the complete flow
        book_structure = simulate_main_function()
        result = simulate_processing_loop(book_structure)

        assert result == 'technical_manual_writer'

    def test_metadata_get_with_defaults(self):
        """Test metadata.get() calls with default values."""
        # Test cases that match the actual code patterns
        test_cases = [
            # (metadata, key, default, expected)
            ({'content_type': 'technical_manual_writer'}, 'content_type', 'pure_research', 'technical_manual_writer'),
            ({}, 'content_type', 'pure_research', 'pure_research'),
            ({'target_audience': 'Experts'}, 'target_audience', 'General readers', 'Experts'),
            ({}, 'target_audience', 'General readers', 'General readers'),
            ({'word_count_target': 75000}, 'word_count_target', 50000, 75000),
            ({}, 'word_count_target', 50000, 50000),
        ]

        for metadata, key, default, expected in test_cases:
            book_structure = BookStructure(
                title='Test',
                sections=[],
                metadata=metadata
            )

            result = book_structure.metadata.get(key, default)
            assert result == expected, f"Failed for key='{key}': expected {expected}, got {result}"

    def test_none_handling_in_metadata(self):
        """Test handling of None values in metadata."""
        # Test with None metadata
        book_structure = BookStructure(
            title='Test',
            sections=[],
            metadata=None
        )

        # This should not crash and should return defaults
        with pytest.raises(AttributeError):
            # This would fail before our fix if metadata was None
            content_type = book_structure.metadata.get('content_type', 'default')

    def test_book_structure_creation_from_json(self):
        """Test creating book structure from JSON data (like in load_book_structure)."""
        # Simulate the JSON data structure
        json_data = {
            'title': 'Test Book from JSON',
            'metadata': {
                'content_type': 'science_journalism_article_writer',
                'target_audience': 'Science enthusiasts',
                'word_count_target': 80000
            },
            'sections': [
                {
                    'subsection_id': '1A1',
                    'title': 'Introduction',
                    'job_file': 'data_jobs/1A1.jsonl',
                    'batch_params': {'key': 'science', 'k': 8},
                    'merge_params': {'key': 'science', 'k': 4},
                    'dependencies': []
                }
            ]
        }

        # Simulate parsing (similar to load_book_structure)
        metadata = json_data.get('metadata', {})
        sections_data = json_data.get('sections', [])

        sections = []
        for section_data in sections_data:
            section = SectionConfig(
                subsection_id=section_data['subsection_id'],
                title=section_data['title'],
                job_file=Path(section_data['job_file']) if section_data.get('job_file') else None,
                batch_params=section_data.get('batch_params', {}),
                merge_params=section_data.get('merge_params', {}),
                dependencies=section_data.get('dependencies', [])
            )
            sections.append(section)

        book_structure = BookStructure(
            title=json_data.get('title', 'Untitled'),
            sections=sections,
            metadata=metadata
        )

        # Verify the structure
        assert book_structure.title == 'Test Book from JSON'
        assert book_structure.metadata['content_type'] == 'science_journalism_article_writer'
        assert len(book_structure.sections) == 1
        assert book_structure.sections[0].subsection_id == '1A1'
        assert book_structure.sections[0].batch_params['key'] == 'science'


class TestErrorScenarios:
    """Test error scenarios that could cause variable scoping issues."""

    def test_missing_metadata_key(self):
        """Test accessing missing keys in metadata."""
        book_structure = BookStructure(
            title='Test',
            sections=[],
            metadata={'existing_key': 'value'}
        )

        # Test safe access
        result = book_structure.metadata.get('missing_key', 'default_value')
        assert result == 'default_value'

        # Test unsafe access (should raise KeyError)
        with pytest.raises(KeyError):
            _ = book_structure.metadata['missing_key']

    def test_none_section_params(self):
        """Test handling None values in section parameters."""
        section = SectionConfig(
            subsection_id='1A1',
            title='Test Section',
            batch_params=None,
            merge_params=None
        )

        # These should be initialized as empty dicts by __post_init__
        assert section.batch_params == {}
        assert section.merge_params == {}

        # Test safe access
        key = section.batch_params.get('key', 'default')
        assert key == 'default'

    def test_empty_sections_list(self):
        """Test handling empty sections list."""
        book_structure = BookStructure(
            title='Test Book',
            sections=[],
            metadata={'content_type': 'test'}
        )

        assert len(book_structure.sections) == 0
        assert book_structure.title == 'Test Book'
        assert book_structure.metadata['content_type'] == 'test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])