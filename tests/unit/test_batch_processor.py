"""
Unit tests for batch processor functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, call
from argparse import Namespace

from src.langchain.lc_batch import (
    load_jsonl_file,
    run_rag_query,
    main
)
from src.config.settings import get_config


class TestJSONLHandling:
    """Test JSONL file handling functionality."""

    def test_load_jsonl_file_valid(self, temp_dir, sample_job_data):
        """Test loading valid JSONL file."""
        # Create test JSONL file
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        # Load and verify
        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert len(loaded_jobs) == 2
        assert loaded_jobs[0]["task"] == sample_job_data[0]["task"]
        assert loaded_jobs[1]["instruction"] == sample_job_data[1]["instruction"]

    def test_load_jsonl_file_with_invalid_lines(self, temp_dir):
        """Test loading JSONL file with some invalid lines."""
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('{"invalid": json}\n')  # Invalid JSON
            f.write('{"also": "valid"}\n')

        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert len(loaded_jobs) == 2  # Only valid lines loaded
        assert loaded_jobs[0]["valid"] == "json"
        assert loaded_jobs[1]["also"] == "valid"

    def test_load_jsonl_file_empty(self, temp_dir):
        """Test loading empty JSONL file."""
        jsonl_file = temp_dir / "empty.jsonl"
        jsonl_file.write_text("")

        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert loaded_jobs == []

    def test_load_jsonl_file_missing(self, temp_dir):
        """Test loading non-existent JSONL file."""
        missing_file = temp_dir / "missing.jsonl"

        with pytest.raises(SystemExit):
            load_jsonl_file(str(missing_file))


class TestRAGQueryIntegration:
    """Test RAG query integration."""

    def test_run_rag_query_failure(self):
        """Test RAG query execution failure."""
        with patch('langchain.lc_batch.get_config'), \
             patch('langchain.lc_batch.validate_collection', side_effect=Exception("Collection validation failed")):

            result = run_rag_query(
                task="Test task",
                instruction="Test instruction"
            )

            assert "error" in result
            assert result["generated_content"] == ""
            assert result["sources"] == []


class TestCommandLineInterface:
    """Test command line interface functionality."""

    def test_main_with_jsonl_file(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test main function with JSONL input file."""
        # Create test JSONL file
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        # Mock command line arguments
        test_args = ['--jobs', str(jsonl_file), '--key', 'test_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            # This should not raise an exception
            main()

    def test_main_with_json_array(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test main function with JSON array input file."""
        # Create test JSON array file
        json_file = temp_dir / "test_jobs.json"
        with open(json_file, 'w') as f:
            json.dump(sample_job_data, f)

        test_args = ['--jobs', str(json_file), '--key', 'test_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            main()

    def test_main_legacy_arguments(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test main function with legacy positional arguments."""
        # Create test JSON array file
        json_file = temp_dir / "test_jobs.json"
        with open(json_file, 'w') as f:
            json.dump(sample_job_data, f)

        # Legacy format: filename key content_type
        test_args = [str(json_file), 'test_key', 'pure_research']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            main()

    def test_main_missing_jobs_argument(self, mock_console):
        """Test main function with missing --jobs argument."""
        with patch('sys.argv', ['lc_batch.py']):
            with pytest.raises(SystemExit):
                main()

    def test_main_invalid_json_file(self, temp_dir, mock_console):
        """Test main function with invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text('{"invalid": json}')

        test_args = ['--jobs', str(invalid_file)]

        with patch('sys.argv', ['lc_batch.py'] + test_args):
            with pytest.raises(SystemExit):
                main()

    def test_main_empty_jobs_file(self, temp_dir, mock_console):
        """Test main function with empty jobs file."""
        empty_file = temp_dir / "empty.jsonl"
        empty_file.write_text("")

        test_args = ['--jobs', str(empty_file)]

        with patch('sys.argv', ['lc_batch.py'] + test_args):
            with pytest.raises(SystemExit):
                main()


class TestBatchProcessing:
    """Test batch processing logic."""

    def test_main_function_exists(self):
        """Test that main function can be imported and called."""
        # Simple test to verify the main function exists and is callable
        assert callable(main)

    def test_load_jsonl_file_function_exists(self):
        """Test that load_jsonl_file function exists."""
        assert callable(load_jsonl_file)

    def test_run_rag_query_function_exists(self):
        """Test that run_rag_query function exists."""
        assert callable(run_rag_query)


class TestSimplifiedBatchProcessor:
    """Simplified tests for batch processor functionality."""

    def test_jsonl_file_loading(self, temp_dir):
        """Test loading JSONL files with valid data."""
        jsonl_file = temp_dir / "test_jobs.jsonl"
        test_data = [
            {"task": "Test task 1", "instruction": "Test instruction 1"},
            {"task": "Test task 2", "instruction": "Test instruction 2"}
        ]

        with open(jsonl_file, 'w') as f:
            for job in test_data:
                f.write(json.dumps(job) + '\n')

        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert len(loaded_jobs) == 2
        assert loaded_jobs[0]["task"] == "Test task 1"
        assert loaded_jobs[1]["instruction"] == "Test instruction 2"

    def test_jsonl_file_with_invalid_lines(self, temp_dir):
        """Test loading JSONL files with some invalid JSON lines."""
        jsonl_file = temp_dir / "test_jobs.jsonl"

        with open(jsonl_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('{"invalid": json}\n')  # Invalid JSON
            f.write('{"also": "valid"}\n')

        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert len(loaded_jobs) == 2  # Only valid lines loaded
        assert loaded_jobs[0]["valid"] == "json"
        assert loaded_jobs[1]["also"] == "valid"

    def test_empty_jsonl_file(self, temp_dir):
        """Test loading empty JSONL file."""
        jsonl_file = temp_dir / "empty.jsonl"
        jsonl_file.write_text("")

        loaded_jobs = load_jsonl_file(str(jsonl_file))
        assert loaded_jobs == []