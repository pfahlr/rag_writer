"""
Unit tests for batch processor functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, call
from argparse import Namespace

from langchain.lc_batch import (
    load_jsonl_file,
    run_lc_ask,
    main
)


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


class TestLCAskIntegration:
    """Test lc_ask integration."""

    def test_run_lc_ask_success(self, mock_subprocess_run, mock_lc_ask_response):
        """Test successful lc_ask execution."""
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            result = run_lc_ask(
                task="Test task",
                instruction="Test instruction",
                key="test_key",
                content_type="pure_research"
            )

            assert result["generated_content"] == "Mock generated content"
            assert result["status"] == "success"

    def test_run_lc_ask_with_topk(self, mock_subprocess_run):
        """Test lc_ask execution with top-k parameter."""
        with patch('subprocess.run', side_effect=mock_subprocess_run) as mock_run:
            result = run_lc_ask(
                task="Test task",
                instruction="Test instruction",
                key="test_key",
                content_type="pure_research",
                topk=5
            )

            # Verify --k parameter was included
            call_args = mock_run.call_args[0][0]
            assert "--k" in call_args
            assert "5" in call_args

    def test_run_lc_ask_failure(self):
        """Test lc_ask execution failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Command failed"

        with patch('subprocess.run', return_value=mock_result):
            result = run_lc_ask(
                task="Test task",
                instruction="Test instruction"
            )

            assert result["error"] == "Command failed"
            assert result["generated_content"] == ""
            assert result["status"] == "error"


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

    def test_process_valid_jobs(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test processing of valid jobs."""
        # Create test JSONL file
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        test_args = ['--jobs', str(jsonl_file), '--key', 'test_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run), \
             patch('langchain.lc_batch.ROOT', temp_dir):

            main()

            # Verify subprocess was called for each job
            # Note: This test would need more sophisticated mocking to verify exact calls

    def test_process_jobs_with_errors(self, temp_dir, mock_console):
        """Test processing jobs that return errors."""
        # Create test file with valid job
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            json.dump({
                "task": "Test task",
                "instruction": "Test instruction"
            }, f)

        # Mock subprocess to return error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Mock error"

        test_args = ['--jobs', str(jsonl_file)]

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', return_value=mock_result), \
             patch('langchain.lc_batch.ROOT', temp_dir):

            main()

            # Should complete without crashing, but with errors recorded

    def test_output_file_generation(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test that output files are generated correctly."""
        # Create test JSONL file
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        test_args = ['--jobs', str(jsonl_file), '--key', 'test_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run), \
             patch('langchain.lc_batch.ROOT', temp_dir):

            main()

            # Check that output file was created
            output_files = list((temp_dir / "output" / "batch").glob("batch_results_*.json"))
            assert len(output_files) == 1

            # Verify output file contents
            with open(output_files[0], 'r') as f:
                results = json.load(f)

            assert len(results) == 2  # Two jobs processed
            assert all('status' in result for result in results)


class TestParameterHandling:
    """Test parameter handling and validation."""

    def test_key_parameter(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test key parameter handling."""
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        test_args = ['--jobs', str(jsonl_file), '--key', 'custom_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run) as mock_run:

            main()

            # Verify the key was passed to lc_ask
            call_args = mock_run.call_args[0][0]
            assert '--key' in call_args
            assert 'custom_key' in call_args

    def test_content_type_parameter(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test content_type parameter handling."""
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        test_args = ['--jobs', str(jsonl_file), '--content-type', 'academic']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run) as mock_run:

            main()

            # Verify the content type was passed to lc_ask
            call_args = mock_run.call_args[0][0]
            assert '--content-type' in call_args
            assert 'academic' in call_args

    def test_topk_parameter(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test top-k parameter handling."""
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        test_args = ['--jobs', str(jsonl_file), '--k', '10']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_subprocess_run) as mock_run:

            main()

            # Verify the top-k was passed to lc_ask
            call_args = mock_run.call_args[0][0]
            assert '--k' in call_args
            assert '10' in call_args


class TestErrorRecovery:
    """Test error recovery and edge cases."""

    def test_partial_job_failure(self, temp_dir, mock_console):
        """Test handling when some jobs fail but others succeed."""
        # Create mixed success/failure scenario
        jsonl_file = temp_dir / "mixed_jobs.jsonl"

        jobs = [
            {"task": "Success task", "instruction": "Success instruction"},
            {"task": "Fail task", "instruction": "Fail instruction"},
            {"task": "Success task 2", "instruction": "Success instruction 2"}
        ]

        with open(jsonl_file, 'w') as f:
            for job in jobs:
                f.write(json.dumps(job) + '\n')

        # Mock subprocess to fail on second job
        def mock_run(cmd, **kwargs):
            mock_result = Mock()
            if "Fail task" in str(cmd):
                mock_result.returncode = 1
                mock_result.stderr = "Mock failure"
            else:
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    "generated_content": "Mock success content",
                    "sources": [],
                    "status": "success"
                })
            mock_result.stderr = ""
            return mock_result

        test_args = ['--jobs', str(jsonl_file)]

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('subprocess.run', side_effect=mock_run), \
             patch('langchain.lc_batch.ROOT', temp_dir):

            main()

            # Should complete and generate output file
            output_files = list((temp_dir / "output" / "batch").glob("batch_results_*.json"))
            assert len(output_files) == 1

            with open(output_files[0], 'r') as f:
                results = json.load(f)

            # Should have results for all jobs (some successful, some failed)
            assert len(results) == 3
            success_count = len([r for r in results if r.get('status') == 'success'])
            error_count = len([r for r in results if r.get('status') == 'error'])
            assert success_count == 2
            assert error_count == 1