"""
Integration tests for the complete content generation pipeline.
"""

import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, Mock

from langchain.lc_outline_converter import (
    parse_text_outline,
    generate_book_structure,
    generate_job_file
)
from langchain.lc_batch import load_jsonl_file, main as batch_main
from langchain.lc_merge_runner import main as merge_main
from langchain.lc_book_runner import load_book_structure, main as book_main


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the complete pipeline from outline to final content."""

    def test_complete_pipeline_text_outline(self, temp_dir, sample_text_outline, mock_subprocess_run, mock_console):
        """Test complete pipeline with text outline input."""
        # Step 1: Parse outline and generate book structure
        sections, metadata = parse_text_outline(sample_text_outline)

        with patch('langchain.lc_outline_converter.ROOT', temp_dir):
            book_structure = generate_book_structure(sections, metadata)

            # Verify book structure
            assert len(book_structure["sections"]) == 6  # 2 chapters + 3 sections + 1 subsection
            assert book_structure["title"] == "Professional Development Handbook for Primary School Educators"

            # Step 2: Generate job files
            generated_jobs = 0
            for section in sections:
                if section.level >= 2:  # Generate jobs for sections and subsections
                    generate_job_file(section, metadata, sections)
                    generated_jobs += 1

            assert generated_jobs == 4  # 2 sections + 2 subsections

            # Verify job files exist
            job_files = list((temp_dir / "data_jobs").glob("*.jsonl"))
            assert len(job_files) == 4

            # Step 3: Simulate batch processing
            # Load one of the job files and verify it contains valid jobs
            sample_job_file = temp_dir / "data_jobs" / "1A1.jsonl"
            assert sample_job_file.exists()

            jobs = load_jsonl_file(str(sample_job_file))
            assert len(jobs) == 4  # 4 standard jobs per section

            # Verify job structure
            for job in jobs:
                assert "task" in job
                assert "instruction" in job
                assert "context" in job
                assert job["context"]["subsection_id"] == "1A1"

    def test_book_runner_integration(self, temp_dir, mock_subprocess_run, mock_console):
        """Test book runner with generated book structure."""
        # Create a minimal book structure
        book_data = {
            "title": "Test Book",
            "metadata": {
                "target_audience": "General",
                "topic": "Testing"
            },
            "sections": [
                {
                    "subsection_id": "1A1",
                    "title": "Test Section",
                    "job_file": "data_jobs/1A1.jsonl",
                    "batch_params": {"key": "test"},
                    "merge_params": {"key": "test"},
                    "dependencies": []
                }
            ]
        }

        book_file = temp_dir / "test_book.json"
        with open(book_file, 'w') as f:
            json.dump(book_data, f)

        # Create corresponding job file
        job_file = temp_dir / "data_jobs" / "1A1.jsonl"
        job_file.parent.mkdir(parents=True, exist_ok=True)

        jobs = [{
            "task": "Test task",
            "instruction": "Test instruction",
            "context": {"subsection_id": "1A1"}
        }]

        with open(job_file, 'w') as f:
            for job in jobs:
                f.write(json.dumps(job) + '\n')

        # Test book structure loading
        with patch('langchain.lc_book_runner.ROOT', temp_dir):
            book_structure = load_book_structure(book_file)

            assert book_structure.title == "Test Book"
            assert len(book_structure.sections) == 1
            assert book_structure.sections[0].subsection_id == "1A1"

    def test_batch_processing_integration(self, temp_dir, sample_job_data, mock_subprocess_run, mock_console):
        """Test batch processing with JSONL input."""
        # Create test JSONL file
        jsonl_file = temp_dir / "test_jobs.jsonl"
        with open(jsonl_file, 'w') as f:
            for job in sample_job_data:
                f.write(json.dumps(job) + '\n')

        # Mock command line arguments for batch processing
        test_args = ['--jobs', str(jsonl_file), '--key', 'test_key']

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('langchain.lc_batch.ROOT', temp_dir), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            # This should complete without errors
            batch_main()

            # Verify output was generated
            output_files = list((temp_dir / "output" / "batch").glob("batch_results_*.json"))
            assert len(output_files) == 1

            # Verify output contents
            with open(output_files[0], 'r') as f:
                results = json.load(f)

            assert len(results) == 2  # Two jobs processed
            assert all('status' in result for result in results)

    def test_merge_pipeline_integration(self, temp_dir, mock_subprocess_run, mock_console):
        """Test merge pipeline with batch results."""
        # Create mock batch results
        batch_results = [
            {
                "section": "1A1",
                "generated_content": "Test content 1",
                "status": "success"
            },
            {
                "section": "1A1",
                "generated_content": "Test content 2",
                "status": "success"
            }
        ]

        batch_file = temp_dir / "output" / "batch" / "batch_results_test.json"
        batch_file.parent.mkdir(parents=True, exist_ok=True)

        with open(batch_file, 'w') as f:
            json.dump(batch_results, f)

        # Create merge types configuration
        merge_types_file = temp_dir / "src" / "tool" / "prompts" / "merge_types.yaml"
        merge_types_file.parent.mkdir(parents=True, exist_ok=True)

        merge_config = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor..."
            }
        }

        with open(merge_types_file, 'w') as f:
            yaml.dump(merge_config, f)

        # Mock user inputs for merge process
        inputs = ['1', 'all', 'Test Chapter', 'Test Section', 'Test Subsection']

        with patch('langchain.lc_merge_runner.ROOT', temp_dir), \
             patch('builtins.input', side_effect=inputs), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            # This should complete without errors
            merge_main()

            # Verify merged output was generated
            merged_files = list((temp_dir / "output" / "merged").glob("merged_content_*.json"))
            assert len(merged_files) >= 1

    def test_advanced_pipeline_integration(self, temp_dir, mock_subprocess_run, mock_console):
        """Test advanced multi-stage pipeline."""
        # Create merge types with advanced pipeline
        merge_types_file = temp_dir / "src" / "tool" / "prompts" / "merge_types.yaml"
        merge_types_file.parent.mkdir(parents=True, exist_ok=True)

        merge_config = {
            "advanced_pipeline": {
                "description": "Multi-stage pipeline",
                "stages": {
                    "critique": {
                        "system_prompt": "You are a senior editor evaluating content quality...",
                        "output_format": "json"
                    },
                    "merge": {
                        "system_prompt": "You are a consolidating editor...",
                        "output_format": "markdown"
                    },
                    "style": {
                        "system_prompt": "You are a line editor harmonizing tone...",
                        "output_format": "markdown"
                    }
                }
            }
        }

        with open(merge_types_file, 'w') as f:
            yaml.dump(merge_config, f)

        # Create mock batch results
        batch_results = [
            {
                "section": "1A1",
                "generated_content": "Test content for advanced pipeline",
                "status": "success"
            }
        ]

        batch_file = temp_dir / "output" / "batch" / "batch_results_test.json"
        batch_file.parent.mkdir(parents=True, exist_ok=True)

        with open(batch_file, 'w') as f:
            json.dump(batch_results, f)

        # Mock user inputs
        inputs = ['2', 'all', 'Test Chapter', 'Test Section', 'Test Subsection']  # Select advanced pipeline

        with patch('langchain.lc_merge_runner.ROOT', temp_dir), \
             patch('builtins.input', side_effect=inputs), \
             patch('subprocess.run', side_effect=mock_subprocess_run):

            merge_main()

            # Verify advanced pipeline was executed
            merged_files = list((temp_dir / "output" / "merged").glob("merged_content_*.json"))
            assert len(merged_files) >= 1

            # Check that pipeline metadata indicates advanced processing
            with open(merged_files[0], 'r') as f:
                merged_data = json.load(f)

            assert merged_data["metadata"]["pipeline_type"] == "advanced"


class TestErrorHandlingIntegration:
    """Test error handling across the pipeline."""

    def test_pipeline_resilience_missing_files(self, temp_dir, mock_console):
        """Test pipeline resilience when files are missing."""
        # Try to run book runner with non-existent book file
        missing_book = temp_dir / "missing_book.json"

        with patch('langchain.lc_book_runner.ROOT', temp_dir):
            with pytest.raises(SystemExit):
                book_main()

    def test_batch_processing_invalid_jobs(self, temp_dir, mock_console):
        """Test batch processing with invalid job files."""
        # Create JSONL file with invalid JSON
        invalid_jsonl = temp_dir / "invalid_jobs.jsonl"
        invalid_jsonl.write_text('{"invalid": json}\n{"valid": "json"}\n')

        test_args = ['--jobs', str(invalid_jsonl)]

        with patch('sys.argv', ['lc_batch.py'] + test_args), \
             patch('langchain.lc_batch.ROOT', temp_dir):

            # Should handle invalid JSON gracefully
            batch_main()

            # Should still produce output (with errors recorded)
            output_files = list((temp_dir / "output" / "batch").glob("batch_results_*.json"))
            assert len(output_files) == 1

    def test_merge_with_no_successful_content(self, temp_dir, mock_console):
        """Test merge processing when no content variations are successful."""
        # Create batch results with all failures
        batch_results = [
            {"section": "1A1", "status": "failed", "generated_content": ""},
            {"section": "1A1", "status": "error", "generated_content": ""}
        ]

        batch_file = temp_dir / "output" / "batch" / "batch_results_test.json"
        batch_file.parent.mkdir(parents=True, exist_ok=True)

        with open(batch_file, 'w') as f:
            json.dump(batch_results, f)

        # Create merge types
        merge_types_file = temp_dir / "src" / "tool" / "prompts" / "merge_types.yaml"
        merge_types_file.parent.mkdir(parents=True, exist_ok=True)

        merge_config = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor..."
            }
        }

        with open(merge_types_file, 'w') as f:
            yaml.dump(merge_config, f)

        # Mock user inputs
        inputs = ['1', 'all', 'Test Chapter', 'Test Section', 'Test Subsection']

        with patch('langchain.lc_merge_runner.ROOT', temp_dir), \
             patch('builtins.input', side_effect=inputs):

            # Should handle gracefully
            merge_main()

            # Should still produce output file
            merged_files = list((temp_dir / "output" / "merged").glob("merged_content_*.json"))
            assert len(merged_files) >= 1


class TestConfigurationIntegration:
    """Test configuration handling across components."""

    def test_yaml_configuration_consistency(self, temp_dir):
        """Test that YAML configurations are consistent across components."""
        # Create merge types configuration
        merge_types_file = temp_dir / "src" / "tool" / "prompts" / "merge_types.yaml"
        merge_types_file.parent.mkdir(parents=True, exist_ok=True)

        merge_config = {
            "educator_handbook": {
                "description": "Optimized for PD handbooks for primary school educators",
                "audience": "primary school teachers",
                "stages": {
                    "critique": {
                        "system_prompt": "You are a senior editor for a PD handbook...",
                        "output_format": "json"
                    },
                    "merge": {
                        "system_prompt": "You are the consolidating editor for educational content...",
                        "output_format": "markdown"
                    }
                }
            }
        }

        with open(merge_types_file, 'w') as f:
            yaml.dump(merge_config, f)

        # Test loading configuration
        from langchain.lc_merge_runner import load_merge_types

        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            loaded_config = load_merge_types()

            assert "educator_handbook" in loaded_config
            handbook_config = loaded_config["educator_handbook"]

            assert handbook_config["description"] == "Optimized for PD handbooks for primary school educators"
            assert handbook_config["audience"] == "primary school teachers"
            assert "stages" in handbook_config
            assert "critique" in handbook_config["stages"]
            assert "merge" in handbook_config["stages"]