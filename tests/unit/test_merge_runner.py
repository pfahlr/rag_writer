"""
Unit tests for merge runner functionality.
"""

import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, Mock, call

from langchain.lc_merge_runner import (
    load_merge_types,
    select_merge_type,
    select_sections,
    select_job_file,
    get_context_info,
    run_pipeline_stage,
    run_advanced_pipeline,
    merge_section_content,
    save_merged_results,
    main
)


class TestMergeTypeLoading:
    """Test merge type configuration loading."""

    def test_load_merge_types_valid(self, temp_dir):
        """Test loading valid merge types YAML."""
        # Create test merge types file in the expected location
        merge_types_dir = temp_dir / "src" / "tool" / "prompts"
        merge_types_dir.mkdir(parents=True, exist_ok=True)
        merge_types_file = merge_types_dir / "merge_types.yaml"

        merge_config = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor..."
            },
            "advanced_pipeline": {
                "description": "Multi-stage pipeline",
                "stages": {
                    "critique": {
                        "system_prompt": "You are a critic...",
                        "output_format": "json"
                    },
                    "merge": {
                        "system_prompt": "You are a merger...",
                        "output_format": "markdown"
                    }
                }
            }
        }

        with open(merge_types_file, 'w') as f:
            yaml.dump(merge_config, f)

        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            merge_types = load_merge_types()

            assert len(merge_types) == 2
            assert "generic_editor" in merge_types
            assert "advanced_pipeline" in merge_types

            # Check simple merge type
            generic = merge_types["generic_editor"]
            assert generic["description"] == "Basic editor merge"
            assert generic["system_prompt"] == "You are a senior editor..."
            assert generic["stages"] is None

            # Check advanced pipeline
            advanced = merge_types["advanced_pipeline"]
            assert advanced["description"] == "Multi-stage pipeline"
            assert advanced["stages"] is not None
            assert "critique" in advanced["stages"]
            assert "merge" in advanced["stages"]

    def test_load_merge_types_missing_file(self, temp_dir, mock_console):
        """Test loading merge types when file doesn't exist."""
        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            merge_types = load_merge_types()

            # Should return default configuration
            assert len(merge_types) == 1
            assert "generic_editor" in merge_types

    def test_load_merge_types_malformed(self, temp_dir, mock_console):
        """Test loading malformed merge types file."""
        merge_types_file = temp_dir / "merge_types.yaml"
        merge_types_file.write_text("invalid: yaml: content: [")

        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            merge_types = load_merge_types()

            # Should return default configuration
            assert len(merge_types) == 1
            assert "generic_editor" in merge_types


class TestMergeTypeSelection:
    """Test merge type selection functionality."""

    def test_select_merge_type_simple(self, mock_console):
        """Test selecting simple merge type."""
        merge_types = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor...",
                "stages": None
            },
            "advanced_pipeline": {
                "description": "Multi-stage pipeline",
                "stages": {"critique": {}, "merge": {}}
            }
        }

        with patch('builtins.input', return_value='1'):  # Select first option
            selected = select_merge_type(merge_types)
            assert selected == "generic_editor"

    def test_select_merge_type_by_name(self, mock_console):
        """Test selecting merge type by name."""
        merge_types = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor...",
                "stages": None
            }
        }

        with patch('builtins.input', return_value='generic_editor'):
            selected = select_merge_type(merge_types)
            assert selected == "generic_editor"

    def test_select_merge_type_invalid_input(self, mock_console):
        """Test handling invalid merge type selection."""
        merge_types = {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor...",
                "stages": None
            }
        }

        inputs = ['invalid', '99', 'generic_editor']  # Invalid, then valid
        with patch('builtins.input', side_effect=inputs):
            selected = select_merge_type(merge_types)
            assert selected == "generic_editor"


class TestSectionSelection:
    """Test section selection functionality."""

    def test_select_sections_specific(self, mock_console):
        """Test selecting specific sections."""
        sections = {
            "1A1": [{"status": "success"}],
            "1A2": [{"status": "success"}],
            "1B1": [{"status": "success"}]
        }

        with patch('builtins.input', return_value='1A1,1B1'):
            selected = select_sections(sections)
            assert set(selected) == {"1A1", "1B1"}

    def test_select_sections_all(self, mock_console):
        """Test selecting all sections."""
        sections = {
            "1A1": [{"status": "success"}],
            "1A2": [{"status": "success"}],
            "1B1": [{"status": "success"}]
        }

        with patch('builtins.input', return_value='all'):
            selected = select_sections(sections)
            assert set(selected) == {"1A1", "1A2", "1B1"}

    def test_select_sections_invalid(self, mock_console):
        """Test handling invalid section selection."""
        sections = {
            "1A1": [{"status": "success"}],
            "1A2": [{"status": "success"}]
        }

        inputs = ['invalid', '1A1']  # Invalid, then valid
        with patch('builtins.input', side_effect=inputs):
            selected = select_sections(sections)
            assert selected == ["1A1"]


class TestJobFileSelection:
    """Test job file selection functionality."""

    def test_select_job_file_custom_path(self, temp_dir, mock_console):
        """Test selecting custom job file path."""
        job_file = temp_dir / "custom_jobs.jsonl"
        job_file.write_text('{"test": "data"}')

        with patch('builtins.input', side_effect=['1', str(job_file)]):
            result = select_job_file()
            assert result == job_file

    def test_select_job_file_default_path(self, mock_console):
        """Test using default job file path."""
        with patch('builtins.input', side_effect=['2', '1A1']):
            result = select_job_file()
            assert result is None  # Will use default path

    def test_select_job_file_skip(self, mock_console):
        """Test skipping job file processing."""
        with patch('builtins.input', return_value='3'):
            result = select_job_file()
            assert result is False


class TestContextInfo:
    """Test context information gathering."""

    def test_get_context_info(self, mock_console):
        """Test gathering context information."""
        inputs = ['Test Chapter', 'Test Section', 'Test Subsection']

        with patch('builtins.input', side_effect=inputs):
            context = get_context_info()

            assert context['chapter'] == 'Test Chapter'
            assert context['section'] == 'Test Section'
            assert context['subsection'] == 'Test Subsection'


class TestPipelineStageExecution:
    """Test pipeline stage execution."""

    def test_run_pipeline_stage_success(self, mock_subprocess_run, mock_lc_ask_response):
        """Test successful pipeline stage execution."""
        stage_config = {
            "system_prompt": "You are a critic...",
            "output_format": "json"
        }

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            result = run_pipeline_stage(
                "critique",
                stage_config,
                "Test content",
                {"chapter": "1", "section": "A", "subsection": "1"},
                "1A1"
            )

            assert result["generated_content"] == "Mock generated content"

    def test_run_pipeline_stage_failure(self, mock_console):
        """Test pipeline stage execution failure."""
        stage_config = {"system_prompt": "Test prompt"}

        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Mock error"

        with patch('subprocess.run', return_value=mock_result):
            result = run_pipeline_stage(
                "critique",
                stage_config,
                "Test content",
                {"chapter": "1", "section": "A", "subsection": "1"},
                "1A1"
            )

            assert result == "Test content"  # Should return original content on failure


class TestAdvancedPipeline:
    """Test advanced pipeline execution."""

    def test_run_advanced_pipeline_simple(self, mock_subprocess_run):
        """Test advanced pipeline with simple content."""
        variations = [
            {"status": "success", "generated_content": "Content 1"},
            {"status": "success", "generated_content": "Content 2"}
        ]

        pipeline_stages = {
            "merge": {
                "system_prompt": "Merge content...",
                "output_format": "markdown"
            }
        }

        context = {"chapter": "1", "section": "A", "subsection": "1"}

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            result = run_advanced_pipeline(variations, context, pipeline_stages, "1A1")

            assert "final_content" in result
            assert "stage_results" in result
            assert result["original_variations"] == 2

    def test_run_advanced_pipeline_with_critique(self, mock_subprocess_run):
        """Test advanced pipeline with critique stage."""
        variations = [
            {"status": "success", "generated_content": "Content 1"},
            {"status": "success", "generated_content": "Content 2"}
        ]

        pipeline_stages = {
            "critique": {
                "system_prompt": "Critique content...",
                "output_format": "json"
            },
            "merge": {
                "system_prompt": "Merge content...",
                "output_format": "markdown"
            }
        }

        context = {"chapter": "1", "section": "A", "subsection": "1"}

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            result = run_advanced_pipeline(variations, context, pipeline_stages, "1A1")

            assert "critique" in result["stage_results"]
            assert "merge" in result["stage_results"]
            assert len(result["stage_results"]["critique"]) == 2  # One critique per variation

    def test_run_advanced_pipeline_no_successful_variations(self):
        """Test advanced pipeline with no successful variations."""
        variations = [
            {"status": "failed", "generated_content": ""},
            {"status": "error", "generated_content": ""}
        ]

        pipeline_stages = {"merge": {"system_prompt": "Merge..."}}
        context = {"chapter": "1", "section": "A", "subsection": "1"}

        result = run_advanced_pipeline(variations, context, pipeline_stages, "1A1")

        assert "error" in result
        assert "No successful content variations" in result["error"]


class TestSimpleMerge:
    """Test simple merge functionality."""

    def test_merge_section_content_success(self, mock_subprocess_run):
        """Test successful simple merge."""
        variations = [
            {"status": "success", "generated_content": "Content 1"},
            {"status": "success", "generated_content": "Content 2"}
        ]

        context = {"chapter": "1", "section": "A", "subsection": "1"}

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            result = merge_section_content(variations, context, "Test prompt")

            assert result == "Mock generated content"

    def test_merge_section_content_no_successful(self):
        """Test merge with no successful variations."""
        variations = [
            {"status": "failed", "generated_content": ""},
            {"status": "error", "generated_content": ""}
        ]

        context = {"chapter": "1", "section": "A", "subsection": "1"}

        result = merge_section_content(variations, context, "Test prompt")

        assert "No successful content variations found" in result


class TestResultSaving:
    """Test merged result saving."""

    def test_save_merged_results(self, temp_dir, mock_console):
        """Test saving merged results."""
        merged_sections = {
            "1A1": {
                "original_variations": 2,
                "merged_content": "Merged content",
                "context": {"chapter": "1", "section": "A", "subsection": "1"},
                "merge_type": "generic_editor",
                "pipeline_type": "simple"
            }
        }

        context = {"chapter": "1", "section": "A", "subsection": "1"}

        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            save_merged_results(merged_sections, context, "generic_editor")

            # Check output file was created
            output_files = list((temp_dir / "output" / "merged").glob("merged_content_*.json"))
            assert len(output_files) == 1

            # Verify contents
            with open(output_files[0], 'r') as f:
                data = json.load(f)

            assert data["metadata"]["merge_type"] == "generic_editor"
            assert data["metadata"]["pipeline_type"] == "simple"
            assert "1A1" in data["sections"]


class TestMainFunction:
    """Test main function integration."""

    def test_main_simple_merge(self, temp_dir, mock_console):
        """Test main function with simple merge."""
        # Create mock batch results
        batch_results = [
            {
                "section": "1A1",
                "generated_content": "Test content",
                "status": "success"
            }
        ]

        batch_file = temp_dir / "output" / "batch" / "batch_results_test.json"
        batch_file.parent.mkdir(parents=True, exist_ok=True)

        with open(batch_file, 'w') as f:
            json.dump(batch_results, f)

        # Create merge types file
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
             patch('builtins.input', side_effect=inputs), \
             patch('subprocess.run') as mock_run:

            # Mock successful subprocess call
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"generated_content": "Merged content"})
            mock_run.return_value = mock_result

            # This should run without errors
            main()

            # Verify merge was called
            assert mock_run.called


class TestErrorHandling:
    """Test error handling in merge runner."""

    def test_main_no_batch_results(self, temp_dir, mock_console):
        """Test main function when no batch results exist."""
        with patch('langchain.lc_merge_runner.ROOT', temp_dir):
            with pytest.raises(SystemExit):
                main()

    def test_pipeline_stage_json_decode_error(self, mock_console):
        """Test handling JSON decode errors in pipeline stages."""
        stage_config = {"system_prompt": "Test prompt"}

        # Mock subprocess with invalid JSON output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"

        with patch('subprocess.run', return_value=mock_result):
            result = run_pipeline_stage(
                "test",
                stage_config,
                "Test content",
                {"chapter": "1", "section": "A", "subsection": "1"},
                "1A1"
            )

            assert result == "Test content"  # Should return original content