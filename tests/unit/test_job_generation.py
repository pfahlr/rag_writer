#!/usr/bin/env python3
"""
Unit tests for job generation functionality, including configurable prompts per section.
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.langchain.job_generation import create_fallback_jobs
from src.config.settings import get_config


class TestConfigurablePrompts:
    """Test the configurable number of prompts per section feature."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_params = {
            "section_title": "Introduction to Machine Learning",
            "book_title": "AI for Everyone",
            "chapter_title": "Chapter 1",
            "section_title_hierarchy": "Section A",
            "subsection_title": "What is Machine Learning?",
            "subsection_id": "1A1",
            "target_audience": "beginners",
            "topic": "artificial intelligence",
        }

    def test_generate_single_prompt(self):
        """Test generating exactly 1 prompt per section."""
        jobs = create_fallback_jobs(num_prompts=1, **self.test_params)

        assert len(jobs) == 1
        assert (
            jobs[0]["task"]
            == f"You are a content writer creating educational material for '{self.test_params['book_title']}'. Focus on practical applications for {self.test_params['target_audience']}."
        )
        assert "introduction" in jobs[0]["instruction"].lower()

    def test_generate_multiple_prompts(self):
        """Test generating multiple prompts per section."""
        for num_prompts in [2, 4, 6, 8]:
            jobs = create_fallback_jobs(num_prompts=num_prompts, **self.test_params)

            assert len(jobs) == num_prompts
            assert all("job_index" in job for job in jobs)
            assert all(job["job_index"] == i + 1 for i, job in enumerate(jobs))

    def test_prompt_diversity(self):
        """Test that different prompts have different instructions."""
        jobs = create_fallback_jobs(num_prompts=4, **self.test_params)

        instructions = [job["instruction"] for job in jobs]
        # Should have at least 2 different instruction types
        unique_instructions = set(instructions)
        assert len(unique_instructions) >= 2

    def test_context_consistency(self):
        """Test that all prompts maintain consistent context."""
        jobs = create_fallback_jobs(num_prompts=3, **self.test_params)

        for job in jobs:
            context = job["context"]
            assert context["book_title"] == self.test_params["book_title"]
            assert context["chapter"] == self.test_params["chapter_title"]
            assert context["section"] == self.test_params["section_title_hierarchy"]
            assert context["subsection"] == self.test_params["subsection_title"]
            assert context["subsection_id"] == self.test_params["subsection_id"]
            assert context["target_audience"] == self.test_params["target_audience"]
            assert context["topic"] == self.test_params["topic"]

    def test_job_structure(self):
        """Test that all jobs have the required structure."""
        jobs = create_fallback_jobs(num_prompts=2, **self.test_params)

        required_fields = ["task", "instruction", "context", "job_index"]
        for job in jobs:
            for field in required_fields:
                assert field in job
            assert isinstance(job["context"], dict)

    def test_cycling_templates(self):
        """Test that templates cycle correctly when requesting more prompts than available templates."""
        jobs = create_fallback_jobs(num_prompts=10, **self.test_params)

        assert len(jobs) == 10
        # Should cycle through the 8 available templates
        template_indices = [job["job_index"] for job in jobs]
        assert template_indices == list(range(1, 11))


class TestConfigurationIntegration:
    """Test integration with the configuration system."""

    def test_config_values(self):
        """Test that configuration values are accessible."""
        config = get_config()

        assert hasattr(config.job_generation, "default_prompts_per_section")
        assert hasattr(config.job_generation, "min_prompts_per_section")
        assert hasattr(config.job_generation, "max_prompts_per_section")

        assert config.job_generation.default_prompts_per_section == 4
        assert config.job_generation.min_prompts_per_section == 1
        assert config.job_generation.max_prompts_per_section == 10

    def test_config_validation(self):
        """Test that configuration values are within valid ranges."""
        config = get_config()

        assert 1 <= config.job_generation.default_prompts_per_section <= 10
        assert config.job_generation.min_prompts_per_section >= 1
        assert config.job_generation.max_prompts_per_section <= 10
        assert (
            config.job_generation.min_prompts_per_section
            <= config.job_generation.default_prompts_per_section
        )


class TestContentTypeLoading:
    """Test on-demand content type loading functionality."""

    def test_load_all_content_types_on_demand(self):
        """Test that all content types can be loaded on-demand."""
        from src.utils.template_engine import (
            get_job_generation_prompt,
            get_job_generation_rag_context,
            get_job_templates,
        )

        content_types = [
            "pure_research",
            "technical_manual_writer",
            "science_journalism_article_writer",
            "folklore_adaptation_and_anthology_editor",
        ]

        for content_type in content_types:
            # All content types should have job generation prompts and RAG context
            prompt = get_job_generation_prompt(content_type)
            rag_context = get_job_generation_rag_context(content_type)

            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert isinstance(rag_context, str)
            assert len(rag_context) > 0

            # Only technical_manual_writer should have job templates
            if content_type == "technical_manual_writer":
                templates = get_job_templates(content_type)
                assert isinstance(templates, list)
                assert len(templates) > 0
            else:
                # Other content types should not have job templates
                try:
                    get_job_templates(content_type)
                    assert (
                        False
                    ), f"Content type {content_type} should not have job templates"
                except ValueError as e:
                    assert "No job templates found" in str(e)

    def test_content_type_error_handling(self):
        """Test error handling for non-existent content types."""
        from src.utils.template_engine import get_job_generation_prompt

        # Should fall back to default.yaml for non-existent content types
        prompt = get_job_generation_prompt("non_existent_content_type")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain content from default.yaml
        assert "book_title" in prompt or "collection_title" in prompt

    def test_template_rendering_with_context(self):
        """Test that templates render correctly with context variables."""
        from src.utils.template_engine import render_job_templates, get_job_templates

        # Get templates for technical_manual_writer
        templates = get_job_templates("technical_manual_writer")

        # Test context
        context = {
            "section_title": "Machine Learning Basics",
            "book_title": "AI Handbook",
            "chapter_title": "Chapter 1",
            "section_title_hierarchy": "Section A",
            "subsection_title": "What is ML?",
            "subsection_id": "1A1",
            "target_audience": "beginners",
            "topic": "artificial intelligence",
        }

        # Render templates
        rendered = render_job_templates(templates[:2], context)

        # Verify rendering worked
        assert len(rendered) == 2
        for job in rendered:
            assert "task" in job
            assert "instruction" in job
            assert "context" in job

            # Check that variables were replaced
            assert "AI Handbook" in job["task"]
            assert "beginners" in job["task"]
            assert (
                "machine learning basics" in job["instruction"]
            )  # Note: template uses |lower filter

            # Verify context structure
            job_context = job["context"]
            assert job_context["book_title"] == "AI Handbook"
            assert job_context["chapter"] == "Chapter 1"
            assert job_context["section"] == "Section A"
            assert job_context["subsection"] == "What is ML?"
            assert job_context["subsection_id"] == "1A1"
            assert job_context["target_audience"] == "beginners"
            assert job_context["topic"] == "artificial intelligence"

    def test_content_type_file_structure(self):
        """Test that content type files are properly structured."""
        from pathlib import Path
        import yaml

        content_types_dir = (
            Path(__file__).parent.parent.parent
            / "src"
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )
        expected_files = [
            "pure_research.yaml",
            "technical_manual_writer.yaml",
            "science_journalism_article_writer.yaml",
            "folklore_adaptation_and_anthology_editor.yaml",
        ]

        # Check that all expected files exist
        for filename in expected_files:
            file_path = content_types_dir / filename
            assert file_path.exists(), f"Missing content type file: {filename}"

            # Check that file is valid YAML and has required structure
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert isinstance(config, dict)
            assert "description" in config
            assert "system_prompt" in config
            assert "job_generation_prompt" in config
            assert "job_generation_rag_context" in config


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_params = {
            "section_title": "Introduction to Machine Learning",
            "book_title": "AI for Everyone",
            "chapter_title": "Chapter 1",
            "section_title_hierarchy": "Section A",
            "subsection_title": "What is Machine Learning?",
            "subsection_id": "1A1",
            "target_audience": "beginners",
            "topic": "artificial intelligence",
            "content_type": "technical_manual_writer",
        }

    def test_zero_prompts(self):
        """Test behavior with 0 prompts (should return empty list)."""
        jobs = create_fallback_jobs(
            num_prompts=0,
            content_type=self.test_params["content_type"],
            **{k: v for k, v in self.test_params.items() if k != "content_type"},
        )
        assert len(jobs) == 0

    def test_large_number_prompts(self):
        """Test behavior with large number of prompts."""
        jobs = create_fallback_jobs(
            num_prompts=20,
            content_type=self.test_params["content_type"],
            **{k: v for k, v in self.test_params.items() if k != "content_type"},
        )
        assert len(jobs) == 20
        # Should cycle through templates multiple times
        assert all(1 <= job["job_index"] <= 20 for job in jobs)

    def test_empty_parameters(self):
        """Test behavior with empty string parameters."""
        params = self.test_params.copy()
        params["section_title"] = ""
        params["target_audience"] = ""

        jobs = create_fallback_jobs(
            num_prompts=1,
            content_type=params["content_type"],
            **{k: v for k, v in params.items() if k != "content_type"},
        )
        assert len(jobs) == 1
        # Should still generate valid job structure
        assert "task" in jobs[0]
        assert "instruction" in jobs[0]
        assert "context" in jobs[0]
