#!/usr/bin/env python3
"""
Template Engine for Dynamic Content Generation

Provides token replacement functionality for YAML-based templates.
Supports variable substitution using {{variable_name}} syntax and filters like {{variable|filter}}.
"""

import re
from typing import Dict, Any, Union
from pathlib import Path
import yaml


class TemplateEngine:
    """Template engine for token replacement in YAML configurations."""

    def __init__(self):
        self.variable_pattern = re.compile(r"\{\{([^}]+)\}\}")

    def render_string(self, template: str, context: Dict[str, Any]) -> str:
        """Render a string template with variable substitution."""

        def replace_match(match):
            expression = match.group(1).strip()

            # Check for filter syntax (e.g., variable|filter)
            if "|" in expression:
                var_name, filter_name = expression.split("|", 1)
                var_name = var_name.strip()
                filter_name = filter_name.strip()

                value = self._get_value(var_name, context)

                # Apply filter
                if filter_name == "lower":
                    return str(value).lower()
                elif filter_name == "upper":
                    return str(value).upper()
                elif filter_name == "title":
                    return str(value).title()
                else:
                    # Unknown filter, return as-is
                    return str(value)
            else:
                # Simple variable substitution
                return str(self._get_value(expression, context))

        return self.variable_pattern.sub(replace_match, template)

    def render_dict(
        self, template_dict: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Render a dictionary template with variable substitution."""
        result = {}

        for key, value in template_dict.items():
            if isinstance(value, str):
                result[key] = self.render_string(value, context)
            elif isinstance(value, dict):
                result[key] = self.render_dict(value, context)
            elif isinstance(value, list):
                result[key] = self.render_list(value, context)
            else:
                result[key] = value

        return result

    def render_list(self, template_list: list, context: Dict[str, Any]) -> list:
        """Render a list template with variable substitution."""
        result = []

        for item in template_list:
            if isinstance(item, str):
                result.append(self.render_string(item, context))
            elif isinstance(item, dict):
                result.append(self.render_dict(item, context))
            elif isinstance(item, list):
                result.append(self.render_list(item, context))
            else:
                result.append(item)

        return result

    def render_template(
        self, template: Union[str, Dict, list], context: Dict[str, Any]
    ) -> Union[str, Dict, list]:
        """Render any template type with variable substitution."""
        if isinstance(template, str):
            return self.render_string(template, context)
        elif isinstance(template, dict):
            return self.render_dict(template, context)
        elif isinstance(template, list):
            return self.render_list(template, context)
        else:
            return template

    def _get_value(self, var_name: str, context: Dict[str, Any]) -> Any:
        """Get a value from the context, supporting nested access with dot notation."""
        if "." in var_name:
            # Handle nested access like "book.title"
            parts = var_name.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return f"{{{{{var_name}}}}}"  # Return original if not found
            return current
        else:
            # Simple variable access
            return context.get(var_name, f"{{{{{var_name}}}}}")


def load_content_type_config(
    content_type: str, config_dir: Path = None
) -> Dict[str, Any]:
    """Load a specific content type configuration file."""
    if config_dir is None:
        # Default path relative to this file
        config_dir = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )

    config_file = config_dir / f"{content_type}.yaml"

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ValueError(f"Content type configuration file not found: {config_file}")
    except Exception as e:
        raise ValueError(
            f"Error loading content type configuration from {config_file}: {e}"
        )


def load_content_types_with_templates(config_path: Path = None) -> Dict[str, Any]:
    """Load content types configuration with template support (legacy compatibility)."""
    if config_path is None:
        # Default path relative to this file
        config_path = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types.yaml"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load content types from {config_path}: {e}")
        return {}


def get_job_templates(
    content_type: str = "technical_manual_writer", config_path: Path = None
) -> list:
    """Get job templates for a specific content type."""
    if config_path is None:
        config_dir = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )
    else:
        config_dir = config_path.parent / "content_types"

    # First try to load from the specific content type
    try:
        content_type_config = load_content_type_config(content_type, config_dir)
        if "job_templates" in content_type_config:
            return content_type_config["job_templates"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to fallback options

    # For job templates, do NOT fall back to default.yaml
    # Only content types that explicitly define job_templates should have them
    raise ValueError(f"No job templates found for content type '{content_type}'")


def get_job_generation_prompt(
    content_type: str = "technical_manual_writer", config_path: Path = None
) -> str:
    """Get job generation prompt for a specific content type."""
    if config_path is None:
        config_dir = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )
    else:
        config_dir = config_path.parent / "content_types"

    # First try to load from the specific content type
    try:
        content_type_config = load_content_type_config(content_type, config_dir)
        if "job_generation_prompt" in content_type_config:
            return content_type_config["job_generation_prompt"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to fallback options

    # If not found in specific content type, try default.yaml
    try:
        default_config = load_content_type_config("default", config_dir)
        if "job_generation_prompt" in default_config:
            return default_config["job_generation_prompt"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to final fallback

    # Final fallback - raise error as before
    raise ValueError(
        f"No job generation prompt found for content type '{content_type}' or in default.yaml"
    )


def get_job_generation_rag_context(
    content_type: str = "technical_manual_writer", config_path: Path = None
) -> str:
    """Get job generation RAG context template for a specific content type."""
    if config_path is None:
        config_dir = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )
    else:
        config_dir = config_path.parent / "content_types"

    # First try to load from the specific content type
    try:
        content_type_config = load_content_type_config(content_type, config_dir)
        if "job_generation_rag_context" in content_type_config:
            return content_type_config["job_generation_rag_context"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to fallback options

    # If not found in specific content type, try default.yaml
    try:
        default_config = load_content_type_config("default", config_dir)
        if "job_generation_rag_context" in default_config:
            return default_config["job_generation_rag_context"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to final fallback

    # Final fallback - raise error as before
    raise ValueError(
        f"No job generation RAG context found for content type '{content_type}' or in default.yaml"
    )


def get_rag_context_query(
    content_type: str = "technical_manual_writer", config_path: Path = None
) -> str:
    """Get RAG context query template for a specific content type."""
    if config_path is None:
        config_dir = (
            Path(__file__).parent.parent
            / "config"
            / "content"
            / "prompts"
            / "content_types"
        )
    else:
        config_dir = config_path.parent / "content_types"

    # First try to load from the specific content type
    try:
        content_type_config = load_content_type_config(content_type, config_dir)
        if "rag_context_query" in content_type_config:
            return content_type_config["rag_context_query"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to fallback options

    # If not found in specific content type, try default.yaml
    try:
        default_config = load_content_type_config("default", config_dir)
        if "rag_context_query" in default_config:
            return default_config["rag_context_query"]
    except (FileNotFoundError, ValueError):
        pass  # Continue to final fallback

    # Final fallback - raise error as before
    raise ValueError(
        f"No RAG context query template found for content type '{content_type}' or in default.yaml"
    )


def render_job_templates(templates: list, context: Dict[str, Any]) -> list:
    """Render job templates with the given context."""
    engine = TemplateEngine()
    return engine.render_list(templates, context)


# Global template engine instance
_template_engine = TemplateEngine()


def render_template(
    template: Union[str, Dict, list], context: Dict[str, Any]
) -> Union[str, Dict, list]:
    """Convenience function to render templates using the global engine."""
    return _template_engine.render_template(template, context)


def render_string_template(template: str, context: Dict[str, Any]) -> str:
    """Convenience function to render string templates."""
    return _template_engine.render_string(template, context)
