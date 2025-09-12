#!/usr/bin/env python3
"""
LangChain Job Generation Module - Shared job file generation functionality

Provides centralized job generation with LLM and RAG support for all runner scripts.
Handles both simple fallback jobs and advanced LLM-generated prompts.

NEW FEATURE: Configurable number of prompts per section (--num-prompts parameter)
- Default: 4 prompts per section
- Range: 1-10 prompts per section
- Supports both LLM-generated and fallback job creation
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from rich.console import Console

# Import our new template engine

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.template_engine import (
    get_job_templates,
    render_job_templates,
    get_job_generation_prompt,
    get_job_generation_rag_context,
    get_rag_context_query,
    render_string_template,
)

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

console = Console()


@dataclass
class SectionConfig:
    """Configuration for a section's processing parameters."""

    subsection_id: str
    title: str
    job_file: Optional[Path] = None
    batch_params: Dict[str, Any] = None
    merge_params: Dict[str, Any] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.batch_params is None:
            self.batch_params = {}
        if self.merge_params is None:
            self.merge_params = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class BookStructure:
    """Represents the hierarchical structure of a book/chapter."""

    title: str
    sections: List[SectionConfig]
    metadata: Dict[str, Any]


@dataclass
class OutlineSection:
    """Represents a section in the outline hierarchy."""

    id: str
    title: str
    level: int
    parent_id: Optional[str] = None
    description: str = ""
    estimated_words: int = 1000


@dataclass
class BookMetadata:
    """Book metadata information."""

    title: str
    topic: str = ""
    target_audience: str = "General readers"
    author_expertise: str = "Intermediate"
    word_count_target: int = 50000
    description: str = ""


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code blocks."""
    import re

    # Look for JSON code blocks
    json_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
    match = re.search(json_block_pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # If no code block found, return original content
    return content


def retrieve_rag_context(
    section_title: str,
    book_title: str,
    target_audience: str,
    rag_key: str,
    content_type: str = "technical_manual_writer",
) -> str:
    """Retrieve relevant context from RAG system for job generation."""
    console.print(f"[dim]Retrieving RAG context for {section_title}...[/dim]")

    # Get the RAG context query template from the content type configuration
    try:
        query_template = get_rag_context_query(
            content_type,
            ROOT / "src" / "config" / "content" / "prompts" / "content_types.yaml",
        )
    except ValueError as e:
        console.print(
            f"[yellow]Warning: Could not load RAG context query template ({e}), using fallback[/yellow]"
        )
        query_template = """
Find relevant information about: {{section_title}}

Context: This is for creating educational content for a book titled "{{book_title}}"
for {{target_audience}}.

Please provide any relevant background information, examples, or context that would be
helpful for writing educational content about this topic.
"""

    # Render the query template with context variables
    context = {
        "section_title": section_title,
        "book_title": book_title,
        "target_audience": target_audience,
    }
    query = render_string_template(query_template, context)

    # Call lc_ask with RAG retrieval
    cmd = [
        sys.executable,
        str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type",
        "pure_research",
        "--key",
        rag_key,
        "--task",
        "You are a research assistant providing relevant context and background information.",
        query,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True
        )

        # Parse the response with better error handling
        stdout_content = result.stdout.strip()
        if not stdout_content:
            console.print(
                "[yellow]Warning: Empty response from RAG subprocess[/yellow]"
            )
            return ""

        try:
            response = json.loads(stdout_content)
        except json.JSONDecodeError:
            # Try to clean up escaped characters
            import re

            cleaned_content = re.sub(r"\\{2,}", r"\\", stdout_content)
            try:
                response = json.loads(cleaned_content)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                json_match = re.search(r"\{.*\}", stdout_content, re.DOTALL)
                if json_match:
                    try:
                        response = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        console.print(
                            f"[yellow]Warning: Could not parse RAG response as JSON: {stdout_content[:200]}...[/yellow]"
                        )
                        return ""
                else:
                    console.print(
                        f"[yellow]Warning: No JSON found in RAG response: {stdout_content[:200]}...[/yellow]"
                    )
                    return ""

        if isinstance(response, dict) and "generated_content" in response:
            return response["generated_content"]
        else:
            return str(response)

    except Exception as e:
        console.print(f"[yellow]Warning: RAG context retrieval failed ({e})[/yellow]")
        return ""


def create_fallback_jobs(
    section_title: str,
    book_title: str,
    chapter_title: str,
    section_title_hierarchy: str,
    subsection_title: str,
    subsection_id: str,
    target_audience: str,
    topic: str = "",
    num_prompts: int = 4,
    content_type: str = "technical_manual_writer",
) -> List[Dict]:
    """Create fallback jobs when LLM generation fails using YAML templates."""
    try:
        # Load job templates from YAML configuration
        job_templates = get_job_templates(
            content_type,
            ROOT / "src" / "config" / "content" / "prompts" / "content_types.yaml",
        )

        # Create context for template rendering
        context = {
            "section_title": section_title,
            "book_title": book_title,
            "chapter_title": chapter_title,
            "section_title_hierarchy": section_title_hierarchy,
            "subsection_title": subsection_title,
            "subsection_id": subsection_id,
            "target_audience": target_audience,
            "topic": topic,
        }

        # Render templates with context
        rendered_templates = render_job_templates(job_templates, context)

        # Return the requested number of jobs, cycling through templates if needed
        jobs = []
        for i in range(num_prompts):
            template_index = i % len(rendered_templates)
            job = rendered_templates[template_index].copy()
            # Add a unique identifier to avoid identical jobs
            job["job_index"] = i + 1
            jobs.append(job)

        return jobs

    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to load YAML templates ({e}), using legacy hardcoded templates[/yellow]"
        )
        return _create_legacy_fallback_jobs(
            section_title,
            book_title,
            chapter_title,
            section_title_hierarchy,
            subsection_title,
            subsection_id,
            target_audience,
            topic,
            num_prompts,
        )


def _create_legacy_fallback_jobs(
    section_title: str,
    book_title: str,
    chapter_title: str,
    section_title_hierarchy: str,
    subsection_title: str,
    subsection_id: str,
    target_audience: str,
    topic: str = "",
    num_prompts: int = 4,
) -> List[Dict]:
    """Legacy fallback job creation for backward compatibility."""
    # Define the base job templates (same as before)
    job_templates = [
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Write an engaging introduction to {section_title.lower()} within the context of {chapter_title} > {section_title_hierarchy}. Hook the reader and establish the importance of this subsection.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Provide detailed explanations and examples for {section_title.lower()} as part of {chapter_title} > {section_title_hierarchy} > {subsection_title}. Include step-by-step processes where applicable.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Create practical exercises, case studies, or activities related to {section_title.lower()} within {chapter_title} > {section_title_hierarchy}. Ensure activities are immediately applicable.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Write a comprehensive summary of {section_title.lower()} that reinforces key concepts from {chapter_title} > {section_title_hierarchy} and provides actionable takeaways.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Analyze key concepts and principles related to {section_title.lower()} within {chapter_title} > {section_title_hierarchy}. Provide clear definitions and explain fundamental ideas.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Discuss real-world applications and use cases for {section_title.lower()} in the context of {chapter_title} > {section_title_hierarchy}. Include industry examples and practical scenarios.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Address common challenges and pitfalls related to {section_title.lower()} within {chapter_title} > {section_title_hierarchy}. Provide troubleshooting guidance and best practices.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
        {
            "task": f"You are a content writer creating educational material for '{book_title}'. Focus on practical applications for {target_audience}.",
            "instruction": f"Explore advanced topics and future developments related to {section_title.lower()} in the context of {chapter_title} > {section_title_hierarchy}. Discuss emerging trends and innovations.",
            "context": {
                "book_title": book_title,
                "chapter": chapter_title,
                "section": section_title_hierarchy,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": target_audience,
                "topic": topic,
            },
        },
    ]

    # Return the requested number of jobs, cycling through templates if needed
    jobs = []
    for i in range(num_prompts):
        template_index = i % len(job_templates)
        job = job_templates[template_index].copy()
        # Add a unique identifier to avoid identical jobs
        job["job_index"] = i + 1
        jobs.append(job)

    return jobs


def generate_llm_job_file(
    section_id: str,
    section_title: str,
    book_title: str,
    chapter_title: str,
    section_title_hierarchy: str,
    subsection_title: str,
    target_audience: str,
    topic: str = "",
    use_rag: bool = False,
    rag_key: str = None,
    base_dir: Path = None,
    num_prompts: int = 4,
    content_type: str = "technical_manual_writer",
) -> Path:
    """Generate a job file for a section using LLM-generated prompts."""
    if base_dir is None:
        base_dir = ROOT

    # Create default job file
    job_file = base_dir / "data_jobs" / f"{section_id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[blue]Generating LLM-based job prompts for {section_id}: {section_title}[/blue]"
    )

    # Get the job generation prompt template from YAML
    try:
        job_generation_prompt_template = get_job_generation_prompt(
            content_type,
            ROOT / "src" / "config" / "content" / "prompts" / "content_types.yaml",
        )
    except ValueError as e:
        console.print(
            f"[yellow]Warning: Could not load job generation prompt template ({e}), using fallback[/yellow]"
        )
        job_generation_prompt_template = """
You are an expert content strategist. Generate {{num_prompts}} diverse, high-quality writing prompts for the section:

**Book Context:**
- Title: {{book_title}}
- Target Audience: {{target_audience}}
- Chapter: {{chapter_title}}
- Section: {{section_title_hierarchy}}
- Subsection: {{subsection_title}} ({{subsection_id}})

**Section to Write About:** {{section_title}}

**Requirements:**
1. Generate exactly {{num_prompts}} different writing prompts that cover different aspects of this section
2. Each prompt should be specific, actionable, and tailored to the context
3. Include variety: introduction, detailed explanation, practical application, and summary/conclusion
4. Consider the hierarchical context (chapter > section > subsection)
5. Ensure prompts are appropriate for the target audience
6. Make prompts engaging and valuable

**Output Format:**
Return a JSON array of exactly {{num_prompts}} job objects, each with:
- "task": Brief description of the writer's role and focus
- "instruction": Specific writing instruction for this aspect of the section
- "context": Object with book_title, chapter, section, subsection, subsection_id, target_audience

Example format:
[
  {
    "task": "You are a content writer...",
    "instruction": "Write an engaging introduction...",
    "context": {
      "book_title": "{{book_title}}",
      "chapter": "{{chapter_title}}",
      "section": "{{section_title_hierarchy}}",
      "subsection": "{{subsection_title}}",
      "subsection_id": "{{subsection_id}}",
      "target_audience": "{{target_audience}}"
    }
  }
]
"""

    # Prepare context for template rendering
    context = {
        "num_prompts": num_prompts,
        "book_title": book_title,
        "target_audience": target_audience,
        "chapter_title": chapter_title,
        "section_title_hierarchy": section_title_hierarchy,
        "subsection_title": subsection_title,
        "subsection_id": section_id,
        "section_title": section_title,
    }

    # Render the job generation prompt
    job_generation_prompt = render_string_template(
        job_generation_prompt_template, context
    )

    # Add RAG context if requested
    if use_rag and rag_key:
        rag_context_content = retrieve_rag_context(
            section_title, book_title, target_audience, rag_key, content_type
        )
        if rag_context_content:
            try:
                rag_context_template = get_job_generation_rag_context(
                    content_type,
                    ROOT
                    / "src"
                    / "config"
                    / "content"
                    / "prompts"
                    / "content_types.yaml",
                )
                rag_context_rendered = render_string_template(
                    rag_context_template, {"rag_context": rag_context_content}
                )
                job_generation_prompt += rag_context_rendered
            except ValueError as e:
                console.print(
                    f"[yellow]Warning: Could not load RAG context template ({e}), using simple format[/yellow]"
                )
                job_generation_prompt += f"""

**Additional Context from RAG:**
Use the following relevant information from the knowledge base to enhance the prompts:

{rag_context_content}

Please incorporate this contextual information naturally into the generated prompts where relevant.
"""

    # Call LLM to generate job prompts
    cmd = [
        sys.executable,
        str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type",
        content_type,
        "--task",
        "You are an expert educational content strategist specializing in creating effective writing prompts.",
        job_generation_prompt,
    ]

    if rag_key and not use_rag:
        cmd.extend(["--key", rag_key])

    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True
        )
        # Parse the LLM response with better error handling
        stdout_content = result.stdout.strip()
        if not stdout_content:
            raise ValueError("Empty response from subprocess")

        # First, try to extract JSON from markdown code blocks
        extracted_json = extract_json_from_markdown(stdout_content)

        try:
            response = json.loads(extracted_json)

        except json.JSONDecodeError:
            # Try to clean up escaped characters that might be in the response
            import re

            # Remove excessive backslashes that might be escaping newlines and quotes
            cleaned_content = re.sub(r"\\{2,}", r"\\", extracted_json)
            try:
                response = json.loads(cleaned_content)

            except json.JSONDecodeError:
                # If still failing, try to extract JSON from the text
                json_match = re.search(r"\{.*\}", extracted_json, re.DOTALL)
                if json_match:
                    try:
                        response = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"Could not parse JSON response: {extracted_json[:200]}..."
                        )
                else:
                    raise ValueError(
                        f"No JSON found in response: {extracted_json[:200]}..."
                    )

        # Handle the parsed response
        if isinstance(response, dict) and "generated_content" in response:
            generated_content = response["generated_content"]
            # If generated_content is a string, try to parse it as JSON
            if isinstance(generated_content, str):
                # First, try to extract JSON from markdown code blocks
                extracted_content = extract_json_from_markdown(generated_content)

                try:
                    # First try direct parsing
                    jobs = json.loads(extracted_content)
                except json.JSONDecodeError:
                    # Try cleaning up escaped characters in the generated content
                    cleaned_generated = re.sub(r"\\{2,}", r"\\", extracted_content)
                    try:
                        jobs = json.loads(cleaned_generated)
                    except json.JSONDecodeError:
                        # Try to extract JSON array from the string
                        json_array_match = re.search(
                            r"\[.*\]", extracted_content, re.DOTALL
                        )
                        if json_array_match:
                            try:
                                jobs = json.loads(json_array_match.group(0))
                            except json.JSONDecodeError:
                                raise ValueError(
                                    f"Could not parse generated_content as JSON: {extracted_content[:200]}..."
                                )
                        else:
                            raise ValueError(
                                f"No JSON array found in generated_content: {extracted_content[:200]}..."
                            )
            elif isinstance(generated_content, list):
                jobs = generated_content
            else:
                raise ValueError(
                    f"Unexpected type for generated_content: {type(generated_content)}"
                )
        elif isinstance(response, list):
            jobs = response
        else:
            # Fallback to parsing as JSON directly
            jobs = json.loads(str(response))

        # Validate that we got the expected structure
        if not isinstance(jobs, list) or len(jobs) != num_prompts:
            console.print(
                f"[yellow]Warning: LLM returned {len(jobs) if isinstance(jobs, list) else 'invalid'} jobs, expected {num_prompts}, using fallback jobs[/yellow]"
            )
            jobs = create_fallback_jobs(
                section_title,
                book_title,
                chapter_title,
                section_title_hierarchy,
                subsection_title,
                section_id,
                target_audience,
                topic,
                num_prompts,
                content_type,
            )

    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to generate LLM jobs ({e}), using fallback[/yellow]"
        )
        jobs = create_fallback_jobs(
            section_title,
            book_title,
            chapter_title,
            section_title_hierarchy,
            subsection_title,
            section_id,
            target_audience,
            topic,
            num_prompts,
        )

    # Write jobs to file
    with open(job_file, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    console.print(f"[green]✓ Generated LLM-based job file: {job_file}[/green]")
    console.print(
        f"[dim]Context: {chapter_title} > {section_title_hierarchy} > {subsection_title}[/dim]"
    )
    console.print(f"[dim]Jobs created: {len(jobs)}[/dim]")
    return job_file


def generate_fallback_job_file(
    section_id: str,
    section_title: str,
    book_title: str,
    chapter_title: str,
    section_title_hierarchy: str,
    subsection_title: str,
    target_audience: str,
    topic: str = "",
    base_dir: Path = None,
    num_prompts: int = 4,
    content_type: str = "technical_manual_writer",
) -> Path:
    """Generate a job file for a section using fallback hard-coded prompts."""
    if base_dir is None:
        base_dir = ROOT

    # Create default job file
    job_file = base_dir / "data_jobs" / f"{section_id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate fallback jobs
    jobs = create_fallback_jobs(
        section_title,
        book_title,
        chapter_title,
        section_title_hierarchy,
        subsection_title,
        section_id,
        target_audience,
        topic,
        num_prompts,
        content_type,
    )

    # Write jobs to file
    with open(job_file, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    console.print(f"[green]Generated fallback job file: {job_file}[/green]")
    console.print(
        f"[dim]Context: {chapter_title} > {section_title_hierarchy} > {subsection_title}[/dim]"
    )
    return job_file


# Legacy compatibility functions for outline converter
def generate_job_file(
    section: OutlineSection,
    metadata: BookMetadata,
    sections: List[OutlineSection],
    use_llm: bool = True,
    use_rag: bool = False,
    rag_key: str = None,
    num_prompts: int = 4,
    content_type: str = "technical_manual_writer",
) -> Path:
    """Legacy function for outline converter compatibility."""
    # Build hierarchical context
    context_parts = []
    current_id = section.id

    # Build context by walking up the hierarchy
    for _ in range(section.level):
        current_section = next((s for s in sections if s.id == current_id), None)
        if current_section:
            if current_section.level == 2:
                context_parts.insert(0, f"Chapter {current_section.id}")
            elif current_section.level == 3:
                context_parts.insert(0, f"Section {current_section.id[1]}")
            elif current_section.level == 4:
                context_parts.insert(0, f"Subsection {current_section.id[2]}")
            elif current_section.level >= 5:
                context_parts.insert(0, f"Sub-subsection {current_section.id[3:]}")

            # Move to parent
            current_section = next(
                (s for s in sections if s.id == current_section.parent_id), None
            )
            if current_section:
                current_id = current_section.id
            else:
                break

    hierarchy_context = " > ".join(context_parts)

    # Extract chapter and section from hierarchy
    chapter_title = context_parts[0] if context_parts else "Chapter 1"
    section_title_hierarchy = (
        context_parts[1] if len(context_parts) > 1 else "Section A"
    )

    if use_llm:
        return generate_llm_job_file(
            section_id=section.id,
            section_title=section.title,
            book_title=metadata.title,
            chapter_title=chapter_title,
            section_title_hierarchy=section_title_hierarchy,
            subsection_title=section.title,
            target_audience=metadata.target_audience,
            topic=metadata.topic,
            use_rag=use_rag,
            rag_key=rag_key,
            num_prompts=num_prompts,
            content_type=content_type,
        )
    else:
        return generate_fallback_job_file(
            section_id=section.id,
            section_title=section.title,
            book_title=metadata.title,
            chapter_title=chapter_title,
            section_title_hierarchy=section_title_hierarchy,
            subsection_title=section.title,
            target_audience=metadata.target_audience,
            topic=metadata.topic,
            num_prompts=num_prompts,
            content_type=content_type,
        )
