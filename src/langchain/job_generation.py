#!/usr/bin/env python3
"""
LangChain Job Generation Module - Shared job file generation functionality

Provides centralized job generation with LLM and RAG support for all runner scripts.
Handles both simple fallback jobs and advanced LLM-generated prompts.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console

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

def retrieve_rag_context(section_title: str, book_title: str, target_audience: str, rag_key: str) -> str:
    """Retrieve relevant context from RAG system for job generation."""
    console.print(f"[dim]Retrieving RAG context for {section_title}...[/dim]")

    # Create a query for relevant context
    query = f"""
Find relevant information about: {section_title}

Context: This is for creating educational content for a book titled "{book_title}"
for {target_audience}.

Please provide any relevant background information, examples, or context that would be
helpful for writing educational content about this topic.
"""

    # Call lc_ask with RAG retrieval
    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type", "pure_research",
        "--key", rag_key,
        "--task", "You are a research assistant providing relevant context and background information.",
        query
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        response = json.loads(result.stdout.strip())
        if isinstance(response, dict) and 'generated_content' in response:
            return response['generated_content']
        else:
            return str(response)

    except Exception as e:
        console.print(f"[yellow]Warning: RAG context retrieval failed ({e})[/yellow]")
        return ""

def create_fallback_jobs(section_title: str, book_title: str, chapter_title: str, section_title_hierarchy: str, subsection_title: str, subsection_id: str, target_audience: str, topic: str = "") -> List[Dict]:
    """Create fallback jobs when LLM generation fails."""
    return [
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
                "topic": topic
            }
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
                "topic": topic
            }
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
                "topic": topic
            }
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
                "topic": topic
            }
        }
    ]

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
    base_dir: Path = None
) -> Path:
    """Generate a job file for a section using LLM-generated prompts."""
    if base_dir is None:
        base_dir = ROOT

    # Create default job file
    job_file = base_dir / "data_jobs" / f"{section_id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Generating LLM-based job prompts for {section_id}: {section_title}[/blue]")

    # Build the job generation prompt
    job_generation_prompt = f"""
You are an expert educational content strategist. Generate 4 diverse, high-quality writing prompts for the section:

**Book Context:**
- Title: {book_title}
- Target Audience: {target_audience}
- Chapter: {chapter_title}
- Section: {section_title_hierarchy}
- Subsection: {subsection_title} ({section_id})

**Section to Write About:** {section_title}

**Requirements:**
1. Generate exactly 4 different writing prompts that cover different aspects of this section
2. Each prompt should be specific, actionable, and tailored to the educational context
3. Include variety: introduction, detailed explanation, practical application, and summary/conclusion
4. Consider the hierarchical context (chapter > section > subsection)
5. Ensure prompts are appropriate for the target audience
6. Make prompts engaging and educationally valuable

**Output Format:**
Return a JSON array of exactly 4 job objects, each with:
- "task": Brief description of the writer's role and focus
- "instruction": Specific writing instruction for this aspect of the section
- "context": Object with book_title, chapter, section, subsection, subsection_id, target_audience

Example format:
[
  {{
    "task": "You are an educational content writer...",
    "instruction": "Write an engaging introduction...",
    "context": {{
      "book_title": "{book_title}",
      "chapter": "{chapter_title}",
      "section": "{section_title_hierarchy}",
      "subsection": "{subsection_title}",
      "subsection_id": "{section_id}",
      "target_audience": "{target_audience}"
    }}
  }},
  ... (3 more jobs)
]
"""

    # Add RAG context if requested
    rag_context = ""
    if use_rag and rag_key:
        rag_context = retrieve_rag_context(section_title, book_title, target_audience, rag_key)
        if rag_context:
            job_generation_prompt += f"""

**Additional Context from RAG:**
Use the following relevant information from the knowledge base to enhance the prompts:

{rag_context}

Please incorporate this contextual information naturally into the generated prompts where relevant.
"""

    # Call LLM to generate job prompts
    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type", "pure_research",
        "--task", "You are an expert educational content strategist specializing in creating effective writing prompts.",
        job_generation_prompt
    ]

    if rag_key and not use_rag:
        cmd.extend(["--key", rag_key])

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the LLM response
        response = json.loads(result.stdout.strip())
        if isinstance(response, dict) and 'generated_content' in response:
            jobs = json.loads(response['generated_content'])
        elif isinstance(response, list):
            jobs = response
        else:
            # Fallback to parsing as JSON directly
            jobs = json.loads(response['generated_content'] if isinstance(response, dict) else str(response))

        # Validate that we got the expected structure
        if not isinstance(jobs, list) or len(jobs) != 4:
            console.print(f"[yellow]Warning: LLM returned unexpected format, using fallback jobs[/yellow]")
            jobs = create_fallback_jobs(section_title, book_title, chapter_title, section_title_hierarchy, subsection_title, section_id, target_audience, topic)

    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate LLM jobs ({e}), using fallback[/yellow]")
        jobs = create_fallback_jobs(section_title, book_title, chapter_title, section_title_hierarchy, subsection_title, section_id, target_audience, topic)

    # Write jobs to file
    with open(job_file, 'w', encoding='utf-8') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    console.print(f"[green]✓ Generated LLM-based job file: {job_file}[/green]")
    console.print(f"[dim]Context: {chapter_title} > {section_title_hierarchy} > {subsection_title}[/dim]")
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
    base_dir: Path = None
) -> Path:
    """Generate a job file for a section using fallback hard-coded prompts."""
    if base_dir is None:
        base_dir = ROOT

    # Create default job file
    job_file = base_dir / "data_jobs" / f"{section_id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate fallback jobs
    jobs = create_fallback_jobs(section_title, book_title, chapter_title, section_title_hierarchy, subsection_title, section_id, target_audience, topic)

    # Write jobs to file
    with open(job_file, 'w', encoding='utf-8') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    console.print(f"[green]Generated fallback job file: {job_file}[/green]")
    console.print(f"[dim]Context: {chapter_title} > {section_title_hierarchy} > {subsection_title}[/dim]")
    return job_file

# Legacy compatibility functions for outline converter
def generate_job_file(section: OutlineSection, metadata: BookMetadata, sections: List[OutlineSection], use_llm: bool = True, use_rag: bool = False, rag_key: str = None) -> Path:
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
            current_section = next((s for s in sections if s.id == current_section.parent_id), None)
            if current_section:
                current_id = current_section.id
            else:
                break

    hierarchy_context = " > ".join(context_parts)

    # Extract chapter and section from hierarchy
    chapter_title = context_parts[0] if context_parts else "Chapter 1"
    section_title_hierarchy = context_parts[1] if len(context_parts) > 1 else "Section A"

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
            rag_key=rag_key
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
            topic=metadata.topic
        )