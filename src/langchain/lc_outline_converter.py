#!/usr/bin/env python3
"""
LangChain Outline Converter - Convert outlines to book structure and job files

Converts various outline formats into:
1. book_structure.json compatible with lc_book_runner.py
2. Individual job files for each subsection with hierarchical context

Supports multiple input formats:
- JSON outlines (from lc_outline_generator.py)
- Markdown outlines
- Text outlines
- Custom structured formats
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

console = Console()

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

def parse_markdown_outline(content: str) -> Tuple[List[OutlineSection], BookMetadata]:
    """Parse markdown outline format."""
    lines = content.strip().split('\n')
    sections = []
    metadata = BookMetadata(title="Converted from Markdown Outline")

    current_chapter = None
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract title if it's the first line
        if not sections and not line.startswith('#'):
            metadata.title = line
            continue

        # Parse markdown headers
        header_match = re.match(r'^(#{1,5})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            if level == 1:
                # Book title
                metadata.title = title
            elif level == 2:
                # Chapter
                current_chapter = len(sections) + 1
                section_id = f"{current_chapter}"
                sections.append(OutlineSection(
                    id=section_id,
                    title=title,
                    level=level,
                    description=f"Chapter {current_chapter}: {title}"
                ))
            elif level == 3:
                # Section
                if current_chapter:
                    current_section = len([s for s in sections if s.level == 3]) + 1
                    section_id = f"{current_chapter}{chr(64 + current_section)}"
                    sections.append(OutlineSection(
                        id=section_id,
                        title=title,
                        level=level,
                        parent_id=str(current_chapter),
                        description=f"Section {chr(64 + current_section)}: {title}"
                    ))
            elif level == 4:
                # Subsection
                if current_chapter and current_section:
                    subsection_num = len([s for s in sections if s.level == 4 and s.parent_id == f"{current_chapter}{chr(64 + current_section)}"]) + 1
                    section_id = f"{current_chapter}{chr(64 + current_section)}{subsection_num}"
                    sections.append(OutlineSection(
                        id=section_id,
                        title=title,
                        level=level,
                        parent_id=f"{current_chapter}{chr(64 + current_section)}",
                        description=f"Subsection {subsection_num}: {title}"
                    ))

    return sections, metadata

def parse_text_outline(content: str) -> Tuple[List[OutlineSection], BookMetadata]:
    """Parse plain text outline format."""
    lines = content.strip().split('\n')
    sections = []
    metadata = BookMetadata(title="Converted from Text Outline")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract title if it's the first line
        if not sections and not any(char in line for char in ['1.', '2.', '3.', 'A.', 'B.', 'C.']):
            metadata.title = line
            continue

        # Parse numbered/bulleted structure
        indent_match = re.match(r'^(\s*)([\d\w]+)[\.\)]\s+(.+)$', line)
        if indent_match:
            indent, number, title = indent_match.groups()
            level = len(indent) // 2 + 1  # 2 spaces per level

            # Convert number to hierarchical ID
            if number.isdigit():
                if level == 1:
                    section_id = number
                elif level == 2:
                    section_id = f"{number[0]}{chr(64 + int(number))}" if len(number) == 1 else number
                elif level == 3:
                    section_id = f"{number[0]}{chr(64 + int(number[1:]))}{number[2:]}" if len(number) > 1 else number
                else:
                    section_id = number
            else:
                # Letter-based numbering
                section_id = number.upper()

            sections.append(OutlineSection(
                id=section_id,
                title=title,
                level=level,
                description=f"Level {level}: {title}"
            ))

    return sections, metadata

def parse_json_outline(content: str) -> Tuple[List[OutlineSection], BookMetadata]:
    """Parse JSON outline format (from lc_outline_generator.py)."""
    try:
        data = json.loads(content)

        # Extract metadata
        metadata = BookMetadata(
            title=data.get('title', 'Converted from JSON Outline'),
            topic=data.get('topic', ''),
            target_audience=data.get('target_audience', 'General readers'),
            description=data.get('description', ''),
            word_count_target=data.get('word_count_target', 50000)
        )

        # Extract sections
        sections = []
        chapters = data.get('chapters', [])

        for chapter in chapters:
            chapter_num = chapter['number']

            # Add chapter as section
            sections.append(OutlineSection(
                id=str(chapter_num),
                title=chapter['title'],
                level=2,
                description=chapter.get('description', ''),
                estimated_words=chapter.get('estimated_words', 5000)
            ))

            for section in chapter.get('sections', []):
                section_letter = section['letter']
                section_id = f"{chapter_num}{section_letter}"

                # Add section
                sections.append(OutlineSection(
                    id=section_id,
                    title=section['title'],
                    level=3,
                    parent_id=str(chapter_num),
                    description=section.get('description', ''),
                    estimated_words=section.get('estimated_words', 2000)
                ))

                for subsection in section.get('subsections', []):
                    subsection_num = subsection['number']
                    subsection_id = f"{section_id}{subsection_num}"

                    # Add subsection
                    sections.append(OutlineSection(
                        id=subsection_id,
                        title=subsection['title'],
                        level=4,
                        parent_id=section_id,
                        description=subsection.get('description', ''),
                        estimated_words=subsection.get('estimated_words', 1000)
                    ))

                    # Add sub-subsections if they exist
                    for subsubsection in subsection.get('subsubsections', []):
                        subsubsection_letter = subsubsection['letter']
                        subsubsection_id = f"{subsection_id}{subsubsection_letter}"

                        sections.append(OutlineSection(
                            id=subsubsection_id,
                            title=subsubsection['title'],
                            level=5,
                            parent_id=subsection_id,
                            description=subsubsection.get('description', ''),
                            estimated_words=subsubsection.get('estimated_words', 500)
                        ))

                        # Add sub-sub-subsections if they exist
                        for subsubsubsection in subsubsection.get('subsubsubsections', []):
                            subsubsubsection_num = subsubsubsection['number']
                            subsubsubsection_id = f"{subsubsection_id}{subsubsubsection_num}"

                            sections.append(OutlineSection(
                                id=subsubsubsection_id,
                                title=subsubsubsection['title'],
                                level=6,
                                parent_id=subsubsection_id,
                                description=subsubsubsection.get('description', ''),
                                estimated_words=subsubsubsection.get('estimated_words', 250)
                            ))

        return sections, metadata

    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON outline: {e}[/red]")
        return [], BookMetadata(title="Error parsing outline")

def detect_outline_format(content: str) -> str:
    """Detect the format of the outline."""
    content = content.strip()

    # Check for JSON
    if content.startswith('{') or content.startswith('['):
        try:
            json.loads(content)
            return 'json'
        except json.JSONDecodeError:
            pass

    # Check for markdown headers
    if re.search(r'^#{1,5}\s+', content, re.MULTILINE):
        return 'markdown'

    # Default to text
    return 'text'

def load_outline_file(file_path: Path) -> Tuple[List[OutlineSection], BookMetadata]:
    """Load and parse outline file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}[/red]")
        sys.exit(1)

    format_type = detect_outline_format(content)
    console.print(f"[dim]Detected format: {format_type}[/dim]")

    if format_type == 'json':
        return parse_json_outline(content)
    elif format_type == 'markdown':
        return parse_markdown_outline(content)
    else:
        return parse_text_outline(content)

def generate_book_structure(sections: List[OutlineSection], metadata: BookMetadata) -> Dict[str, Any]:
    """Generate book structure JSON compatible with lc_book_runner.py."""
    console.print("[blue]Generating book structure...[/blue]")

    # Convert sections to book structure format
    book_sections = []

    for section in sections:
        # Skip top-level sections (chapters) as they're not processed individually
        if section.level <= 2:
            continue

        # Determine topic for batch/merge parameters
        topic_key = metadata.topic.lower().replace(' ', '_') if metadata.topic else 'general'

        section_entry = {
            "subsection_id": section.id,
            "title": section.title,
            "job_file": f"data_jobs/{section.id}.jsonl",
            "batch_params": {
                "key": topic_key,
                "k": 5
            },
            "merge_params": {
                "key": topic_key,
                "k": 3
            },
            "dependencies": []
        }

        # Add dependencies based on parent relationships
        if section.parent_id:
            # Find parent section
            parent = next((s for s in sections if s.id == section.parent_id), None)
            if parent and parent.level >= 3:
                section_entry["dependencies"].append(parent.id)

        book_sections.append(section_entry)

    # Create the complete book structure
    book_structure = {
        "title": metadata.title,
        "metadata": {
            "author": "AI Content Generator",
            "version": "1.0",
            "target_audience": metadata.target_audience,
            "word_count_target": metadata.word_count_target,
            "created_date": json.dumps(None),  # Will be set when saved
            "description": metadata.description,
            "topic": metadata.topic,
            "author_expertise": metadata.author_expertise,
            "source": "outline_converter"
        },
        "sections": book_sections
    }

    return book_structure

def generate_job_file(section: OutlineSection, metadata: BookMetadata, sections: List[OutlineSection]) -> Path:
    """Generate a job file for a section with hierarchical context."""
    job_file = ROOT / "data_jobs" / f"{section.id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

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

    # Generate contextual jobs
    jobs = [
        {
            "task": f"You are a content writer creating educational material for '{metadata.title}'. Focus on practical applications for {metadata.target_audience}.",
            "instruction": f"Write an engaging introduction to {section.title.lower()} within the context of {hierarchy_context}. Hook the reader and establish the importance of this subsection.",
            "context": {
                "book_title": metadata.title,
                "hierarchy": hierarchy_context,
                "subsection_id": section.id,
                "target_audience": metadata.target_audience,
                "topic": metadata.topic
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{metadata.title}'. Focus on practical applications for {metadata.target_audience}.",
            "instruction": f"Provide detailed explanations and examples for {section.title.lower()} as part of {hierarchy_context}. Include step-by-step processes where applicable.",
            "context": {
                "book_title": metadata.title,
                "hierarchy": hierarchy_context,
                "subsection_id": section.id,
                "target_audience": metadata.target_audience,
                "topic": metadata.topic
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{metadata.title}'. Focus on practical applications for {metadata.target_audience}.",
            "instruction": f"Create practical exercises, case studies, or activities related to {section.title.lower()} within {hierarchy_context}. Ensure activities are immediately applicable.",
            "context": {
                "book_title": metadata.title,
                "hierarchy": hierarchy_context,
                "subsection_id": section.id,
                "target_audience": metadata.target_audience,
                "topic": metadata.topic
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{metadata.title}'. Focus on practical applications for {metadata.target_audience}.",
            "instruction": f"Write a comprehensive summary of {section.title.lower()} that reinforces key concepts from {hierarchy_context} and provides actionable takeaways.",
            "context": {
                "book_title": metadata.title,
                "hierarchy": hierarchy_context,
                "subsection_id": section.id,
                "target_audience": metadata.target_audience,
                "topic": metadata.topic
            }
        }
    ]

    # Write jobs to file
    with open(job_file, 'w', encoding='utf-8') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    console.print(f"[green]Generated job file: {job_file}[/green]")
    console.print(f"[dim]Context: {hierarchy_context}[/dim]")

    return job_file

def display_conversion_summary(sections: List[OutlineSection], metadata: BookMetadata, book_structure: Dict[str, Any]):
    """Display summary of the conversion."""
    console.print("\n[bold green]📋 Conversion Summary[/bold green]")

    # Book info
    info_table = Table()
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Title", metadata.title)
    info_table.add_row("Topic", metadata.topic or "Not specified")
    info_table.add_row("Target Audience", metadata.target_audience)
    info_table.add_row("Word Count Target", f"{metadata.word_count_target:,}")

    console.print(info_table)

    # Section breakdown
    console.print("\n[bold]Section Breakdown:[/bold]")

    level_counts = {}
    for section in sections:
        level_counts[section.level] = level_counts.get(section.level, 0) + 1

    breakdown_table = Table()
    breakdown_table.add_column("Level", style="cyan")
    breakdown_table.add_column("Count", style="magenta")
    breakdown_table.add_column("Description", style="white")

    level_names = {
        1: "Book Title",
        2: "Chapters",
        3: "Sections",
        4: "Subsections",
        5: "Sub-subsections",
        6: "Sub-sub-subsections"
    }

    for level in sorted(level_counts.keys()):
        breakdown_table.add_row(
            str(level),
            str(level_counts[level]),
            level_names.get(level, f"Level {level}")
        )

    console.print(breakdown_table)

    # Generated files
    console.print(f"\n[dim]Generated {len(book_structure['sections'])} book sections[/dim]")
    console.print(f"[dim]Generated {len(book_structure['sections'])} job files[/dim]")

def main():
    """Main conversion workflow."""
    import argparse
    ap = argparse.ArgumentParser(description="Convert outlines to book structure and job files")
    ap.add_argument("--outline", required=True, help="Input outline file (JSON, Markdown, or Text)")
    ap.add_argument("--output", help="Output book structure JSON file")
    ap.add_argument("--title", help="Override book title")
    ap.add_argument("--topic", help="Override book topic")
    ap.add_argument("--audience", help="Override target audience")
    ap.add_argument("--wordcount", type=int, help="Override word count target")
    args = ap.parse_args()

    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LangChain Outline Converter[/bold cyan]\n"
        "[dim]Convert outlines to book structure and job files[/dim]",
        border_style="cyan"
    ))

    # Load and parse outline
    outline_file = Path(args.outline)
    if not outline_file.exists():
        console.print(f"[red]Outline file not found: {outline_file}[/red]")
        sys.exit(1)

    console.print(f"[dim]Loading outline from {outline_file}[/dim]")
    sections, metadata = load_outline_file(outline_file)

    if not sections:
        console.print("[red]No sections found in outline file.[/red]")
        sys.exit(1)

    # Apply overrides
    if args.title:
        metadata.title = args.title
    if args.topic:
        metadata.topic = args.topic
    if args.audience:
        metadata.target_audience = args.audience
    if args.wordcount:
        metadata.word_count_target = args.wordcount

    # Generate book structure
    book_structure = generate_book_structure(sections, metadata)

    # Generate job files for each section
    console.print("[blue]Generating job files...[/blue]")
    generated_jobs = 0
    for section in sections:
        if section.level >= 3:  # Only generate jobs for subsections and below
            generate_job_file(section, metadata, sections)
            generated_jobs += 1

    # Save book structure
    output_file = Path(args.output) if args.output else ROOT / "outlines" / "converted_structures" / f"{metadata.title.replace(' ', '_')}_structure.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(book_structure, f, indent=2, ensure_ascii=False)

    # Display summary
    display_conversion_summary(sections, metadata, book_structure)

    # Success message
    console.print()
    console.print(Panel(
        f"[green]🎉 Conversion completed![/green]\n"
        f"[dim]Book structure: {output_file}[/dim]\n"
        f"[dim]Job files: {generated_jobs}[/dim]\n"
        f"[dim]Ready for: python src/langchain/lc_book_runner.py --book {output_file}[/dim]",
        title="[bold green]Success[/bold green]",
        border_style="green"
    ))

if __name__ == "__main__":
    main()