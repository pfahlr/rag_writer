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
    """Parse markdown outline format with proper hierarchical structure."""
    lines = content.strip().split('\n')
    sections = []
    metadata = BookMetadata(title="Converted from Markdown Outline")

    # Track hierarchy levels
    level_stack = []  # [(level, section_id), ...]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract title if it's the first line and doesn't start with #
        if not sections and not line.startswith('#'):
            metadata.title = line
            continue

        # Parse markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            if level == 1:
                # Book title
                metadata.title = title
                continue  # Don't add book title as a section

            # Generate hierarchical ID
            section_id = generate_markdown_hierarchical_id(level, level_stack)

            # Find parent ID
            parent_id = None
            if level > 2 and level_stack:
                # Find the most recent parent at the previous level
                for stack_level, stack_parent in reversed(level_stack):
                    if stack_level == level - 1:
                        parent_id = stack_parent
                        break

            # Update level stack
            # Remove any levels deeper than current
            level_stack = [(l, p) for l, p in level_stack if l < level]
            # Add current level
            level_stack.append((level, section_id))

            sections.append(OutlineSection(
                id=section_id,
                title=title,
                level=level,
                parent_id=parent_id,
                description=f"Level {level}: {title}"
            ))

    return sections, metadata

def generate_markdown_hierarchical_id(level: int, level_stack: List[Tuple[int, str]]) -> str:
    """Generate hierarchical ID for markdown headers."""
    if level == 2:
        # Chapter level - count existing chapters
        chapter_num = len([s for s in level_stack if s[0] == 2]) + 1
        return str(chapter_num)

    elif level == 3:
        # Section level - find parent chapter
        parent_chapter = "1"  # Default
        for stack_level, stack_id in level_stack:
            if stack_level == 2:
                parent_chapter = stack_id
                break

        # Count sections under this chapter by looking at recent sections with same parent
        section_num = 1
        for stack_level, stack_id in reversed(level_stack):
            if stack_level == 3 and stack_id.startswith(parent_chapter):
                # Extract the letter part and increment
                if len(stack_id) > len(parent_chapter):
                    letter_part = stack_id[len(parent_chapter):]
                    if letter_part.isalpha():
                        section_num = ord(letter_part.upper()) - 64 + 1
                        break
            elif stack_level == 2:
                # We've gone past sections of this chapter, start over
                break

        return f"{parent_chapter}{chr(64 + section_num)}"  # 1A, 1B, 2A, etc.

    elif level == 4:
        # Subsection level - find parent section
        parent_section = "1A"  # Default
        for stack_level, stack_id in level_stack:
            if stack_level == 3:
                parent_section = stack_id

        # Count subsections under this section by looking at recent subsections
        subsection_num = 1
        for stack_level, stack_id in reversed(level_stack):
            if stack_level == 4 and stack_id.startswith(parent_section):
                # Extract the number part and increment
                if len(stack_id) > len(parent_section):
                    num_part = stack_id[len(parent_section):]
                    if num_part.isdigit():
                        subsection_num = int(num_part) + 1
                        break
            elif stack_level == 3:
                # We've gone past subsections of this section, start over
                break

        return f"{parent_section}{subsection_num}"  # 1A1, 1A2, 1B1, etc.

    elif level == 5:
        # Sub-subsection level - find parent subsection
        parent_subsection = "1A1"  # Default
        for stack_level, stack_id in level_stack:
            if stack_level == 4:
                parent_subsection = stack_id

        # Count sub-subsections under this subsection
        subsubsection_num = 1
        for stack_level, stack_id in reversed(level_stack):
            if stack_level == 5 and stack_id.startswith(parent_subsection):
                # Extract the letter part and increment
                if len(stack_id) > len(parent_subsection):
                    letter_part = stack_id[len(parent_subsection):]
                    if letter_part.isalpha():
                        subsubsection_num = ord(letter_part.lower()) - 96 + 1
                        break
            elif stack_level == 4:
                # We've gone past sub-subsections of this subsection
                break

        return f"{parent_subsection}{chr(96 + subsubsection_num)}"  # 1A1a, 1A1b, etc.

    elif level == 6:
        # Sub-sub-subsection level - find parent sub-subsection
        parent_subsubsection = "1A1a"  # Default
        for stack_level, stack_id in level_stack:
            if stack_level == 5:
                parent_subsubsection = stack_id

        # Count sub-sub-subsections under this sub-subsection
        subsubsubsection_num = 1
        for stack_level, stack_id in reversed(level_stack):
            if stack_level == 6 and stack_id.startswith(parent_subsubsection):
                # Extract the number part and increment
                if len(stack_id) > len(parent_subsubsection):
                    num_part = stack_id[len(parent_subsubsection):]
                    if num_part.isdigit():
                        subsubsubsection_num = int(num_part) + 1
                        break
            elif stack_level == 5:
                # We've gone past sub-sub-subsections of this sub-subsection
                break

        return f"{parent_subsubsection}{subsubsection_num}"  # 1A1a1, 1A1a2, etc.

    else:
        # Fallback for any other levels
        return f"L{level}_{len(level_stack) + 1}"

def parse_text_outline(content: str) -> Tuple[List[OutlineSection], BookMetadata]:
    """Parse plain text outline format with proper hierarchical structure."""
    lines = content.strip().split('\n')
    sections = []
    metadata = BookMetadata(title="Converted from Text Outline")

    # Track hierarchy levels
    level_stack = []  # [(level, parent_id), ...]

    for line in lines:
        # Only strip trailing whitespace, preserve leading indentation
        line = line.rstrip()
        if not line:
            continue

        # Extract title if it's the first line and doesn't look like an outline item
        if not sections and not re.match(r'^\s*[\d\w]+[\.\)]\s+', line):
            metadata.title = line
            continue

        # Parse outline structure with comprehensive regex
        # Matches: "1. Title", "  1A. Subtitle", "    1A1. Sub-sub", etc.
        outline_match = re.match(r'^(\s*)([\d]+[A-Za-z\d]*|[A-Za-z]+)[\.\)]\s*(.+)$', line)

        if outline_match:
            indent, number, title = outline_match.groups()

            # Calculate level based on indentation pattern
            indent_len = len(indent)
            if indent_len == 0:
                level = 1  # "1. Title" - chapter level
            elif indent_len == 2:
                level = 2  # "  1A. Title" - section level
            elif indent_len == 4:
                level = 3  # "    1A1. Title" - subsection level
            elif indent_len == 6:
                level = 4  # "      1A1a. Title" - sub-subsection level
            else:
                level = min(5, (indent_len // 2) + 1)  # Fallback for deeper levels

            # Generate hierarchical ID
            section_id = generate_hierarchical_id(number, level, level_stack)

            # Find parent ID
            parent_id = None
            if level > 1 and level_stack:
                # Find the most recent parent at the previous level
                for stack_level, stack_parent in reversed(level_stack):
                    if stack_level == level - 1:
                        parent_id = stack_parent
                        break

            # Update level stack
            # Remove any levels deeper than current
            level_stack = [(l, p) for l, p in level_stack if l < level]
            # Add current level
            level_stack.append((level, section_id))

            sections.append(OutlineSection(
                id=section_id,
                title=title,
                level=level,
                parent_id=parent_id,
                description=f"Level {level}: {title}"
            ))

    return sections, metadata

def generate_hierarchical_id(number: str, level: int, level_stack: List[Tuple[int, str]]) -> str:
    """Generate a hierarchical ID based on outline numbering and current stack."""
    # Handle alphanumeric combinations like "1A", "2B", "1A1", "1A2", etc.
    if re.match(r'^\d+[A-Za-z]+\d*$', number):
        # This is already a hierarchical ID like "1A", "2B", "1A1", "1A2"
        return number

    elif number.isdigit():
        # Pure numeric (1, 2, 3, etc.)
        num = int(number)

        if level == 1:
            return str(num)
        elif level == 2:
            # Get parent chapter number
            parent_chapter = "1"  # Default
            for stack_level, stack_id in level_stack:
                if stack_level == 1:
                    parent_chapter = stack_id
                    break
            return f"{parent_chapter}{chr(64 + num)}"  # 1A, 1B, 2A, etc.
        elif level == 3:
            # Get parent section
            parent_section = "1A"  # Default
            for stack_level, stack_id in level_stack:
                if stack_level == 2:
                    parent_section = stack_id
                    break
            return f"{parent_section}{num}"  # 1A1, 1A2, 1B1, etc.
        elif level == 4:
            # Get parent subsection
            parent_subsection = "1A1"  # Default
            for stack_level, stack_id in level_stack:
                if stack_level == 3:
                    parent_subsection = stack_id
                    break
            return f"{parent_subsection}{chr(96 + num)}"  # 1A1a, 1A1b, etc.
        else:
            # For deeper levels, just append the number
            return f"{level_stack[-1][1] if level_stack else '1'}{num}"

    elif number.isalpha():
        # Letter-based numbering (A, B, C, etc.)
        letter = number.upper()

        if level == 1:
            return letter
        elif level == 2:
            parent_chapter = "1"  # Default
            for stack_level, stack_id in level_stack:
                if stack_level == 1:
                    parent_chapter = stack_id
                    break
            return f"{parent_chapter}{letter}"
        else:
            # For deeper levels with letters
            parent_id = level_stack[-1][1] if level_stack else "1"
            return f"{parent_id}{letter.lower()}"

    else:
        # Roman numerals or other formats - treat as generic
        if level == 1:
            return f"1{number}"
        else:
            parent_id = level_stack[-1][1] if level_stack else "1"
            return f"{parent_id}{number}"
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
        # Include level 1+ sections (chapters, sections, and subsections)
        # Level 1 = chapters, Level 2 = sections, Level 3+ = subsections
        if section.level < 1:
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
        1: "Chapters",
        2: "Sections",
        3: "Subsections",
        4: "Sub-subsections",
        5: "Sub-sub-subsections",
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
        if section.level >= 1:  # Generate jobs for chapters, sections and subsections
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