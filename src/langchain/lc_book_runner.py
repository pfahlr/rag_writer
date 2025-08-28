#!/usr/bin/env python3
"""
LangChain Book Runner - High-level orchestration for entire books/chapters

Orchestrates the complete content generation pipeline for books and chapters:
1. Parses hierarchical book structure (4 levels deep)
2. Generates job files for each sub-sub-section
3. Runs batch processing for content generation
4. Runs merge processing for content refinement
5. Aggregates all results into final markdown document

Supports both embedded job definitions and external JSON file references.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.prompt import Prompt, Confirm

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

def load_book_structure(book_file: Path) -> BookStructure:
    """Load book structure from JSON file."""
    console.print(f"[dim]Loading book structure from {book_file}[/dim]")

    try:
        with open(book_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse metadata
        metadata = data.get('metadata', {})

        # Parse sections
        sections = []
        for section_data in data.get('sections', []):
            section = SectionConfig(
                subsection_id=section_data['subsection_id'],
                title=section_data['title'],
                job_file=Path(section_data['job_file']) if section_data.get('job_file') else None,
                batch_params=section_data.get('batch_params', {}),
                merge_params=section_data.get('merge_params', {}),
                dependencies=section_data.get('dependencies', [])
            )
            sections.append(section)

        return BookStructure(
            title=data.get('title', 'Untitled Book'),
            sections=sections,
            metadata=metadata
        )

    except Exception as e:
        console.print(f"[red]Error loading book structure: {e}[/red]")
        sys.exit(1)

def generate_job_file(section: SectionConfig, book_structure: BookStructure, base_dir: Path) -> Path:
    """Generate a job file for a section if it doesn't exist."""
    if section.job_file and section.job_file.exists():
        return section.job_file

    # Create default job file
    job_file = base_dir / "data_jobs" / f"{section.subsection_id}.jsonl"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    # Parse hierarchical context from subsection_id
    # Format: Chapter + Section + Subsection + Sub-subsection (e.g., "1A1", "2B3a")
    subsection_id = section.subsection_id

    # Extract hierarchical levels
    chapter_num = subsection_id[0] if subsection_id[0].isdigit() else "1"
    section_letter = subsection_id[1] if len(subsection_id) > 1 and subsection_id[1].isalpha() else "A"
    subsection_num = subsection_id[2] if len(subsection_id) > 2 and subsection_id[2].isdigit() else "1"
    subsubsection = subsection_id[3:] if len(subsection_id) > 3 else ""

    # Build hierarchical context
    chapter_title = f"Chapter {chapter_num}"
    section_title = f"Section {section_letter}"
    subsection_title = f"Subsection {subsection_num}"
    if subsubsection:
        subsection_title += f"{subsubsection}"

    # Generate contextualized jobs
    jobs = [
        {
            "task": f"You are a content writer creating educational material for '{book_structure.title}'. Focus on practical applications for {book_structure.metadata.get('target_audience', 'professionals')}.",
            "instruction": f"Write an engaging introduction to {section.title.lower()} within the context of {chapter_title} > {section_title}. Hook the reader and establish the importance of this subsection.",
            "context": {
                "book_title": book_structure.title,
                "chapter": chapter_title,
                "section": section_title,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": book_structure.metadata.get('target_audience', 'general audience')
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{book_structure.title}'. Focus on practical applications for {book_structure.metadata.get('target_audience', 'professionals')}.",
            "instruction": f"Provide detailed explanations and examples for {section.title.lower()} as part of {chapter_title} > {section_title} > {subsection_title}. Include step-by-step processes where applicable.",
            "context": {
                "book_title": book_structure.title,
                "chapter": chapter_title,
                "section": section_title,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": book_structure.metadata.get('target_audience', 'general audience')
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{book_structure.title}'. Focus on practical applications for {book_structure.metadata.get('target_audience', 'professionals')}.",
            "instruction": f"Create practical exercises, case studies, or activities related to {section.title.lower()} within {chapter_title} > {section_title}. Ensure activities are immediately applicable.",
            "context": {
                "book_title": book_structure.title,
                "chapter": chapter_title,
                "section": section_title,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": book_structure.metadata.get('target_audience', 'general audience')
            }
        },
        {
            "task": f"You are a content writer creating educational material for '{book_structure.title}'. Focus on practical applications for {book_structure.metadata.get('target_audience', 'professionals')}.",
            "instruction": f"Write a comprehensive summary of {section.title.lower()} that reinforces key concepts from {chapter_title} > {section_title} and provides actionable takeaways.",
            "context": {
                "book_title": book_structure.title,
                "chapter": chapter_title,
                "section": section_title,
                "subsection": subsection_title,
                "subsection_id": subsection_id,
                "target_audience": book_structure.metadata.get('target_audience', 'general audience')
            }
        }
    ]

    # Write jobs to file
    with open(job_file, 'w', encoding='utf-8') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    console.print(f"[green]Generated contextualized job file: {job_file}[/green]")
    console.print(f"[dim]Context: {chapter_title} > {section_title} > {subsection_title}[/dim]")
    return job_file

def run_batch_processing(section: SectionConfig, job_file: Path, progress: Progress, task: TaskID) -> bool:
    """Run batch processing for a section."""
    console.print(f"[blue]Running batch processing for {section.subsection_id}[/blue]")

    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_batch.py"),
        "--jobs", str(job_file)
    ]

    # Add batch parameters
    if section.batch_params.get('key'):
        cmd.extend(["--key", section.batch_params['key']])
    if section.batch_params.get('content_type'):
        cmd.extend(["--content-type", section.batch_params['content_type']])
    if section.batch_params.get('k'):
        cmd.extend(["--k", str(section.batch_params['k'])])

    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        progress.update(task, advance=50)
        console.print(f"[green]✓ Batch processing completed for {section.subsection_id}[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Batch processing failed for {section.subsection_id}: {e}[/red]")
        console.print(f"[red]STDERR: {e.stderr}[/red]")
        return False

def run_merge_processing(section: SectionConfig, progress: Progress, task: TaskID) -> Tuple[bool, str]:
    """Run merge processing for a section."""
    console.print(f"[blue]Running merge processing for {section.subsection_id}[/blue]")

    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_merge_runner.py"),
        "--sub", section.subsection_id
    ]

    # Add merge parameters
    if section.merge_params.get('key'):
        cmd.extend(["--key", section.merge_params['key']])
    if section.merge_params.get('k'):
        cmd.extend(["--k", str(section.merge_params['k'])])

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        progress.update(task, advance=50)

        # Extract merged content from result
        merged_file = ROOT / "output" / "merged" / f"merged_content_{int(time.time())}.json"
        if merged_file.exists():
            with open(merged_file, 'r', encoding='utf-8') as f:
                merge_data = json.load(f)
                content = ""
                for section_data in merge_data.get('sections', {}).values():
                    content += section_data.get('merged_content', '')

            console.print(f"[green]✓ Merge processing completed for {section.subsection_id}[/green]")
            return True, content
        else:
            console.print(f"[yellow]⚠ Merge completed but no output file found for {section.subsection_id}[/yellow]")
            return True, ""

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Merge processing failed for {section.subsection_id}: {e}[/red]")
        console.print(f"[red]STDERR: {e.stderr}[/red]")
        return False, ""

def aggregate_book_content(book_structure: BookStructure, section_contents: Dict[str, str]) -> str:
    """Aggregate all section contents into a final book document."""
    console.print("[blue]Aggregating book content...[/blue]")

    # Create front matter
    front_matter = f"""# {book_structure.title}

**Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Sections**: {len(book_structure.sections)}

---

"""

    # Add table of contents
    toc = "## Table of Contents\n\n"
    for section in book_structure.sections:
        toc += f"- [{section.title}](#{section.subsection_id})\n"
    toc += "\n---\n\n"

    # Combine all sections
    content = front_matter + toc

    for section in book_structure.sections:
        section_content = section_contents.get(section.subsection_id, "")
        if section_content:
            content += f"## {section.title} {{#{section.subsection_id}}}\n\n"
            content += section_content
            content += "\n\n---\n\n"

    return content

def save_final_book(content: str, book_structure: BookStructure, output_file: Optional[Path] = None) -> Path:
    """Save the final book content to a markdown file."""
    if output_file is None:
        timestamp = int(time.time())
        output_file = ROOT / "exports" / "books" / f"{book_structure.title.replace(' ', '_')}_{timestamp}.md"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    console.print(f"[green]✓ Final book saved to: {output_file}[/green]")
    return output_file

def main():
    """Main orchestration flow."""
    import argparse
    ap = argparse.ArgumentParser(description="High-level book/chapter content generation orchestrator")
    ap.add_argument("--book", required=True, help="JSON file defining book structure")
    ap.add_argument("--output", help="Output markdown file path")
    ap.add_argument("--force", action="store_true", help="Force regeneration of all content")
    ap.add_argument("--skip-merge", action="store_true", help="Skip merge processing, only run batch")
    args = ap.parse_args()

    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LangChain Book Runner[/bold cyan]\n"
        "[dim]Complete book/chapter generation orchestration[/dim]",
        border_style="cyan"
    ))

    # Load book structure
    book_file = Path(args.book)
    if not book_file.exists():
        console.print(f"[red]Book structure file not found: {book_file}[/red]")
        sys.exit(1)

    book_structure = load_book_structure(book_file)

    console.print(f"[green]Loaded book: {book_structure.title}[/green]")
    console.print(f"[dim]Sections: {len(book_structure.sections)}[/dim]")

    # Display section overview
    table = Table()
    table.add_column("Section ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Job File", style="yellow")
    table.add_column("Dependencies", style="magenta")

    for section in book_structure.sections:
        job_status = "✓" if section.job_file and section.job_file.exists() else "Generate"
        deps = ", ".join(section.dependencies) if section.dependencies else "None"
        table.add_row(section.subsection_id, section.title, job_status, deps)

    console.print(table)

    # Confirm execution
    if not args.force and not Confirm.ask("Proceed with content generation?"):
        console.print("[yellow]Operation cancelled.[/yellow]")
        return

    # Process sections with progress tracking
    section_contents = {}
    failed_sections = []

    with Progress() as progress:
        main_task = progress.add_task("Processing book sections...", total=len(book_structure.sections))

        for section in book_structure.sections:
            progress.update(main_task, description=f"Processing {section.subsection_id}...")

            # Generate job file if needed
            job_file = generate_job_file(section, book_structure, ROOT)

            # Run batch processing
            batch_success = run_batch_processing(section, job_file, progress, main_task)

            if not batch_success:
                failed_sections.append(section.subsection_id)
                progress.update(main_task, advance=1)
                continue

            # Run merge processing (unless skipped)
            if not args.skip_merge:
                merge_success, content = run_merge_processing(section, progress, main_task)
                if merge_success:
                    section_contents[section.subsection_id] = content
                else:
                    failed_sections.append(section.subsection_id)
            else:
                # For skip-merge, just mark as successful with empty content
                section_contents[section.subsection_id] = ""

            progress.update(main_task, advance=1)

    # Report results
    console.print()
    console.print("[bold]Processing Complete[/bold]")
    console.print(f"[green]Successful sections: {len(section_contents)}[/green]")

    if failed_sections:
        console.print(f"[red]Failed sections: {len(failed_sections)}[/red]")
        for failed in failed_sections:
            console.print(f"[red]  • {failed}[/red]")

    # Generate final book if we have content
    if section_contents:
        final_content = aggregate_book_content(book_structure, section_contents)
        output_file = save_final_book(final_content, book_structure, Path(args.output) if args.output else None)

        console.print()
        console.print(Panel(
            f"[green]🎉 Book generation completed![/green]\n"
            f"[dim]Output: {output_file}[/dim]\n"
            f"[dim]Sections: {len(section_contents)}/{len(book_structure.sections)}[/dim]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        ))
    else:
        console.print("[red]No sections were successfully processed. Cannot generate final book.[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()