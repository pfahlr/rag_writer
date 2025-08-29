#!/usr/bin/env python3
"""
LangChain Book Runner - High-level orchestration for entire books/chapters

Refactored to use centralized core modules for better maintainability and performance.

Orchestrates the complete content generation pipeline for books and chapters:
1. Parses hierarchical book structure (4 levels deep)
2. Generates job files for each sub-sub-section
3. Runs batch processing for content generation (direct calls, no subprocess)
4. Runs merge processing for content refinement (direct calls, no subprocess)
5. Aggregates all results into final markdown document

Supports both embedded job definitions and external JSON file references.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.prompt import Prompt, Confirm

from src.langchain.job_generation import generate_llm_job_file, generate_fallback_job_file

# Import our new core modules
from src.config.settings import get_config
from src.utils.error_handler import handle_and_exit

# Get centralized configuration
config = get_config()
console = Console()

# Keep ROOT for backward compatibility with tests
ROOT = config.paths.root_dir

# Content generation logging system
CONTENT_LOG_FILE = ROOT / "logs" / "content_generation.jsonl"

def log_content_generation(content: str, content_label: str, job_file: str, job_file_modified_time: float, content_type: str = "merged", batch_results: Optional[List] = None):
    """Log content generation to the centralized log file."""
    CONTENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": time.time(),
        "content": content,
        "content_label": content_label,
        "job_file": job_file,
        "job_file_modified_time": job_file_modified_time,
        "content_type": content_type,  # "pre_merge" or "merged"
        "batch_results_count": len(batch_results) if batch_results else 0
    }

    with open(CONTENT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        f.flush()  # Ensure immediate write to disk

    # Also log individual pre-merge content if available
    if batch_results and content_type == "merged":
        for i, result in enumerate(batch_results):
            if result.get('status') == 'success' and result.get('generated_content'):
                pre_merge_entry = {
                    "timestamp": time.time(),
                    "content": result['generated_content'],
                    "content_label": f"{content_label} - Prompt {i+1}",
                    "job_file": job_file,
                    "job_file_modified_time": job_file_modified_time,
                    "content_type": "pre_merge",
                    "prompt_index": i + 1,
                    "job_data": result.get('job', {})
                }

                with open(CONTENT_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(pre_merge_entry, ensure_ascii=False) + '\n')
                    f.flush()

def check_content_cache(job_file: str, job_file_modified_time: float) -> Optional[str]:
    """Check if content has already been generated for this job file version."""
    if not CONTENT_LOG_FILE.exists():
        return None

    try:
        with open(CONTENT_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    if (entry['job_file'] == job_file and
                        entry['job_file_modified_time'] == job_file_modified_time):
                        return entry['content']
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read content cache: {e}[/yellow]")

    return None

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


def run_batch_processing(section: SectionConfig, job_file: Path, book_structure: BookStructure) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run batch processing for a section using direct function calls."""
    console.print(f"[blue]Running batch processing for {section.subsection_id}[/blue]")

    try:
        # Import and use the batch processing functions directly
        from src.langchain.lc_batch import load_jsonl_file, process_items_sequential

        # Load job data
        job_data = load_jsonl_file(str(job_file))

        # Add section information to each job item
        for job_item in job_data:
            job_item['section'] = section.subsection_id

        # Prepare arguments for batch processing
        class Args:
            def __init__(self):
                self.key = section.batch_params.get('key', config.rag_key)
                self.content_type = book_structure.metadata.get('content_type', 'technical_manual_writer')
                self.k = section.batch_params.get('k', config.retriever.default_k)
                self.parallel = 1  # Sequential for book processing
                self.output_dir = None

        args = Args()

        # Process items
        results, errors = process_items_sequential(job_data, args, console)

        # Debug: Check what we got back
        console.print(f"[dim]Batch processing returned {len(results)} results and {len(errors)} errors[/dim]")
        if results:
            successful_results = [r for r in results if r.get('status') == 'success']
            console.print(f"[dim]Successful results: {len(successful_results)}[/dim]")
            if successful_results:
                console.print(f"[dim]Sample result keys: {list(successful_results[0].keys())}[/dim]")

        if errors:
            console.print(f"[red]✗ Batch processing failed for {section.subsection_id}: {len(errors)} errors[/red]")
            for error in errors[:3]:  # Show first 3 errors
                console.print(f"[red]  • {error}[/red]")
            return False, results
        else:
            console.print(f"[green]✓ Batch processing completed for {section.subsection_id}[/green]")
            console.print(f"[dim]Processed {len(results)} items successfully[/dim]")
            return True, results

    except Exception as e:
        console.print(f"[red]✗ Batch processing failed for {section.subsection_id}: {e}[/red]")
        return False, []

def run_merge_processing(section: SectionConfig, book_structure: BookStructure) -> Tuple[bool, str]:
    """Run merge processing for a section with improved error handling."""
    console.print(f"[blue]Running merge processing for {section.subsection_id}[/blue]")

    try:
        # Parse hierarchical context from subsection_id
        subsection_id = section.subsection_id
        chapter_num = subsection_id[0] if subsection_id[0].isdigit() else "1"
        section_letter = subsection_id[1] if len(subsection_id) > 1 and subsection_id[1].isalpha() else "A"
        subsection_num = subsection_id[2] if len(subsection_id) > 2 and subsection_id[2].isdigit() else "1"

        # Build hierarchical context using actual titles from book structure
        chapter_title = f"Chapter {chapter_num}"
        section_title = f"Section {section_letter}"
        subsection_title = section.title  # Use the actual section title

        # For now, we'll use subprocess but with better error handling
        # TODO: Refactor lc_merge_runner.py to use direct function calls
        cmd = [
            sys.executable, str(config.paths.root_dir / "src/langchain/lc_merge_runner.py"),
            "--batch-only",  # Force batch-only mode to avoid interactive prompts
            "--chapter", chapter_title,
            "--section", section_title,
            "--subsection", subsection_title
        ]

        # Add merge parameters
        if section.merge_params.get('key'):
            cmd.extend(["--key", section.merge_params['key']])
        if section.merge_params.get('k'):
            cmd.extend(["--k", str(section.merge_params['k'])])

        console.print(f"[dim]Running merge command...[/dim]")

        result = subprocess.run(
            cmd,
            cwd=config.paths.root_dir,
            capture_output=True,
            text=True,
            check=True
        )

        # Extract merged content from result
        # Look for the most recent merged content file
        merged_dir = config.paths.output_dir / "merged"
        if merged_dir.exists():
            merged_files = list(merged_dir.glob("merged_content_*.json"))
            if merged_files:
                # Get the most recent file
                merged_file = max(merged_files, key=lambda f: f.stat().st_mtime)
                with open(merged_file, 'r', encoding='utf-8') as f:
                    merge_data = json.load(f)
                    content = ""
                    for section_data in merge_data.get('sections', {}).values():
                        content += section_data.get('merged_content', '')

                console.print(f"[green]✓ Merge processing completed for {section.subsection_id}[/green]")
                return True, content

        console.print(f"[yellow]⚠ Merge completed but no output file found for {section.subsection_id}[/yellow]")
        return True, ""

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Merge processing failed for {section.subsection_id}: {e}[/red]")
        if e.stderr:
            console.print(f"[red]STDERR: {e.stderr}[/red]")
        return False, ""
    except Exception as e:
        console.print(f"[red]✗ Unexpected error in merge processing for {section.subsection_id}: {e}[/red]")
        return False, ""

def get_batch_content_for_section(subsection_id: str, batch_results: List[Dict[str, Any]] = None) -> Optional[str]:
    """Try to extract content from batch results for a section when merge is skipped."""
    console.print(f"[dim]Looking for batch content for section: {subsection_id}[/dim]")

    # If batch_results are provided directly, use them
    if batch_results is not None:
        console.print(f"[dim]Using provided batch results: {len(batch_results)} items[/dim]")
        batch_data = batch_results
    else:
        # Fall back to file-based approach
        try:
            # Look for batch result files
            output_dir = config.paths.output_dir / "batch"
            console.print(f"[dim]Checking output dir: {output_dir}[/dim]")
            if not output_dir.exists():
                console.print(f"[dim]Output dir doesn't exist: {output_dir}[/dim]")
                return None

            # Find the most recent batch results file
            batch_files = list(output_dir.glob("batch_results_*.json"))
            console.print(f"[dim]Found {len(batch_files)} batch files[/dim]")
            if not batch_files:
                return None

            # Get the most recent file
            batch_file = max(batch_files, key=lambda f: f.stat().st_mtime)
            console.print(f"[dim]Using batch file: {batch_file}[/dim]")

            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)

            console.print(f"[dim]Loaded {len(batch_data)} batch results[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load batch file for {subsection_id}: {e}[/yellow]")
            return None

    # Look for content related to this subsection
    content_parts = []
    for result in batch_data:
        section_name = result.get('section', '')
        console.print(f"[dim]Checking result for section: {section_name}[/dim]")
        if section_name == subsection_id or section_name.startswith(subsection_id) or subsection_id in section_name:
            console.print(f"[dim]Found matching section: {section_name}[/dim]")
            status = result.get('status')
            generated_content = result.get('generated_content', '')
            error = result.get('error', '')
            console.print(f"[dim]Result status: {status}, content length: {len(generated_content)}[/dim]")
            if error:
                console.print(f"[dim]Result error: {error}[/dim]")
            if generated_content:
                console.print(f"[dim]Content preview: {generated_content[:100]}...[/dim]")
            if status == 'success' and generated_content:
                content_parts.append(generated_content)
                console.print(f"[dim]Added content, total parts: {len(content_parts)}[/dim]")
            else:
                console.print(f"[dim]Skipping result: status={status}, has_content={bool(generated_content)}, has_error={bool(error)}[/dim]")

    if content_parts:
        final_content = '\n\n'.join(content_parts)
        console.print(f"[dim]Returning combined content, length: {len(final_content)}[/dim]")
        return final_content

    console.print(f"[dim]No content found for section: {subsection_id}[/dim]")
    return None

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
        content += f"## {section.title} {{#{section.subsection_id}}}\n\n"
        if section_content:
            content += section_content
        else:
            content += "*No content generated for this section.*\n\n"
        content += "\n\n---\n\n"

    return content

def save_final_book(content: str, book_structure: BookStructure, output_file: Optional[Path] = None) -> Path:
    """Save the final book content to a markdown file using centralized config."""
    if output_file is None:
        timestamp = int(time.time())
        safe_title = book_structure.title.replace(' ', '_').replace('/', '_')
        output_file = config.paths.exports_dir / "books" / f"{safe_title}_{timestamp}.md"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        console.print(f"[green]✓ Final book saved to: {output_file}[/green]")
        return output_file
    except Exception as e:
        handle_and_exit(e, f"saving final book to {output_file}")

def main():
    """Main orchestration flow."""
    import argparse
    ap = argparse.ArgumentParser(description="High-level book/chapter content generation orchestrator")
    ap.add_argument("--book", required=True, help="JSON file defining book structure")
    ap.add_argument("--output", help="Output markdown file path")
    ap.add_argument("--force", action="store_true", help="Force regeneration of all content")
    ap.add_argument("--skip-merge", action="store_true", help="Skip merge processing, only run batch")
    ap.add_argument("--use-rag", action="store_true", help="Use RAG for additional context when generating job prompts")
    ap.add_argument("--rag-key", help="Collection key for RAG retrieval (required if --use-rag is specified)")
    ap.add_argument("--num-prompts", type=int, help="Number of prompts to generate per section (default: 4)")
    args = ap.parse_args()

    # Validate RAG arguments
    if args.use_rag and not args.rag_key:
        ap.error("--rag-key is required when --use-rag is specified")

    # Validate num_prompts
    num_prompts = args.num_prompts or config.job_generation.default_prompts_per_section
    if num_prompts < config.job_generation.min_prompts_per_section or num_prompts > config.job_generation.max_prompts_per_section:
        ap.error(f"--num-prompts must be between {config.job_generation.min_prompts_per_section} and {config.job_generation.max_prompts_per_section}")

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

            # Parse hierarchical context from subsection_id
            subsection_id = section.subsection_id
            chapter_num = subsection_id[0] if subsection_id[0].isdigit() else "1"
            section_letter = subsection_id[1] if len(subsection_id) > 1 and subsection_id[1].isalpha() else "A"
            subsection_num = subsection_id[2] if len(subsection_id) > 2 and subsection_id[2].isdigit() else "1"

            # Build hierarchical context
            chapter_title = f"Chapter {chapter_num}"
            section_title = f"Section {section_letter}"

            # Generate job file if needed
            job_file = generate_llm_job_file(
                section_id=section.subsection_id,
                section_title=section.title,
                book_title=book_structure.title,
                chapter_title=chapter_title,
                section_title_hierarchy=section_title,
                subsection_title=section.title,
                target_audience=book_structure.metadata.get('target_audience', 'general audience'),
                topic=book_structure.metadata.get('topic', ''),
                use_rag=args.use_rag,
                rag_key=args.rag_key,
                base_dir=config.paths.root_dir,
                num_prompts=num_prompts,
                content_type=book_structure.metadata.get('content_type', 'technical_manual_writer')
            )

            # Run batch processing
            batch_success, batch_results = run_batch_processing(section, job_file, book_structure)

            if not batch_success:
                failed_sections.append(section.subsection_id)
                progress.update(main_task, advance=1)
                continue

            # Check cache first to avoid re-generation
            job_file_modified_time = job_file.stat().st_mtime if job_file.exists() else 0
            cached_content = check_content_cache(str(job_file), job_file_modified_time)

            if cached_content and not args.force:
                console.print(f"[green]✓ Using cached content for {section.subsection_id}[/green]")
                section_contents[section.subsection_id] = cached_content
                progress.update(main_task, advance=1)
                continue

            # Run merge processing (unless skipped)
            if not args.skip_merge:
                merge_success, content = run_merge_processing(section, book_structure)
                console.print(f"[dim]Merge result: success={merge_success}, content_length={len(content) if content else 0}[/dim]")
                if merge_success:
                    section_contents[section.subsection_id] = content
                    # Log the generated content
                    log_content_generation(
                        content=content,
                        content_label=f"{section.subsection_id}: {section.title}",
                        job_file=str(job_file),
                        job_file_modified_time=job_file_modified_time,
                        content_type="merged",
                        batch_results=batch_results
                    )
                else:
                    failed_sections.append(section.subsection_id)
            else:
                # For skip-merge, try to get content from batch results
                content = get_batch_content_for_section(section.subsection_id, batch_results)
                console.print(f"[dim]Batch content extraction: found={bool(content)}, length={len(content) if content else 0}[/dim]")
                if content:
                    section_contents[section.subsection_id] = content
                    # Log the generated content
                    log_content_generation(
                        content=content,
                        content_label=f"{section.subsection_id}: {section.title}",
                        job_file=str(job_file),
                        job_file_modified_time=job_file_modified_time,
                        content_type="batch_only",
                        batch_results=batch_results
                    )
                else:
                    # No content available, mark as empty but still log
                    section_contents[section.subsection_id] = ""
                    log_content_generation(
                        content="",
                        content_label=f"{section.subsection_id}: {section.title} (no content)",
                        job_file=str(job_file),
                        job_file_modified_time=job_file_modified_time,
                        content_type="batch_only",
                        batch_results=batch_results
                    )

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