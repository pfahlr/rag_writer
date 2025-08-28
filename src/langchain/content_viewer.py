#!/usr/bin/env python3
"""
Content Viewer for lc_batch.py output

A terminal user interface to browse batch-generated content by sections.
Uses rich and textual libraries for enhanced display and navigation.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

console = Console()

def load_batch_results() -> Dict[str, List[Dict[str, Any]]]:
    """Load all batch result files and group by section."""
    output_dir = ROOT / "output"
    if not output_dir.exists():
        console.print("[red]No output directory found. Run lc-batch first.[/red]")
        sys.exit(1)

    sections = defaultdict(list)

    # Load all batch result files
    for json_file in output_dir.glob("batch_results_*.json"):
        try:
            # Extract timestamp from filename
            timestamp = int(json_file.stem.split('_')[-1])

            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            for i, result in enumerate(results):
                section = result.get('section', 'unknown')
                # Add metadata for sorting
                result['_timestamp'] = timestamp
                result['_file_position'] = i
                result['_filename'] = json_file.name
                # Format date for display
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp)
                result['_formatted_date'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                sections[section].append(result)

        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {json_file}: {e}[/yellow]")

    if not sections:
        console.print("[red]No batch results found in output directory.[/red]")
        sys.exit(1)

    return dict(sections)

def display_sections(sections: Dict[str, List[Dict[str, Any]]]) -> str:
    """Display available sections and let user choose one."""
    console.clear()

    # Create a table of sections
    table = Table(title="Available Sections")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Variations", style="magenta")
    table.add_column("Latest File", style="green")

    for section_name, variations in sorted(sections.items()):
        # Find the most recent variation
        latest_timestamp = 0
        latest_file = ""
        for variation in variations:
            # Extract timestamp from file path if available
            pass

        table.add_row(
            section_name,
            str(len(variations)),
            latest_file or "N/A"
        )

    console.print(table)
    console.print()

    # Get user selection
    section_names = sorted(sections.keys())
    for i, name in enumerate(section_names, 1):
        console.print(f"[cyan]{i}[/cyan]. {name}")

    console.print(f"[cyan]{len(section_names) + 1}[/cyan]. Exit")

    while True:
        try:
            choice = Prompt.ask("\nSelect a section (number or name) or 'exit' to quit", console=console)
            if choice.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                sys.exit(0)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(section_names):
                    return section_names[idx]
                elif idx == len(section_names):  # Exit option
                    console.print("[yellow]Goodbye![/yellow]")
                    sys.exit(0)
            elif choice in section_names:
                return choice
            console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            sys.exit(0)

def display_variations(section_name: str, variations: List[Dict[str, Any]]):
    """Display numbered variations for a section."""
    console.clear()

    console.print(f"[bold cyan]Section: {section_name}[/bold cyan]")
    console.print(f"[dim]Found {len(variations)} variations[/dim]\n")

    for i, variation in enumerate(variations, 1):
        content = variation.get('generated_content', 'No content')
        task = variation.get('task', 'No task')
        instruction = variation.get('instruction', 'No instruction')

        # Truncate content for preview
        preview = content[:200] + "..." if len(content) > 200 else content

        panel = Panel(
            f"[bold]Task:[/bold] {task}\n"
            f"[bold]Instruction:[/bold] {instruction}\n"
            f"[bold]Content:[/bold] {preview}",
            title=f"Variation {i}",
            border_style="blue"
        )
        console.print(panel)
        console.print()

    # Let user choose viewing mode
    while True:
        try:
            choice = Prompt.ask("Choose view mode: (number) for detailed view, 'slideshow' for sequential view, 'consolidated' for all-in-one view, or 'back' to return", console=console)
            if choice.lower() == 'back':
                return
            elif choice.lower() == 'slideshow':
                slideshow_variations(section_name, variations)
                return
            elif choice.lower() == 'consolidated':
                consolidated_view(section_name, variations)
                return
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(variations):
                    display_full_variation(section_name, variations[idx], idx + 1)
                    return
            console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Returning to section list...[/yellow]")
            return

def display_full_variation(section_name: str, variation: Dict[str, Any], variation_num: int):
    """Display a single variation in full detail."""
    console.clear()

    content = variation.get('generated_content', 'No content')
    task = variation.get('task', 'No task')
    instruction = variation.get('instruction', 'No instruction')
    sources = variation.get('sources', [])

    console.print(f"[bold cyan]Section: {section_name} - Variation {variation_num}[/bold cyan]\n")

    # Display task and instruction
    console.print("[bold]Task:[/bold]")
    console.print(f"  {task}\n")

    console.print("[bold]Instruction:[/bold]")
    console.print(f"  {instruction}\n")

    # Display full content
    console.print("[bold]Generated Content:[/bold]")
    console.print(Panel(content, border_style="green"))

    # Display sources (if any)
    if sources:
        console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown')
            page = source.get('page', 'N/A')
            source_path = source.get('source', 'N/A')
            console.print(f"  {i}. {title} (p.{page}) :: {source_path}")

    console.print("\n[dim]Press Enter to return to variations list...[/dim]")
    input()

def slideshow_variations(section_name: str, variations: List[Dict[str, Any]]):
    """Display variations in slideshow mode without sources."""
    current_idx = 0
    total_variations = len(variations)

    while True:
        console.clear()
        variation = variations[current_idx]
        content = variation.get('generated_content', 'No content')
        task = variation.get('task', 'No task')
        instruction = variation.get('instruction', 'No instruction')

        # Display header with progress
        console.print(f"[bold cyan]Section: {section_name}[/bold cyan]")
        console.print(f"[dim]Variation {current_idx + 1} of {total_variations}[/dim]\n")

        # Display task and instruction (brief)
        console.print(f"[bold]Task:[/bold] {task}")
        console.print(f"[bold]Instruction:[/bold] {instruction}\n")

        # Display content without sources
        console.print("[bold]Generated Content:[/bold]")
        console.print(Panel(content, border_style="green", padding=(1, 2)))

        # Navigation instructions
        console.print("\n[dim]Navigation: 'n' (next), 'p' (previous), 'b' (back to variations), 'q' (quit)[/dim]")

        try:
            nav = Prompt.ask("Navigate", console=console, default="n").lower()
            if nav in ['n', 'next', '']:
                current_idx = (current_idx + 1) % total_variations
            elif nav in ['p', 'previous']:
                current_idx = (current_idx - 1) % total_variations
            elif nav in ['b', 'back']:
                return
            elif nav in ['q', 'quit']:
                console.print("\n[yellow]Goodbye![/yellow]")
                sys.exit(0)
            else:
                console.print("[red]Invalid navigation command.[/red]")
                continue
        except KeyboardInterrupt:
            console.print("\n[yellow]Returning to variations list...[/yellow]")
            return

def consolidated_view(section_name: str, variations: List[Dict[str, Any]]):
    """Display all variations in one consolidated view with save functionality."""
    # Sort variations by timestamp and file position
    sorted_variations = sorted(variations, key=lambda x: (x['_timestamp'], x['_file_position']))

    while True:
        console.clear()

        console.print(f"[bold cyan]Consolidated View - Section: {section_name}[/bold cyan]")
        console.print(f"[dim]Showing {len(sorted_variations)} variations[/dim]\n")

        # Build the full content for display and potential saving
        full_content = []
        for i, variation in enumerate(sorted_variations, 1):
            content = variation.get('generated_content', 'No content')
            timestamp = variation['_timestamp']
            date_str = variation.get('_formatted_date', 'Unknown date')

            # Create separator
            separator = f"---\nsection {section_name}, variation {i}, date {date_str}\n---"

            full_content.append(separator)
            full_content.append(content)
            full_content.append("")  # Empty line after content

        # Display the content
        for line in full_content:
            console.print(line)

        console.print("\n[dim]Navigation: 'w' (write to file), 'back' (return to variations)[/dim]")

        try:
            choice = Prompt.ask("Choose action", console=console, default="back").lower()
            if choice in ['back', 'b', '']:
                return
            elif choice in ['w', 'write']:
                # Prompt for filename
                filename_base = Prompt.ask("Enter filename (without extension)", console=console)
                if filename_base.strip():
                    # Generate filename with timestamp and .md extension
                    import time
                    timestamp = int(time.time())
                    filename = f"{filename_base}_{timestamp}.md"

                    # Write content to file
                    try:
                        with open('./output/'+filename, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(full_content))
                        console.print(f"[green]Content saved to: {filename}[/green]")
                        console.print("[dim]Press Enter to continue...[/dim]")
                        input()
                    except Exception as e:
                        console.print(f"[red]Error saving file: {e}[/red]")
                        console.print("[dim]Press Enter to continue...[/dim]")
                        input()
                else:
                    console.print("[yellow]Filename cannot be empty.[/yellow]")
                    console.print("[dim]Press Enter to continue...[/dim]")
                    input()
            else:
                console.print("[red]Invalid choice. Use 'w' to write or 'back' to return.[/red]")
                console.print("[dim]Press Enter to continue...[/dim]")
                input()
        except KeyboardInterrupt:
            console.print("\n[yellow]Returning to variations list...[/yellow]")
            return

def main():
    """Main application loop."""
    try:
        sections = load_batch_results()

        while True:
            section_name = display_sections(sections)
            variations = sections[section_name]
            display_variations(section_name, variations)

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()