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
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            for result in results:
                section = result.get('section', 'unknown')
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

    while True:
        try:
            choice = Prompt.ask("\nSelect a section (number or name)", console=console)
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(section_names):
                    return section_names[idx]
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

    # Let user select a variation to view in full
    while True:
        try:
            choice = Prompt.ask("Select variation to view in full (number) or 'back' to return", console=console)
            if choice.lower() == 'back':
                return
            if choice.isdigit():
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