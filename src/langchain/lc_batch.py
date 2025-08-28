#!/usr/bin/env python3
"""
LangChain batch processor for multiple lc-ask calls

Reads a JSON array of objects with 'task', 'instruction', and 'section' fields,
calls lc-ask for each item, and writes results to a timestamped JSON file.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.prompt import Confirm

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

console = Console()

def run_lc_ask(task: str, instruction: str, key: str = "default", content_type: str = "pure_research", topk: int = None):
    """Run lc-ask with given parameters and return parsed JSON result."""
    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_ask.py"),
        "ask",  # Specify the ask command
    ]

    # Only add --task if it has a value
    if task and task.strip():
        cmd.extend(["--task", task])

    # Add key, content type, and instruction
    cmd.extend(["--key", key, "--content-type", content_type])

    # Add top-k if specified
    if topk:
        cmd.extend(["--k", str(topk)])

    # Add instruction as the final argument
    cmd.append(instruction)

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        # Return error info instead of printing
        return {"error": str(e), "generated_content": "", "sources": []}
    except json.JSONDecodeError as e:
        # Return error info instead of printing
        return {"error": f"JSON decode error: {e}", "generated_content": "", "sources": []}

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file (one JSON object per line)."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]Warning: Invalid JSON on line {line_num}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}[/red]")
        sys.exit(1)

    return data

def main():
    # Parse command line arguments
    import argparse
    ap = argparse.ArgumentParser(description="Process multiple queries and save results to timestamped files")
    ap.add_argument("--jobs", help="JSON or JSONL file containing job definitions")
    ap.add_argument("--key", default="default", help="Collection key for lc_ask")
    ap.add_argument("--content-type", default="pure_research", help="Content type for lc_ask")
    ap.add_argument("--k", type=int, help="Retriever top-k for lc_ask")

    # For backward compatibility, also accept positional arguments
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # Legacy mode: positional arguments
        args = ap.parse_args([])
        input_file = sys.argv[1]
        args.key = sys.argv[2] if len(sys.argv) > 2 else "default"
        args.content_type = sys.argv[3] if len(sys.argv) > 3 else "pure_research"
    else:
        # New mode: named arguments
        args = ap.parse_args()

        if not args.jobs:
            console.print("[red]Error: --jobs argument is required[/red]")
            sys.exit(1)
        input_file = args.jobs

    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LangChain Batch Processor[/bold cyan]\n"
        "[dim]Process multiple queries and save results to timestamped files[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Input handling
    console.print(f"[dim]Reading from file: {input_file}[/dim]")

    # Try JSON first, then JSONL
    data = None
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            console.print("[red]Error: JSON file must contain an array[/red]")
            sys.exit(1)
        console.print("[dim]Detected JSON array format[/dim]")
    except json.JSONDecodeError:
        # Try JSONL format
        console.print("[dim]JSON array failed, trying JSONL format...[/dim]")
        data = load_jsonl_file(input_file)
        if not data:
            console.print("[red]Error: No valid data found in file[/red]")
            sys.exit(1)
        console.print("[dim]Detected JSONL format[/dim]")

    console.print(f"[dim]Using collection key: {args.key}[/dim]")
    console.print(f"[dim]Using content type: {args.content_type}[/dim]")
    if hasattr(args, 'k') and args.k:
        console.print(f"[dim]Using retriever top-k: {args.k}[/dim]")
    console.print()

    # Validate data
    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            console.print(f"[yellow]Warning: Skipping non-object item at index {i}[/yellow]")
            continue

        instruction = item.get('instruction', '')
        if not instruction:
            console.print(f"[yellow]Warning: Skipping item without instruction at index {i}[/yellow]")
            continue

        valid_items.append(item)

    if not valid_items:
        console.print("[red]Error: No valid items found in input[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(valid_items)} valid items to process[/green]")
    console.print()

    # Process items with progress bar
    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("Processing items...", total=len(valid_items))

        for item in valid_items:
            section = item.get('section', 'unknown')
            progress.update(task, description=f"Processing section: {section}")

            task_text = item.get('task', '')
            instruction = item.get('instruction', '')

            try:
                # Run lc-ask
                result = run_lc_ask(task_text, instruction, args.key, args.content_type, getattr(args, 'k', None))

                # Add metadata to result
                result['section'] = section
                result['task'] = task_text
                result['instruction'] = instruction
                result['status'] = 'success'

                results.append(result)

            except Exception as e:
                error_result = {
                    'section': section,
                    'task': task_text,
                    'instruction': instruction,
                    'status': 'error',
                    'error': str(e),
                    'generated_content': '',
                    'sources': []
                }
                results.append(error_result)
                errors.append(f"Section '{section}': {e}")

            progress.update(task, advance=1)

    # Display results summary
    console.print()
    success_count = len([r for r in results if r.get('status') == 'success'])
    error_count = len([r for r in results if r.get('status') == 'error'])

    # Create summary table
    summary_table = Table(title="Batch Processing Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="magenta", justify="right")

    summary_table.add_row("Total Items", str(len(valid_items)))
    summary_table.add_row("Successful", str(success_count))
    summary_table.add_row("Errors", str(error_count))

    console.print(summary_table)

    # Display errors if any
    if errors:
        console.print()
        console.print("[red]Errors encountered:[/red]")
        for error in errors:
            console.print(f"  • {error}")

    # Write results to timestamped file
    console.print()
    console.print("[dim]Saving results...[/dim]")

    output_dir = ROOT / "output/batch"
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    output_file = output_dir / f"batch_results_{timestamp}.json"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Success message with file info
        file_size = output_file.stat().st_size
        console.print()
        console.print(Panel(
            f"[green]✓ Batch processing complete![/green]\n"
            f"[dim]Results saved to: {output_file}[/dim]\n"
            f"[dim]File size: {file_size:,} bytes[/dim]\n"
            f"[dim]Timestamp: {timestamp}[/dim]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        ))

    except Exception as e:
        console.print()
        console.print(Panel(
            f"[red]✗ Failed to save results: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
        sys.exit(1)

if __name__ == "__main__":
    main()