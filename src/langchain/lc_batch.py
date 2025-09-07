#!/usr/bin/env python3
"""
LangChain batch processor for multiple RAG queries

Refactored to use centralized core modules for better maintainability and performance.

Features:
- Direct RAG processing (no subprocess calls)
- Parallel processing support
- Centralized configuration management
- Standardized error handling
- Source citation extraction and deduplication
"""

import os
import sys
import json
import time
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich.prompt import Confirm

# Import our new core modules
from src.core.retriever import RetrieverFactory, RetrieverConfig
from src.core.llm import LLMFactory, LLMConfig
from src.config.settings import get_config
from src.utils.error_handler import handle_and_exit, validate_collection
from src.tool import ToolRegistry
from src.tool.prompts import generate_tool_prompt

console = Console()

# Get centralized configuration
config = get_config()

# Keep ROOT for backward compatibility with tests
ROOT = config.paths.root_dir

# Constants
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]*[a-z0-9]\b", re.I)
USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Use the context below to answer. Include page-cited quotes for key claims.\n\n"
    "Context:\n{context}"
)


def load_content_types():
    """Load content types from YAML files using centralized config."""
    content_types_dir = config.paths.root_dir / "src/tool/prompts/content_types"
    content_types = {}

    try:
        import yaml

        # Load all YAML files from the content_types directory
        if content_types_dir.exists():
            for yaml_file in content_types_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                        # Use filename without extension as the content type key
                        content_type_key = yaml_file.stem
                        content_types[content_type_key] = data
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not load content type from {yaml_file}: {e}[/yellow]"
                    )
        else:
            console.print(
                f"[yellow]Warning: Content types directory not found: {content_types_dir}[/yellow]"
            )

        return content_types
    except Exception as e:
        console.print(f"[red]Error loading content types: {e}[/red]")
        return {}


def get_system_prompt(content_type: str) -> str:
    """Get system prompt for a content type."""
    content_types = load_content_types()

    if content_type not in content_types:
        available_types = list(content_types.keys())
        raise ValueError(
            f"Unknown content type '{content_type}'. Available types: {', '.join(available_types)}"
        )

    system_prompt_parts = content_types[content_type].get("system_prompt", [])
    if not system_prompt_parts:
        raise ValueError(f"No system prompt defined for content type '{content_type}'")

    return "".join(system_prompt_parts)


def _norm(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r"\W+", " ", (s or "")).strip().lower()


def _extract_cited(docs, answer: str):
    """Return (cited_docs, uncited_docs) based on title/DOI matches in the answer text."""
    if not answer or not docs:
        return [], docs

    ans = answer.lower()
    ans_norm = _norm(answer)
    cited, uncited = [], []

    for d in docs:
        title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
        title_norm = _norm(title)
        doi = (d.metadata.get("doi") or "").lower()

        hit = False
        # 1) Exact/normalized title presence
        if title and (title.lower() in ans or title_norm and title_norm in ans_norm):
            hit = True
        # 2) DOI presence
        if not hit and doi:
            if doi in ans or any(
                m.group(0).lower() == doi for m in DOI_RE.finditer(ans)
            ):
                hit = True

        if hit:
            cited.append(d)
        else:
            uncited.append(d)

    return cited, uncited


def _fmt_doc_for_context(d):
    """Format document for context inclusion."""
    title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
    page = d.metadata.get("page")
    header = f"[{title}, p.{page}]" if page else f"[{title}]"
    return f"{header}\n{d.page_content}"


def _format_context(docs):
    """Format documents into context string."""
    return "\n\n---\n\n".join(_fmt_doc_for_context(d) for d in docs)


def run_rag_query(
    task: str,
    instruction: str,
    key: str = "default",
    content_type: str = "pure_research",
    topk: int = None,
    tool_registry: Optional[ToolRegistry] = None,
) -> Dict[str, Any]:
    """Run RAG query directly using core modules (no subprocess calls).

    If a ``ToolRegistry`` is provided, its descriptions are appended to the
    system prompt so the model knows how to format tool calls.
    """
    try:
        # Validate collection exists
        validate_collection(key, config.paths.storage_dir)

        # Initialize retriever
        factory = RetrieverFactory(config.paths.root_dir)
        retriever_config = RetrieverConfig(
            key=key,
            k=topk or config.retriever.default_k,
            embedding_model=config.embedding.model_name,
            openai_model=config.llm.openai_model,
            rerank_model=config.retriever.rerank_model,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            use_reranking=config.retriever.use_reranking,
        )
        retriever = factory.create_hybrid_retriever(retriever_config)

        # Initialize LLM
        llm_config = LLMConfig(
            model=config.llm.openai_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            openai_api_key=config.openai_api_key,
            ollama_model=config.llm.ollama_model,
        )
        llm_factory = LLMFactory(llm_config)
        backend, llm = llm_factory.create_llm()

        # Retrieve documents
        docs = []
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(instruction)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(instruction)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(instruction)
        else:
            docs = retriever(instruction)

        context_text = _format_context(docs) if docs else ""

        # Build final question
        final_question = (
            f"{task} {instruction}".strip() if task and task.strip() else instruction
        )

        # Get system prompt and append tool descriptions if provided
        system_prompt = get_system_prompt(content_type)
        if tool_registry is not None:
            tool_prompt = generate_tool_prompt(tool_registry)
            tool_prompt = tool_prompt.replace("{", "{{").replace("}", "}}")
            system_prompt = f"{system_prompt}\n\n{tool_prompt}"

        # Create prompt and generate response
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", USER_PROMPT),
            ]
        )

        messages = prompt.format_messages(question=final_question, context=context_text)

        # Generate content
        if backend in ("lc_openai", "ollama"):
            resp = llm.invoke(messages)
            generated_content = resp.content
        elif backend == "raw_openai":
            msgs = [
                {
                    "role": "system" if m.type == "system" else "user",
                    "content": m.content,
                }
                for m in messages
            ]
            content = (
                llm.chat.completions.create(
                    model=config.llm.openai_model,
                    messages=msgs,
                    temperature=config.llm.temperature,
                )
                .choices[0]
                .message.content
            )
            generated_content = content
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Extract cited sources
        cited_docs, _ = _extract_cited(docs, generated_content)

        # Deduplicate sources by article
        sources = []
        seen_articles = set()
        for d in cited_docs:
            title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
            source_path = d.metadata.get("source", "")

            article_id = title if title else source_path
            if article_id and article_id not in seen_articles:
                seen_articles.add(article_id)
                sources.append({"title": title, "source": source_path})

        return {"generated_content": generated_content, "sources": sources}

    except Exception as e:
        return {"error": str(e), "generated_content": "", "sources": []}


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file (one JSON object per line)."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        console.print(
                            f"[yellow]Warning: Invalid JSON on line {line_num}: {e}[/yellow]"
                        )
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}[/red]")
        sys.exit(1)

    return data


def process_items_sequential(valid_items: List[Dict[str, Any]], args, console: Console):
    """Process items sequentially with progress tracking."""
    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing items...", total=len(valid_items))

        for item in valid_items:
            section = item.get("section", "unknown")
            progress.update(task, description=f"Processing section: {section}")

            task_text = item.get("task", "")
            instruction = item.get("instruction", "")

            try:
                # Run RAG query directly
                result = run_rag_query(
                    task_text, instruction, args.key, args.content_type, args.k
                )

                # Add metadata to result
                result["section"] = section
                result["task"] = task_text
                result["instruction"] = instruction
                result["status"] = "success"

                results.append(result)

            except Exception as e:
                error_result = {
                    "section": section,
                    "task": task_text,
                    "instruction": instruction,
                    "status": "error",
                    "error": str(e),
                    "generated_content": "",
                    "sources": [],
                }
                results.append(error_result)
                errors.append(f"Section '{section}': {e}")

            progress.update(task, advance=1)

    return results, errors


def process_items_parallel(valid_items: List[Dict[str, Any]], args, console: Console):
    """Process items in parallel with progress tracking."""
    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing items...", total=len(valid_items))

        # Process items in parallel
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    run_rag_query,
                    item.get("task", ""),
                    item.get("instruction", ""),
                    args.key,
                    args.content_type,
                    args.k,
                ): item
                for item in valid_items
            }

            # Process completed tasks
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                section = item.get("section", "unknown")
                progress.update(task, description=f"Processing section: {section}")

                try:
                    result = future.result()

                    # Add metadata to result
                    result["section"] = section
                    result["task"] = item.get("task", "")
                    result["instruction"] = item.get("instruction", "")
                    result["status"] = "success"

                    results.append(result)

                except Exception as e:
                    error_result = {
                        "section": section,
                        "task": item.get("task", ""),
                        "instruction": item.get("instruction", ""),
                        "status": "error",
                        "error": str(e),
                        "generated_content": "",
                        "sources": [],
                    }
                    results.append(error_result)
                    errors.append(f"Section '{section}': {e}")

                progress.update(task, advance=1)

    return results, errors


def main():
    """Main batch processing function with improved configuration and parallel processing."""
    # Parse command line arguments
    import argparse

    ap = argparse.ArgumentParser(
        description="Process multiple RAG queries and save results to timestamped files"
    )
    ap.add_argument("--jobs", help="JSON or JSONL file containing job definitions")
    ap.add_argument(
        "--key", default=config.rag_key, help="Collection key for RAG queries"
    )
    ap.add_argument(
        "--content-type", default="pure_research", help="Content type for queries"
    )
    ap.add_argument(
        "--k", type=int, default=config.retriever.default_k, help="Retriever top-k"
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential)",
    )
    ap.add_argument("--output-dir", help="Custom output directory")

    # For backward compatibility, also accept positional arguments
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Legacy mode: positional arguments
        args = ap.parse_args([])
        input_file = sys.argv[1]
        args.key = sys.argv[2] if len(sys.argv) > 2 else config.rag_key
        args.content_type = sys.argv[3] if len(sys.argv) > 3 else "pure_research"
        args.parallel = 1  # Sequential for legacy mode
    else:
        # New mode: named arguments
        args = ap.parse_args()

        if not args.jobs:
            console.print("[red]Error: --jobs argument is required[/red]")
            sys.exit(1)
        input_file = args.jobs

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]LangChain Batch Processor[/bold cyan]\n"
            "[dim]Process multiple queries and save results to timestamped files[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Input handling
    console.print(f"[dim]Reading from file: {input_file}[/dim]")

    # Try JSON first, then JSONL
    data = None
    try:
        with open(input_file, "r", encoding="utf-8") as f:
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
    console.print(f"[dim]Using retriever top-k: {args.k}[/dim]")
    console.print(f"[dim]Using parallel workers: {args.parallel}[/dim]")
    console.print()

    # Validate data
    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            console.print(
                f"[yellow]Warning: Skipping non-object item at index {i}[/yellow]"
            )
            continue

        instruction = item.get("instruction", "")
        if not instruction:
            console.print(
                f"[yellow]Warning: Skipping item without instruction at index {i}[/yellow]"
            )
            continue

        valid_items.append(item)

    if not valid_items:
        console.print("[red]Error: No valid items found in input[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(valid_items)} valid items to process[/green]")
    console.print()

    # Process items (sequential or parallel)
    results = []
    errors = []

    if args.parallel == 1:
        # Sequential processing
        results, errors = process_items_sequential(valid_items, args, console)
    else:
        # Parallel processing
        results, errors = process_items_parallel(valid_items, args, console)

    # Display results summary
    console.print()
    success_count = len([r for r in results if r.get("status") == "success"])
    error_count = len([r for r in results if r.get("status") == "error"])

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

    # Use custom output directory if specified, otherwise use default
    if hasattr(args, "output_dir") and args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.paths.output_dir / "batch"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_file = output_dir / f"batch_results_{timestamp}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Success message with file info
        file_size = output_file.stat().st_size
        console.print()
        console.print(
            Panel(
                f"[green]✓ Batch processing complete![/green]\n"
                f"[dim]Results saved to: {output_file}[/dim]\n"
                f"[dim]File size: {file_size:,} bytes[/dim]\n"
                f"[dim]Timestamp: {timestamp}[/dim]\n"
                f"[dim]Processing mode: {'Parallel' if args.parallel > 1 else 'Sequential'}[/dim]",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print()
        console.print(
            Panel(
                f"[red]✗ Failed to save results: {e}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
