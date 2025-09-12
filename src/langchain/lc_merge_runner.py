#!/usr/bin/env python3
"""
LangChain Merge Runner - Intelligent content merging for batch results

Reads batch result files, groups content by section, and uses AI to merge
variations into cohesive subsections with proper editorial context.
"""

import sys
import json
import time
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# Import config to get centralized paths
from src.config.settings import get_config


# Function to get ROOT dynamically to allow for testing
def get_root():
    """Get the root directory from config."""
    config = get_config()
    return config.paths.root_dir


# Use config-based ROOT for consistency with other modules
# Note: This will be called at import time, but we can mock get_config for testing
ROOT = get_root()

console = Console()


def load_batch_results() -> Dict[str, List[Dict[str, Any]]]:
    """Load all batch result files and group by section."""
    output_dir = ROOT / "output" / "batch"
    if not output_dir.exists():
        console.print("[red]No batch output directory found. Run lc-batch first.[/red]")
        sys.exit(1)

    sections = defaultdict(list)

    # Load all batch result files
    for json_file in output_dir.glob("batch_results_*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            for result in results:
                section = result.get("section", "unknown")
                sections[section].append(result)

        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {json_file}: {e}[/yellow]")

    if not sections:
        console.print("[red]No batch results found in output/batch directory.[/red]")
        sys.exit(1)

    return dict(sections)


def display_sections(sections: Dict[str, List[Dict[str, Any]]]):
    """Display available sections with variation counts."""
    console.print()
    console.print("[bold cyan]Available Sections for Merging[/bold cyan]")
    console.print()

    table = Table()
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Variations", style="magenta", justify="center")
    table.add_column("Status", style="green")

    for section_name, variations in sorted(sections.items()):
        variation_count = len(variations)
        success_count = len([v for v in variations if v.get("status") == "success"])
        status = f"{success_count}/{variation_count} successful"
        table.add_row(section_name, str(variation_count), status)

    console.print(table)
    console.print()


def load_merge_types() -> Dict[str, Dict]:
    """Load merge types and their configurations from YAML file."""
    merge_types_file = (
        ROOT / "src" / "config" / "content" / "prompts" / "merge_types.yaml"
    )

    if not merge_types_file.exists():
        console.print(
            f"[yellow]Warning: Merge types file not found at {merge_types_file}[/yellow]"
        )
        console.print("[yellow]Using default merge type.[/yellow]")
        return {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor for a publisher...",
                "stages": None,  # Simple single-stage merge
            }
        }

    try:
        with open(merge_types_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        merge_types = {}
        for merge_type, config in data.items():
            merge_config = {}

            # Handle description
            merge_config["description"] = config.get("description", merge_type)

            # Handle stages (for advanced pipelines)
            if "stages" in config:
                merge_config["stages"] = config["stages"]
                merge_config["system_prompt"] = None  # Multi-stage, no single prompt
            else:
                # Handle simple system_prompt (backward compatibility)
                if "system_prompt" in config:
                    if isinstance(config["system_prompt"], list):
                        merge_config["system_prompt"] = "".join(config["system_prompt"])
                    else:
                        merge_config["system_prompt"] = config["system_prompt"]
                    merge_config["stages"] = None

            merge_types[merge_type] = merge_config

        if not merge_types:
            console.print(
                "[yellow]Warning: No merge types found in YAML file.[/yellow]"
            )
            return {
                "generic_editor": {
                    "description": "Basic editor merge",
                    "system_prompt": "You are a senior editor for a publisher...",
                    "stages": None,
                }
            }

        return merge_types

    except Exception as e:
        console.print(f"[yellow]Warning: Could not load merge types: {e}[/yellow]")
        return {
            "generic_editor": {
                "description": "Basic editor merge",
                "system_prompt": "You are a senior editor for a publisher...",
                "stages": None,
            }
        }


def select_merge_type(merge_types: Dict[str, Dict]) -> str:
    """Prompt user to select which merge type to use."""
    console.print()
    console.print("[bold green]Merge Type Selection[/bold green]")
    console.print("Choose the type of merge strategy to use:")
    console.print()

    # Display merge types with numbers for reference
    for i, (merge_type, config) in enumerate(merge_types.items(), 1):
        description = config.get("description", merge_type)
        console.print(f"[cyan]{i}.[/cyan] [bold]{merge_type}[/bold]")
        console.print(f"[dim]   {description}[/dim]")

        # Show if it has stages (advanced pipeline)
        if config.get("stages"):
            stage_names = list(config["stages"].keys())
            console.print(f"[dim]   Pipeline stages: {', '.join(stage_names)}[/dim]")

        console.print()

    while True:
        selection = Prompt.ask("[cyan]Merge type[/cyan]").strip()

        # Try to parse as number first
        try:
            index = int(selection) - 1
            if 0 <= index < len(merge_types):
                return list(merge_types.keys())[index]
        except ValueError:
            pass

        # Try to match by name
        if selection in merge_types:
            return selection

        console.print(
            f"[red]Invalid selection. Please enter a number (1-{len(merge_types)}) or merge type name.[/red]"
        )


def select_job_file() -> Path:
    """Prompt user to select or specify a JSONL job file."""
    console.print()
    console.print("[bold green]Job File Selection[/bold green]")
    console.print("You can either:")
    console.print("1. Specify a custom job file path")
    console.print("2. Use default location (data_jobs/<subsection>.jsonl)")
    console.print("3. Skip job file processing (use existing batch results)")
    console.print()

    while True:
        choice = Prompt.ask("[cyan]Choice (1/2/3)[/cyan]").strip()

        if choice == "1":
            file_path = Prompt.ask("[cyan]Job file path[/cyan]").strip()
            job_file = Path(file_path)
            if job_file.exists():
                return job_file
            else:
                console.print(f"[red]File not found: {file_path}[/red]")
                continue

        elif choice == "2":
            # Will use default path based on subsection
            return None

        elif choice == "3":
            return False  # Skip job processing

        else:
            console.print("[red]Please enter 1, 2, or 3[/red]")


def select_sections(sections: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Prompt user to select which sections to merge."""
    console.print()
    console.print("[bold green]Section Selection[/bold green]")
    console.print("Enter the section names you want to merge, separated by commas.")
    console.print("Or enter 'all' to merge all sections:")
    console.print()

    # Display sections with numbers for reference
    for i, section_name in enumerate(sorted(sections.keys()), 1):
        console.print(f"[cyan]{i}.[/cyan] {section_name}")

    console.print()

    while True:
        selection = Prompt.ask("[cyan]Sections to merge[/cyan]").strip()

        if selection.lower() == "all":
            return list(sections.keys())

        # Parse comma-separated input
        selected_sections = [s.strip() for s in selection.split(",") if s.strip()]

        # Validate selections
        invalid_sections = [s for s in selected_sections if s not in sections]
        if invalid_sections:
            console.print(
                f"[red]Invalid section(s): {', '.join(invalid_sections)}[/red]"
            )
            console.print("Please try again.")
            continue

        if not selected_sections:
            console.print("[red]No sections selected. Please try again.[/red]")
            continue

        return selected_sections


def get_context_info() -> Dict[str, str]:
    """Get chapter, section, and subsection titles from user."""
    console.print("[bold green]Context Information[/bold green]")
    console.print("Please provide the hierarchical context for the content sections:")
    console.print()

    chapter = Prompt.ask("[cyan]Chapter title[/cyan]")
    section = Prompt.ask("[cyan]Section title[/cyan]")
    subsection = Prompt.ask("[cyan]Subsection title[/cyan]")

    return {"chapter": chapter, "section": section, "subsection": subsection}


def run_pipeline_stage(
    stage_name: str,
    stage_config: Dict,
    content: str,
    context: Dict[str, str],
    subsection_id: str,
) -> Dict[str, Any]:
    """Execute a single pipeline stage using lc_ask.py."""
    console.print(f"[dim]Running {stage_name} stage...[/dim]")

    # Build the system prompt
    system_prompt = stage_config.get("system_prompt", "")
    if isinstance(system_prompt, list):
        system_prompt = "".join(system_prompt)

    # Create stage-specific instruction
    chapter = context["chapter"]
    section = context["section"]
    subsection = context["subsection"]

    base_instruction = f"""
Chapter: {chapter}
Section: {section}
Subsection: {subsection} ({subsection_id})

"""

    # Add stage-specific content and instructions
    if stage_name.startswith("critique") and "scoring_instruction" in stage_config:
        # For critique stage, add scoring instruction
        instruction = (
            base_instruction
            + f"""
{content}

{stage_config['scoring_instruction']}
"""
        )
    else:
        # For other stages, just add the content
        instruction = base_instruction + content

    # Call lc_ask.py with the stage-specific prompt
    cmd = [
        sys.executable,
        str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type",
        "pure_research",
        "--task",
        system_prompt,
        instruction,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True
        )

        # Parse the response with better error handling
        stdout_content = result.stdout.strip()
        if not stdout_content:
            console.print(
                "[yellow]Warning: Empty response from pipeline stage subprocess[/yellow]"
            )
            return content  # Return original content on error

        try:
            response = json.loads(stdout_content)
        except json.JSONDecodeError:
            # Try to clean up escaped characters
            import re

            cleaned_content = re.sub(r"\\{2,}", r"\\", stdout_content)
            try:
                response = json.loads(cleaned_content)
            except json.JSONDecodeError:
                console.print(
                    f"[yellow]Warning: Could not parse pipeline stage response as JSON: {stdout_content[:200]}...[/yellow]"
                )
                return content  # Return original content on error

        # Handle different output formats
        output_format = stage_config.get("output_format", "markdown")
        if output_format == "json":
            return response  # Return parsed JSON
        else:
            return response.get("generated_content", content)  # Return text content

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error in {stage_name} stage: {e}[/red]")
        return content  # Return original content on error
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing {stage_name} response: {e}[/red]")
        return content  # Return original content on error


def load_jsonl_jobs(jobs_file: Path) -> List[Dict[str, Any]]:
    """Load jobs from a JSONL file."""
    jobs = []
    try:
        with open(jobs_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        job = json.loads(line)
                        jobs.append(job)
                    except json.JSONDecodeError as e:
                        console.print(
                            f"[yellow]Warning: Invalid JSON on line {line_num} of {jobs_file}: {e}[/yellow]"
                        )
    except Exception as e:
        console.print(f"[red]Error reading jobs file {jobs_file}: {e}[/red]")
        return []

    console.print(f"[green]Loaded {len(jobs)} jobs from {jobs_file}[/green]")
    return jobs


def run_job(job: Dict[str, Any], key: str = None, topk: int = None) -> str:
    """Execute a single job using lc_ask.py."""
    # Create temporary job file
    tmpdir = ROOT / "generated" / "_tmp_jobs"
    tmpdir.mkdir(parents=True, exist_ok=True)

    job_path = tmpdir / f"job_{int(time.time() * 1000)}.json"
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)

    # Build command
    cmd = [
        sys.executable,
        str(ROOT / "src/langchain/lc_ask.py"),
        "--json",
        str(job_path),
    ]

    if key:
        cmd.extend(["--key", key])
    if topk:
        cmd.extend(["--k", str(topk)])

    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Job execution failed: {e}[/red]")
        console.print(f"[red]STDERR: {e.stderr}[/red]")
        return ""
    finally:
        # Clean up temporary file
        if job_path.exists():
            job_path.unlink()


def generate_content_from_jobs(
    jobs: List[Dict[str, Any]], subsection_id: str, key: str = None, topk: int = None
) -> List[Dict[str, Any]]:
    """Generate content variations from JSONL jobs."""
    console.print(f"[blue]Generating content from {len(jobs)} jobs...[/blue]")

    generated_content = []
    gen_dir = ROOT / "generated" / subsection_id
    gen_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs, 1):
        console.print(f"[dim]Processing job {i}/{len(jobs)}...[/dim]")

        content = run_job(job, key, topk)

        if content:
            # Save individual generation
            output_file = gen_dir / "02d"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content + "\n")

            generated_content.append(
                {
                    "job_index": i,
                    "job": job,
                    "generated_content": content,
                    "file_path": str(output_file),
                    "status": "success",
                }
            )
        else:
            generated_content.append(
                {
                    "job_index": i,
                    "job": job,
                    "generated_content": "",
                    "status": "failed",
                }
            )

    successful = len([g for g in generated_content if g["status"] == "success"])
    console.print(
        f"[green]Generated {successful}/{len(jobs)} successful content variations[/green]"
    )

    return generated_content


def score_content_variations(
    variations: List[Dict[str, Any]],
    critique_config: Dict,
    context: Dict[str, str],
    subsection_id: str,
) -> List[Dict[str, Any]]:
    """Score content variations using critique prompts."""
    console.print(f"[blue]Scoring {len(variations)} content variations...[/blue]")

    scored_variations = []

    for i, variation in enumerate(variations, 1):
        if variation.get("status") != "success" or not variation.get(
            "generated_content"
        ):
            # Skip failed variations
            scored_variations.append(
                {
                    **variation,
                    "score": 0.0,
                    "critique": "Failed variation - no content to evaluate",
                }
            )
            continue

        content = variation["generated_content"]
        console.print(f"[dim]Scoring variation {i}/{len(variations)}...[/dim]")

        # Run critique
        critique_result = run_pipeline_stage(
            f"critique_{i}", critique_config, content, context, subsection_id
        )

        # Extract score from critique result
        score = 0.0
        if isinstance(critique_result, dict):
            # JSON format critique
            score = float(critique_result.get("score", 0))
        else:
            # Try to extract score from text
            import re

            score_match = re.search(r'"score"\s*:\s*([0-9.]+)', critique_result)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    pass

        scored_variations.append(
            {**variation, "score": score, "critique": critique_result}
        )

    # Sort by score (highest first)
    scored_variations.sort(key=lambda x: x["score"], reverse=True)

    successful_scores = [v for v in scored_variations if v["score"] > 0]
    if successful_scores:
        avg_score = sum(v["score"] for v in successful_scores) / len(successful_scores)
        console.print(f"[green]Average score: {avg_score:.1f}/10[/green]")
        console.print(
            f"[green]Best score: {successful_scores[0]['score']:.1f}/10[/green]"
        )

    return scored_variations


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Remove common stop words for better comparison
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
    }
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 and not words2:
        return 1.0  # Both empty = identical

    if not words1 or not words2:
        return 0.0  # One empty = completely different

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def select_top_variations(
    scored_variations: List[Dict[str, Any]],
    top_n: int = 3,
    similarity_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Select top-N variations based on scores, with Jaccard similarity de-duplication."""
    console.print(
        f"[blue]Selecting top {top_n} variations (similarity threshold: {similarity_threshold})...[/blue]"
    )

    # Start with highest-scored variations
    selected = []

    for variation in scored_variations:
        if len(selected) >= top_n:
            break

        if variation["score"] <= 0:
            continue

        content = variation["generated_content"]

        # Check similarity against already selected variations
        is_too_similar = False
        for selected_variation in selected:
            similarity = jaccard_similarity(
                content, selected_variation["generated_content"]
            )
            if similarity >= similarity_threshold:
                console.print(
                    f"[dim]Skipping variation (similarity: {similarity:.2f} >= {similarity_threshold})[/dim]"
                )
                is_too_similar = True
                break

        if not is_too_similar:
            selected.append(variation)
            console.print(
                f"[dim]Selected variation (score: {variation['score']:.1f})[/dim]"
            )

    if len(selected) < top_n:
        console.print(
            f"[yellow]Only found {len(selected)} non-similar variations (requested {top_n})[/yellow]"
        )

    console.print(f"[green]Selected {len(selected)} variations for merging[/green]")
    return selected


def run_advanced_pipeline(
    variations: List[Dict[str, Any]],
    context: Dict[str, str],
    pipeline_config: Dict,
    subsection_id: str,
) -> Dict[str, Any]:
    """Run the complete multi-stage pipeline with scoring and selection."""
    pipeline_stages = pipeline_config.get("stages", {})
    parameters = pipeline_config.get("parameters", {})

    # Get pipeline parameters with defaults
    top_n = parameters.get("top_n_variations", 3)
    similarity_threshold = parameters.get("similarity_threshold", 0.85)

    # Collect all successful content variations
    content_parts = []
    successful_variations = []
    for variation in variations:
        if variation.get("status") == "success" and variation.get("generated_content"):
            content_parts.append(variation["generated_content"])
            successful_variations.append(variation)

    if not content_parts:
        return {"error": "No successful content variations found for this section."}

    # Execute each pipeline stage in order
    stage_results = {}
    current_content = "\n\n---\n\n".join(content_parts)

    for stage_name, stage_config in pipeline_stages.items():
        console.print(f"[blue]Executing stage: {stage_name}[/blue]")

        if stage_name == "critique":
            # Critique and scoring stage
            scored_variations = score_content_variations(
                successful_variations, stage_config, context, subsection_id
            )

            # Select top variations for merging using configured parameters
            top_variations = select_top_variations(
                scored_variations, top_n, similarity_threshold
            )

            # Update content_parts to use only top variations
            content_parts = [v["generated_content"] for v in top_variations]
            current_content = "\n\n---\n\n".join(content_parts)

            stage_results[stage_name] = {
                "all_scores": scored_variations,
                "selected_variations": top_variations,
                "parameters": {
                    "top_n": top_n,
                    "similarity_threshold": similarity_threshold,
                },
            }

        elif stage_name == "merge":
            # Merge stage - combine selected content
            current_content = run_pipeline_stage(
                stage_name, stage_config, current_content, context, subsection_id
            )
            stage_results[stage_name] = current_content

        elif stage_name in ["style", "images"]:
            # Other stages - process the current content
            current_content = run_pipeline_stage(
                stage_name, stage_config, current_content, context, subsection_id
            )
            stage_results[stage_name] = current_content

    return {
        "final_content": current_content,
        "stage_results": stage_results,
        "original_variations": len(successful_variations),
        "selected_variations": (
            len(content_parts)
            if "critique" in pipeline_stages
            else len(successful_variations)
        ),
        "pipeline_parameters": parameters,
    }


def merge_section_content(
    variations: List[Dict[str, Any]], context: Dict[str, str], system_prompt: str
) -> str:
    """Merge all variations for a section using AI (simple/single-stage version)."""
    # Collect all successful content variations
    content_parts = []
    for variation in variations:
        if variation.get("status") == "success" and variation.get("generated_content"):
            content_parts.append(variation["generated_content"])

    if not content_parts:
        return "No successful content variations found for this section."

    # Create the merging prompt
    chapter = context["chapter"]
    section = context["section"]
    subsection = context["subsection"]

    merge_instruction = f"""
Chapter: {chapter}
Section: {section}
Subsection: {subsection}

Please merge the following content variations into a single, cohesive subsection that fits within the context of "{chapter}" > "{section}". The merged content should:

1. Include all unique concepts from the individual variations
2. Flow naturally as a single piece of writing
3. Maintain consistency in tone and style
4. Be appropriate for the book's audience and purpose
5. Eliminate redundancy while preserving important details

Content variations to merge:
"""

    for i, content in enumerate(content_parts, 1):
        merge_instruction += f"\n--- Variation {i} ---\n{content}\n"

    # Call lc_ask.py with the selected system prompt
    cmd = [
        sys.executable,
        str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type",
        "pure_research",
        "--task",
        system_prompt,
        merge_instruction,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=True
        )

        # Parse the response with better error handling
        stdout_content = result.stdout.strip()
        if not stdout_content:
            console.print(
                "[yellow]Warning: Empty response from pipeline stage subprocess[/yellow]"
            )
            return content  # Return original content on error

        try:
            response = json.loads(stdout_content)
        except json.JSONDecodeError:
            # Try to clean up escaped characters
            import re

            cleaned_content = re.sub(r"\\{2,}", r"\\", stdout_content)
            try:
                response = json.loads(cleaned_content)
            except json.JSONDecodeError:
                console.print(
                    f"[yellow]Warning: Could not parse pipeline stage response as JSON: {stdout_content[:200]}...[/yellow]"
                )
                return content  # Return original content on error
        return response.get("generated_content", "Failed to generate merged content.")
    except subprocess.CalledProcessError as e:
        return f"Error during merging: {e}"
    except json.JSONDecodeError as e:
        return f"Error parsing response: {e}"


def save_merged_results(
    merged_sections: Dict[str, Dict[str, Any]], context: Dict[str, str], merge_type: str
):
    """Save merged results to a timestamped file."""
    timestamp = int(time.time())

    # Determine if any sections used advanced pipeline
    has_advanced_pipeline = any(
        section_data.get("pipeline_type") == "advanced"
        for section_data in merged_sections.values()
    )

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "chapter": context["chapter"],
            "section": context["section"],
            "subsection": context["subsection"],
            "merge_tool": "lc_merge_runner",
            "merge_type": merge_type,
            "pipeline_type": "advanced" if has_advanced_pipeline else "simple",
        },
        "sections": merged_sections,
    }

    output_dir = ROOT / "output" / "merged"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"merged_content_{timestamp}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        console.print()
        console.print(
            Panel(
                f"[green]✓ Merged content saved![/green]\n"
                f"[dim]File: {output_file}[/dim]\n"
                f"[dim]Timestamp: {timestamp}[/dim]\n"
                f"[dim]Pipeline: {'Advanced' if has_advanced_pipeline else 'Simple'}[/dim]",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]Error saving merged results: {e}[/red]")
        sys.exit(1)


def main():
    """Main application flow with command-line argument support."""
    # Parse command line arguments
    import argparse

    ap = argparse.ArgumentParser(
        description="Advanced merge runner for content processing with multi-stage pipelines"
    )
    ap.add_argument("--sub", help="Subsection ID (e.g., 1A1) for job file processing")
    ap.add_argument("--jobs", help="Path to JSONL jobs file")
    ap.add_argument("--key", help="Collection key for lc_ask")
    ap.add_argument("--k", type=int, help="Retriever top-k for lc_ask")
    ap.add_argument(
        "--batch-only",
        action="store_true",
        help="Force use of batch results only (skip job file prompts)",
    )
    ap.add_argument("--chapter", help="Chapter title for context")
    ap.add_argument("--section", help="Section title for context")
    ap.add_argument("--subsection", help="Subsection title for context")
    args = ap.parse_args()

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Advanced LangChain Merge Runner[/bold cyan]\n"
            "[dim]Multi-stage content processing with YAML-driven pipelines[/dim]",
            border_style="cyan",
        )
    )

    # Load batch results (always available as fallback)
    console.print()
    console.print("[dim]Loading batch results...[/dim]")
    try:
        sections = load_batch_results()
    except SystemExit:
        # If no batch results and we're in batch-only mode, exit gracefully
        if args.batch_only:
            console.print(
                "[yellow]No batch results found. Nothing to process.[/yellow]"
            )
            return
        else:
            # Re-raise for interactive mode
            raise

    # Load available merge types
    merge_types = load_merge_types()

    # Select merge type (use default in batch-only mode)
    if args.batch_only:
        selected_merge_type = list(merge_types.keys())[
            0
        ]  # Use first available merge type
        console.print(
            f"[dim]Batch-only mode: Using default merge type '{selected_merge_type}'[/dim]"
        )
    else:
        selected_merge_type = select_merge_type(merge_types)
    merge_config = merge_types[selected_merge_type]

    console.print()
    console.print(f"[green]Selected merge type: {selected_merge_type}[/green]")
    console.print(f"[dim]{merge_config['description']}[/dim]")

    # Determine processing mode
    use_job_processing = False

    if args.batch_only:
        # Force batch-only mode
        console.print("[yellow]Using batch results only (--batch-only flag)[/yellow]")
        use_job_processing = False
    elif args.jobs or args.sub:
        # Command line job specification
        use_job_processing = True
        if args.jobs:
            job_file = Path(args.jobs)
        else:
            job_file = ROOT / "data_jobs" / f"{args.sub}.jsonl"
    else:
        # Interactive mode - let user choose
        if sections:
            console.print(
                "[dim]Batch results available. You can also process job files.[/dim]"
            )
        else:
            console.print(
                "[yellow]No batch results found. Job file processing required.[/yellow]"
            )
            use_job_processing = True

    # Handle job file processing mode
    if use_job_processing:
        console.print()
        console.print("[green]Job File Processing Mode[/green]")

        if not (args.jobs or args.sub):
            # Interactive job file selection
            job_file = select_job_file()
            if job_file is False:
                console.print("[red]Job file processing cancelled. Exiting.[/red]")
                return
            elif job_file is None:
                subsection_id = Prompt.ask(
                    "[cyan]Subsection ID (e.g., 1A1)[/cyan]"
                ).strip()
                job_file = ROOT / "data_jobs" / f"{subsection_id}.jsonl"
        else:
            # Use command line specified job file
            job_file = (
                job_file
                if "job_file" in locals()
                else ROOT / "data_jobs" / f"{args.sub}.jsonl"
            )

        if not job_file.exists():
            console.print(f"[red]Job file not found: {job_file}[/red]")
            return

        # Load and validate job file
        jobs = load_jsonl_jobs(job_file)
        if not jobs:
            console.print("[red]No valid jobs found in the file.[/red]")
            return

        # Get context information (use command line args if provided, otherwise prompt)
        if args.chapter and args.section and args.subsection:
            context = {
                "chapter": args.chapter,
                "section": args.section,
                "subsection": args.subsection,
            }
            console.print(
                f"[dim]Using provided context: {args.chapter} > {args.section} > {args.subsection}[/dim]"
            )
        elif args.batch_only:
            context = {
                "chapter": "Chapter",
                "section": "Section",
                "subsection": "Subsection",
            }
            console.print("[dim]Batch-only mode: Using default context[/dim]")
        else:
            context = get_context_info()

        # Set processing parameters
        subsection_id = job_file.stem
        key = args.key
        topk = args.k

        # Generate content from jobs
        variations = generate_content_from_jobs(jobs, subsection_id, key, topk)

        # Process the generated content through pipeline
        console.print()
        console.print(
            f"[bold blue]Processing generated content for: {subsection_id}[/bold blue]"
        )

        merged_sections = {}

        if merge_config.get("stages"):
            console.print(
                f"[green]Using advanced pipeline with stages: {', '.join(merge_config['stages'].keys())}[/green]"
            )

            pipeline_result = run_advanced_pipeline(
                variations, context, merge_config, subsection_id
            )

            merged_content = pipeline_result.get(
                "final_content", "Pipeline execution failed"
            )
            stage_results = pipeline_result.get("stage_results", {})

            merged_sections[subsection_id] = {
                "original_variations": len(variations),
                "merged_content": merged_content,
                "stage_results": stage_results,
                "context": context,
                "merge_type": selected_merge_type,
                "pipeline_type": "advanced",
                "source": "job_file",
                "job_file": str(job_file),
                "jobs_processed": len(jobs),
            }
        else:
            # Use simple single-stage merge
            selected_prompt = merge_config.get("system_prompt", "You are an editor...")
            merged_content = merge_section_content(variations, context, selected_prompt)

            merged_sections[subsection_id] = {
                "original_variations": len(variations),
                "merged_content": merged_content,
                "context": context,
                "merge_type": selected_merge_type,
                "pipeline_type": "simple",
                "source": "job_file",
                "job_file": str(job_file),
                "jobs_processed": len(jobs),
            }

        # Display preview of merged content
        preview = (
            merged_content[:200] + "..."
            if len(merged_content) > 200
            else merged_content
        )
        console.print(f"[dim]Merged content preview:[/dim] {preview}")

    else:
        # Batch processing mode
        if not sections:
            console.print(
                "[red]No batch results found. Please run lc-batch first or specify job files.[/red]"
            )
            return

        # Display available sections
        display_sections(sections)

        # Select sections to merge
        if args.batch_only:
            # In batch-only mode, process all sections automatically
            selected_sections = list(sections.keys())
            console.print(
                f"[green]Batch-only mode: Processing all {len(selected_sections)} sections[/green]"
            )
        else:
            selected_sections = select_sections(sections)

        # Filter sections based on selection
        sections_to_process = {name: sections[name] for name in selected_sections}

        if not sections_to_process:
            console.print("[yellow]No sections selected for processing.[/yellow]")
            return

        console.print()
        console.print(
            f"[green]Selected {len(selected_sections)} section(s) for merging:[/green]"
        )
        for section in selected_sections:
            console.print(f"[dim]• {section}[/dim]")
        console.print()

        # Get context information (use command line args if provided, otherwise prompt)
        if args.chapter and args.section and args.subsection:
            context = {
                "chapter": args.chapter,
                "section": args.section,
                "subsection": args.subsection,
            }
            console.print(
                f"[dim]Using provided context: {args.chapter} > {args.section} > {args.subsection}[/dim]"
            )
        elif args.batch_only:
            context = {
                "chapter": "Chapter",
                "section": "Section",
                "subsection": "Subsection",
            }
            console.print("[dim]Batch-only mode: Using default context[/dim]")
        else:
            context = get_context_info()

        # Check if this is a multi-stage pipeline
        is_advanced_pipeline = merge_config.get("stages") is not None

        # Process sections using existing batch results
        merged_sections = {}

        for section_name, variations in sorted(sections_to_process.items()):
            console.print()
            console.print(f"[bold blue]Processing section: {section_name}[/bold blue]")

            if is_advanced_pipeline:
                # Use advanced multi-stage pipeline
                console.print(
                    f"[green]Using advanced pipeline with stages: {', '.join(merge_config['stages'].keys())}[/green]"
                )

                pipeline_result = run_advanced_pipeline(
                    variations, context, merge_config, section_name
                )

                merged_content = pipeline_result.get(
                    "final_content", "Pipeline execution failed"
                )
                stage_results = pipeline_result.get("stage_results", {})

                merged_sections[section_name] = {
                    "original_variations": len(variations),
                    "merged_content": merged_content,
                    "stage_results": stage_results,
                    "context": context,
                    "merge_type": selected_merge_type,
                    "pipeline_type": "advanced",
                    "source": "batch_results",
                }
            else:
                # Use simple single-stage merge
                selected_prompt = merge_config.get(
                    "system_prompt", "You are an editor..."
                )
                merged_content = merge_section_content(
                    variations, context, selected_prompt
                )

                merged_sections[section_name] = {
                    "original_variations": len(variations),
                    "merged_content": merged_content,
                    "context": context,
                    "merge_type": selected_merge_type,
                    "pipeline_type": "simple",
                    "source": "batch_results",
                }

            # Display preview of merged content
            preview = (
                merged_content[:200] + "..."
                if len(merged_content) > 200
                else merged_content
            )
            console.print(f"[dim]Merged content preview:[/dim] {preview}")

    # Save results
    save_merged_results(merged_sections, context, selected_merge_type)

    console.print()
    console.print("[green]All sections have been successfully merged![/green]")


if __name__ == "__main__":
    main()
