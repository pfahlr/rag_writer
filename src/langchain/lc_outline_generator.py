#!/usr/bin/env python3
"""
LangChain Outline Generator - Interactive book outline creation

Generates detailed book outlines using LangChain's indexed knowledge:
1. Prompts user for comprehensive book details
2. Allows selection of outline depth (3-5 levels)
3. Uses indexed content to generate intelligent outlines
4. Outputs JSON format compatible with lc_book_runner.py

Supports both interactive and programmatic usage.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text
project_base = os.getcwd()
print(project_base)
exit(0)

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

console = Console()

@dataclass
class BookDetails:
    """Comprehensive book information collected from user."""
    title: str
    topic: str
    description: str
    target_audience: str
    author_expertise: str
    word_count_target: int
    key_objectives: List[str]
    special_considerations: List[str]
    outline_depth: int

def collect_book_details() -> BookDetails:
    """Interactive collection of comprehensive book details."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LangChain Outline Generator[/bold cyan]\n"
        "[dim]Create intelligent book outlines from indexed knowledge[/dim]",
        border_style="cyan"
    ))

    console.print("\n[bold green]📚 Book Information[/bold green]")

    # Basic information
    title = Prompt.ask("[cyan]Book title[/cyan]").strip()
    topic = Prompt.ask("[cyan]Primary topic/subject[/cyan]").strip()

    console.print("\n[dim]Provide a detailed description of what the book will cover:[/dim]")
    description = Prompt.ask("[cyan]Book description[/cyan]").strip()

    # Audience and expertise
    target_audience = Prompt.ask("[cyan]Target audience[/cyan]", default="General readers").strip()
    author_expertise = Prompt.ask("[cyan]Author expertise level[/cyan]", default="Intermediate").strip()

    # Word count target
    word_count_target = IntPrompt.ask("[cyan]Target word count[/cyan]", default=50000)

    # Key objectives
    console.print("\n[dim]What are the 3-5 main objectives of this book? (press Enter after each)[/dim]")
    key_objectives = []
    for i in range(5):
        objective = Prompt.ask(f"[cyan]Objective {i+1}[/cyan]", default="")
        if objective.strip():
            key_objectives.append(objective.strip())
        else:
            break

    # Special considerations
    console.print("\n[dim]Any special considerations? (press Enter after each)[/dim]")
    special_considerations = []
    for i in range(3):
        consideration = Prompt.ask(f"[cyan]Consideration {i+1}[/cyan]", default="")
        if consideration.strip():
            special_considerations.append(consideration.strip())
        else:
            break

    # Outline depth selection
    console.print("\n[bold green]📋 Outline Configuration[/bold green]")
    console.print("Select outline depth (number of hierarchical levels):")
    console.print("• [cyan]3 levels[/cyan]: Chapter > Section > Subsection")
    console.print("• [cyan]4 levels[/cyan]: Chapter > Section > Subsection > Sub-subsection")
    console.print("• [cyan]5 levels[/cyan]: Chapter > Section > Subsection > Sub-subsection > Sub-sub-subsection")

    outline_depth = IntPrompt.ask("[cyan]Outline depth[/cyan]", choices=["3", "4", "5"], default=4)

    return BookDetails(
        title=title,
        topic=topic,
        description=description,
        target_audience=target_audience,
        author_expertise=author_expertise,
        word_count_target=word_count_target,
        key_objectives=key_objectives,
        special_considerations=special_considerations,
        outline_depth=outline_depth
    )

def generate_outline_prompt(book_details: BookDetails) -> str:
    """Generate a comprehensive prompt for outline creation."""

    depth_descriptions = {
        3: "3-level hierarchy: Chapters > Sections > Subsections",
        4: "4-level hierarchy: Chapters > Sections > Subsections > Sub-subsections",
        5: "5-level hierarchy: Chapters > Sections > Subsections > Sub-subsections > Sub-sub-subsections"
    }

    prompt = f"""
You are an expert book outline creator with extensive knowledge of educational content and publishing best practices.

Create a comprehensive, well-structured outline for a book with the following specifications:

**Book Details:**
- Title: {book_details.title}
- Topic: {book_details.topic}
- Description: {book_details.description}
- Target Audience: {book_details.target_audience}
- Author Expertise: {book_details.author_expertise}
- Target Word Count: {book_details.word_count_target:,} words

**Key Objectives:**
"""

    for i, objective in enumerate(book_details.key_objectives, 1):
        prompt += f"{i}. {objective}\n"

    if book_details.special_considerations:
        prompt += "\n**Special Considerations:**\n"
        for consideration in book_details.special_considerations:
            prompt += f"• {consideration}\n"

    prompt += f"""
**Outline Requirements:**
- Structure: {depth_descriptions[book_details.outline_depth]}
- Depth: {book_details.outline_depth} hierarchical levels
- Logical flow from foundational to advanced concepts
- Balanced chapter lengths (aim for {book_details.word_count_target // 8:,} - {book_details.word_count_target // 6:,} words per chapter)
- Practical, actionable content suitable for {book_details.target_audience}
- Clear learning progression throughout the book

**Output Format:**
Return a JSON object with this exact structure:
{{
  "chapters": [
    {{
      "number": 1,
      "title": "Chapter Title",
      "description": "Brief chapter description",
      "estimated_words": 5000,
      "sections": [
        {{
          "letter": "A",
          "title": "Section Title",
          "description": "Brief section description",
          "estimated_words": 2000,
          "subsections": [
            {{
              "number": 1,
              "title": "Subsection Title",
              "description": "Brief subsection description",
              "estimated_words": 1000
"""

    if book_details.outline_depth >= 4:
        prompt += """,
              "subsubsections": [
                {
                  "letter": "a",
                  "title": "Sub-subsection Title",
                  "description": "Brief sub-subsection description",
                  "estimated_words": 500
                }
              ]"""

    if book_details.outline_depth >= 5:
        prompt += """,
                "subsubsubsections": [
                  {
                    "number": 1,
                    "title": "Sub-sub-subsection Title",
                    "description": "Brief sub-sub-subsection description",
                    "estimated_words": 250
                  }
                ]"""

    prompt += """
            }
          ]
        }
      ]
    }
  ]
}

Ensure the outline is comprehensive, logically structured, and suitable for a {book_details.word_count_target:,}-word book aimed at {book_details.target_audience}.
"""

    return prompt

def generate_outline_with_langchain(book_details: BookDetails) -> Dict[str, Any]:
    """Generate outline using LangChain's indexed knowledge."""
    console.print(f"\n[blue]Generating {book_details.outline_depth}-level outline using indexed knowledge...[/blue]")

    # Create the outline generation prompt
    prompt = generate_outline_prompt(book_details)

    # Use lc_ask.py to generate the outline
    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_ask.py"),
        "ask",
        "--content-type", "pure_research",
        "--task", prompt
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the JSON response with better error handling
        stdout_content = result.stdout.strip()
        if not stdout_content:
            console.print("[yellow]⚠ Empty response from subprocess, using fallback structure[/yellow]")
            return create_fallback_outline(book_details)

        try:
            response = json.loads(stdout_content)
        except json.JSONDecodeError:
            # Try to clean up escaped characters
            import re
            cleaned_content = re.sub(r'\\{2,}', r'\\', stdout_content)
            try:
                response = json.loads(cleaned_content)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                json_match = re.search(r'\{.*\}', stdout_content, re.DOTALL)
                if json_match:
                    try:
                        response = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        console.print(f"[yellow]⚠ Could not parse JSON response, trying to extract from text[/yellow]")
                        response = {"generated_content": stdout_content}
                else:
                    console.print(f"[yellow]⚠ No JSON found in response, using raw content[/yellow]")
                    response = {"generated_content": stdout_content}

        outline_data = response.get('generated_content', '')

        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_start = outline_data.find('{')
            json_end = outline_data.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = outline_data[json_start:json_end]
                outline = json.loads(json_str)
                console.print("[green]✓ Outline generated successfully[/green]")
                return outline
            else:
                console.print("[yellow]⚠ Could not extract JSON from response, using fallback structure[/yellow]")
                return create_fallback_outline(book_details)

        except json.JSONDecodeError as e:
            console.print(f"[yellow]⚠ JSON parsing failed: {e}, using fallback structure[/yellow]")
            return create_fallback_outline(book_details)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Outline generation failed: {e}[/red]")
        console.print(f"[red]STDERR: {e.stderr}[/red]")
        return create_fallback_outline(book_details)

def create_fallback_outline(book_details: BookDetails) -> Dict[str, Any]:
    """Create a basic fallback outline when AI generation fails."""
    console.print("[yellow]Creating fallback outline structure...[/yellow]")

    chapters = []
    words_per_chapter = book_details.word_count_target // 6  # Assume 6 chapters

    for i in range(1, 7):
        chapter = {
            "number": i,
            "title": f"Chapter {i}: {book_details.topic} - Part {i}",
            "description": f"Comprehensive coverage of {book_details.topic} concepts and applications",
            "estimated_words": words_per_chapter,
            "sections": []
        }

        # Add sections based on depth
        for j, section_letter in enumerate(['A', 'B', 'C'], 1):
            section = {
                "letter": section_letter,
                "title": f"Section {section_letter}: Key Concepts and Foundations",
                "description": f"Essential {book_details.topic} concepts for {book_details.target_audience}",
                "estimated_words": words_per_chapter // 3,
                "subsections": []
            }

            # Add subsections
            for k in range(1, 4):
                subsection = {
                    "number": k,
                    "title": f"Subsection {k}: Practical Applications",
                    "description": f"Hands-on {book_details.topic} applications and examples",
                    "estimated_words": words_per_chapter // 9
                }

                if book_details.outline_depth >= 4:
                    subsection["subsubsections"] = [
                        {
                            "letter": "a",
                            "title": "Advanced Concepts",
                            "description": "Deep dive into complex topics",
                            "estimated_words": words_per_chapter // 18
                        }
                    ]

                if book_details.outline_depth >= 5:
                    if "subsubsections" in subsection:
                        subsection["subsubsections"][0]["subsubsubsections"] = [
                            {
                                "number": 1,
                                "title": "Expert Insights",
                                "description": "Advanced practitioner knowledge",
                                "estimated_words": words_per_chapter // 36
                            }
                        ]

                section["subsections"].append(subsection)

            chapter["sections"].append(section)
        chapters.append(chapter)

    return {"chapters": chapters}

def convert_outline_to_book_structure(outline: Dict[str, Any], book_details: BookDetails) -> Dict[str, Any]:
    """Convert generated outline to book structure format compatible with lc_book_runner.py."""
    console.print("[blue]Converting outline to book structure format...[/blue]")

    sections = []
    section_counter = 1

    for chapter in outline.get("chapters", []):
        chapter_num = chapter["number"]

        for section in chapter.get("sections", []):
            section_letter = section["letter"]

            for subsection in section.get("subsections", []):
                subsection_num = subsection["number"]

                # Create subsection entry
                subsection_id = f"{chapter_num}{section_letter}{subsection_num}"

                section_entry = {
                    "subsection_id": subsection_id,
                    "title": subsection["title"],
                    "job_file": f"data_jobs/{subsection_id}.jsonl",
                    "batch_params": {
                        "key": book_details.topic.lower().replace(" ", "_"),
                        "k": 5
                    },
                    "merge_params": {
                        "key": book_details.topic.lower().replace(" ", "_"),
                        "k": 3
                    },
                    "dependencies": []
                }

                # Add sub-subsections if they exist
                if book_details.outline_depth >= 4:
                    for subsubsection in subsection.get("subsubsections", []):
                        subsubsection_id = f"{subsection_id}{subsubsection['letter']}"
                        subsubsection_entry = {
                            "subsection_id": subsubsection_id,
                            "title": subsubsection["title"],
                            "job_file": f"data_jobs/{subsubsection_id}.jsonl",
                            "batch_params": {
                                "key": book_details.topic.lower().replace(" ", "_"),
                                "k": 5
                            },
                            "merge_params": {
                                "key": book_details.topic.lower().replace(" ", "_"),
                                "k": 3
                            },
                            "dependencies": [subsection_id]
                        }
                        sections.append(subsubsection_entry)

                        # Add sub-sub-subsections if they exist
                        if book_details.outline_depth >= 5:
                            for subsubsubsection in subsubsection.get("subsubsubsections", []):
                                subsubsubsection_id = f"{subsubsection_id}{subsubsubsection['number']}"
                                subsubsubsection_entry = {
                                    "subsection_id": subsubsubsection_id,
                                    "title": subsubsubsection["title"],
                                    "job_file": f"data_jobs/{subsubsubsection_id}.jsonl",
                                    "batch_params": {
                                        "key": book_details.topic.lower().replace(" ", "_"),
                                        "k": 5
                                    },
                                    "merge_params": {
                                        "key": book_details.topic.lower().replace(" ", "_"),
                                        "k": 3
                                    },
                                    "dependencies": [subsubsection_id]
                                }
                                sections.append(subsubsubsection_entry)

                sections.append(section_entry)
                section_counter += 1

    # Create the final book structure
    book_structure = {
        "title": book_details.title,
        "metadata": {
            "author": "AI Content Generator",
            "version": "1.0",
            "target_audience": book_details.target_audience,
            "word_count_target": book_details.word_count_target,
            "created_date": time.strftime('%Y-%m-%d'),
            "description": book_details.description,
            "topic": book_details.topic,
            "author_expertise": book_details.author_expertise,
            "key_objectives": book_details.key_objectives,
            "special_considerations": book_details.special_considerations,
            "outline_depth": book_details.outline_depth
        },
        "sections": sections
    }

    return book_structure

def save_book_structure(book_structure: Dict[str, Any], output_file: Optional[Path] = None) -> Path:
    """Save the book structure to a JSON file."""
    if output_file is None:
        timestamp = int(time.time())
        safe_title = book_structure["title"].replace(" ", "_").replace("/", "_")
        output_file = ROOT / "outlines" / "book_structures" / f"{safe_title}_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(book_structure, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Book structure saved to: {output_file}[/green]")
    return output_file

def display_outline_summary(book_structure: Dict[str, Any], book_details: BookDetails):
    """Display a summary of the generated outline."""
    console.print("\n[bold green]📋 Outline Summary[/bold green]")

    table = Table()
    table.add_column("Level", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="center")
    table.add_column("Description", style="white")

    total_sections = len(book_structure["sections"])
    chapters = len(set(s["subsection_id"][0] for s in book_structure["sections"]))
    sections = len(set(s["subsection_id"][:2] for s in book_structure["sections"]))
    subsections = len(set(s["subsection_id"][:3] for s in book_structure["sections"]))

    table.add_row("Chapters", str(chapters), "Major book divisions")
    table.add_row("Sections", str(sections), "Chapter subdivisions")
    table.add_row("Subsections", str(subsections), "Detailed topic areas")

    if book_details.outline_depth >= 4:
        subsubsections = len([s for s in book_structure["sections"] if len(s["subsection_id"]) >= 4])
        table.add_row("Sub-subsections", str(subsubsections), "Granular topics")

    if book_details.outline_depth >= 5:
        subsubsubsections = len([s for s in book_structure["sections"] if len(s["subsection_id"]) >= 5])
        table.add_row("Sub-sub-subsections", str(subsubsubsections), "Expert-level details")

    table.add_row("Total Sections", str(total_sections), "All outline items")

    console.print(table)

    # Word count estimate
    total_words = sum(s.get("estimated_words", 0) for s in book_structure["sections"])
    console.print(f"\n[dim]Estimated total words: {total_words:,}[/dim]")
    console.print(f"[dim]Target word count: {book_details.word_count_target:,}[/dim]")

def main():
    """Main outline generation workflow."""
    import argparse
    ap = argparse.ArgumentParser(description="Interactive book outline generator using LangChain")
    ap.add_argument("--output", help="Output JSON file path")
    ap.add_argument("--non-interactive", action="store_true", help="Skip interactive prompts")
    args = ap.parse_args()

    try:
        # Collect book details
        if args.non_interactive:
            console.print("[red]Non-interactive mode not yet implemented[/red]")
            return

        book_details = collect_book_details()

        # Display collected information
        console.print("\n[bold green]📖 Book Details Summary[/bold green]")
        info_table = Table()
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Title", book_details.title)
        info_table.add_row("Topic", book_details.topic)
        info_table.add_row("Target Audience", book_details.target_audience)
        info_table.add_row("Author Expertise", book_details.author_expertise)
        info_table.add_row("Word Count Target", f"{book_details.word_count_target:,}")
        info_table.add_row("Outline Depth", f"{book_details.outline_depth} levels")

        console.print(info_table)

        # Confirm before proceeding
        if not Confirm.ask("\nProceed with outline generation?"):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

        # Generate outline
        outline = generate_outline_with_langchain(book_details)

        # Convert to book structure format
        book_structure = convert_outline_to_book_structure(outline, book_details)

        # Display summary
        display_outline_summary(book_structure, book_details)

        # Save the structure
        output_file = save_book_structure(book_structure, Path(args.output) if args.output else None)

        # Success message
        console.print()
        console.print(Panel(
            f"[green]🎉 Outline generation completed![/green]\n"
            f"[dim]Output: {output_file}[/dim]\n"
            f"[dim]Ready for: python src/langchain/lc_book_runner.py --book {output_file}[/dim]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()