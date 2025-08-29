#!/usr/bin/env python3
"""
LangChain ask CLI (hybrid FAISS + BM25, optional Flashrank reranker)

Refactored to use centralized core modules for better maintainability.

Features:
- Loads FAISS index for a collection key (default="default")
- Hybrid retrieval: FAISS + BM25 with optional reranking
- LLM selection with automatic backend detection
- Content type system for different writing styles
- Source citation extraction and deduplication
"""

import os
from pathlib import Path
import typer
import json
import re
import yaml

from langchain_core.prompts import ChatPromptTemplate

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import our new core modules
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable LangChain debug mode to prevent compatibility issues
try:
    import langchain
    # Try to disable debug mode if the attribute exists
    if hasattr(langchain, 'debug'):
        langchain.debug = False
    if hasattr(langchain, 'verbose'):
        langchain.verbose = False
    if hasattr(langchain, 'llm_cache'):
        langchain.llm_cache = None
except ImportError:
    # LangChain not available, skip
    pass
except Exception as e:
    # Any other error during LangChain configuration, skip silently
    pass

from core.retriever import RetrieverFactory, RetrieverConfig
from core.llm import LLMFactory, LLMConfig
from config.settings import get_config
from utils.error_handler import handle_and_exit, validate_collection
from utils.template_engine import render_string_template

console = Console()

# Get centralized configuration
config = get_config()

def load_content_types():
    """Load content types from individual YAML files in content_types/ subdirectory."""
    content_types_dir = config.paths.root_dir / "src/tool/prompts/content_types"
    content_types = {}

    try:
        if not content_types_dir.exists():
            console.print(f"[yellow]Warning: Content types directory not found: {content_types_dir}[/yellow]")
            return {}

        # Load each YAML file in the content_types directory
        for yaml_file in content_types_dir.glob("*.yaml"):
            if yaml_file.name == "default.yaml":
                continue  # Skip default.yaml as it's not a content type

            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content_type_data = yaml.safe_load(f) or {}

                # Use filename (without .yaml extension) as the content type key
                content_type_name = yaml_file.stem
                content_types[content_type_name] = content_type_data

            except yaml.YAMLError as e:
                console.print(f"[red]Error parsing {yaml_file.name}: {e}[/red]")
                continue
            except Exception as e:
                console.print(f"[red]Error loading {yaml_file.name}: {e}[/red]")
                continue

        return content_types

    except Exception as e:
        console.print(f"[red]Error loading content types: {e}[/red]")
        return {}

def get_system_prompt(content_type: str, context: dict = None) -> str:
    """Get system prompt for a content type with optional token replacement."""
    content_types = load_content_types()

    if content_type not in content_types:
        available_types = list(content_types.keys())
        raise ValueError(f"Unknown content type '{content_type}'. Available types: {', '.join(available_types)}")

    system_prompt_parts = content_types[content_type].get("system_prompt", [])
    if not system_prompt_parts:
        raise ValueError(f"No system prompt defined for content type '{content_type}'")

    # Join the parts and apply token replacement if context is provided
    system_prompt = "".join(system_prompt_parts)

    if context:
        system_prompt = render_string_template(system_prompt, context)

    return system_prompt

def list_content_types():
    """List all available content types with Rich formatting."""
    content_types = load_content_types()

    if not content_types:
        console.print("[red]No content types found.[/red]")
        return

    # Create a table for content types
    table = Table(title="Available Content Types")
    table.add_column("Content Type", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("System Prompt Length", style="green", justify="center")

    for name, config in sorted(content_types.items()):
        description = config.get("description", "No description available")
        # Count total characters in system prompt parts
        system_prompt_parts = config.get("system_prompt", [])
        if isinstance(system_prompt_parts, list):
            prompt_length = sum(len(str(part)) for part in system_prompt_parts)
        else:
            prompt_length = len(str(system_prompt_parts))
        table.add_row(name, description, f"{prompt_length} chars")

    # Display the table in a panel
    panel = Panel(
        table,
        title="[bold blue]Content Type Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    )

    console.print(panel)
    console.print(f"\n[dim]Use with: --content-type [cyan]<type>[/cyan][/dim]")
    console.print(f"[dim]Example: --content-type [cyan]technical_manual_writer[/cyan][/dim]")

USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Use the context below to answer. Include page-cited quotes for key claims.\n\n"
    "Context:\n{context}"
)


DOI_RE = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-z0-9]*[a-z0-9]\b', re.I)

def _norm(s: str) -> str:
    return re.sub(r'\W+', ' ', (s or '')).strip().lower()

def _extract_cited(docs, answer: str):
    """Return (cited_docs, uncited_docs) based on title/DOI matches in the answer text."""
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
            if doi in ans or any(m.group(0).lower() == doi for m in DOI_RE.finditer(ans)):
                hit = True

        if hit:
            cited.append(d)
        else:
            uncited.append(d)

    return cited, uncited

def _fmt_cited(d):
    title = d.metadata.get("title") or Path(d.metadata.get("source"," ")).stem
    page = d.metadata.get("page")
    src  = d.metadata.get("source")
    doi  = d.metadata.get("doi")
    bits = [f"{title}"]
    if page: bits.append(f"p.{page}")
    if doi: bits.append(f"doi:{doi}")
    if src: bits.append(str(src))
    return " (" + " · ".join(bits) + ")"


def make_retriever(key: str, k: int = 30):
    """Create a retriever using the centralized factory."""
    try:
        # Validate collection exists
        validate_collection(key, config.paths.storage_dir)

        # Use centralized retriever factory
        factory = RetrieverFactory(config.paths.root_dir)
        retriever_config = RetrieverConfig(
            key=key,
            k=k,
            embedding_model=config.embedding.model_name,
            openai_model=config.llm.openai_model,
            rerank_model=config.retriever.rerank_model,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            use_reranking=config.retriever.use_reranking
        )
        return factory.create_hybrid_retriever(retriever_config)
    except Exception as e:
        handle_and_exit(e, f"creating retriever for collection '{key}'")

def _fmt_doc_for_context(d):
  title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
  page = d.metadata.get("page")
  header = f"[{title}, p.{page}]" if page else f"[{title}]"
  return f"{header}\n{d.page_content}"

def _format_context(docs):
  return "\n\n---\n\n".join(_fmt_doc_for_context(d) for d in docs)

def _select_backend():
    """Select LLM backend using centralized factory."""
    try:
        llm_config = LLMConfig(
            model=config.llm.openai_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            openai_api_key=config.openai_api_key,
            ollama_model=config.llm.ollama_model
        )
        factory = LLMFactory(llm_config)
        return factory.create_llm()
    except Exception as e:
        handle_and_exit(e, "selecting LLM backend")

app = typer.Typer(add_completion=False)

@app.command()
def list_types():
    """List all available content types."""
    list_content_types()

@app.command()
def ask(
  question: str = typer.Argument(..., help="Your question or instruction to retrieve on"),
  conttype: str = typer.Option('pure_research',"--content-type", "-ct", help="The type of writing to perform"),
  key: str = typer.Option(config.rag_key, "--key", "-k", help="Collection key"),
  k: int = typer.Option(config.retriever.default_k, help="Top-k to retrieve"),
  task: str = typer.Option("", "--task", help="Optional task prefix to prepend to final LLM question (excluded from retriever)"),
  file: str = typer.Option("", "--file", help="File containing prompt question")
  ):
  """
  CLI entrypoint that supports a separate 'task' prefix which is:
   - excluded from the retriever query (used only to fetch context)
   - prepended to the final question sent to the LLM

  Behavior:
   - If --file is provided, it loads JSON and looks for 'instruction' (used for retrieval)
     and optional 'task' (used as prefix). CLI --task overrides file 'task'.
   - Otherwise, the positional 'question' argument is used as the retrieval instruction,
     and optional --task is prepended only to the final LLM question.
  """
  try:
    backend, llm = _select_backend()
    retriever = make_retriever(key, k=k)

    # Determine retrieval instruction and task prefix
    if file:
      with open(file, "r", encoding="utf-8") as f:
        directive = json.load(f)
      instruction = (directive.get("instruction") or directive.get("question") or "").strip()
      file_task = (directive.get("task") or "").strip()
      final_task = task.strip() or file_task
    else:
      instruction = question.strip()
      final_task = task.strip()

    # Retrieve documents using ONLY the instruction (task is excluded)
    docs = []
    try:
      # Try invoke first (newer LangChain method)
      if hasattr(retriever, "invoke"):
        docs = retriever.invoke(instruction)
      elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(instruction)
      elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(instruction)
      else:
        # Fallback: try calling retriever as a function
        docs = retriever(instruction)
    except Exception as e:
      handle_and_exit(e, "retrieving documents")

    context_text = _format_context(docs) if docs else ""

    # Build the final question for the LLM by prepending the task (if any)
    final_question = f"{final_task} {instruction}".strip() if final_task else instruction

    # Get system prompt from YAML configuration
    try:
      system_prompt = get_system_prompt(conttype)
    except ValueError as e:
      handle_and_exit(e, f"loading content type '{conttype}'")

    prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", USER_PROMPT),
    ])

    messages = prompt.format_messages(question=final_question, context=context_text)

    # Invoke the selected LLM backend
    try:
      if backend in ("lc_openai", "ollama"):
        resp = llm.invoke(messages)
        generated_content = resp.content
      elif backend == "raw_openai":
        # Convert LangChain messages into OpenAI API schema
        msgs = [{"role": "system" if m.type == "system" else "user", "content": m.content}
                for m in messages]
        content = llm.chat.completions.create(
          model=config.llm.openai_model,
          messages=msgs,
          temperature=config.llm.temperature,
        ).choices[0].message.content
        generated_content = content
      else:
        raise ValueError(f"Unsupported backend: {backend}")
    except Exception as e:
      handle_and_exit(e, "generating content with LLM")

    # Filter sources to only include those referenced in the generated content
    cited_docs, _ = _extract_cited(docs, generated_content)

    # Collect only the cited sources, deduplicated by article (not by page)
    sources = []
    seen_articles = set()

    for d in cited_docs:
      title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
      source_path = d.metadata.get("source", "")

      # Use title or source path as the unique identifier for the article
      article_id = title if title else source_path

      # Only add if we haven't seen this article before
      if article_id and article_id not in seen_articles:
        seen_articles.add(article_id)
        source_info = {
          "title": title,
          "source": source_path
        }
        sources.append(source_info)

    # Output as JSON
    output = {
      "generated_content": generated_content,
      "sources": sources
    }
    print(json.dumps(output, indent=2))

  except Exception as e:
    handle_and_exit(e, "processing ask request")

if __name__ == "__main__":
    app()
