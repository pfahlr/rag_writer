#!/usr/bin/env python3
"""
Interactive Shell for RAG Writer

This module contains the interactive shell functionality extracted from the
monolithic cli.py file for better maintainability and separation of concerns.
"""

import json
from typing import Optional

from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from jinja2 import Environment, BaseLoader

from ..core.retriever import RetrieverFactory, RetrieverConfig
from ..core.llm import LLMFactory, LLMConfig
from ..config.settings import get_config
from ..utils.error_handler import handle_and_exit, validate_collection


# Global configuration
config = get_config()
env = Environment(
    loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True
)

COMMON_SYSTEM = (
    "You are a careful research assistant. Use ONLY the provided context. "
    "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y). "
    "If context is insufficient or conflicting, state what is missing and stop."
)


def _paths(key: str) -> tuple:
    """Get paths for a given collection key."""
    faiss_dir = config.paths.storage_dir / f"faiss_{key}"
    playbooks = config.paths.root_dir / "src/config/content/prompts/playbooks.yaml"
    return faiss_dir, playbooks


def _load_retriever(key: str, k: int = 10, multiquery: bool = True):
    """Load and configure retriever with comprehensive error handling."""
    try:
        factory = RetrieverFactory(config.paths.root_dir)
        retriever_config = RetrieverConfig(
            key=key,
            k=k,
            multiquery=multiquery,
            embedding_model=config.embedding.model_name,
            openai_model=config.llm.openai_model,
            rerank_model=config.retriever.rerank_model,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
        )
        return factory.create_hybrid_retriever(retriever_config)
    except Exception as e:
        handle_and_exit(e, f"loading retriever for collection '{key}'")


def _llm():
    """Load LLM with comprehensive error handling."""
    try:
        llm_config = LLMConfig(
            model=config.llm.openai_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            openai_api_key=config.openai_api_key,
            ollama_model=config.llm.ollama_model,
        )
        factory = LLMFactory(llm_config)
        return factory.create_llm()
    except Exception as e:
        handle_and_exit(e, "initializing LLM")


def _rag_answer(
    key: str,
    retrieval_query: str,
    system_prompt: str,
    final_question: Optional[str] = None,
    k: int = 10,
) -> dict:
    """Generate a RAG answer while allowing separation of retrieval_query and final_question."""
    try:
        retriever = _load_retriever(key, k=k)
    except FileNotFoundError:
        return {
            "error": f"Collection '{key}' not found. Run 'make lc-index {key}' to create it.",
            "answer": "Failed to load collection.",
            "context": [],
        }
    except Exception as e:
        return {
            "error": f"Failed to load retriever: {str(e)}",
            "answer": "Failed to load retriever.",
            "context": [],
        }

    # Fetch documents using ONLY the retrieval_query
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(retrieval_query)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(retrieval_query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(retrieval_query)
        else:
            docs = retriever(retrieval_query)
    except Exception as e:
        return {
            "error": f"Retriever failed: {str(e)}",
            "answer": "Failed to retrieve documents.",
            "context": [],
        }

    # Initialize LLM
    try:
        backend, llm = _llm()
    except Exception as e:
        return {
            "error": f"LLM initialization failed: {str(e)}",
            "answer": "Failed to initialize LLM.",
            "context": docs,
        }

    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\nReturn a grounded, citation-rich answer.",
            ),
        ]
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # Final question passed to the LLM (task may be prepended)
    q = (
        final_question
        if final_question is not None and final_question != ""
        else retrieval_query
    )

    try:
        out = doc_chain.invoke({"input_documents": docs, "question": q})
        # Normalize output into a string answer
        if isinstance(out, dict):
            if "answer" in out:
                answer = out["answer"]
            elif "output" in out:
                answer = out["output"]
            else:
                val = next((v for v in out.values() if isinstance(v, str)), None)
                answer = val if val is not None else str(out)
        elif isinstance(out, str):
            answer = out
        else:
            answer = str(out)
        return {"answer": answer, "context": docs}
    except Exception as e:
        return {
            "error": f"RAG generation failed: {str(e)}",
            "answer": "An error occurred while generating the answer.",
            "context": docs,
        }


def _collect_inputs(inputs_spec, context):
    """Ask user for variables as defined in inputs_spec. Update and return context dict."""
    for inp in inputs_spec:
        name = inp.get("name")
        if not name:
            continue

        # Skip if already provided
        if name in context:
            continue

        label = inp.get("prompt", name)
        default = inp.get("default")
        type_ = (inp.get("type") or "str").lower()
        choices = inp.get("choices")
        multi = inp.get("multi", False)

        if choices:
            prompt_text = f"{label} (choices: {', '.join(map(str, choices))})"
        else:
            prompt_text = label

        try:
            ans = Prompt.ask(
                prompt_text, default=str(default) if default is not None else None
            )

            if multi and isinstance(ans, str):
                vals = [a.strip() for a in ans.split(",") if a.strip()]
            else:
                vals = ans

            # Validate choices if specified
            if (
                choices
                and vals not in choices
                and (not multi or not all(v in choices for v in vals))
            ):
                print(
                    f"[red]Invalid choice. Must be one of: {', '.join(map(str, choices))}[/]"
                )
                continue

            # cast type with better error handling
            try:
                if type_ == "int":
                    vals = (
                        [int(v) for v in vals] if isinstance(vals, list) else int(vals)
                    )
                elif type_ == "float":
                    vals = (
                        [float(v) for v in vals]
                        if isinstance(vals, list)
                        else float(vals)
                    )
            except (ValueError, TypeError) as e:
                print(f"[red]Invalid {type_} value: {ans}. Error: {str(e)}[/]")
                continue

            context[name] = vals

        except KeyboardInterrupt:
            print("\n[yellow]Input cancelled. Using default value if available.[/]")
            if default is not None:
                context[name] = default
            break
        except Exception as e:
            print(f"[red]Input error: {str(e)}[/]")
            if default is not None:
                context[name] = default

    return context


def _render(template_str: str, context: dict) -> str:
    """Render a Jinja2 template with the given context."""
    try:
        return env.from_string(template_str).render(**context)
    except Exception:
        return template_str  # fall back if templating fails


def _validate_collection(key: str) -> bool:
    """Validate that the collection exists and is accessible."""
    try:
        validate_collection(key, config.paths.storage_dir)
        return True
    except Exception as e:
        print(
            f"[red]Error: Collection '{key}' exists but cannot be loaded: {str(e)}[/]"
        )
        print(f"[yellow]Try rebuilding the index: make lc-index {key}[/]")
        return False


def _display_error_with_suggestions(error_msg: str, key: str = None):
    """Display error message with helpful suggestions."""
    print(f"\n[red]Error: {error_msg}[/]")

    if "Collection" in error_msg or "FAISS" in error_msg:
        if key:
            print("[yellow]Suggestions:[/]")
            print(f"  • Run 'make lc-index {key}' to create the collection")
            print("  • Check if PDFs exist in data_raw/ directory")
            print(f"  • Verify the collection key '{key}' is correct")
        else:
            print("[yellow]Suggestions:[/]")
            print("  • Run 'make lc-index <key>' to create a collection")
            print("  • Check if PDFs exist in data_raw/ directory")

    elif "OPENAI_API_KEY" in error_msg:
        print("[yellow]Suggestions:[/]")
        print("  • Set OPENAI_API_KEY environment variable")
        print("  • Check your .env file or environment configuration")
        print("  • Verify your OpenAI API key is valid")

    elif "embedding model" in error_msg.lower():
        print("[yellow]Suggestions:[/]")
        print("  • Check your internet connection")
        print("  • Verify the EMBED_MODEL setting")
        print("  • Try a different embedding model")

    print()  # Add spacing


def _safe_metadata_access(obj, key: str, default: str = "Unknown"):
    """Safely access metadata with fallback."""
    try:
        if hasattr(obj, "metadata") and obj.metadata:
            return obj.metadata.get(key, default)
        return default
    except Exception:
        return default


def _graceful_shutdown():
    """Handle graceful shutdown with cleanup."""
    print("\n[yellow]Shutting down gracefully...[/]")
    # Add any cleanup code here if needed
    print("Goodbye!")


def shell(key: str = None):
    """Interactive shell using the specified collection key."""
    if key is None:
        key = config.rag_key

    try:
        faiss_dir, playbooks_path = _paths(key)

        # Validate collection before starting
        if not _validate_collection(key):
            print(
                f"\n[yellow]Shell will start but collection '{key}' may not work properly.[/]"
            )
            print("[yellow]You can still use 'help' and 'presets' commands.[/]\n")

        # Load playbooks
        try:
            import yaml

            if playbooks_path.exists():
                with playbooks_path.open("r", encoding="utf-8") as f:
                    playbooks = yaml.safe_load(f) or {}
            else:
                playbooks = {}
        except Exception as e:
            print(f"[yellow]Warning: Could not load playbooks: {e}[/]")
            playbooks = {}

        base_cmds = [
            "help",
            "ask",
            "compare",
            "summarize",
            "outline",
            "sources",
            "presets",
            "preset",
            "quit",
        ]
        completer = WordCompleter(base_cmds + list(playbooks.keys()), ignore_case=True)

        session = PromptSession()
        last_result = None

        banner = Panel(
            f"[bold cyan]RAG Tool Shell[/]\nROOT: {config.paths.root_dir}\nKEY: {key}\nIndex: {faiss_dir}"
        )
        print(banner)

        if not _validate_collection(key):
            print(
                f"[yellow]⚠️  Collection '{key}' has issues. Some commands may fail.[/]\n"
            )

        while True:
            try:
                cmdline = session.prompt("rag> ", completer=completer).strip()
            except (EOFError, KeyboardInterrupt):
                _graceful_shutdown()
                break
            if not cmdline:
                continue
            parts = cmdline.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in {"quit", "exit"}:
                break

            if cmd == "help":
                tbl = Table(show_header=False)
                tbl.add_row("ask <q>", "General RAG answer with citations")
                tbl.add_row(
                    "compare <topic>",
                    "Contrast positions/methods/results across sources",
                )
                tbl.add_row("summarize <topic>", "High-level summary with quotes")
                tbl.add_row(
                    "outline <topic>", "Book/essay outline with evidence bullets"
                )
                tbl.add_row("presets", "List dynamic presets from playbooks.yaml")
                tbl.add_row(
                    "preset <name> [topic]",
                    "Run a guided multi-step preset (with interactive inputs)",
                )
                tbl.add_row("sources", "Show sources from last answer")
                tbl.add_row("quit", "Exit")
                print(tbl)
                continue

            if cmd == "sources":
                try:
                    if last_result is None:
                        print("No result yet.")
                    else:
                        context = last_result.get("context", [])
                        if not context:
                            print("No sources found in the last result.")
                        else:
                            for i, d in enumerate(context):
                                try:
                                    title = _safe_metadata_access(d, "title", "Unknown")
                                    page = _safe_metadata_access(d, "page", "?")
                                    src = _safe_metadata_access(
                                        d, "source", "Unknown source"
                                    )
                                    print(f"- {title} (p.{page}) :: {src}")
                                except Exception as e:
                                    print(
                                        f"- Source {i+1}: Error displaying metadata - {str(e)}"
                                    )
                except Exception as e:
                    print(f"[red]Error displaying sources: {str(e)}[/]")
                continue

            if cmd == "presets":
                tbl = Table(title=f"Presets (key={key})")
                tbl.add_column("Name")
                tbl.add_column("Label")
                tbl.add_column("Description")
                for name, pb in playbooks.items():
                    tbl.add_row(name, pb.get("label", ""), pb.get("description", ""))
                print(tbl)
                continue

            if cmd == "preset":
                try:
                    if not arg:
                        name = Prompt.ask("Preset name (type 'presets' to list)")
                        topic = Prompt.ask("Topic/Question")
                    else:
                        name, *rest = arg.split(" ", 1)
                        topic = rest[0] if rest else Prompt.ask("Topic/Question")

                    pb = playbooks.get(name)
                    if not pb:
                        print(f"[red]Unknown preset:[/] {name}")
                        continue

                    system_prompt = pb.get("system_prompt", COMMON_SYSTEM)
                    steps = pb.get("steps", [])

                    if not steps:
                        print(
                            f"[yellow]Warning: Playbook '{name}' has no steps defined[/]"
                        )
                        continue

                    context = {"topic": topic}
                    # Preset-level inputs
                    try:
                        context = _collect_inputs(pb.get("inputs", []), context)
                    except Exception as e:
                        print(f"[red]Error collecting preset inputs: {str(e)}[/]")
                        continue

                    previous = {}
                    aggregate = []

                    for i, step in enumerate(steps, start=1):
                        try:
                            step_name = step.get("name", f"step_{i}")

                            # Step-level interactive inputs
                            try:
                                context = _collect_inputs(
                                    step.get("inputs", []), context
                                )
                            except Exception as e:
                                print(
                                    f"[yellow]Warning: Error collecting step inputs: {str(e)}[/]"
                                )

                            step_prompt_raw = step.get("prompt", "")
                            if not step_prompt_raw:
                                print(
                                    f"[yellow]Warning: Step '{step_name}' has no prompt, skipping[/]"
                                )
                                continue

                            step_prompt = _render(
                                step_prompt_raw, {**context, "previous": previous}
                            )
                            composed = f"{topic}\n\n{step_prompt}\n\nUse earlier results if helpful:\n{json.dumps(previous, ensure_ascii=False)[:2000]}"

                            result = _rag_answer(key, composed, system_prompt)

                            if "error" in result:
                                print(
                                    f"[red]Error in step '{step_name}': {result['error']}[/]"
                                )
                                break

                            text = result.get("answer", "No answer generated")
                            aggregate.append({"step": step_name, "output": text})
                            previous[step_name] = text
                            print(
                                Panel.fit(
                                    text,
                                    title=f"{name} :: {step_name}",
                                    border_style="green",
                                )
                            )

                        except Exception as e:
                            print(
                                f"[red]Error executing step '{step_name}': {str(e)}[/]"
                            )
                            break

                    if aggregate and pb.get("stitch_final", True):
                        try:
                            final_prompt = pb.get(
                                "final_prompt",
                                "Synthesize the step outputs into a single, well-structured deliverable with citations. Avoid duplication.",
                            )
                            final_prompt = _render(
                                final_prompt, {**context, "previous": previous}
                            )
                            composed = f"{topic}\n\n{final_prompt}\n\n{json.dumps(aggregate, ensure_ascii=False)[:4000]}"

                            last_result = _rag_answer(key, composed, system_prompt)

                            if "error" in last_result:
                                _display_error_with_suggestions(
                                    last_result["error"], key
                                )
                                last_result = {
                                    "answer": aggregate[-1]["output"],
                                    "context": [],
                                }
                            else:
                                print(
                                    Panel.fit(
                                        last_result["answer"],
                                        title=f"{name} :: final",
                                        border_style="cyan",
                                    )
                                )

                        except Exception as e:
                            print(f"[red]Error in final synthesis: {str(e)}[/]")
                            last_result = {
                                "answer": aggregate[-1]["output"],
                                "context": [],
                            }
                    elif aggregate:
                        last_result = {"answer": aggregate[-1]["output"], "context": []}

                except KeyboardInterrupt:
                    print("\n[yellow]Preset execution cancelled by user[/]")
                except Exception as e:
                    print(f"[red]Unexpected error in preset execution: {str(e)}[/]")
                continue

            if cmd in {"ask", "compare", "summarize", "outline"}:
                try:
                    question = arg or Prompt.ask("Enter topic/question")
                    if not question.strip():
                        print("[yellow]No question provided, skipping[/]")
                        continue

                    user_suffix = ""
                    if cmd == "compare":
                        user_suffix = (
                            "\n\nDecompose into: claims, methods/evidence, limitations, agreements, disagreements. "
                            "Return a table where possible and include page-cited quotes."
                        )
                    elif cmd == "summarize":
                        user_suffix = "\n\nProvide a layered summary (1 paragraph → bullets → key quotes)."
                    elif cmd == "outline":
                        user_suffix = "\n\nDraft a 10–14 chapter outline with 2–3 bullets each, each bullet with page-cited evidence."

                    q = f"{question}{user_suffix}"
                    last_result = _rag_answer(key, q, COMMON_SYSTEM)

                    if "error" in last_result:
                        _display_error_with_suggestions(last_result["error"], key)
                    else:
                        print("\n" + last_result.get("answer", "No answer generated"))
                        print(
                            "\n[dim]Type 'sources' to list retrieved source chunks.[/dim]"
                        )

                except KeyboardInterrupt:
                    print("\n[yellow]Command cancelled by user[/]")
                except Exception as e:
                    print(f"\n[red]Unexpected error: {str(e)}[/]")
                continue

            print(f"Unknown command: {cmd}. Type 'help'.")

    except KeyboardInterrupt:
        print("\n[yellow]Shell interrupted by user[/]")
    except Exception as e:
        print(f"\n[red]Critical error in shell: {str(e)}[/]")
        print("[red]Please check your configuration and try again.[/]")
        return 1
    return 0
