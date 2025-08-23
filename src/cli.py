## src/tool/cli.py
#!/usr/bin/env python3
"""
Unified RAG CLI + Interactive Shell (env-configurable, multi-collection)
- ROOT path from RAG_ROOT; collection key from RAG_KEY or --key in shell
- Dynamic presets from YAML
- NEW: Playbook-driven interactive inputs using Jinja2 templating
"""
import os, sys, json
from pathlib import Path
import typer, yaml
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from jinja2 import Environment, BaseLoader

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

app = typer.Typer(add_completion=False)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
ROOT = Path(root_dir)

DEFAULT_KEY = os.getenv("RAG_KEY", "default")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)

COMMON_SYSTEM = (
    "You are a careful research assistant. Use ONLY the provided context. "
    "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y). "
    "If context is insufficient or conflicting, state what is missing and stop."
)


def load_playbooks(playbooks_path: Path):
    """Load playbooks from YAML file with comprehensive error handling."""
    try:
        if not playbooks_path.exists():
            print(f"[yellow]Warning: Playbooks file not found: {playbooks_path}[/]")
            return {}
            
        try:
            with playbooks_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"[red]Error parsing playbooks YAML: {str(e)}[/]")
            return {}
        except UnicodeDecodeError as e:
            print(f"[red]Error reading playbooks file (encoding issue): {str(e)}[/]")
            return {}
        except Exception as e:
            print(f"[red]Error reading playbooks file: {str(e)}[/]")
            return {}
            
        # Validate and set defaults for each playbook
        for name, pb in data.items():
            if not isinstance(pb, dict):
                print(f"[yellow]Warning: Invalid playbook format for '{name}', skipping[/]")
                continue
                
            pb.setdefault("label", name)
            pb.setdefault("description", "")
            pb.setdefault("system_prompt", COMMON_SYSTEM)
            pb.setdefault("steps", [])
            pb.setdefault("stitch_final", True)
            pb.setdefault("inputs", [])
            
            # Validate steps
            if not isinstance(pb["steps"], list):
                print(f"[yellow]Warning: Invalid steps format for playbook '{name}', using empty list[/]")
                pb["steps"] = []
                
            for step in pb["steps"]:
                if isinstance(step, dict):
                    step.setdefault("inputs", [])
                else:
                    print(f"[yellow]Warning: Invalid step format in playbook '{name}', skipping[/]")
                    
        return data
        
    except Exception as e:
        print(f"[red]Unexpected error loading playbooks: {str(e)}[/]")
        return {}


def _paths(key: str):
    faiss_dir = ROOT / f"storage/faiss_{key}"
    playbooks = ROOT / "src/tool/prompts/playbooks.yaml"
    return faiss_dir, playbooks


def _load_retriever(key: str, k: int = 10, multiquery: bool = True):
    """Load and configure retriever with comprehensive error handling."""
    try:
        faiss_dir, _ = _paths(key)
        
        # Check if FAISS directory exists
        if not faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {faiss_dir}")
        
        # Load embeddings model
        try:
            emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{EMBED_MODEL}': {str(e)}")
        
        # Load FAISS index
        try:
            vs = FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {faiss_dir}: {str(e)}")
        
        vector = vs.as_retriever(search_kwargs={"k": k})

        # Create BM25 retriever
        try:
            docs = list(vs.docstore._dict.values())
            if not docs:
                raise ValueError("No documents found in the index")
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = k
        except Exception as e:
            raise RuntimeError(f"Failed to create BM25 retriever: {str(e)}")

        # Create hybrid retriever
        hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])
        
        # Add multi-query if available
        if multiquery and os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
            try:
                llm_for_multi = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
                hybrid = MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm_for_multi)
            except Exception as e:
                print(f"[yellow]Warning: Multi-query retriever failed, falling back to hybrid: {str(e)}[/]")

        # Add reranking
        try:
            rerank = FlashrankRerank(top_n=k)
            compressor = DocumentCompressorPipeline(transformers=[rerank])
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid)
        except Exception as e:
            print(f"[yellow]Warning: Reranking failed, falling back to hybrid retriever: {str(e)}[/]")
            return hybrid
            
    except Exception as e:
        raise RuntimeError(f"Failed to load retriever for collection '{key}': {str(e)}")


def _llm():
    """Load LLM with comprehensive error handling."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        if ChatOpenAI is None:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        try:
            return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
            
    except Exception as e:
        raise RuntimeError(f"LLM initialization failed: {str(e)}. Set OPENAI_API_KEY or add a local LLM.")


def _rag_answer(key: str, question: str, system_prompt: str, k: int = 10) -> dict:
    """Generate RAG answer with comprehensive error handling."""
    try:
        retriever = _load_retriever(key, k=k)
        llm = _llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question:\n{question}\n\nReturn a grounded, citation-rich answer."),
        ])
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        return rag_chain.invoke({"question": question})
    except FileNotFoundError:
        return {
            "error": f"Collection '{key}' not found. Run 'make lc-index {key}' to create it.",
            "answer": "Failed to load collection."
        }
    except Exception as e:
        return {
            "error": f"RAG generation failed: {str(e)}",
            "answer": "An error occurred while generating the answer."
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
            # simple comma-separated selection
            prompt_text = f"{label} (choices: {', '.join(map(str, choices))})"
        else:
            prompt_text = label
            
        try:
            ans = Prompt.ask(prompt_text, default=str(default) if default is not None else None)
            
            if multi and isinstance(ans, str):
                vals = [a.strip() for a in ans.split(",") if a.strip()]
            else:
                vals = ans
                
            # Validate choices if specified
            if choices and vals not in choices and (not multi or not all(v in choices for v in vals)):
                print(f"[red]Invalid choice. Must be one of: {', '.join(map(str, choices))}[/]")
                continue
                
            # cast type with better error handling
            try:
                if type_ == "int":
                    vals = [int(v) for v in vals] if isinstance(vals, list) else int(vals)
                elif type_ == "float":
                    vals = [float(v) for v in vals] if isinstance(vals, list) else float(vals)
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
    try:
        return env.from_string(template_str).render(**context)
    except Exception as e:
        return template_str  # fall back if templating fails


def _validate_collection(key: str) -> bool:
    """Validate that the collection exists and is accessible."""
    try:
        faiss_dir, _ = _paths(key)
        if not faiss_dir.exists():
            print(f"[red]Error: Collection '{key}' not found at {faiss_dir}[/]")
            print(f"[yellow]To create this collection, run: make lc-index {key}[/]")
            return False
            
        # Try to load a small retriever to test
        try:
            test_retriever = _load_retriever(key, k=1, multiquery=False)
            return True
        except Exception as e:
            print(f"[red]Error: Collection '{key}' exists but cannot be loaded: {str(e)}[/]")
            print(f"[yellow]Try rebuilding the index: make lc-index {key}[/]")
            return False
            
    except Exception as e:
        print(f"[red]Error validating collection '{key}': {str(e)}[/]")
        return False


def _display_error_with_suggestions(error_msg: str, key: str = None):
    """Display error message with helpful suggestions."""
    print(f"\n[red]Error: {error_msg}[/]")
    
    # Provide specific suggestions based on error type
    if "Collection" in error_msg or "FAISS" in error_msg:
        if key:
            print(f"[yellow]Suggestions:[/]")
            print(f"  • Run 'make lc-index {key}' to create the collection")
            print(f"  • Check if PDFs exist in data_raw/ directory")
            print(f"  • Verify the collection key '{key}' is correct")
        else:
            print(f"[yellow]Suggestions:[/]")
            print(f"  • Run 'make lc-index <key>' to create a collection")
            print(f"  • Check if PDFs exist in data_raw/ directory")
            
    elif "OPENAI_API_KEY" in error_msg:
        print(f"[yellow]Suggestions:[/]")
        print(f"  • Set OPENAI_API_KEY environment variable")
        print(f"  • Check your .env file or environment configuration")
        print(f"  • Verify your OpenAI API key is valid")
        
    elif "embedding model" in error_msg.lower():
        print(f"[yellow]Suggestions:[/]")
        print(f"  • Check your internet connection")
        print(f"  • Verify the EMBED_MODEL setting")
        print(f"  • Try a different embedding model")
        
    print()  # Add spacing


def _safe_metadata_access(obj, key: str, default: str = "Unknown"):
    """Safely access metadata with fallback."""
    try:
        if hasattr(obj, 'metadata') and obj.metadata:
            return obj.metadata.get(key, default)
        return default
    except Exception:
        return default


def _graceful_shutdown():
    """Handle graceful shutdown with cleanup."""
    print("\n[yellow]Shutting down gracefully...[/]")
    # Add any cleanup code here if needed
    print("Goodbye!")


@app.command()
def shell(key: str = typer.Option(DEFAULT_KEY, "--key", "-k", help="Collection key (e.g., llms_education)")):
    """Interactive shell using the specified collection key."""
    try:
        faiss_dir, playbooks_path = _paths(key)
        
        # Validate collection before starting
        if not _validate_collection(key):
            print(f"\n[yellow]Shell will start but collection '{key}' may not work properly.[/]")
            print("[yellow]You can still use 'help' and 'presets' commands.[/]\n")
        
        playbooks = load_playbooks(playbooks_path)

        base_cmds = ["help", "ask", "compare", "summarize", "outline", "sources", "presets", "preset", "quit"]
        completer = WordCompleter(base_cmds + list(playbooks.keys()), ignore_case=True)

        session = PromptSession()
        last_result = None

        banner = Panel(f"[bold cyan]RAG Tool Shell[/]\nROOT: {ROOT}\nKEY: {key}\nIndex: {faiss_dir}")
        print(banner)
        
        if not _validate_collection(key):
            print(f"[yellow]⚠️  Collection '{key}' has issues. Some commands may fail.[/]\n")

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

            if cmd in {"quit", "exit"}: break

            if cmd == "help":
                tbl = Table(show_header=False)
                tbl.add_row("ask <q>", "General RAG answer with citations")
                tbl.add_row("compare <topic>", "Contrast positions/methods/results across sources")
                tbl.add_row("summarize <topic>", "High-level summary with quotes")
                tbl.add_row("outline <topic>", "Book/essay outline with evidence bullets")
                tbl.add_row("presets", "List dynamic presets from playbooks.yaml")
                tbl.add_row("preset <name> [topic]", "Run a guided multi-step preset (with interactive inputs)")
                tbl.add_row("sources", "Show sources from last answer")
                tbl.add_row("quit", "Exit")
                print(tbl); continue

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
                                    src = _safe_metadata_access(d, "source", "Unknown source")
                                    print(f"- {title} (p.{page}) :: {src}")
                                except Exception as e:
                                    print(f"- Source {i+1}: Error displaying metadata - {str(e)}")
                except Exception as e:
                    print(f"[red]Error displaying sources: {str(e)}[/]")
                continue

            if cmd == "presets":
                tbl = Table(title=f"Presets (key={key})")
                tbl.add_column("Name"); tbl.add_column("Label"); tbl.add_column("Description")
                for name, pb in playbooks.items():
                    tbl.add_row(name, pb.get("label",""), pb.get("description",""))
                print(tbl); continue

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
                        print(f"[yellow]Warning: Playbook '{name}' has no steps defined[/]")
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
                                context = _collect_inputs(step.get("inputs", []), context)
                            except Exception as e:
                                print(f"[yellow]Warning: Error collecting step inputs: {str(e)}[/]")
                                
                            step_prompt_raw = step.get("prompt", "")
                            if not step_prompt_raw:
                                print(f"[yellow]Warning: Step '{step_name}' has no prompt, skipping[/]")
                                continue
                                
                            step_prompt = _render(step_prompt_raw, {**context, "previous": previous})
                            composed = f"{topic}\n\n{step_prompt}\n\nUse earlier results if helpful:\n{json.dumps(previous, ensure_ascii=False)[:2000]}"
                            
                            result = _rag_answer(key, composed, system_prompt)
                            
                            if "error" in result:
                                print(f"[red]Error in step '{step_name}': {result['error']}[/]")
                                break
                                
                            text = result.get("answer", "No answer generated")
                            aggregate.append({"step": step_name, "output": text})
                            previous[step_name] = text
                            print(Panel.fit(text, title=f"{name} :: {step_name}", border_style="green"))
                            
                        except Exception as e:
                            print(f"[red]Error executing step '{step_name}': {str(e)}[/]")
                            break
                            
                    if aggregate and pb.get("stitch_final", True):
                        try:
                            final_prompt = pb.get("final_prompt", "Synthesize the step outputs into a single, well-structured deliverable with citations. Avoid duplication.")
                            final_prompt = _render(final_prompt, {**context, "previous": previous})
                            composed = f"{topic}\n\n{final_prompt}\n\n{json.dumps(aggregate, ensure_ascii=False)[:4000]}"
                            
                            last_result = _rag_answer(key, composed, system_prompt)
                            
                            if "error" in last_result:
                                _display_error_with_suggestions(last_result['error'], key)
                                last_result = {"answer": aggregate[-1]["output"], "context": []}
                            else:
                                print(Panel.fit(last_result["answer"], title=f"{name} :: final", border_style="cyan"))
                                
                        except Exception as e:
                            print(f"[red]Error in final synthesis: {str(e)}[/]")
                            last_result = {"answer": aggregate[-1]["output"], "context": []}
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
                        user_suffix = ("\n\nDecompose into: claims, methods/evidence, limitations, agreements, disagreements. "
                                        "Return a table where possible and include page-cited quotes.")
                    elif cmd == "summarize":
                        user_suffix = "\n\nProvide a layered summary (1 paragraph → bullets → key quotes)."
                    elif cmd == "outline":
                        user_suffix = ("\n\nDraft a 10–14 chapter outline with 2–3 bullets each, each bullet with page-cited evidence.")
                        
                    q = f"{question}{user_suffix}"
                    last_result = _rag_answer(key, q, COMMON_SYSTEM)
                    
                    if "error" in last_result:
                        _display_error_with_suggestions(last_result['error'], key)
                    else:
                        print("\n" + last_result.get("answer", "No answer generated"))
                        print("\n[dim]Type 'sources' to list retrieved source chunks.[/dim]")
                        
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


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        print("\n[yellow]Application interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[red]Critical application error: {str(e)}[/]")
        print("[red]Please check your configuration and try again.[/]")
        sys.exit(1)
