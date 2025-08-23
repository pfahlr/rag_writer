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
from langchain.retrievers.document_compressors import DocumentReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import SentenceTransformerRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

app = typer.Typer(add_completion=False)

ROOT = Path(os.getenv("RAG_ROOT", "/run/host/var/srv/IOMEGA_EXTERNAL/rag"))
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
    if not playbooks_path.exists():
        return {}
    with playbooks_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for name, pb in data.items():
        pb.setdefault("label", name)
        pb.setdefault("description", "")
        pb.setdefault("system_prompt", COMMON_SYSTEM)
        pb.setdefault("steps", [])
        pb.setdefault("stitch_final", True)
        pb.setdefault("inputs", [])  # NEW: preset-level inputs
        for step in pb["steps"]:
            step.setdefault("inputs", [])  # NEW: step-level inputs
    return data


def _paths(key: str):
    faiss_dir = ROOT / f"storage/faiss_{key}"
    playbooks = ROOT / "src/tool/prompts/playbooks.yaml"
    return faiss_dir, playbooks


def _load_retriever(key: str, k: int = 10, multiquery: bool = True):
    faiss_dir, _ = _paths(key)
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)
    vector = vs.as_retriever(search_kwargs={"k": k})

    docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k

    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])
    if multiquery and os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
        llm_for_multi = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
        hybrid = MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm_for_multi)

    rerank = SentenceTransformerRerank(model_name="BAAI/bge-reranker-base", top_n=k)
    compressor = DocumentReranker(rerank)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid)


def _llm():
    if os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    raise SystemExit("No OPENAI_API_KEY set and no local LLM wired. Set a key or add a local LLM.")


def _rag_answer(key: str, question: str, system_prompt: str, k: int = 10) -> dict:
    retriever = _load_retriever(key, k=k)
    llm = _llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question:
{question}

Return a grounded, citation-rich answer."),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain.invoke({"question": question})


def _collect_inputs(inputs_spec, context):
    """Ask user for variables as defined in inputs_spec. Update and return context dict."""
    for inp in inputs_spec:
        name = inp.get("name")
        if not name:
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
        ans = Prompt.ask(prompt_text, default=str(default) if default is not None else None)
        if multi and isinstance(ans, str):
            vals = [a.strip() for a in ans.split(",") if a.strip()]
        else:
            vals = ans
        # cast type
        try:
            if type_ == "int":
                vals = [int(v) for v in vals] if isinstance(vals, list) else int(vals)
            elif type_ == "float":
                vals = [float(v) for v in vals] if isinstance(vals, list) else float(vals)
        except Exception:
            pass
        context[name] = vals
    return context


def _render(template_str: str, context: dict) -> str:
    try:
        return env.from_string(template_str).render(**context)
    except Exception as e:
        return template_str  # fall back if templating fails


@app.command()
def shell(key: str = typer.Option(DEFAULT_KEY, "--key", "-k", help="Collection key (e.g., llms_education)")):
    """Interactive shell using the specified collection key."""
    faiss_dir, playbooks_path = _paths(key)
    playbooks = load_playbooks(playbooks_path)

    base_cmds = ["help", "ask", "compare", "summarize", "outline", "sources", "presets", "preset", "quit"]
    completer = WordCompleter(base_cmds + list(playbooks.keys()), ignore_case=True)

    session = PromptSession()
    last_result = None

    banner = Panel(f"[bold cyan]RAG Tool Shell[/]
ROOT: {ROOT}
KEY: {key}
Index: {faiss_dir}")
    print(banner)

    while True:
        try:
            cmdline = session.prompt("rag> ", completer=completer).strip()
        except (EOFError, KeyboardInterrupt):
            print("
Bye.")
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
            if last_result is None:
                print("No result yet.")
            else:
                for d in last_result.get("context", []):
                    title = d.metadata.get("title"); page = d.metadata.get("page"); src = d.metadata.get("source")
                    print(f"- {title} (p.{page}) :: {src}")
            continue

        if cmd == "presets":
            tbl = Table(title=f"Presets (key={key})")
            tbl.add_column("Name"); tbl.add_column("Label"); tbl.add_column("Description")
            for name, pb in playbooks.items():
                tbl.add_row(name, pb.get("label",""), pb.get("description",""))
            print(tbl); continue

        if cmd == "preset":
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
            context = {"topic": topic}
            # Preset-level inputs
            context = _collect_inputs(pb.get("inputs", []), context)
            previous = {}
            aggregate = []
            for i, step in enumerate(steps, start=1):
                step_name = step.get("name", f"step_{i}")
                # Step-level interactive inputs
                context = _collect_inputs(step.get("inputs", []), context)
                step_prompt_raw = step.get("prompt", "")
                step_prompt = _render(step_prompt_raw, {**context, "previous": previous})
                composed = f"{topic}

{step_prompt}

Use earlier results if helpful:
{json.dumps(previous, ensure_ascii=False)[:2000]}"
                result = _rag_answer(key, composed, system_prompt)
                text = result["answer"]
                aggregate.append({"step": step_name, "output": text})
                previous[step_name] = text
                print(Panel.fit(text, title=f"{name} :: {step_name}", border_style="green"))
            if pb.get("stitch_final", True):
                final_prompt = pb.get("final_prompt", "Synthesize the step outputs into a single, well-structured deliverable with citations. Avoid duplication.")
                final_prompt = _render(final_prompt, {**context, "previous": previous})
                composed = f"{topic}

{final_prompt}

{json.dumps(aggregate, ensure_ascii=False)[:4000]}"
                last_result = _rag_answer(key, composed, system_prompt)
                print(Panel.fit(last_result["answer"], title=f"{name} :: final", border_style="cyan"))
            else:
                last_result = {"answer": aggregate[-1]["output"], "context": []}
            continue

        if cmd in {"ask", "compare", "summarize", "outline"}:
            question = arg or Prompt.ask("Enter topic/question")
            user_suffix = ""
            if cmd == "compare":
                user_suffix = ("

Decompose into: claims, methods/evidence, limitations, agreements, disagreements. "
                                "Return a table where possible and include page-cited quotes.")
            elif cmd == "summarize":
                user_suffix = "

Provide a layered summary (1 paragraph → bullets → key quotes)."
            elif cmd == "outline":
                user_suffix = ("

Draft a 10–14 chapter outline with 2–3 bullets each, each bullet with page-cited evidence.")
            q = f"{question}{user_suffix}"
            last_result = _rag_answer(key, q, COMMON_SYSTEM)
            print("
" + last_result["answer"])
            print("
[dim]Type 'sources' to list retrieved source chunks.[/dim]")
            continue

        print(f"Unknown command: {cmd}. Type 'help'.")


if __name__ == "__main__":
    app()
