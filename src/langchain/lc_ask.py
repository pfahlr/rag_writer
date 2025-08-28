#!/usr/bin/env python3
"""
LangChain ask CLI (hybrid FAISS + BM25, optional Flashrank reranker)

- ROOT resolved relative to repo (two levels up from this file)
- Loads FAISS index for a collection key (default="default")
- Hybrid retrieval: FAISS + BM25
- Optional reranker: Flashrank if installed
- LLM selection:
    1) langchain_openai.ChatOpenAI  (pip install langchain-openai)
    2) langchain_community.chat_models.ChatOllama (if Ollama running)
    3) raw openai client (pip install openai)
"""

import os
from pathlib import Path
import typer
import json
import re 

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.prompts import ChatPromptTemplate

# --- Optional reranker: Flashrank ---
_have_flashrank = False
_flashrank_cls = None
try:
  from langchain_community.document_transformers import FlashrankRerank
  _flashrank_cls = FlashrankRerank
  _have_flashrank = True
except Exception:
    pass

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

DEFAULT_KEY = "default"
EMBED_MODEL = "BAAI/bge-small-en"  # CPU-friendly
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a careful research assistant.\n"
    "Use ONLY the provided context\n."
    "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y).\n"
    "If the context is insufficient or conflicting, state what is missing and stop."
    "Do not cite sources you did not use. Aim to use at least two distinct sources when available.\n"
)
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
  idx_dir = ROOT / f"storage/faiss_{key}"
  if not idx_dir.exists():
    raise SystemExit(
      f"FAISS index not found for key '{key}': {idx_dir}\n"
      f"Build it first with: make lc-index {key}"
    )
  
  embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
  vs = FAISS.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
  vector = vs.as_retriever(search_kwargs={"k": k})
  
  # BM25 over same docs
  all_docs = list(vs.docstore._dict.values())
  bm25 = BM25Retriever.from_documents(all_docs); bm25.k = k
  
  base = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])
  
  # Optional reranker with Flashrank
  if _have_flashrank:
    try:
      rerank = _flashrank_cls(top_n=k)
      compressor = DocumentCompressorPipeline(transformers=[rerank])
      return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base
      )
    except Exception:
      return base
  return base

def _fmt_doc_for_context(d):
  title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
  page = d.metadata.get("page")
  header = f"[{title}, p.{page}]" if page else f"[{title}]"
  return f"{header}\n{d.page_content}"

def _format_context(docs):
  return "\n\n---\n\n".join(_fmt_doc_for_context(d) for d in docs)

def _select_backend():
  # 1) Try langchain-openai
  try:
    from langchain_openai import ChatOpenAI
    if os.getenv("OPENAI_API_KEY"):
      return ("lc_openai", ChatOpenAI(model=LLM_MODEL, temperature=0))
  except Exception:
    pass
  
  # 2) Try Ollama
  try:
    from langchain_community.chat_models import ChatOllama
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return ("ollama", ChatOllama(model=ollama_model, temperature=0))
  except Exception:
    pass
  
  # 3) Raw OpenAI
  try:
    import openai
    if os.getenv("OPENAI_API_KEY"):
      return ("raw_openai", openai.OpenAI())
  except Exception:
    pass
  
  raise SystemExit(
    "No usable LLM backend. Install one of:\n"
    "  pip install langchain-openai   (preferred)\n"
    "  pip install openai             (fallback)\n"
    "Or run Ollama locally."
  )

app = typer.Typer(add_completion=False)

@app.command()
def main(
  question: str = typer.Argument(..., help="Your question or instruction to retrieve on"),
  key: str = typer.Option(DEFAULT_KEY, "--key", "-k", help="Collection key"),
  k: int = typer.Option(15, help="Top-k to retrieve"),
  file: str = typer.Option("", help="File containing prompt question"),
  task: str = typer.Option("", help="Optional task prefix to prepend to final LLM question (excluded from retriever)")
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
    if hasattr(retriever, "get_relevant_documents"):
      docs = retriever.get_relevant_documents(instruction)
    elif hasattr(retriever, "invoke"):
      docs = retriever.invoke(instruction)
    elif hasattr(retriever, "retrieve"):
      docs = retriever.retrieve(instruction)
    else:
      # Fallback: try calling retriever as a function
      docs = retriever(instruction)
  except Exception as e:
    raise SystemExit(f"Retriever failed: {e}")

  context_text = _format_context(docs) if docs else ""

  # Build the final question for the LLM by prepending the task (if any)
  final_question = f"{final_task} {instruction}".strip() if final_task else instruction

  prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
  ])

  messages = prompt.format_messages(question=final_question, context=context_text)

  # Invoke the selected LLM backend
  if backend in ("lc_openai", "ollama"):
    resp = llm.invoke(messages)
    generated_content = resp.content
  elif backend == "raw_openai":
    # Convert LangChain messages into OpenAI API schema
    msgs = [{"role": "system" if m.type == "system" else "user", "content": m.content}
            for m in messages]
    content = llm.chat.completions.create(
      model=LLM_MODEL,
      messages=msgs,
      temperature=0,
    ).choices[0].message.content
    generated_content = content

  # Collect sources
  sources = []
  for d in docs:
    title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
    page = d.metadata.get("page")
    source_info = {
      "title": title,
      "page": page,
      "source": d.metadata.get("source")
    }
    sources.append(source_info)

  # Output as JSON
  output = {
    "generated_content": generated_content,
    "sources": sources
  }
  print(json.dumps(output, indent=2))

if __name__ == "__main__":
    app()
