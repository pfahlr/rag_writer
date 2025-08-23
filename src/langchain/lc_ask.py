#!/usr/bin/env python3
"""
LangChain ask CLI (hybrid FAISS + BM25, optional reranker, LCEL-only)

- ROOT is resolved relative to repo (two levels up from this file)
- Loads FAISS index for a collection key (default="default")
- Hybrid retrieval: FAISS + BM25, optional MultiQuery (if you want later)
- Optional reranker: Flashrank if installed; otherwise none
- Uses LCEL-style prompt -> llm without create_* helpers (avoids 0.2.x import issues)
"""

import os
from pathlib import Path
import typer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.prompts import ChatPromptTemplate

# Optional reranker: Flashrank
_have_flashrank = False
_flashrank_cls = None
try:
    from langchain_community.document_transformers import FlashrankRerank
    _flashrank_cls = FlashrankRerank
    _have_flashrank = True
except Exception:
    _flashrank_cls = None
    _have_flashrank = False

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from llama_index.llms.openai import OpenAI  # swap to your provider if desired

app = typer.Typer(add_completion=False)

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

DEFAULT_KEY = "default"
EMBED_MODEL = "BAAI/bge-small-en"  # CPU-friendly
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a careful research assistant. Use ONLY the provided context.\n"
    "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y).\n"
    "If the context is insufficient or conflicting, state what is missing and stop."
)
USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Use the context below to answer. Include page-cited quotes for key claims.\n\n"
    "Context:\n{context}"
)

def make_retriever(key: str, k: int = 10):
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
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k

    base = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])

    # Optional reranker with Flashrank as a compression stage
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
    else:
        return base

def _fmt_doc_for_context(d):
    title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
    page = d.metadata.get("page")
    header = f"[{title}, p.{page}]" if page else f"[{title}]"
    return f"{header}\n{d.page_content}"

def _format_context(docs):
    # Compact but keeps per-doc headers for easier citation
    return "\n\n---\n\n".join(_fmt_doc_for_context(d) for d in docs)

@app.command()
def main(
    question: str = typer.Argument(..., help="Your question"),
    key: str = typer.Option(DEFAULT_KEY, "--key", "-k", help="Collection key"),
    k: int = typer.Option(10, help="Top-k to retrieve")
):
    if not USE_OPENAI:
        raise SystemExit("No OPENAI_API_KEY set and no local LLM configured.")

    retriever = make_retriever(key, k=k)
    # Pull documents (so we can both pass context and print sources deterministically)
    docs = retriever.invoke(question)
    context_text = _format_context(docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    messages = prompt.format_messages(question=question, context=context_text)
    resp = llm.invoke(messages)

    print(resp.content)

    # Show sources
    print("\nSOURCES:")
    for d in docs:
        title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
        page = d.metadata.get("page")
        print(f"- {title} (p.{page}) :: {d.metadata.get('source')}")

if __name__ == "__main__":
    app()
