#!/usr/bin/env python3
"""
LlamaIndex ask CLI (argparse version)
- Loads LanceDB vector store for a given collection key (default from RAG_KEY or "default")
- Builds a hybrid retriever (BM25 + vector) when chunk artifacts exist; degrades
  gracefully to vector-only if BM25 can't be built
- Applies cross-encoder reranking (SentenceTransformerRerank)
- Enforces grounded answers with inline page citations

Examples
--------
python src/llamaindex/ask.py -k llms_education \
  "What is the most agreed upon fact about the use of AI in the classroom? Include in-text citations."

# or using env
RAG_KEY=llms_education python src/llamaindex/ask.py "Compare positions across sources"
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI  # swap to your provider if desired

# ---------- Defaults from environment ----------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ROOT = Path(root_dir)
ENV_KEY = (os.getenv("RAG_KEY", "default") or "default").strip()
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_RERANK = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Configure global embedding model (CPU-friendly by default)
Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED)

# Configure LLM if OpenAI key is present
if os.getenv("OPENAI_API_KEY"):
    Settings.llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

PROMPT_RULES = (
    "You must answer ONLY using the retrieved context. "
    "Every claim must include inline citations like (Title, p.X–Y). "
    "If the context lacks support, explicitly say what is missing."
)


# ---------- Helpers ----------

def load_index(key: str) -> VectorStoreIndex:
  db_dir = ROOT / f"storage/lancedb_{key}"
  if not db_dir.exists():
    raise SystemExit(
      f"[ask.py] LanceDB directory not found for key '{key}': {db_dir}"
      f"Build it with: make index {key}"
    )
  vector_store = LanceDBVectorStore(uri=str(db_dir), table_name="chunks")
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def _load_chunks_for_key(key: str) -> List[TextNode]:
  """Load keyed chunk artifacts to build BM25 if index.docstore is empty.
  Falls back to unkeyed chunks.jsonl for backward compatibility.
  """
  candidates = [
    ROOT / f"data_processed/chunks_{key}.jsonl",
    ROOT / "data_processed/chunks.jsonl",
  ]
  for path in candidates:
    if path.exists():
      nodes: List[TextNode] = []
      with path.open("r", encoding="utf-8") as f:
        for line in f:
          r = json.loads(line)
          nodes.append(
            TextNode(
              text=r["text"],
              metadata={
                "doc_id": r.get("doc_id"),
                "title": r.get("title"),
                "source_path": r.get("source_path")
                or r.get("metadata", {}).get("source"),
                "page_start": r.get("page_start")
                or r.get("metadata", {}).get("page"),
                "page_end": r.get("page_end")
                or r.get("metadata", {}).get("page"),
              },
            )
          )
      return nodes
  return []


def make_retriever(index: VectorStoreIndex, key: str, k: int, use_bm25: bool) -> object:
  vec = index.as_retriever(similarity_top_k=k)
  
  if not use_bm25:
    return vec
  
  # Try to build BM25 from chunks
  try:
    nodes = _load_chunks_for_key(key)
    if nodes:
      bm25 = BM25Retriever.from_defaults(documents=nodes, similarity_top_k=k)
      return QueryFusionRetriever(retrievers=[bm25, vec], similarity_top_k=k, num_queries=1)
    else:
      print("[ask.py] Note: no chunk JSONL found for BM25; using vector-only.")
      return vec
  except Exception as e:
    print(f"[ask.py] BM25 unavailable ({type(e).__name__}: {e}); using vector-only.")
    return vec


# ---------- CLI ----------

def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Query a LlamaIndex LanceDB collection with grounded citations.")
  p.add_argument("question", help="Your question (use quotes)")
  p.add_argument("-k", "--key", default=ENV_KEY, help="Collection key (default: env RAG_KEY or 'default')")
  p.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "10")), help="Top-k retrieved before rerank")
  p.add_argument("--no-bm25", action="store_true", help="Disable BM25 and use vector-only retrieval")
  p.add_argument("--rerank-topn", type=int, default=8, help="Top-N to keep after cross-encoder rerank")
  p.add_argument("--embed-model", default=DEFAULT_EMBED, help="HF embed model (env EMBED_MODEL)")
  p.add_argument("--rerank-model", default=DEFAULT_RERANK, help="Cross-encoder rerank model")
  p.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="LLM model (uses OpenAI if key present)")
  return p


def main():
  args = build_arg_parser().parse_args()
  
  # Apply runtime model choices if they differ from env-configured Settings
  if args.embed_model != DEFAULT_EMBED:
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
  if os.getenv("OPENAI_API_KEY") and args.model != DEFAULT_OPENAI_MODEL:
    Settings.llm = OpenAI(model=args.model)
  
  index = load_index(args.key)
  retriever = make_retriever(index, args.key, k=args.top_k, use_bm25=not args.no_bm25)
  
  reranker = SentenceTransformerRerank(top_n=min(args.rerank_topn, args.top_k), model=args.rerank_model)
  synth = get_response_synthesizer()
  
  qe = index.as_query_engine(
    retriever=retriever,
    response_mode="compact",
    node_postprocessors=[reranker],
    similarity_top_k=args.top_k,
    response_synthesizer=synth,
  )
  
  q_full = f"{PROMPT_RULES} User question: {args.question}"
  
  resp = qe.query(q_full)
  
  print(str(resp))
  print("SOURCES:")
  for src in getattr(resp, "source_nodes", []):
    m = src.node.metadata
    title = m.get("title")
    p1, p2 = m.get("page_start"), m.get("page_end")
    pages = (f"pp. {p1}-{p2}" if p1 and p2 else (f"p. {p1}" if p1 else "p. ?"))
    sp = m.get("source_path")
    print(f"- {title} ({pages}) :: {sp}")


if __name__ == "__main__":
  main()
