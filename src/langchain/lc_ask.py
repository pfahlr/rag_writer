#!/usr/bin/env python3
"""Simple LangChain RAG ask CLI."""

from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Ensure project root on path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.langchain.retriever_factory import make_retriever


def _load_chunks_jsonl(path: Path) -> list[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["text"], metadata=rec.get("metadata", {})))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--json", dest="json_path", help="JSON job file containing 'question'")
    parser.add_argument("--key", required=True, help="collection key used at index time")
    parser.add_argument(
        "--mode",
        default="faiss",
        choices=[
            "faiss",
            "bm25",
            "hybrid",
            "parent",
            "faiss+compression",
            "hybrid+compression",
        ],
    )
    parser.add_argument("--rerank", default="none", choices=["none", "ce"])
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    if args.json_path:
        with open(args.json_path, "r", encoding="utf-8") as f:
            job = json.load(f)
        question = (
            job.get("instruction")
            or job.get("question")
            or job.get("prompt")
            or ""
        )
    else:
        question = args.question or ""
    if not question:
        raise SystemExit("No question provided")

    chunks_path = Path(f"generated/lc_chunks_{args.key}.jsonl")
    if not chunks_path.exists():
        raise SystemExit(
            f"[lc_ask] chunks not found: {chunks_path} – run lc_build_index for KEY={args.key}"
        )
    docs = _load_chunks_jsonl(chunks_path)

    emb_name_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.embed_model)
    faiss_dir = Path(f"storage/faiss_{args.key}__{emb_name_safe}")
    if not faiss_dir.exists():
        raise SystemExit(
            f"[lc_ask] FAISS dir not found: {faiss_dir} – rebuild index with --embed-model {args.embed_model}"
        )
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model)
    vectorstore = FAISS.load_local(
        str(faiss_dir), embeddings=embedder, allow_dangerous_deserialization=True
    )

    retriever = make_retriever(
        mode=args.mode,
        vectorstore=vectorstore,
        docs=docs,
        k=args.k,
        rerank=(None if args.rerank == "none" else args.rerank),
        ce_model=args.ce_model,
    )

    llm = ChatOpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])

    output = {
        "answer": answer,
        "sources": [
            {"text": d.page_content, "metadata": d.metadata} for d in sources
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

