#!/usr/bin/env python3
"""Simple LangChain RAG ask CLI."""

from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# Prefer langchain-huggingface (new home), fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Ensure project root ('/app') is on sys.path so we can import 'src.*'
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.langchain.retriever_factory import make_retriever

ROOT = project_root


def _fs_safe(value: str) -> str:
    """Return a filesystem-safe slug for embedding model names."""

    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def _resolve_paths(
    key: str,
    embed_model: str,
    *,
    chunks_dir: Path,
    index_dir: Path,
) -> tuple[Path, Path, Path]:
    """Derive chunk metadata and FAISS directories based on CLI arguments."""

    chunk_path = Path(chunks_dir) / f"lc_chunks_{key}.jsonl"
    base_dir = Path(index_dir) / f"faiss_{key}__{_fs_safe(embed_model)}"
    repacked_dir = base_dir.parent / f"{base_dir.name}_repacked"
    return chunk_path, base_dir, repacked_dir


def _get_embedding_dimension(embedder: HuggingFaceEmbeddings) -> int | None:
    """Return the output dimension for a HuggingFace embedding model."""

    client = getattr(embedder, "client", None)
    if client is None:
        return None

    getter = getattr(client, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return None

    # Fallback for SentenceTransformer-like clients that expose `embedding_dim`
    dim = getattr(client, "embedding_dim", None)
    if isinstance(dim, int):
        return dim

    return None


def _validate_index_embedding_compatibility(
    embedder: HuggingFaceEmbeddings, vectorstore: FAISS, index_path: Path
) -> None:
    """Ensure the FAISS index dimension matches the embedding model output."""

    index_dim = getattr(getattr(vectorstore, "index", None), "d", None)
    embed_dim = _get_embedding_dimension(embedder)

    if index_dim is None or embed_dim is None:
        return

    if index_dim != embed_dim:
        model_name = getattr(embedder, "model_name", "(unknown)")
        raise SystemExit(
            "[lc_ask] Embedding dimension mismatch: "
            f"index at {index_path} expects dimension {index_dim}, "
            f"but embedding model '{model_name}' produces {embed_dim}.\n"
            "  • Pass --embed-model with the model used to build the index, "
            "or rebuild the index for the requested model."
        )


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
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT / "data_raw"),
        help="Path to directory containing source files for index",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=str(ROOT / "data_processed"),
        help="Path to directory to store chunks",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(ROOT / "storage"),
        help=(
            "Path to directory containing index directories (i.e., storage) not "
            "individual index directories, the collection of them"
        ),
    )
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

    chunks_path, base_dir, repacked_dir = _resolve_paths(
        key=args.key,
        embed_model=args.embed_model,
        chunks_dir=Path(args.chunks_dir),
        index_dir=Path(args.index_dir),
    )
    if not chunks_path.exists():
        raise SystemExit(
            f"[lc_ask] chunks not found: {chunks_path} – run lc_build_index for KEY={args.key}"
        )
    docs = _load_chunks_jsonl(chunks_path)

    # Prefer a repacked/merged index if available
    faiss_dir = None
    for cand in (repacked_dir, base_dir):
        if (cand / "index.faiss").exists():
            faiss_dir = cand
            break

    if faiss_dir is None:
        if base_dir.exists():
            shards = [p for p in base_dir.iterdir() if p.is_dir()]
            if shards:
                raise SystemExit(
                    f"[lc_ask] FAISS shards found but no merged index: {base_dir}\n"
                    "  • Merge shards before querying (merge step not completed)"
                )
        raise SystemExit(
            "[lc_ask] FAISS dir not found: "
            f"{base_dir} (or repacked: {repacked_dir}).\n"
            f"  • If you upgraded LangChain, try: make repack-faiss KEY={args.key} EMBED_MODEL={args.embed_model}\n"
            f"  • Or rebuild the index: python src/langchain/lc_build_index.py {args.key}"
        )
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model)
    vectorstore = FAISS.load_local(
        str(faiss_dir), embeddings=embedder, allow_dangerous_deserialization=True
    )
    _validate_index_embedding_compatibility(embedder, vectorstore, faiss_dir)

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
