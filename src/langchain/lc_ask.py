#!/usr/bin/env python3
"""Simple LangChain RAG ask CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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
from src.langchain.trace import configure_emitter


def _fs_safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)

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


def _infer_key_from_index_dir(path: Path) -> str | None:
    name = path.name
    if name == "index.faiss":
        name = path.parent.name
    if name.endswith("_repacked"):
        name = name[: -len("_repacked")]
    if name.startswith("faiss_"):
        name = name[len("faiss_") :]
        if "__" in name:
            name = name.split("__", 1)[0]
    if not name:
        return None
    return _fs_safe(name)


def _resolve_faiss_directory(path: Path) -> tuple[Path, str | None]:
    if not path.exists():
        raise SystemExit(f"[lc_ask] index path not found: {path}")
    if path.is_file():
        if path.name != "index.faiss":
            raise SystemExit(
                "[lc_ask] --index must point to a directory or index.faiss file"
            )
        path = path.parent
    if (path / "index.faiss").exists():
        return path, _infer_key_from_index_dir(path)
    candidates = [p for p in path.iterdir() if p.is_dir() and (p / "index.faiss").exists()]
    if len(candidates) == 1:
        return candidates[0], _infer_key_from_index_dir(candidates[0])
    if not candidates:
        raise SystemExit(
            f"[lc_ask] index.faiss not found in {path}. Provide --index pointing to the directory containing index.faiss"
        )
    raise SystemExit(
        "[lc_ask] multiple FAISS directories found; pass --index with the desired directory"
    )


def _locate_chunks_file(
    *,
    explicit_path: str | None,
    chunks_dir: Path,
    key_safe: str | None,
    index_dir: Path | None,
) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.exists():
            raise SystemExit(f"[lc_ask] chunks file not found: {candidate}")
        return candidate
    if key_safe:
        candidate = chunks_dir / f"lc_chunks_{key_safe}.jsonl"
        if candidate.exists():
            return candidate
    if index_dir is not None:
        for pattern in ("lc_chunks_*.jsonl", "*.jsonl"):
            matches = sorted(index_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _extract_docs_from_vectorstore(vectorstore) -> list[Document] | None:
    docstore = getattr(vectorstore, "docstore", None)
    if docstore is None:
        return None
    records = None
    if hasattr(docstore, "_dict"):
        records = list(getattr(docstore, "_dict").values())
    elif hasattr(docstore, "values"):
        records = list(docstore.values())
    if not records:
        return None
    docs: list[Document] = []
    for item in records:
        if isinstance(item, Document):
            docs.append(item)
        elif isinstance(item, dict) and "page_content" in item:
            docs.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {}),
                )
            )
    return docs or None


def main():
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", metavar="QUESTION", help="Question to ask")
    parser.add_argument(
        "-q",
        "--question",
        dest="question_opt",
        help="Question to ask (overrides positional QUESTION)",
    )
    parser.add_argument("--json", dest="json_path", help="JSON job file containing 'question'")
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--key", help="collection key used at index time")


    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")

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
    parser.add_argument("--trace", action="store_true", help="Emit TRACE events to stderr")
    parser.add_argument(
        "--trace-file",
        help="Optional path to tee TRACE events to disk",
    )
    parser.add_argument(
        "--chunks-dir",
        default=str(root / "data_processed"),
        help="Directory containing lc_build_index chunk outputs",
    )
    parser.add_argument(
        "--chunks-file",
        dest="chunks_file",
        help="Explicit path to chunk JSONL (overrides --chunks-dir lookup)",
    )

    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT / "data_raw"),
        help="Path to directory containing source files for index",
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

    emitter = configure_emitter(args.trace, trace_file=args.trace_file)
    qid = os.getenv("TRACE_QID")

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
        question = args.question_opt or args.question or ""
    if not question:
        raise SystemExit("No question provided")

    chunks_dir = Path(args.chunks_dir).expanduser()
    index_dir = Path(args.index_dir).expanduser()

    docs: list[Document] | None = None
    faiss_dir: Path | None = None
    key_safe: str | None = None

    if args.key:
        key_safe = _fs_safe(args.key)
        emb_name_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.embed_model)
        base_dir = index_dir / f"faiss_{key_safe}__{emb_name_safe}"
        repacked_dir = base_dir.parent / f"{base_dir.name}_repacked"

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
    else:
        faiss_dir, key_safe = _resolve_faiss_directory(Path(args.index_path).expanduser())

    chunks_path, base_dir, repacked_dir = _resolve_paths(
        key=args.key,
        embed_model=args.embed_model,
        chunks_dir=Path(args.chunks_dir),
        index_dir=Path(args.index_dir),
    )
    if chunks_path is not None:
        docs = _load_chunks_jsonl(chunks_path)

    if not chunks_path.exists():
        raise SystemExit(
            f"[lc_ask] chunks not found: {chunks_path} – run lc_build_index for KEY={args.key}"
        )
 
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model)
    vectorstore = FAISS.load_local(
        str(faiss_dir), embeddings=embedder, allow_dangerous_deserialization=True
    )
    _validate_index_embedding_compatibility(embedder, vectorstore, faiss_dir)

    if docs is None:
        docs = _extract_docs_from_vectorstore(vectorstore)

    docs_for_retriever = docs or []
    modes_requiring_docs = {"bm25", "hybrid", "parent", "hybrid+compression"}
    if not docs_for_retriever and args.mode in modes_requiring_docs:
        raise SystemExit(
            f"[lc_ask] Document chunks required for mode '{args.mode}'. Provide --key or --chunks-file"
        )

    retriever = make_retriever(
        mode=args.mode,
        vectorstore=vectorstore,
        docs=docs_for_retriever,
        k=args.k,
        rerank=(None if args.rerank == "none" else args.rerank),
        ce_model=args.ce_model,
        trace_emitter=emitter,
        trace_context={"qid": qid, "backend": args.mode, "top_k": args.k},
    )

    llm = ChatOpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
    )

    with emitter:
        span_id = emitter.make_span("llm.ask") if emitter.enabled else None
        if emitter.enabled:
            emitter.emit(
                {
                    "qid": qid,
                    "span": span_id,
                    "parent": "root",
                    "role": "user",
                    "type": "llm.prompt",
                    "name": "langchain.RetrievalQA",
                    "detail": {
                        "model": getattr(llm, "model_name", "unknown"),
                        "messages": [
                            {"role": "system", "content": "RetrievalQA"},
                            {"role": "user", "content": question},
                        ],
                        "params": {"temperature": getattr(llm, "temperature", None)},
                    },
                }
            )
        start = time.perf_counter()
        result = chain.invoke({"query": question})
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result["result"]
        if emitter.enabled:
            emitter.emit(
                {
                    "qid": qid,
                    "span": span_id,
                    "parent": "root",
                    "role": "assistant",
                    "type": "llm.completion",
                    "name": "langchain.RetrievalQA",
                    "detail": {"content": answer, "finish_reason": "stop"},
                    "metrics": {"latency_ms": round(latency_ms, 2)},
                }
            )
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
