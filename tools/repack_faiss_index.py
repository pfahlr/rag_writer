#!/usr/bin/env python3
"""
Repack a FAISS index saved under older LangChain/Pydantic versions so it
loads cleanly with current versions, without re-embedding.

It reconstructs a fresh InMemoryDocstore from the chunks JSONL and aligns
doc IDs to FAISS row order, then saves a new vectorstore directory.

Usage examples:
  # Derive paths from KEY and EMBED_MODEL
  python tools/repack_faiss_index.py --key chatgptedu2 --embed-model BAAI/bge-small-en-v1.5

  # Explicit paths
  python tools/repack_faiss_index.py --faiss-dir storage/faiss_chatgptedu2__BAAI-bge-small-en-v1.5 \
                                     --chunks data_processed/lc_chunks_chatgptedu2.jsonl \
                                     --out-dir storage/faiss_chatgptedu2__BAAI-bge-small-en-v1.5_repacked

Notes:
  - No embeddings are computed. A dummy embedder is used just to satisfy
    the FAISS class constructor.
  - If your chunk count differs from index rows (ntotal), use --truncate
    to align to the smaller of the two.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import faiss  # type: ignore

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def _fs_safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)


def _load_docstore_impl():
    # Try several locations depending on LangChain version
    try:
        from langchain_community.docstore.in_memory import InMemoryDocstore  # type: ignore
        return InMemoryDocstore
    except Exception:
        pass
    try:
        from langchain_community.docstores import InMemoryDocstore  # type: ignore
        return InMemoryDocstore
    except Exception:
        pass
    try:
        from langchain.docstore.in_memory import InMemoryDocstore  # type: ignore
        return InMemoryDocstore
    except Exception as e:
        raise RuntimeError(
            "Unable to import InMemoryDocstore from known locations; ensure a compatible LangChain version is installed"
        ) from e


class _DummyEmbeddings:
    """Minimal embeddings stub for saving the vector store.

    FAISS.save_local does not call embedding methods; real embeddings
    will be supplied on load via FAISS.load_local.
    """

    def embed_documents(self, texts):  # pragma: no cover
        raise NotImplementedError("Not used during repack")

    def embed_query(self, text):  # pragma: no cover
        raise NotImplementedError("Not used during repack")


def repack(faiss_dir: Path, chunks_path: Path, out_dir: Path, truncate: bool = False) -> Path:
    if not faiss_dir.exists():
        raise FileNotFoundError(f"FAISS directory not found: {faiss_dir}")
    index_path = faiss_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    ntotal = int(index.ntotal)

    # Build doc mapping aligned to row order
    docs_map = {}
    index_to_docstore_id = {}

    with chunks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not truncate and i >= ntotal:
                break
            rec = json.loads(line)
            doc_id = str(i)
            doc = Document(page_content=rec.get("text", ""), metadata=rec.get("metadata", {}))
            docs_map[doc_id] = doc
            index_to_docstore_id[i] = doc_id
            if truncate and i + 1 >= ntotal:
                break

    if not truncate and len(docs_map) != ntotal:
        raise ValueError(
            f"Row count mismatch: FAISS rows={ntotal}, chunks={len(docs_map)}. "
            "Use --truncate to align to the smaller count if necessary."
        )

    InMemoryDocstore = _load_docstore_impl()
    docstore = InMemoryDocstore(docs_map)

    # Handle constructor signature differences across versions
    try:
        vs = FAISS(embedding=_DummyEmbeddings(), index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)  # type: ignore
    except TypeError:
        vs = FAISS(embedding_function=_DummyEmbeddings(), index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Repack a FAISS index to current LangChain format without re-embedding.")
    ap.add_argument("--key", help="Collection key (used to derive default paths)")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="Embedding model used for index (for path derivation)")
    ap.add_argument("--faiss-dir", help="Explicit path to FAISS directory")
    ap.add_argument("--chunks", help="Path to chunks JSONL (default: data_processed/lc_chunks_<key>.jsonl)")
    ap.add_argument("--out-dir", help="Output directory for repacked index (default: <faiss_dir>_repacked)")
    ap.add_argument("--truncate", action="store_true", help="Truncate to min(row_count, chunks_count) instead of erroring on mismatch")
    args = ap.parse_args()

    if args.faiss_dir:
        faiss_dir = Path(args.faiss_dir)
        if args.key:
            # Optional chunks default if key provided
            chunks_path = Path(args.chunks) if args.chunks else Path(f"data_processed/lc_chunks_{args.key}.jsonl")
        else:
            if not args.chunks:
                ap.error("--chunks is required when --faiss-dir is used without --key")
            chunks_path = Path(args.chunks)
    else:
        if not args.key:
            ap.error("Provide either --faiss-dir or --key")
        emb_name = _fs_safe(args.embed_model)
        faiss_dir = Path(f"storage/faiss_{args.key}__{emb_name}")
        chunks_path = Path(args.chunks) if args.chunks else Path(f"data_processed/lc_chunks_{args.key}.jsonl")

    out_dir = Path(args.out_dir) if args.out_dir else Path(str(faiss_dir) + "_repacked")

    new_dir = repack(faiss_dir, chunks_path, out_dir, truncate=args.truncate)
    print(f"[ok] Repacked index saved to: {new_dir}")


if __name__ == "__main__":
    main()
