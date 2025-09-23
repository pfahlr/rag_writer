from __future__ import annotations
import time
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Try EnsembleRetriever from multiple locations (version differences)
try:
    from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore
except Exception:
    try:
        from langchain_community.retrievers import EnsembleRetriever  # type: ignore
    except Exception:
        EnsembleRetriever = None  # type: ignore

# ContextualCompressionRetriever moved across versions
try:
    from langchain.retrievers import ContextualCompressionRetriever  # type: ignore
except Exception:
    try:
        from langchain_community.retrievers import ContextualCompressionRetriever  # type: ignore
    except Exception:
        ContextualCompressionRetriever = None  # type: ignore

# ParentDocumentRetriever location varies by version
try:
    from langchain.retrievers import ParentDocumentRetriever  # type: ignore
except Exception:
    try:
        from langchain_community.retrievers import ParentDocumentRetriever  # type: ignore
    except Exception:
        ParentDocumentRetriever = None  # type: ignore

# Cross-encoder reranker optional
try:
    from langchain_community.document_transformers import CrossEncoderReranker  # type: ignore
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # type: ignore
except Exception:
    CrossEncoderReranker = None  # type: ignore
    HuggingFaceCrossEncoder = None  # type: ignore

try:
    from .trace import TraceEmitter
except Exception:  # pragma: no cover - allows usage without trace module
    TraceEmitter = None  # type: ignore


class _TracingRetriever:
    def __init__(
        self,
        retriever,
        *,
        emitter: TraceEmitter | None,
        backend: str,
        top_k: int,
        context: dict | None = None,
    ) -> None:
        self._retriever = retriever
        self._emitter = emitter
        self._backend = backend
        self._top_k = top_k
        self._context = context or {}

    def get_relevant_documents(self, query: str):  # type: ignore[override]
        if not self._emitter or not getattr(self._emitter, "enabled", False):
            return self._retriever.get_relevant_documents(query)
        qid = self._context.get("qid")
        parent = self._context.get("parent")
        span = self._emitter.make_span("vector.query", parent=parent)
        self._emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": parent or "root",
                "role": "retriever",
                "type": "vector.query",
                "name": f"{self._backend}.search",
                "detail": {
                    "backend": self._backend,
                    "top_k": self._top_k,
                    "query_text": query,
                },
            }
        )
        start = time.perf_counter()
        docs = self._retriever.get_relevant_documents(query)
        latency_ms = (time.perf_counter() - start) * 1000
        hits = []
        for idx, doc in enumerate(docs):
            meta = doc.metadata or {}
            hits.append(
                {
                    "rank": idx + 1,
                    "doc_id": meta.get("doc_id") or meta.get("source") or meta.get("doc"),
                    "title": meta.get("title"),
                    "score": meta.get("score"),
                    "chunk_id": meta.get("chunk_id"),
                    "offset": meta.get("offset"),
                }
            )
        self._emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": parent or "root",
                "role": "retriever",
                "type": "vector.results",
                "name": f"{self._backend}.search",
                "detail": {"hits": hits},
                "metrics": {"latency_ms": round(latency_ms, 2)},
            }
        )
        return docs

    def __getattr__(self, name):
        return getattr(self._retriever, name)

def _build_faiss(vectorstore: FAISS, k: int):
    return vectorstore.as_retriever(search_kwargs={"k": k})

def _build_bm25(docs: List[Document], k: int) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(docs)  # pure-Python BM25 (rank_bm25)
    bm25.k = k
    return bm25

def _build_hybrid(bm25: BM25Retriever, faiss_retriever, weights: Tuple[float, float] = (0.5, 0.5)):
    if EnsembleRetriever is None:
        print("[yellow]Warning: EnsembleRetriever not available; falling back to FAISS retriever only.[/yellow]")
        return faiss_retriever
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=list(weights))

def _build_ce_compression(base_retriever, ce_model: str):
    if CrossEncoderReranker is None or HuggingFaceCrossEncoder is None or ContextualCompressionRetriever is None:
        print("[yellow]Warning: CE reranking not available; using base retriever.[/yellow]")
        return base_retriever
    ce = HuggingFaceCrossEncoder(model_name=ce_model)
    compressor = CrossEncoderReranker(cross_encoder=ce)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

def _build_parent_child(vectorstore: FAISS, docs: List[Document],
                        child_chunk: int = 300, parent_chunk: int = 1200, k: int = 8):
    # Retrieve on small “child” chunks but return larger parent spans
    if ParentDocumentRetriever is None:
        print("[yellow]Warning: ParentDocumentRetriever not available; using FAISS retriever only.[/yellow]")
        return vectorstore.as_retriever(search_kwargs={"k": k})
    return ParentDocumentRetriever.from_documents(
        docs, vectorstore,
        child_splitter_kwargs={"chunk_size": child_chunk, "chunk_overlap": 40},
        parent_splitter_kwargs={"chunk_size": parent_chunk, "chunk_overlap": 80},
        search_kwargs={"k": k},
    )

def make_retriever(
    mode: str,
    vectorstore: Optional[FAISS],
    docs: Optional[List[Document]],
    k: int = 10,
    rerank: Optional[str] = None,
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    trace_emitter: TraceEmitter | None = None,
    trace_context: dict | None = None,
):
    """
    mode: 'faiss' | 'bm25' | 'hybrid' | 'parent' | 'faiss+compression' | 'hybrid+compression'
    rerank: None | 'ce'
    """
    if mode == "faiss":
        base = _build_faiss(vectorstore, k)
    elif mode == "bm25":
        base = _build_bm25(docs, k)
    elif mode == "hybrid":
        base = _build_hybrid(_build_bm25(docs, k), _build_faiss(vectorstore, k))
    elif mode == "parent":
        base = _build_parent_child(vectorstore, docs, k=k)
    elif mode == "faiss+compression":
        base = _build_faiss(vectorstore, k)
        if rerank == "ce":
            return _build_ce_compression(base, ce_model)
        return base
    elif mode == "hybrid+compression":
        base = _build_hybrid(_build_bm25(docs, k), _build_faiss(vectorstore, k))
    else:
        raise ValueError(f"Unknown retriever mode: {mode}")

    if rerank == "ce":
        base = _build_ce_compression(base, ce_model)
    if trace_emitter is not None:
        context = dict(trace_context or {})
        context.setdefault("backend", mode)
        context.setdefault("top_k", k)
        base = _TracingRetriever(
            base,
            emitter=trace_emitter,
            backend=context.get("backend", mode),
            top_k=context.get("top_k", k),
            context=context,
        )
    return base
