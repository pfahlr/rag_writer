from __future__ import annotations
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever, ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def _build_faiss(vectorstore: FAISS, k: int):
    return vectorstore.as_retriever(search_kwargs={"k": k})

def _build_bm25(docs: List[Document], k: int) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(docs)  # pure-Python BM25 (rank_bm25)
    bm25.k = k
    return bm25

def _build_hybrid(bm25: BM25Retriever, faiss_retriever, weights: Tuple[float, float] = (0.5, 0.5)):
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=list(weights))

def _build_ce_compression(base_retriever, ce_model: str):
    ce = HuggingFaceCrossEncoder(model_name=ce_model)
    compressor = CrossEncoderReranker(cross_encoder=ce)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

def _build_parent_child(vectorstore: FAISS, docs: List[Document],
                        child_chunk: int = 300, parent_chunk: int = 1200, k: int = 8):
    # Retrieve on small “child” chunks but return larger parent spans
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
        return _build_ce_compression(base, ce_model)
    return base
