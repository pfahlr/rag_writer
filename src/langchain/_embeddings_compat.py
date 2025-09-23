"""Compatibility helpers for LangChain embedding interfaces."""

from __future__ import annotations

from typing import Iterable, Sequence

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS


def _call_embedder(embedding_fn, method: str, *args):
    if isinstance(embedding_fn, Embeddings):
        return getattr(embedding_fn, method)(*args)
    if hasattr(embedding_fn, method):
        return getattr(embedding_fn, method)(*args)
    if callable(embedding_fn):
        return embedding_fn(*args)
    raise TypeError(
        f"Embedding function must implement '{method}' or be callable; got {type(embedding_fn)}"
    )


def _embed_query(self: FAISS, text: str) -> Sequence[float]:  # type: ignore[override]
    return _call_embedder(self.embedding_function, "embed_query", text)


def _embed_documents(self: FAISS, texts: Iterable[str]):  # type: ignore[override]
    return _call_embedder(self.embedding_function, "embed_documents", list(texts))


if not getattr(FAISS, "_ragwriter_embed_patch", False):
    FAISS._embed_query = _embed_query  # type: ignore[assignment]
    FAISS._embed_documents = _embed_documents  # type: ignore[assignment]
    FAISS._ragwriter_embed_patch = True

