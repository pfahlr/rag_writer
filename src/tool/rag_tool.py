"""RAG retrieval tool built on top of :mod:`src.core.retriever`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .base import Tool, ToolSpec
from ..core.retriever import RetrieverFactory, RetrieverConfig


def create_rag_retrieve_tool(key: str, index_dir: Optional[Path | str] = None) -> Tool:
    """Create a tool that retrieves documents from the local vector database.

    Parameters
    ----------
    key:
        Collection key used to locate the FAISS index.
    """

    storage_dir_path: Path | None = None
    if index_dir is not None:
        storage_dir_path = Path(index_dir)

    factory = RetrieverFactory(storage_dir=storage_dir_path)

    def _run(
        query: str,
        k: int = 5,
        retriever_profile: str = "hybrid",
        rerank: bool = True,
    ) -> Dict[str, Any]:
        profile = retriever_profile.lower()
        config = RetrieverConfig(
            key=key,
            k=k,
            multiquery=False,
            use_bm25=profile in {"hybrid", "bm25"},
            use_reranking=rerank,
        )
        if profile == "vector":
            retriever = factory.create_vector_retriever(config)
        elif profile == "bm25":
            retriever = factory.create_bm25_retriever(config)
        elif profile == "hybrid":
            retriever = factory.create_hybrid_retriever(config)
        else:
            raise ValueError(f"Unknown retriever_profile: {retriever_profile}")

        docs: List[Document] = retriever.get_relevant_documents(query)
        out_docs = []
        for d in docs:
            out_docs.append(
                {
                    "source_id": d.metadata.get("source", ""),
                    "title": d.metadata.get("title", ""),
                    "text": d.page_content,
                    "metadata": d.metadata,
                }
            )
        return {"docs": out_docs}

    spec = ToolSpec(
        name="rag_retrieve",
        description="Retrieve relevant documents from the vector database",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "retriever_profile": {
                    "type": "string",
                    "enum": ["vector", "hybrid", "bm25"],
                    "default": "hybrid",
                },
                "rerank": {"type": "boolean", "default": True},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "docs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["text", "metadata"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["docs"],
            "additionalProperties": False,
        },
    )

    return Tool(spec, _run)
