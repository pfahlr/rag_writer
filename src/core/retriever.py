#!/usr/bin/env python3
"""
Unified Retriever Factory for RAG Writer

This module provides a centralized factory for creating various types of retrievers
used across the application, eliminating code duplication between different modules.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

# Import langchain first to ensure proper initialization
import langchain

# Disable debug mode to prevent compatibility issues
try:
    langchain.debug = False
except AttributeError:
    # debug attribute doesn't exist in this version, skip
    pass
try:
    langchain.llm_cache = None
except AttributeError:
    # llm_cache attribute doesn't exist in this version, skip
    pass

# Try to import langchain.retrievers, but don't fail if it's not available
try:
    import langchain.retrievers  # Initialize the retrievers module
except ImportError:
    pass  # Continue without langchain.retrievers

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from . import faiss_utils

# Try to import HuggingFaceEmbeddings from the new package first
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to the old package
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import retriever classes with fallbacks
try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain_community.retrievers import MultiQueryRetriever
    except ImportError:
        MultiQueryRetriever = None

try:
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline
except ImportError:
    try:
        from langchain_community.document_compressors import DocumentCompressorPipeline
    except ImportError:
        DocumentCompressorPipeline = None

try:
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain_community.retrievers import ContextualCompressionRetriever
    except ImportError:
        ContextualCompressionRetriever = None

# Optional imports
try:
    from langchain_community.document_compressors import FlashrankRerank
    _flashrank_available = True
except ImportError:
    _flashrank_available = False

try:
    from langchain_openai import ChatOpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


@dataclass
class RetrieverConfig:
    """Configuration for retriever creation."""
    key: str
    k: int = 10
    multiquery: bool = True
    use_bm25: bool = True
    use_reranking: bool = True
    embedding_model: str = "BAAI/bge-small-en"
    rerank_model: str = "BAAI/bge-reranker-base"
    openai_model: str = "gpt-4o-mini"
    vector_weight: float = 0.6
    bm25_weight: float = 0.4


class RetrieverFactory:
    """Factory for creating various types of retrievers with consistent configuration."""

    def __init__(self, root_dir: Optional[Path] = None):
        """Initialize the factory with root directory."""
        if root_dir is None:
            # Calculate root relative to this file (two levels up)
            root_dir = Path(__file__).parent.parent.parent
        self.root_dir = root_dir

    def _get_storage_path(self, key: str) -> Path:
        """Get the storage path for a given collection key."""
        return self.root_dir / "storage" / f"faiss_{key}"

    def _load_embeddings(self, model_name: str) -> HuggingFaceEmbeddings:
        """Load and return the embedding model."""
        try:
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}")

    def _load_faiss_index(self, faiss_dir: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
        """Load FAISS index from directory."""
        try:
            vectorstore = FAISS.load_local(
                str(faiss_dir), embeddings, allow_dangerous_deserialization=True
            )
            if faiss_utils.is_faiss_gpu_available():
                gpu_index = faiss_utils.clone_index_to_gpu(vectorstore.index)
                if gpu_index is not None:
                    vectorstore.index = gpu_index
            return vectorstore
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {faiss_dir}: {str(e)}")

    def _create_bm25_retriever(self, docs: list, k: int) -> BM25Retriever:
        """Create BM25 retriever from documents."""
        try:
            if not docs:
                raise ValueError("No documents found in the index")
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = k
            return bm25
        except Exception as e:
            raise RuntimeError(f"Failed to create BM25 retriever: {str(e)}")

    def _create_multiquery_retriever(self, base_retriever: Any, config: RetrieverConfig) -> Any:
        """Add multi-query capability if available."""
        if not config.multiquery or not os.getenv("OPENAI_API_KEY") or not _openai_available or MultiQueryRetriever is None:
            return base_retriever

        try:
            llm_for_multi = ChatOpenAI(model=config.openai_model, temperature=0)
            return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm_for_multi)
        except Exception as e:
            print(f"[yellow]Warning: Multi-query retriever failed, falling back to base retriever: {str(e)}[/]")
            return base_retriever

    def _add_reranking(self, base_retriever: Any, config: RetrieverConfig) -> Any:
        """Add reranking capability if available."""
        if not config.use_reranking or not _flashrank_available or DocumentCompressorPipeline is None or ContextualCompressionRetriever is None:
            return base_retriever

        try:
            rerank = FlashrankRerank(top_n=config.k)
            compressor = DocumentCompressorPipeline(transformers=[rerank])
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except Exception as e:
            print(f"[yellow]Warning: Reranking failed, falling back to base retriever: {str(e)}[/]")
            return base_retriever

    def create_hybrid_retriever(self, config: RetrieverConfig) -> Any:
        """
        Create a hybrid retriever with FAISS vector search, BM25, and optional reranking.

        Args:
            config: Configuration for the retriever

        Returns:
            Configured retriever object

        Raises:
            FileNotFoundError: If FAISS index directory doesn't exist
            RuntimeError: If any component fails to initialize
        """
        faiss_dir = self._get_storage_path(config.key)

        # Check if FAISS directory exists
        if not faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {faiss_dir}")

        # Load embeddings and FAISS index
        embeddings = self._load_embeddings(config.embedding_model)
        vs = self._load_faiss_index(faiss_dir, embeddings)

        # Create vector retriever
        vector_retriever = vs.as_retriever(search_kwargs={"k": config.k})

        if not config.use_bm25 or EnsembleRetriever is None:
            # Simple vector-only retriever
            retriever = vector_retriever
        else:
            # Create BM25 retriever
            docs = list(vs.docstore._dict.values())
            bm25_retriever = self._create_bm25_retriever(docs, config.k)

            # Create hybrid ensemble retriever
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[config.vector_weight, config.bm25_weight]
            )

        # Add multi-query capability
        retriever = self._create_multiquery_retriever(retriever, config)

        # Add reranking
        retriever = self._add_reranking(retriever, config)

        return retriever

    def create_vector_retriever(self, config: RetrieverConfig) -> Any:
        """
        Create a simple vector-only retriever.

        Args:
            config: Configuration for the retriever

        Returns:
            Vector retriever object
        """
        # Disable BM25 and other features for vector-only
        vector_config = RetrieverConfig(
            key=config.key,
            k=config.k,
            multiquery=False,
            use_bm25=False,
            use_reranking=False,
            embedding_model=config.embedding_model
        )
        return self.create_hybrid_retriever(vector_config)

    def create_bm25_retriever(self, config: RetrieverConfig) -> BM25Retriever:
        """
        Create a BM25-only retriever.

        Args:
            config: Configuration for the retriever

        Returns:
            BM25 retriever object
        """
        faiss_dir = self._get_storage_path(config.key)

        if not faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {faiss_dir}")

        embeddings = self._load_embeddings(config.embedding_model)
        vs = self._load_faiss_index(faiss_dir, embeddings)

        docs = list(vs.docstore._dict.values())
        return self._create_bm25_retriever(docs, config.k)


# Convenience functions for backward compatibility
def create_hybrid_retriever(key: str, k: int = 10, multiquery: bool = True,
                          embedding_model: str = "BAAI/bge-small-en") -> Any:
    """Convenience function to create a hybrid retriever with default settings."""
    factory = RetrieverFactory()
    config = RetrieverConfig(
        key=key,
        k=k,
        multiquery=multiquery,
        embedding_model=embedding_model
    )
    return factory.create_hybrid_retriever(config)


def create_vector_retriever(key: str, k: int = 10,
                          embedding_model: str = "BAAI/bge-small-en") -> Any:
    """Convenience function to create a vector-only retriever."""
    factory = RetrieverFactory()
    config = RetrieverConfig(
        key=key,
        k=k,
        embedding_model=embedding_model
    )
    return factory.create_vector_retriever(config)