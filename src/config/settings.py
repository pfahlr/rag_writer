#!/usr/bin/env python3
"""
Centralized Configuration Management for RAG Writer

This module provides a single source of truth for all application configuration,
eliminating scattered environment variables and hardcoded values throughout the codebase.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    root_dir: Path
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    storage_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    exports_dir: Path = field(init=False)
    outlines_dir: Path = field(init=False)
    data_jobs_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_raw_dir = self.root_dir / "data_raw"
        self.data_processed_dir = self.root_dir / "data_processed"
        self.storage_dir = self.root_dir / "storage"
        self.output_dir = self.root_dir / "output"
        self.exports_dir = self.root_dir / "exports"
        self.outlines_dir = self.root_dir / "outlines"
        self.data_jobs_dir = self.root_dir / "data_jobs"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "BAAI/bge-small-en"
    batch_size: int = 128
    device: str = "cpu"  # or "cuda" for GPU


@dataclass
class LLMConfig:
    """Configuration for LLM backends."""
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@dataclass
class RetrieverConfig:
    """Configuration for retrievers."""
    default_k: int = 10
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    use_multiquery: bool = True
    use_reranking: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"


@dataclass
class IndexingConfig:
    """Configuration for document indexing."""
    chunk_size: int = 1200
    chunk_overlap: int = 200
    separators: list = field(default_factory=lambda: ["\n\n", "\n", " ", ""])


@dataclass
class JobGenerationConfig:
    """Configuration for job generation."""
    default_prompts_per_section: int = 4
    max_prompts_per_section: int = 10
    min_prompts_per_section: int = 1
    use_llm_for_job_generation: bool = True
    use_rag_for_job_generation: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    # Core paths
    paths: PathConfig = field(init=False)

    # Component configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    job_generation: JobGenerationConfig = field(default_factory=JobGenerationConfig)

    # Environment variables
    rag_key: str = "default"
    openai_api_key: Optional[str] = None

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = False
    debug_mode: bool = False
    parallel_workers: int = field(init=False)

    def __post_init__(self):
        """Initialize configuration from environment."""
        # Determine root directory
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

        # Initialize paths
        self.paths = PathConfig(root_dir)

        # Load environment variables
        self.rag_key = os.getenv("RAG_KEY", "default")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Override embedding model if specified
        if os.getenv("EMBED_MODEL"):
            self.embedding.model_name = os.getenv("EMBED_MODEL")

        # Override LLM model if specified
        if os.getenv("OPENAI_MODEL"):
            self.llm.openai_model = os.getenv("OPENAI_MODEL")

        # Override Ollama model if specified
        if os.getenv("OLLAMA_MODEL"):
            self.llm.ollama_model = os.getenv("OLLAMA_MODEL")

        # Override indexing parameters
        if os.getenv("EMBED_BATCH"):
            self.embedding.batch_size = int(os.getenv("EMBED_BATCH"))

        # Set debug mode
        self.debug_mode = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")

        # Determine parallel workers default
        self.parallel_workers = self._determine_parallel_workers()

    @staticmethod
    def _clamp_parallel_workers(value: int) -> int:
        """Clamp parallel worker counts to sensible operational bounds."""

        min_workers = 1
        max_workers = 32
        return max(min_workers, min(max_workers, value))

    def _determine_parallel_workers(self) -> int:
        """Resolve the default parallel worker count from env or system state."""

        for env_var in ("RAG_PARALLEL_WORKERS", "PARALLEL_WORKERS_COUNT"):
            env_value = os.getenv(env_var)
            if env_value is None or str(env_value).strip() == "":
                continue
            try:
                return self._clamp_parallel_workers(int(env_value))
            except (TypeError, ValueError):
                continue

        cpu_count = os.cpu_count() or 1
        return self._clamp_parallel_workers(cpu_count)

    def get_storage_path(self, key: Optional[str] = None) -> Path:
        """Get storage path for a collection key."""
        collection_key = key or self.rag_key
        return self.paths.storage_dir / f"faiss_{collection_key}"

    def get_processed_data_path(self, key: Optional[str] = None) -> Path:
        """Get processed data path for a collection key."""
        collection_key = key or self.rag_key
        return self.paths.data_processed_dir / f"lc_chunks_{collection_key}.jsonl"

    def ensure_directories_exist(self):
        """Ensure all required directories exist."""
        directories = [
            self.paths.data_raw_dir,
            self.paths.data_processed_dir,
            self.paths.storage_dir,
            self.paths.output_dir,
            self.paths.exports_dir,
            self.paths.outlines_dir,
            self.paths.data_jobs_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience functions for common access patterns
def get_root_dir() -> Path:
    """Get the root directory."""
    return get_config().paths.root_dir


def get_storage_dir() -> Path:
    """Get the storage directory."""
    return get_config().paths.storage_dir


def get_data_raw_dir() -> Path:
    """Get the raw data directory."""
    return get_config().paths.data_raw_dir


def get_output_dir() -> Path:
    """Get the output directory."""
    return get_config().paths.output_dir


def get_rag_key() -> str:
    """Get the current RAG collection key."""
    return get_config().rag_key


def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key."""
    return get_config().openai_api_key


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_config().debug_mode
