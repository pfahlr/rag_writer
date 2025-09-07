#!/usr/bin/env python3
"""Remove FAISS shard directories for a specific collection key and embedding model.

This script deletes temporary shard directories produced during multi-stage
index builds. These directories follow the pattern::

    storage/faiss_<KEY>__<EMB_MODEL_SAFE>__*

Usage:
    python src/langchain/cleanup_shards.py KEY EMB_MODEL

Example:
    python src/langchain/cleanup_shards.py science BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _fs_safe(s: str) -> str:
    """Return a filesystem safe version of *s*."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)


def cleanup_shards(key: str, embed_model: str, storage_dir: Path | None = None) -> int:
    """Remove shard directories for *key* and *embed_model*.

    Args:
        key: Collection key.
        embed_model: Embedding model name.
        storage_dir: Base storage directory (defaults to project ``storage``).

    Returns:
        Number of shard directories removed.
    """
    storage = storage_dir or ROOT / "storage"
    emb_safe = _fs_safe(embed_model)
    prefix = f"faiss_{key}__{emb_safe}__"
    removed = 0
    for path in storage.glob(prefix + "*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed {path}")
            removed += 1
    if removed == 0:
        print("No shard directories found.")
    else:
        print(f"Removed {removed} shard directory{'ies' if removed != 1 else ''}.")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="Collection key")
    parser.add_argument("embed_model", help="Embedding model name")
    parser.add_argument("--storage", default=str(ROOT / "storage"), help="Storage directory")
    args = parser.parse_args()
    cleanup_shards(args.key, args.embed_model, Path(args.storage))


if __name__ == "__main__":
    main()
