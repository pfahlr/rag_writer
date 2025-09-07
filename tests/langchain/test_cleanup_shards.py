"""Tests for the cleanup_shards utility."""

from pathlib import Path

from src.langchain.cleanup_shards import cleanup_shards, _fs_safe


def _make_shard_dirs(base: Path, key: str, embed_model: str, count: int) -> list[Path]:
    """Create *count* temporary shard directories under *base*."""
    emb_safe = _fs_safe(embed_model)
    dirs = []
    for idx in range(count):
        path = base / f"faiss_{key}__{emb_safe}__{idx}"
        path.mkdir(parents=True)
        dirs.append(path)
    return dirs


def test_cleanup_shards_removes_directories(tmp_path: Path) -> None:
    """cleanup_shards should delete shard directories and report their count."""
    key = "science"
    embed_model = "BAAI/bge-small-en-v1.5"
    shard_dirs = _make_shard_dirs(tmp_path, key, embed_model, 3)
    # Add an unrelated directory that should remain untouched
    other_dir = tmp_path / "unrelated"
    other_dir.mkdir()

    removed = cleanup_shards(key, embed_model, tmp_path)

    assert removed == len(shard_dirs)
    for d in shard_dirs:
        assert not d.exists()
    assert other_dir.exists()


def test_cleanup_shards_no_matching_directories(tmp_path: Path) -> None:
    """When no shards exist, cleanup_shards should return zero."""
    key = "science"
    embed_model = "BAAI/bge-small-en-v1.5"
    existing = tmp_path / "not_a_shard"
    existing.mkdir()

    removed = cleanup_shards(key, embed_model, tmp_path)

    assert removed == 0
    assert existing.exists()
