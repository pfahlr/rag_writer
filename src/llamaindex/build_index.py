#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
from tqdm import tqdm

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ROOT = Path(root_dir)
KEY = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("RAG_KEY", "default")).strip() or "default"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")

DATA = ROOT / f"data_processed/chunks_{KEY}.jsonl"
# Backward compat: if keyed file doesn't exist, fall back to unkeyed
if not DATA.exists():
    DATA = ROOT / "data_processed/chunks.jsonl"

DB_DIR = ROOT / f"storage/lancedb_{KEY}"
DB_DIR.mkdir(parents=True, exist_ok=True)

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
EMB = Settings.embed_model
BATCH = int(os.getenv("EMBED_BATCH", "128"))


def load_records():
    with DATA.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build():
    vector_store = LanceDBVectorStore(uri=str(DB_DIR), table_name="chunks")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    recs = list(load_records())
    print(f"Preparing {len(recs)} nodes… (key={KEY}) from {DATA.name}")
    nodes = [
        TextNode(
            text=r["text"],
            metadata={
                "doc_id": r.get("doc_id"),
                "title": r.get("title"),
                "source_path": r.get("source_path") or r.get("metadata", {}).get("source"),
                "page_start": r.get("page_start") or r.get("metadata", {}).get("page"),
                "page_end": r.get("page_end") or r.get("metadata", {}).get("page"),
            },
        )
        for r in recs
    ]

    print(f"Embedding {len(nodes)} nodes in batches of {BATCH}…")
    for i in tqdm(range(0, len(nodes), BATCH), desc="Embedding", unit="batch"):
        batch = nodes[i:i+BATCH]
        embs = EMB.get_text_embedding_batch([n.get_content() for n in batch])
        for n, e in zip(batch, embs):
            n.embedding = e
        vector_store.add(nodes=batch)

    VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    storage_context.persist(persist_dir=str(DB_DIR))
    print("LanceDB index built at", DB_DIR)


if __name__ == "__main__":
    build()
