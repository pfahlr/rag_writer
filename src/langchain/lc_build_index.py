#!/usr/bin/env python3
from pathlib import Path
import sys, json, math
from typing import List
from tqdm import tqdm
from rich.pretty import pprint
import re
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# Prefer langchain-huggingface (new home), fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Helpers for chunk export and multi-model FAISS building
# ---------------------------------------------------------------------------

def _fs_safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)


def write_chunks_jsonl(chunks: List[Document], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, d in enumerate(chunks):
            rec = {
                "id": d.metadata.get("id") or f"chunk-{i}",
                "text": d.page_content,
                "metadata": d.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_faiss_for_models(
    chunks: List[Document],
    key: str,
    embedding_models: list[str],
    shard_size: int = 1000,
):
    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]
    num_shards = max(1, math.ceil(len(texts) / shard_size))
    for emb in tqdm(embedding_models, desc="Embeddings", leave=True):
        emb_name = _fs_safe(emb)
        vs_dir = Path(f"storage/faiss_{key}__{emb_name}")
        vs_dir.mkdir(parents=True, exist_ok=True)
        embedder = HuggingFaceEmbeddings(model_name=emb)
        vs = None
        shard_pbar = tqdm(
            range(num_shards),
            leave=False,
            desc=f"{emb_name} shard 1/{num_shards}",
        )
        for shard_idx in shard_pbar:
            shard_pbar.set_description(
                f"{emb_name} shard {shard_idx + 1}/{num_shards}"
            )
            start = shard_idx * shard_size
            end = min(start + shard_size, len(texts))
            shard_vs = FAISS.from_texts(
                texts=texts[start:end],
                embedding=embedder,
                metadatas=metadatas[start:end],
            )
            if vs is None:
                vs = shard_vs
            else:
                vs.merge_from(shard_vs)
        if vs is not None:
            vs.save_local(str(vs_dir))
            print(f"[build] wrote FAISS: {vs_dir}")


# ---------------------------------------------------------------------------
# Existing PDF ingestion + chunking
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
KEY = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("RAG_KEY", "default")).strip() or "default"
PDF_DIR = f"{ROOT}/data_raw"

DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]')


def load_pdfs() -> List[Document]:
    docs = []
    for pdf in sorted(Path(PDF_DIR).glob("**/*.pdf")):
        pprint("loading pdf: " + str(pdf))
        loader = PyMuPDFLoader(str(pdf))
        per_page = loader.load()
        doi = get_doi(per_page)
        notified_missing_metadata = False
        if doi:
            pprint("found DOI (this is only a guess, you must verify):" + doi)
        for d in per_page:
            meta = dict(d.metadata)
            try:
                meta_extended = json.loads(meta['subject'])
                meta['doi'] = meta_extended['doi']
                meta['isbn'] = meta_extended['isbn']
            except ValueError as e: 
                if not notified_missing_metadata:
                 print('pdf from older version, has no extended metadata')
                 notified_missing_metadata = True
            meta["title"] = pdf.stem
            meta["source"] = str(pdf)
            if 'doi' not in meta or meta['doi'] == "":
                meta['doi'] = doi
            d.metadata = meta
        docs.extend(per_page)
    return docs


def get_doi(pages) -> str:
    count = 0
    for p in pages:
        match = DOI_REGEX.search(p.page_content)
        if match:
            doi = match.group(0).lower()
            return doi
        if count > 1:
            return ""
        count = count + 1


# ---------------------------------------------------------------------------

def main():
    print("Parsing PDFs…")
    pages = load_pdfs()
    print(f"Splitting {len(pages)} pages into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = list(tqdm(splitter.split_documents(pages), desc="Splitting", unit="chunk"))

    # Normalize chunks JSONL and build FAISS indexes for multiple models
    chunks_out = Path(f"data_processed/lc_chunks_{KEY}.jsonl")
    write_chunks_jsonl(chunks, chunks_out)

    build_faiss_for_models(
        chunks,
        KEY,
        embedding_models=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ],
    )


if __name__ == "__main__":
    main()
