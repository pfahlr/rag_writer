#!/usr/bin/env python3
from pathlib import Path
import json
from typing import List
from tqdm import tqdm
from rich.pretty import pprint
import re
import os
import math
import shutil
import argparse

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
    resume: bool = False,
    keep_shards: bool = False,
):
    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]
    n_shards = math.ceil(len(texts) / shard_size)

    for emb in tqdm(
        embedding_models, desc="Embedding models", unit="model", leave=True
    ):
        emb_name = _fs_safe(emb)
        base_dir = Path(f"storage/faiss_{key}__{emb_name}")
        shards_dir = base_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        embedder = HuggingFaceEmbeddings(model_name=emb)
        completed_index_file = base_dir / "index.faiss"
        
        if resume and completed_index_file.exists():
            continue

        existing_shards = 0
        if resume:
            existing_shards = sum(
                1
                for p in shards_dir.glob("shard_*")
                if (p / "index.faiss").exists()
            )

        shard_range = range(existing_shards * shard_size, len(texts), shard_size)
        shard_bar = tqdm(
            shard_range,
            total=max(n_shards - existing_shards, 0),
            unit="shard",
            leave=False,
        )
        for shard_idx, start in enumerate(shard_bar, existing_shards):
            shard_bar.set_description(f"{emb_name} shard {shard_idx}")
            shard_path = shards_dir / f"shard_{shard_idx:03d}"
            if resume and (shard_path / "index.faiss").exists():
                continue
            slice_texts = texts[start : start + shard_size]
            slice_metas = metadatas[start : start + shard_size]
            vs = FAISS.from_texts(
                texts=slice_texts, embedding=embedder, metadatas=slice_metas
            )
            vs.save_local(str(shard_path))
        shard_bar.close()

        shard_paths = sorted(shards_dir.glob("shard_*"))
        vectorstore = None
        for shard_path in tqdm(
            shard_paths, desc=f"Merging {emb_name}", unit="shard"
        ):
            vs = FAISS.load_local(
                str(shard_path),
                embeddings=embedder,
                allow_dangerous_deserialization=True,
            )
            if vectorstore is None:
                vectorstore = vs
            else:
                vectorstore.merge_from(vs)

        base_dir.mkdir(parents=True, exist_ok=True)
        if vectorstore is not None:
            vectorstore.save_local(str(base_dir))
            print(f"[build] wrote FAISS: {base_dir}")

        if not keep_shards:
            shutil.rmtree(shards_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Existing PDF ingestion + chunking
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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
            except ValueError:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "key",
        nargs="?",
        default=os.getenv("RAG_KEY", "default"),
        help="Storage key prefix",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of chunks per shard",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip shards already built"
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Do not delete shard directories after merge",
    )
    args = parser.parse_args()

    key = (args.key or "default").strip() or "default"

    print("Parsing PDFs…")
    pages = load_pdfs()
    print(f"Splitting {len(pages)} pages into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = list(
        tqdm(splitter.split_documents(pages), desc="Splitting", unit="chunk")
    )

    # Normalize chunks JSONL and build FAISS indexes for multiple models
    chunks_out = Path(f"data_processed/lc_chunks_{key}.jsonl")
    write_chunks_jsonl(chunks, chunks_out)

    build_faiss_for_models(
        chunks,
        key,
        embedding_models=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ],
        shard_size=args.shard_size,
        resume=args.resume,
        keep_shards=args.keep_shards,
    )


if __name__ == "__main__":
    main()
