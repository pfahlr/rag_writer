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
import logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.core import faiss_utils
# Prefer langchain-huggingface (new home), fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for chunk export and multi-model FAISS building
# ---------------------------------------------------------------------------


def pick_device(no_gpu: bool) -> str:
    """Select the torch device for embedding inference."""

    if no_gpu:
        device = "cpu"
    else:
        try:
            import torch  # type: ignore
        except ImportError:
            device = "cpu"
        else:
            cuda_runtime = getattr(torch, "cuda", None)
            has_cuda = bool(
                cuda_runtime
                and callable(getattr(cuda_runtime, "is_available", None))
                and cuda_runtime.is_available()
            )
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            has_mps = bool(
                mps_backend
                and callable(getattr(mps_backend, "is_available", None))
                and mps_backend.is_available()
            )

            if has_cuda:
                device = "cuda"
            elif has_mps:
                device = "mps"
            else:
                device = "cpu"

    logger.info("Selected embedding device: %s", device)
    return device

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
    device: str = "cpu",
    serve_gpu: bool = False,
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
        embedder = HuggingFaceEmbeddings(
            model_name=emb,
            model_kwargs={"device": device},
            encode_kwargs={"device": device},
        )
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
        logger.info("Merging %s shards on CPU", emb_name)
        for shard_path in tqdm(
            shard_paths, desc=f"Merging {emb_name}", unit="shard"
        ):
            vs = FAISS.load_local(
                str(shard_path),
                embeddings=embedder,
                allow_dangerous_deserialization=True,
            )
            vs.index = faiss_utils.ensure_cpu_index(vs.index)
            if vectorstore is None:
                vectorstore = vs
            else:
                vectorstore.index = faiss_utils.ensure_cpu_index(vectorstore.index)
                vectorstore.merge_from(vs)

        base_dir.mkdir(parents=True, exist_ok=True)
        if vectorstore is not None:
            vectorstore.index = faiss_utils.ensure_cpu_index(vectorstore.index)
            logger.info("Saving FAISS index for %s to %s", emb_name, base_dir)
            vectorstore.save_local(str(base_dir))
            print(f"[build] wrote FAISS: {base_dir}")

            if serve_gpu:
                gpu_index = faiss_utils.try_index_cpu_to_gpu(vectorstore.index)
                if gpu_index is not None:
                    vectorstore.index = gpu_index
                    logger.info(
                        "Copied FAISS index for %s to GPU memory for serving", emb_name
                    )
                else:
                    logger.info(
                        "Skipped GPU copy for %s (GPU runtime unavailable)", emb_name
                    )

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
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Skip shards already built (accepts optional value for compatibility)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Do not delete shard directories after merge",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU embeddings even when accelerators are available",
    )
    parser.add_argument(
        "--serve-gpu",
        action="store_true",
        help="Copy the final FAISS index to GPU memory for serving",
    )
    parser.add_argument(
        "--faiss-threads",
        type=int,
        default=None,
        help="Thread count for FAISS operations (default: CPU count)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    key = (args.key or "default").strip() or "default"
    device = pick_device(args.no_gpu)
    faiss_threads = args.faiss_threads or os.cpu_count() or 1
    faiss_utils.set_faiss_threads(faiss_threads)
    logger.info("Configured FAISS thread count: %s", faiss_threads)

    print("Parsing PDFs…")
    pages = load_pdfs()
    print(f"Splitting {len(pages)} pages into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = list(
        tqdm(splitter.split_documents(pages), desc="Splitting", unit="chunk")
    )

    resume_flag = bool(args.resume)
    
    # Normalize chunks JSONL and build FAISS indexes for multiple models
    chunks_out = Path(f"data_processed/lc_chunks_{key}.jsonl")

    # don't rebuild this on --resume
    if not resume_flag or chunks_out.file_size < 100: 
        write_chunks_jsonl(chunks, chunks_out)

    build_faiss_for_models(
        chunks,
        key,
        embedding_models=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ],
        shard_size=args.shard_size,
        resume=resume_flag,
        keep_shards=args.keep_shards,
        device=device,
        serve_gpu=args.serve_gpu,
    )


if __name__ == "__main__":
    main()
