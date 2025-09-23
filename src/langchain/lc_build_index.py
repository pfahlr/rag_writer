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
import sys
import inspect

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# your existing helpers
from src.core import faiss_utils
# robust CPU-only FAISS merge helper
from src.core.faiss_merge_helpers import merge_faiss_vectorstores_cpu

# Prefer langchain-huggingface (new home), fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# FAISS threading control (respect optional env override)
# ---------------------------------------------------------------------------
try:
    VIRTUAL_CPU_COUNT = int(os.getenv("VIRTUAL_CPU_COUNT", os.cpu_count() or 1))
except ValueError:
    VIRTUAL_CPU_COUNT = os.cpu_count() or 1
if VIRTUAL_CPU_COUNT > (os.cpu_count() or 1):
    VIRTUAL_CPU_COUNT = os.cpu_count() or 1
faiss_utils.set_faiss_threads(VIRTUAL_CPU_COUNT)

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR: Path | str = ROOT / "data_raw"
CHUNKS_DIR: Path | str = ROOT / "data_processed"
INDEX_DIR: Path | str = ROOT / "storage"

DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
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

    key_safe = _fs_safe(key)

    for emb in tqdm(
        embedding_models, desc="Embedding models", unit="model", leave=True
    ):
        emb_name = _fs_safe(emb)
        base_dir = Path(INDEX_DIR) / f"faiss_{key_safe}__{emb_name}"
        shards_dir = base_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        # Put embeddings on chosen device (GPU/MPS/CPU). encode_kwargs not required.
        try:
            embedder = HuggingFaceEmbeddings(
                model_name=emb,
                model_kwargs={"device": device},
            )
        except TypeError:
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
            if not slice_texts:
                continue
            slice_metas = metadatas[start : start + shard_size]
            vs = FAISS.from_texts(
                texts=slice_texts, embedding=embedder, metadatas=slice_metas
            )
            # Save shard (CPU format on disk)
            vs.save_local(str(shard_path))
        shard_bar.close()

        shard_paths = sorted(shards_dir.glob("shard_*"))
        vectorstore = None
        logger.info("Merging %s shards on CPU", emb_name)

        # Guard to ensure we didn't accidentally turn a vectorstore into a raw faiss.Index
        def _assert_is_vectorstore(obj, label):
            if not hasattr(obj, "index"):
                raise TypeError(
                    f"{label} is not a LangChain-compatible vectorstore (got {type(obj)})"
                )

        for shard_path in tqdm(
            shard_paths, desc=f"Merging {emb_name}", unit="shard"
        ):
            vs = FAISS.load_local(
                str(shard_path),
                embeddings=embedder,
                allow_dangerous_deserialization=True,
            )
            # Ensure CPU index for robust merging
            vs.index = faiss_utils.ensure_cpu_index(vs.index)
            _assert_is_vectorstore(vs, "vs")

            if faiss_utils.is_faiss_gpu_available():
                probe_gpu = faiss_utils.clone_index_to_gpu(vs.index)
                if probe_gpu is not None:
                    vs.index = faiss_utils.clone_index_to_cpu(probe_gpu)

            if vectorstore is None:
                vectorstore = vs
            else:
                # Keep accumulator on CPU and use a FAISS-native robust merge when
                # working with real LangChain vectorstores. Dummy test doubles fall
                # back to their native merge implementation.
                gpu_index = None
                if faiss_utils.is_faiss_gpu_available():
                    gpu_index = faiss_utils.clone_index_to_gpu(vectorstore.index)
                    if gpu_index is not None:
                        vectorstore.index = gpu_index

                if hasattr(vectorstore, "docstore") and hasattr(vectorstore, "index_to_docstore_id"):
                    vectorstore.index = faiss_utils.ensure_cpu_index(vectorstore.index)
                    _assert_is_vectorstore(vectorstore, "vectorstore")
                    vectorstore = merge_faiss_vectorstores_cpu(vectorstore, vs)
                else:
                    vectorstore.merge_from(vs)

                if gpu_index is not None:
                    vectorstore.index = faiss_utils.clone_index_to_cpu(vectorstore.index)

        base_dir.mkdir(parents=True, exist_ok=True)
        if vectorstore is not None:
            _assert_is_vectorstore(vectorstore, "vectorstore (pre-save)")
            # Always save CPU index to disk
            vectorstore.index = faiss_utils.ensure_cpu_index(vectorstore.index)
            logger.info("Saving FAISS index for %s to %s", emb_name, base_dir)
            vectorstore.save_local(str(base_dir))
            print(f"[build] wrote FAISS: {base_dir}")

            # Optional: after saving, copy to GPU for serving (in-memory only)
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
# PDF ingestion + chunking
# ---------------------------------------------------------------------------

def load_pdfs() -> List[Document]:
    docs: list[Document] = []
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
    global ROOT, PDF_DIR, CHUNKS_DIR, INDEX_DIR
    ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
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
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT / "data_raw"),
        help="Path to directory containing source files for index",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=str(ROOT / "data_processed"),
        help="Path to directory to store chunks",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(ROOT / "storage"),
        help=(
            "Path to directory containing index directories (i.e., storage) not "
            "individual index directories, the collection of them"
        ),
    )
    args = parser.parse_args()
    PDF_DIR = Path(args.input_dir).expanduser()
    CHUNKS_DIR = Path(args.chunks_dir).expanduser()
    INDEX_DIR = Path(args.index_dir).expanduser()

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
    key_safe = _fs_safe(key)
    chunks_out = Path(CHUNKS_DIR) / f"lc_chunks_{key_safe}.jsonl"
    write_chunks_jsonl(chunks, chunks_out)

    build_kwargs: dict[str, object] = {}
    build_signature = inspect.signature(build_faiss_for_models)
    if "device" in build_signature.parameters:
        build_kwargs["device"] = device
    if "serve_gpu" in build_signature.parameters:
        build_kwargs["serve_gpu"] = args.serve_gpu

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
        **build_kwargs,
    )


if __name__ == "__main__":
    main()

