#!/usr/bin/env python3
from pypdf import PdfReader
from pathlib import Path
import json, re, hashlib

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ROOT = Path(root_dir)

RAW_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "data_processed"
OUT_PATH = OUT_DIR / "chunks.jsonl"  # LlamaIndex chunk artifact

def read_pdf_with_pages(path: Path):
    pdf = PdfReader(str(path))
    for i, page in enumerate(pdf.pages, start=1):
        yield i, (page.extract_text() or "")

def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # de-hyphenate across line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def to_chunks_by_page(doc_path: Path, max_len_chars=3000, overlap=300):
    buf, start_page, last_page, total = [], None, None, 0
    for page_no, raw in read_pdf_with_pages(doc_path):
        txt = clean(raw)
        if not txt:
            continue
        for para in txt.split("\n"):
            if not buf:
                start_page = page_no
            buf.append(para)
            total += len(para) + 1
            last_page = page_no
            if total >= max_len_chars:
                chunk = "\n".join(buf)
                yield chunk, start_page, last_page
                keep = chunk[-overlap:]
                buf, total, start_page = [keep], len(keep), last_page
    if buf:
        yield "\n".join(buf), start_page, last_page

def process_dir():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for pdf in RAW_DIR.glob("**/*.pdf"):
            doc_id = hashlib.sha256(str(pdf).encode()).hexdigest()[:12]
            for chunk, pstart, pend in to_chunks_by_page(pdf):
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(pdf),
                    "title": pdf.stem,
                    "page_start": pstart,
                    "page_end": pend,
                    "text": chunk,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote -> {OUT_PATH}")

if __name__ == "__main__":
    process_dir()
