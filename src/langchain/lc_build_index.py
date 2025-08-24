#!/usr/bin/env python3
from pathlib import Path
import sys, json, hashlib
from typing import List
from tqdm import tqdm
from rich.pretty import pprint
import re


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

KEY = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("RAG_KEY", "default")).strip() or "default"

PDF_DIR = f"{ROOT}/data_raw"
IDX_DIR = f"{ROOT}/storage/faiss_{KEY}"
ART_DIR = f"{ROOT}/data_processed"
CHUNKS =  f"{ART_DIR}/lc_chunks_{KEY}.jsonl"
Path(IDX_DIR).mkdir(parents=True, exist_ok=True)
Path(ART_DIR).mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
BATCH = int(os.getenv("EMBED_BATCH", "128"))

DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]')

def load_pdfs() -> List[Document]:
  docs = []
  for pdf in sorted(Path(PDF_DIR).glob("**/*.pdf")):
    pprint("loading pdf: "+str(pdf))
    loader = PyPDFLoader(str(pdf))
    per_page = loader.load()
    doi = get_doi(per_page)
    if doi: 
      pprint("found DOI (this is only a guess, you must verify):"+ doi)
    for d in per_page:
      meta = dict(d.metadata)
      meta["title"] = pdf.stem
      meta["source"] = str(pdf)
      meta['DOI'] = doi
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
    count = count+1
    

def write_chunks(chunks: List[Document]):
  with Path(CHUNKS).open("w", encoding="utf-8") as out:
    for d in chunks:
      h = hashlib.sha256((d.page_content + str(d.metadata)).encode()).hexdigest()[:12]
      rec = {"id": h, "text": d.page_content, "metadata": d.metadata}
      out.write(json.dumps(rec, ensure_ascii=False) + '\n')
  print(f"Wrote {len(chunks)} chunks -> {CHUNKS}")


def read_chunks() -> List[Document]:
  docs = []
  with Path(CHUNKS).open("r", encoding="utf-8") as f:
    for line in f:
      r = json.loads(line)
      docs.append(Document(page_content=r["text"], metadata=r["metadata"]))
  print(f"Loaded {len(docs)} chunks from {CHUNKS}")
  return docs


def main():
  if Path(CHUNKS).exists():
    print(f"Loading cached chunks ({Path(CHUNKS).name})…")
    chunks = read_chunks()
  else:
    print("Parsing PDFs…")
    pages = load_pdfs()
    print(f"Splitting {len(pages)} pages into chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = list(tqdm(splitter.split_documents(pages), desc="Splitting", unit="chunk"))
    #pprint(chunks)
    write_chunks(chunks)

  texts = [d.page_content for d in chunks]
  metas = [d.metadata for d in chunks]


  print(f"Embedding {len(texts)} chunks in batches of {BATCH}… (key={KEY})")
  emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

  vectors = []
  for i in tqdm(range(0, len(texts), BATCH), desc="Embedding", unit="batch"):
    batch = texts[i:i+BATCH]
    vectors.extend(emb.embed_documents(batch))

  txt_embeddings = list(zip(texts, vectors))

  print("Building FAISS index…")
  try:
    vs = FAISS.from_embeddings(embeddings=vectors, metadatas=metas, embedding=emb, text_embeddings=txt_embeddings)
  except TypeError:
    vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)

  print("Saving FAISS index…")
  vs.save_local(str(IDX_DIR))
  print("LangChain FAISS index saved to", IDX_DIR)


if __name__ == "__main__":
  main()
