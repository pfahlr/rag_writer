#!/usr/bin/env python3
import os, sys
from pathlib import Path
import typer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import SentenceTransformerRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from langchain_openai import ChatOpenAI

app = typer.Typer(add_completion=False)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ROOT = Path(root_dir)

DEFAULT_KEY = os.getenv("RAG_KEY", "default")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a careful research assistant. Use ONLY the provided context. "
    "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y). "
    "If the context is insufficient or conflicting, say what is missing and stop."
)
USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Answer with bullet points or a short structured narrative. Include page-cited quotes for key claims."
)

def make_retriever(key: str, k: int = 10):
    idx_dir = ROOT / f"storage/faiss_{key}"
    if not idx_dir.exists():
        raise SystemExit(f"FAISS index not found for key '{key}': {idx_dir}\n"
                         f"Build it first with: make lc-index {key}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
    vector = vs.as_retriever(search_kwargs={"k": k})

    # BM25 over the same docs
    all_docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs); bm25.k = k

    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])

    # Optional multi-query expansion (requires OpenAI key)
    if USE_OPENAI:
        mq_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        hybrid = MultiQueryRetriever.from_llm(retriever=hybrid, llm=mq_llm)

    # Rerank top-k with a cross-encoder (DocumentTransformer) wrapped as a compressor pipeline
    rerank = SentenceTransformerRerank(model_name="BAAI/bge-reranker-base", top_n=k)
    compressor = DocumentCompressorPipeline(transformers=[rerank])

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid
    )

@app.command()
def main(
    question: str = typer.Argument(..., help="Your question"),
    key: str = typer.Option(DEFAULT_KEY, "--key", "-k", help="Collection key (e.g., llms_education)"),
    k: int = typer.Option(10, help="Top-k to retrieve before compression/rerank")
):
    retriever = make_retriever(key, k=k)

    if USE_OPENAI:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    else:
        raise SystemExit("No OPENAI_API_KEY set and no local LLM configured.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    result = rag_chain.invoke({"question": question})
    print(result["answer"])

    print("\nSOURCES:")
    for d in result["context"]:
        title = d.metadata.get("title") or Path(d.metadata.get("source"," ")).stem
        page = d.metadata.get("page")
        print(f"- {title} (p.{page}) :: {d.metadata.get('source')}")

if __name__ == "__main__":
    app()
