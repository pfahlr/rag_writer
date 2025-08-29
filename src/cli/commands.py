#!/usr/bin/env python3
"""
CLI Commands for RAG Writer

This module contains the command-line interface commands extracted from the
monolithic cli.py file for better maintainability and separation of concerns.
"""

import os
import sys
import json
import typer
from pathlib import Path
from typing import Optional

from ..core.retriever import RetrieverFactory, RetrieverConfig
from ..core.llm import LLMFactory, LLMConfig
from ..config.settings import get_config
from ..utils.error_handler import handle_and_exit


# Global configuration
config = get_config()

# Initialize Typer app
app = typer.Typer(add_completion=False)


def _load_retriever(key: str, k: int = 10, multiquery: bool = True):
    """Load and configure retriever with comprehensive error handling."""
    try:
        factory = RetrieverFactory(config.paths.root_dir)
        retriever_config = RetrieverConfig(
            key=key,
            k=k,
            multiquery=multiquery,
            embedding_model=config.embedding.model_name,
            openai_model=config.llm.openai_model,
            rerank_model=config.retriever.rerank_model,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight
        )
        return factory.create_hybrid_retriever(retriever_config)
    except Exception as e:
        handle_and_exit(e, f"loading retriever for collection '{key}'")


def _llm():
    """Load LLM with comprehensive error handling."""
    try:
        llm_config = LLMConfig(
            model=config.llm.openai_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            openai_api_key=config.openai_api_key,
            ollama_model=config.llm.ollama_model
        )
        factory = LLMFactory(llm_config)
        return factory.create_llm()
    except Exception as e:
        handle_and_exit(e, "initializing LLM")


def _rag_answer(key: str, retrieval_query: str, system_prompt: str, final_question: Optional[str] = None, k: int = 10) -> dict:
    """Generate a RAG answer while allowing separation of retrieval_query and final_question."""
    try:
        retriever = _load_retriever(key, k=k)
    except FileNotFoundError:
        return {
            "error": f"Collection '{key}' not found. Run 'make lc-index {key}' to create it.",
            "answer": "Failed to load collection.",
            "context": []
        }
    except Exception as e:
        return {
            "error": f"Failed to load retriever: {str(e)}",
            "answer": "Failed to load retriever.",
            "context": []
        }

    # Fetch documents using ONLY the retrieval_query
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(retrieval_query)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(retrieval_query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(retrieval_query)
        else:
            docs = retriever(retrieval_query)
    except Exception as e:
        return {"error": f"Retriever failed: {str(e)}", "answer": "Failed to retrieve documents.", "context": []}

    # Initialize LLM
    try:
        backend, llm = _llm()
    except Exception as e:
        return {"error": f"LLM initialization failed: {str(e)}", "answer": "Failed to initialize LLM.", "context": docs}

    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question:\n{question}\n\nReturn a grounded, citation-rich answer."),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # Final question passed to the LLM (task may be prepended)
    q = final_question if final_question is not None and final_question != "" else retrieval_query

    try:
        out = doc_chain.invoke({"input_documents": docs, "question": q})
        # Normalize output into a string answer
        if isinstance(out, dict):
            if "answer" in out:
                answer = out["answer"]
            elif "output" in out:
                answer = out["output"]
            else:
                val = next((v for v in out.values() if isinstance(v, str)), None)
                answer = val if val is not None else str(out)
        elif isinstance(out, str):
            answer = out
        else:
            answer = str(out)
        return {"answer": answer, "context": docs}
    except Exception as e:
        return {"error": f"RAG generation failed: {str(e)}", "answer": "An error occurred while generating the answer.", "context": docs}


def _display_error_with_suggestions(error_msg: str, key: str = None):
    """Display error message with helpful suggestions."""
    print(f"\n[red]Error: {error_msg}[/]")

    if "Collection" in error_msg or "FAISS" in error_msg:
        if key:
            print(f"[yellow]Suggestions:[/]")
            print(f"  • Run 'make lc-index {key}' to create the collection")
            print(f"  • Check if PDFs exist in data_raw/ directory")
            print(f"  • Verify the collection key '{key}' is correct")
        else:
            print(f"[yellow]Suggestions:[/]")
            print(f"  • Run 'make lc-index <key>' to create a collection")
            print(f"  • Check if PDFs exist in data_raw/ directory")

    elif "OPENAI_API_KEY" in error_msg:
        print(f"[yellow]Suggestions:[/]")
        print(f"  • Set OPENAI_API_KEY environment variable")
        print(f"  • Check your .env file or environment configuration")
        print(f"  • Verify your OpenAI API key is valid")

    elif "embedding model" in error_msg.lower():
        print(f"[yellow]Suggestions:[/]")
        print(f"  • Check your internet connection")
        print(f"  • Verify the EMBED_MODEL setting")
        print(f"  • Try a different embedding model")

    print()  # Add spacing


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question or instruction to retrieve on"),
    key: str = typer.Option(config.rag_key, "--key", "-k", help="Collection key"),
    k: int = typer.Option(15, help="Top-k to retrieve"),
    task: str = typer.Option("", "--task", help="Optional task prefix to prepend to final LLM question (excluded from retrieval)"),
    file: str = typer.Option("", "--file", help="File containing prompt question")
):
    """
    CLI entrypoint that supports a separate 'task' prefix which is:
    - excluded from the retriever query (used only to fetch context)
    - prepended to the final question sent to the LLM
    """
    # Determine retrieval instruction and task prefix
    if file:
        with open(file, "r", encoding="utf-8") as f:
            directive = json.load(f)
        instruction = (directive.get("instruction") or directive.get("question") or "").strip()
        file_task = (directive.get("task") or "").strip()
        final_task = task.strip() or file_task
    else:
        instruction = question.strip()
        final_task = task.strip()

    # Use the common system prompt
    system_prompt = (
        "You are a careful research assistant. Use ONLY the provided context. "
        "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y). "
        "If context is insufficient or conflicting, state what is missing and stop."
    )

    result = _rag_answer(key, instruction, system_prompt, final_question=final_task if final_task else None, k=k)

    if "error" in result:
        _display_error_with_suggestions(result['error'], key)
        print(result.get("answer", "No answer generated"))
    else:
        print("\n" + result.get("answer", "No answer generated"))
        print("\n[dim]Type 'sources' to list retrieved source chunks.[/dim]")


if __name__ == "__main__":
    app()