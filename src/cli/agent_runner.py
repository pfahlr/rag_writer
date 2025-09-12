"""Simple agent orchestrator leveraging a RAG retrieval tool."""

from __future__ import annotations

from typing import Any

from src.tool import ToolRegistry, create_rag_retrieve_tool, run_agent


def run_agent_with_retrieval(question: str, llm: Any, faiss_key: str) -> str:
    """Run the tool-enabled agent for a question using a FAISS index key.

    Parameters
    ----------
    question:
        The user question posed to the agent.
    llm:
        Language model instance compatible with :func:`run_agent`.
    faiss_key:
        Key of the FAISS index used to create the retrieval tool.
    """
    registry = ToolRegistry()
    registry.register(create_rag_retrieve_tool(faiss_key))
    return run_agent(llm, registry, question)
