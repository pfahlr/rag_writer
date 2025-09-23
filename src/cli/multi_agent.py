"""CLI entry point for a tool-enabled multi-agent conversation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ..core.llm import LLMFactory
from ..tool import ToolRegistry, create_rag_retrieve_tool, run_agent

app = typer.Typer(add_completion=False)

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "storage"


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to pass to the agent"),
    key: str = typer.Option("default", "--key", "-k", help="FAISS index key"),
    mcp: str | None = typer.Option(
        None, "--mcp", help="Path to MCP server executable to register tools from"
    ),
    index: Path = typer.Option(
        DEFAULT_INDEX_DIR,
        "--index",
        help="Directory containing FAISS index directories (default: [repo]/storage)",
    ),
) -> None:
    """Run the agent loop with RAG and optional MCP tools."""
    registry = ToolRegistry()
    registry.register(create_rag_retrieve_tool(key, index_dir=index))
    if mcp:
        asyncio.run(registry.register_mcp_server(mcp))

    factory = LLMFactory()
    _, llm = factory.create_llm()
    result = run_agent(llm, registry, question)
    typer.echo(result)


if __name__ == "__main__":
    app()
