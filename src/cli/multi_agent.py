"""CLI entry point for a tool-enabled multi-agent conversation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ..core.llm import LLMFactory
from ..tool import ToolRegistry, create_rag_retrieve_tool, run_agent

app = typer.Typer(add_completion=False)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to pass to the agent"),
    key: str = typer.Option("default", "--key", "-k", help="FAISS index key"),
    index: Path | None = typer.Option(
        None,
        "--index",
        help=(
            "Directory containing FAISS index folders. Defaults to ./storage inside the"
            " project root."
        ),
    ),
    mcp: str | None = typer.Option(
        None, "--mcp", help="Path to MCP server executable to register tools from"
    ),
) -> None:
    """Run the agent loop with RAG and optional MCP tools."""
    registry = ToolRegistry()
    repo_root = Path(__file__).resolve().parents[2]
    if index is None:
        index_dir = repo_root / "storage"
    else:
        index_dir = index if index.is_absolute() else (repo_root / index).resolve()

    registry.register(create_rag_retrieve_tool(key, index_dir=index_dir))
    if mcp:
        asyncio.run(registry.register_mcp_server(mcp))

    factory = LLMFactory()
    _, llm = factory.create_llm()
    result = run_agent(llm, registry, question)
    typer.echo(result)


if __name__ == "__main__":
    app()
