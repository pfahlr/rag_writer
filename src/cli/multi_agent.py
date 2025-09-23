"""CLI entry point for a tool-enabled multi-agent conversation."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import typer

from ..core.llm import LLMFactory
from ..tool import ToolRegistry, create_rag_retrieve_tool, run_agent
from ..langchain.trace import configure_emitter


DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "storage"

app = typer.Typer(add_completion=False)


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
    trace: bool = typer.Option(False, "--trace", help="Emit TRACE events"),
    trace_file: str | None = typer.Option(
        None,
        "--trace-file",
        help="Optional path to tee TRACE events",
    ),
) -> None:
    """Run the agent loop with RAG and optional MCP tools."""
    emitter = configure_emitter(trace, trace_file=trace_file)
    qid = os.getenv("TRACE_QID")

    registry = ToolRegistry()
    registry.register(create_rag_retrieve_tool(key, index_dir=index))
    if mcp:
        asyncio.run(registry.register_mcp_server(mcp))

    factory = LLMFactory()
    _, llm = factory.create_llm()
    with emitter:
        traced_llm = _wrap_llm_with_trace(llm, emitter, qid)
        _wrap_registry_with_trace(registry, emitter, qid)
        result = run_agent(traced_llm, registry, question)
    typer.echo(result)


def _wrap_llm_with_trace(llm, emitter, qid: str | None):
    if not emitter.enabled:
        return llm

    def _invoke(messages):
        span = emitter.make_span("llm.turn")
        emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": "root",
                "role": "user",
                "type": "llm.prompt",
                "name": "multi_agent.llm",
                "detail": {"messages": messages},
            }
        )
        start = time.perf_counter()
        if hasattr(llm, "invoke"):
            response = llm.invoke(messages)  # type: ignore[attr-defined]
        else:
            response = llm(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = str(response)
        emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": "root",
                "role": "assistant",
                "type": "llm.completion",
                "name": "multi_agent.llm",
                "detail": {"content": content, "finish_reason": "stop"},
                "metrics": {"latency_ms": round(latency_ms, 2)},
            }
        )
        return response

    class _LLMWrapper:
        def invoke(self, messages):
            return _invoke(messages)

    return _LLMWrapper()


def _wrap_registry_with_trace(registry: ToolRegistry, emitter, qid: str | None) -> None:
    if not emitter.enabled:
        return

    original_run = registry.run

    def traced_run(name: str, **kwargs):
        span = emitter.make_span("tool.call")
        event_type = "mcp.call" if name in getattr(registry, "_mcp_tools", {}) else "tool.call"
        emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": "root",
                "role": "mcp" if event_type.startswith("mcp") else "tool",
                "type": event_type,
                "name": name,
                "detail": {"args": kwargs},
            }
        )
        start = time.perf_counter()
        try:
            result = original_run(name, **kwargs)
        except Exception as exc:
            emitter.emit(
                {
                    "qid": qid,
                    "span": span,
                    "parent": "root",
                    "role": "mcp" if event_type.startswith("mcp") else "tool",
                    "type": "mcp.result" if event_type.startswith("mcp") else "tool.result",
                    "name": name,
                    "detail": {"ok": False, "error": str(exc)},
                }
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        detail = {"ok": True}
        if isinstance(result, dict):
            if "items" in result and isinstance(result["items"], list):
                detail["item_count"] = len(result["items"])
            for key in ("items", "result", "output"):
                if key in result:
                    detail[key] = result[key]
                    break
        emitter.emit(
            {
                "qid": qid,
                "span": span,
                "parent": "root",
                "role": "mcp" if event_type.startswith("mcp") else "tool",
                "type": "mcp.result" if event_type.startswith("mcp") else "tool.result",
                "name": name,
                "detail": detail,
                "metrics": {"latency_ms": round(latency_ms, 2)},
            }
        )
        return result

    registry.run = traced_run  # type: ignore[assignment]


if __name__ == "__main__":
    app()
