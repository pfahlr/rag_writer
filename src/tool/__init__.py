"""Tool interface and built-in tools for RAG Writer."""

from .base import Tool, ToolSpec, ToolRegistry
from .agent import run_agent
from .rag_tool import create_rag_retrieve_tool
from .mcp_client import connect, fetch_tools, call_tool

__all__ = [
    "Tool",
    "ToolSpec",
    "ToolRegistry",
    "run_agent",
    "create_rag_retrieve_tool",
    "connect",
    "fetch_tools",
    "call_tool",
]
