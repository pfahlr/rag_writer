#!/usr/bin/env python3
"""
Refactored CLI Entry Point for RAG Writer

This is the new, clean entry point that uses the extracted modules
for better maintainability and separation of concerns.
"""

import os
import sys
import argparse
from pathlib import Path

from .config.settings import get_config
from .cli.shell import shell
from .cli.commands import ask
from .utils.error_handler import handle_and_exit


def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Unified RAG CLI + Interactive Shell")
    subparsers = parser.add_subparsers(dest="cmd", help="Sub-commands")

    # Interactive shell (default)
    p_shell = subparsers.add_parser("shell", help="Start interactive shell")
    p_shell.add_argument("--key", "-k", default=None, help="Collection key (e.g., llms_education)")

    # One-shot ask via CLI
    p_ask = subparsers.add_parser("ask", help="Run a single RAG query (non-interactive)")
    p_ask.add_argument("--key", "-k", default=None, help="Collection key")
    p_ask.add_argument("--instruction", "-i", required=True, help="Instruction/query to retrieve on")
    p_ask.add_argument("--task", "-t", default="", help="Optional task prefix prepended to the final LLM question")
    p_ask.add_argument("--k", type=int, default=10, help="Top-k for retriever")

    args = parser.parse_args()

    # Get configuration
    config = get_config()

    # Use provided key or default
    key = args.key or config.rag_key

    if args.cmd == "ask":
        # Handle one-shot ask command
        try:
            # Import the ask function and call it directly
            from .cli.commands import _rag_answer

            system_prompt = (
                "You are a careful research assistant. Use ONLY the provided context. "
                "Every claim MUST include inline citations like (Title, p.X) or (Title, pp.X–Y). "
                "If context is insufficient or conflicting, state what is missing and stop."
            )

            retrieval_query = args.instruction
            final_question = f"{args.task} {retrieval_query}".strip() if args.task else None

            result = _rag_answer(key, retrieval_query, system_prompt, final_question=final_question, k=args.k)

            if "error" in result:
                from .cli.commands import _display_error_with_suggestions
                _display_error_with_suggestions(result['error'], key)
                print(result.get("answer", "No answer generated"))
            else:
                print("\n" + result.get("answer", "No answer generated"))
                print("\n[dim]Type 'sources' to list retrieved source chunks.[/dim]")

        except Exception as e:
            handle_and_exit(e, "running ask command")

    else:
        # Default to interactive shell
        try:
            exit_code = shell(key)
            sys.exit(exit_code)
        except Exception as e:
            handle_and_exit(e, "starting interactive shell")


if __name__ == "__main__":
    main()
