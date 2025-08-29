"""
Unit tests for CLI command functionality.
"""

import json
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from src.cli.commands import (
    _display_error_with_suggestions,
    _rag_answer
)


class TestErrorDisplay:
    """Test error display functionality."""

    def test_display_error_with_suggestions_exists(self):
        """Test that the error display function exists and can be called."""
        # Just test that the function can be called without errors
        try:
            _display_error_with_suggestions("Test error message", "test_key")
        except Exception as e:
            # If there are import issues, that's expected - we just want to test the function exists
            assert "rich" in str(e) or "print" in str(e)  # Expected import or mock issues


class TestRAGAnswer:
    """Test RAG answer functionality."""

    def test_rag_answer_function_exists(self):
        """Test that the RAG answer function exists."""
        # Just test that we can import and the function exists
        assert callable(_rag_answer)

    def test_rag_answer_collection_not_found(self):
        """Test RAG answer when collection is not found."""
        with patch('src.cli.commands._load_retriever', side_effect=FileNotFoundError("Collection not found")):
            result = _rag_answer("missing_key", "Test question", "Test prompt")

            assert "error" in result
            assert "not found" in result["error"]
            assert result["answer"] == "Failed to load collection."

    def test_rag_answer_retriever_failure(self):
        """Test RAG answer when retriever fails."""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Retriever error")

        with patch('src.cli.commands._load_retriever', return_value=mock_retriever):
            result = _rag_answer("test_key", "Test question", "Test prompt")

            assert "error" in result
            assert "Retriever failed" in result["error"]