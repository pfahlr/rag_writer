"""
Unit tests for error handler functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.utils.error_handler import (
    ErrorHandler,
    RAGError,
    ConfigurationError,
    RetrievalError,
    IndexingError,
    LLMError,
    FileSystemError,
    ValidationError,
    handle_and_exit,
    validate_collection,
    validate_openai_key
)


class TestErrorHandler:
    """Test ErrorHandler class functionality."""

    def test_get_error_suggestions_collection_not_found(self):
        """Test suggestions for collection not found errors."""
        error = FileNotFoundError("Collection 'test' not found")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 3
        assert any("make lc-index" in s for s in suggestions)
        assert any("data_raw" in s for s in suggestions)

    def test_get_error_suggestions_api_key_missing(self):
        """Test suggestions for API key errors."""
        error = ValueError("OPENAI_API_KEY not found")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 3
        assert any("OPENAI_API_KEY" in s for s in suggestions)
        assert any(".env" in s for s in suggestions)

    def test_get_error_suggestions_embedding_model(self):
        """Test suggestions for embedding model errors."""
        error = Exception("embedding model download failed")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 3
        assert any("internet connection" in s for s in suggestions)
        assert any("EMBED_MODEL" in s for s in suggestions)

    def test_get_error_suggestions_file_not_found(self):
        """Test suggestions for file not found errors."""
        error = FileNotFoundError("file.txt not found")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 2
        assert any("file/directory exists" in s for s in suggestions)

    def test_get_error_suggestions_memory_error(self):
        """Test suggestions for memory-related errors."""
        error = MemoryError("out of memory")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 3
        assert any("EMBED_BATCH" in s for s in suggestions)
        assert any("RAM" in s for s in suggestions)

    def test_get_error_suggestions_network_error(self):
        """Test suggestions for network-related errors."""
        error = ConnectionError("network timeout")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 2
        assert any("internet connection" in s for s in suggestions)

    def test_get_error_suggestions_generic_error(self):
        """Test suggestions for generic errors."""
        error = Exception("some random error")
        suggestions = ErrorHandler._get_error_suggestions(error)

        assert len(suggestions) >= 2
        assert any("DEBUG=1" in s for s in suggestions)

    def test_validate_collection_exists_success(self, tmp_path):
        """Test successful collection validation."""
        collection_dir = tmp_path / "faiss_test"
        collection_dir.mkdir()

        # Should not raise an exception
        ErrorHandler.validate_collection_exists("test", tmp_path)

    def test_validate_collection_exists_missing(self, tmp_path):
        """Test validation when collection doesn't exist."""
        with pytest.raises(FileSystemError) as exc_info:
            ErrorHandler.validate_collection_exists("missing", tmp_path)

        assert "not found" in str(exc_info.value)
        assert exc_info.value.error_code == "COLLECTION_NOT_FOUND"

    def test_validate_collection_exists_not_directory(self, tmp_path):
        """Test validation when path exists but is not a directory."""
        collection_file = tmp_path / "faiss_test"
        collection_file.write_text("not a directory")

        with pytest.raises(FileSystemError) as exc_info:
            ErrorHandler.validate_collection_exists("test", tmp_path)

        assert "not a directory" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_COLLECTION_PATH"

    def test_validate_api_key_success(self):
        """Test successful API key validation."""
        # Should not raise an exception
        ErrorHandler.validate_api_key("sk-test123456789", "OpenAI")

    def test_validate_api_key_missing(self):
        """Test validation with missing API key."""
        with pytest.raises(ConfigurationError) as exc_info:
            ErrorHandler.validate_api_key(None, "OpenAI")

        assert "not found" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_API_KEY"

    def test_validate_api_key_invalid_format(self):
        """Test validation with invalid API key format."""
        with pytest.raises(ConfigurationError) as exc_info:
            ErrorHandler.validate_api_key("invalid-key", "OpenAI")

        assert "Invalid OpenAI API key format" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_API_KEY_FORMAT"

    def test_validate_api_key_custom_service(self):
        """Test validation with custom service name."""
        # Should not raise an exception for non-OpenAI services
        ErrorHandler.validate_api_key("any-key", "CustomService")

    def test_create_error_message(self):
        """Test error message creation."""
        error = ValueError("test error")
        message = ErrorHandler.create_error_message(error, "test operation")

        assert "Failed to test operation: test error" == message

    def test_create_error_message_no_operation(self):
        """Test error message creation without operation context."""
        error = ValueError("test error")
        message = ErrorHandler.create_error_message(error, None)

        assert "Failed to None: test error" == message


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_rag_error_creation(self):
        """Test RAGError creation."""
        error = RAGError("test message", "TEST_CODE", {"key": "value"})

        assert str(error) == "test message"
        assert error.error_code == "TEST_CODE"
        assert error.details == {"key": "value"}

    def test_configuration_error_creation(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError("config error")

        assert str(error) == "config error"
        assert isinstance(error, RAGError)

    def test_retrieval_error_creation(self):
        """Test RetrievalError creation."""
        error = RetrievalError("retrieval error")

        assert str(error) == "retrieval error"
        assert isinstance(error, RAGError)

    def test_indexing_error_creation(self):
        """Test IndexingError creation."""
        error = IndexingError("indexing error")

        assert str(error) == "indexing error"
        assert isinstance(error, RAGError)

    def test_llm_error_creation(self):
        """Test LLMError creation."""
        error = LLMError("llm error")

        assert str(error) == "llm error"
        assert isinstance(error, RAGError)

    def test_file_system_error_creation(self):
        """Test FileSystemError creation."""
        error = FileSystemError("file system error")

        assert str(error) == "file system error"
        assert isinstance(error, RAGError)

    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationError("validation error")

        assert str(error) == "validation error"
        assert isinstance(error, RAGError)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_handle_and_exit_calls_error_handler(self):
        """Test that handle_and_exit calls ErrorHandler."""
        error = ValueError("test error")

        with patch('src.utils.error_handler.ErrorHandler.handle_error') as mock_handle:
            with patch('sys.exit') as mock_exit:
                handle_and_exit(error, "test context", 42)

        mock_handle.assert_called_once_with(error, "test context", 42)

    def test_validate_collection_calls_error_handler(self, tmp_path):
        """Test that validate_collection calls ErrorHandler."""
        collection_dir = tmp_path / "faiss_test"
        collection_dir.mkdir()

        with patch('src.utils.error_handler.ErrorHandler.validate_collection_exists') as mock_validate:
            validate_collection("test", tmp_path)

        mock_validate.assert_called_once_with("test", tmp_path)

    def test_validate_openai_key_calls_error_handler(self):
        """Test that validate_openai_key calls ErrorHandler."""
        with patch('src.utils.error_handler.ErrorHandler.validate_api_key') as mock_validate:
            validate_openai_key("sk-test123")

        mock_validate.assert_called_once_with("sk-test123", "OpenAI")