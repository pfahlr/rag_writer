#!/usr/bin/env python3
"""
Standardized Error Handling for RAG Writer

This module provides consistent error handling patterns and user-friendly
error messages across the application.
"""

import sys
from typing import Optional, Dict, Any
from pathlib import Path


class RAGError(Exception):
    """Base exception class for RAG Writer errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(RAGError):
    """Error related to configuration issues."""
    pass


class RetrievalError(RAGError):
    """Error related to document retrieval."""
    pass


class IndexingError(RAGError):
    """Error related to document indexing."""
    pass


class LLMError(RAGError):
    """Error related to LLM operations."""
    pass


class FileSystemError(RAGError):
    """Error related to file system operations."""
    pass


class ValidationError(RAGError):
    """Error related to data validation."""
    pass


class ErrorHandler:
    """Centralized error handling with user-friendly messages and suggestions."""

    @staticmethod
    def handle_error(error: Exception, context: Optional[str] = None, exit_code: int = 1) -> None:
        """
        Handle an error with appropriate user messaging and suggestions.

        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            exit_code: Exit code to use if terminating the program
        """
        error_type = type(error).__name__

        # Create context prefix
        context_prefix = f"[{context}] " if context else ""

        print(f"\n[red]Error: {context_prefix}{str(error)}[/red]")

        # Provide specific suggestions based on error type
        suggestions = ErrorHandler._get_error_suggestions(error)
        if suggestions:
            print(f"\n[yellow]Suggestions:[/yellow]")
            for suggestion in suggestions:
                print(f"  • {suggestion}")

        print()  # Add spacing

        if exit_code != 0:
            sys.exit(exit_code)

    @staticmethod
    def _get_error_suggestions(error: Exception) -> list:
        """Get user-friendly suggestions based on error type and message."""
        error_msg = str(error).lower()
        error_type = type(error).__name__

        suggestions = []

        # Collection/Index related errors
        if any(keyword in error_msg for keyword in ["collection", "faiss", "index"]):
            if "not found" in error_msg:
                suggestions.extend([
                    "Run 'make lc-index <key>' to create the collection",
                    "Check if PDFs exist in data_raw/ directory",
                    "Verify the collection key is correct"
                ])
            else:
                suggestions.extend([
                    "Try rebuilding the index: make lc-index <key>",
                    "Check available disk space",
                    "Verify embedding model is properly installed"
                ])

        # API/OpenAI related errors
        elif any(keyword in error_msg for keyword in ["openai", "api", "key"]):
            suggestions.extend([
                "Set OPENAI_API_KEY environment variable",
                "Check your .env file or environment configuration",
                "Verify your OpenAI API key is valid and has sufficient credits",
                "Check your internet connection"
            ])

        # Embedding model related errors
        elif any(keyword in error_msg for keyword in ["embedding", "model"]):
            suggestions.extend([
                "Check your internet connection for model downloads",
                "Verify the EMBED_MODEL setting is valid",
                "Try a different embedding model",
                "Check available memory (some models require significant RAM)"
            ])

        # File system related errors
        elif any(keyword in error_msg for keyword in ["file", "directory", "path"]):
            if "not found" in error_msg:
                suggestions.extend([
                    "Check if the file/directory exists",
                    "Verify file permissions",
                    "Ensure correct relative paths"
                ])
            elif "permission" in error_msg:
                suggestions.extend([
                    "Check file/directory permissions",
                    "Ensure write access to the target directory",
                    "Try running with appropriate permissions"
                ])

        # Memory related errors
        elif any(keyword in error_msg for keyword in ["memory", "ram", "out of memory"]):
            suggestions.extend([
                "Reduce batch size with EMBED_BATCH environment variable",
                "Process data in smaller chunks",
                "Close other memory-intensive applications",
                "Consider using a machine with more RAM"
            ])

        # Network related errors
        elif any(keyword in error_msg for keyword in ["network", "connection", "timeout"]):
            suggestions.extend([
                "Check your internet connection",
                "Try again in a few moments",
                "Consider using a different network or VPN if applicable"
            ])

        # Generic fallback suggestions
        if not suggestions:
            suggestions.extend([
                "Check the logs for more detailed error information",
                "Try running with DEBUG=1 for verbose output",
                "Ensure all required dependencies are installed"
            ])

        return suggestions

    @staticmethod
    def validate_collection_exists(collection_key: str, storage_dir: Path) -> None:
        """
        Validate that a collection exists and is accessible.

        Args:
            collection_key: The collection key to validate
            storage_dir: The storage directory path

        Raises:
            FileSystemError: If collection doesn't exist or isn't accessible
        """
        faiss_dir = storage_dir / f"faiss_{collection_key}"

        if not faiss_dir.exists():
            raise FileSystemError(
                f"Collection '{collection_key}' not found at {faiss_dir}",
                error_code="COLLECTION_NOT_FOUND",
                details={"collection_key": collection_key, "faiss_dir": str(faiss_dir)}
            )

        # Check if directory is readable
        if not faiss_dir.is_dir():
            raise FileSystemError(
                f"Path exists but is not a directory: {faiss_dir}",
                error_code="INVALID_COLLECTION_PATH",
                details={"collection_key": collection_key, "faiss_dir": str(faiss_dir)}
            )

    @staticmethod
    def validate_api_key(api_key: Optional[str], service_name: str = "OpenAI") -> None:
        """
        Validate that an API key is available.

        Args:
            api_key: The API key to validate
            service_name: Name of the service for error messages

        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        if not api_key:
            raise ConfigurationError(
                f"{service_name} API key not found",
                error_code="MISSING_API_KEY",
                details={"service": service_name}
            )

        # Basic format validation for OpenAI keys
        if service_name == "OpenAI" and not api_key.startswith("sk-"):
            raise ConfigurationError(
                f"Invalid {service_name} API key format",
                error_code="INVALID_API_KEY_FORMAT",
                details={"service": service_name}
            )

    @staticmethod
    def create_error_message(error: Exception, operation: str) -> str:
        """Create a formatted error message with context."""
        return f"Failed to {operation}: {str(error)}"


# Convenience functions for common error handling patterns
def handle_and_exit(error: Exception, context: Optional[str] = None, exit_code: int = 1) -> None:
    """Handle an error and exit the program."""
    ErrorHandler.handle_error(error, context, exit_code)


def validate_collection(collection_key: str, storage_dir: Path) -> None:
    """Validate collection exists and is accessible."""
    ErrorHandler.validate_collection_exists(collection_key, storage_dir)


def validate_openai_key(api_key: Optional[str]) -> None:
    """Validate OpenAI API key."""
    ErrorHandler.validate_api_key(api_key, "OpenAI")