#!/usr/bin/env python3
"""
Unified LLM Factory for RAG Writer

This module provides a centralized factory for creating various LLM backends
used across the application, eliminating code duplication and providing
consistent error handling.
"""

import os
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM initialization."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    openai_api_key: Optional[str] = None
    ollama_model: str = "llama3.1:8b"
    use_openai: bool = True
    use_ollama: bool = True
    use_raw_openai: bool = True


class LLMFactory:
    """Factory for creating various LLM backends with consistent configuration."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the factory with configuration."""
        if config is None:
            config = LLMConfig()
        self.config = config

    def _try_openai_langchain(self) -> Tuple[str, Any]:
        """Try to initialize OpenAI via LangChain."""
        if not self.config.use_openai:
            raise ImportError("OpenAI LangChain backend disabled")

        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return ("lc_openai", llm)
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI LangChain client: {str(e)}")

    def _try_ollama(self) -> Tuple[str, Any]:
        """Try to initialize Ollama via LangChain."""
        if not self.config.use_ollama:
            raise ImportError("Ollama backend disabled")

        try:
            # Try the new langchain-ollama package first
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=self.config.ollama_model,
                temperature=self.config.temperature
            )
            return ("ollama", llm)
        except ImportError:
            try:
                # Fallback to langchain-community
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(
                    model=self.config.ollama_model,
                    temperature=self.config.temperature
                )
                return ("ollama", llm)
            except ImportError:
                raise ImportError("Ollama package not available. Install with: pip install langchain-ollama")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama client: {str(e)}")

    def _try_raw_openai(self) -> Tuple[str, Any]:
        """Try to initialize raw OpenAI client."""
        if not self.config.use_raw_openai:
            raise ImportError("Raw OpenAI backend disabled")

        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            return ("raw_openai", client)
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize raw OpenAI client: {str(e)}")

    def create_llm(self) -> Tuple[str, Any]:
        """
        Create an LLM instance using the best available backend.

        Returns:
            Tuple of (backend_name, llm_instance)

        Raises:
            RuntimeError: If no usable LLM backend can be initialized
        """
        backends_to_try = [
            ("OpenAI (LangChain)", self._try_openai_langchain),
            ("Ollama", self._try_ollama),
            ("OpenAI (Raw)", self._try_raw_openai)
        ]

        errors = []

        for backend_name, backend_func in backends_to_try:
            try:
                backend, llm = backend_func()
                #print(f"[green]✓ Using {backend_name} backend ({backend})[/green]")
                return backend, llm
            except Exception as e:
                error_msg = f"{backend_name}: {str(e)}"
                errors.append(error_msg)
                print(f"[yellow]⚠ {backend_name} unavailable: {str(e)}[/yellow]")

        # If we get here, no backend worked
        error_summary = "\n".join(f"  • {error}" for error in errors)
        raise RuntimeError(
            f"No usable LLM backend found. Tried:\n{error_summary}\n\n"
            "Please install one of:\n"
            "  • pip install langchain-openai (preferred)\n"
            "  • pip install openai (fallback)\n"
            "  • Install and run Ollama locally\n"
            "And ensure OPENAI_API_KEY is set for OpenAI backends."
        )

    def get_available_backends(self) -> Dict[str, bool]:
        """Check which LLM backends are available."""
        backends = {}

        # Test OpenAI LangChain
        try:
            self._try_openai_langchain()
            backends["openai_langchain"] = True
        except Exception:
            backends["openai_langchain"] = False

        # Test Ollama
        try:
            self._try_ollama()
            backends["ollama"] = True
        except Exception:
            backends["ollama"] = False

        # Test Raw OpenAI
        try:
            self._try_raw_openai()
            backends["raw_openai"] = True
        except Exception:
            backends["raw_openai"] = False

        return backends


# Convenience functions for backward compatibility
def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> Tuple[str, Any]:
    """Convenience function to create an LLM with default settings."""
    config = LLMConfig(model=model, temperature=temperature)
    factory = LLMFactory(config)
    return factory.create_llm()


def get_available_backends() -> Dict[str, bool]:
    """Convenience function to check available backends."""
    factory = LLMFactory()
    return factory.get_available_backends()
