"""
Unit tests for core LLM functionality.
"""

import pytest
from unittest.mock import patch, Mock

from src.core.llm import (
    LLMConfig,
    LLMFactory,
    create_llm,
    get_available_backends
)


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_default_config(self):
        """Test default LLMConfig values."""
        config = LLMConfig()

        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.openai_api_key is None
        assert config.ollama_model == "llama3.1:8b"
        assert config.use_openai is True
        assert config.use_ollama is True
        assert config.use_raw_openai is True

    def test_custom_config(self):
        """Test custom LLMConfig values."""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            openai_api_key="sk-test123",
            ollama_model="llama3.2:8b",
            use_openai=False,
            use_ollama=True,
            use_raw_openai=False
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.openai_api_key == "sk-test123"
        assert config.ollama_model == "llama3.2:8b"
        assert config.use_openai is False
        assert config.use_ollama is True
        assert config.use_raw_openai is False


class TestLLMFactory:
    """Test LLMFactory class."""

    def test_factory_init_default_config(self):
        """Test factory initialization with default config."""
        factory = LLMFactory()

        assert isinstance(factory.config, LLMConfig)
        assert factory.config.model == "gpt-4o-mini"

    def test_factory_init_custom_config(self):
        """Test factory initialization with custom config."""
        config = LLMConfig(model="gpt-4", temperature=0.5)
        factory = LLMFactory(config)

        assert factory.config.model == "gpt-4"
        assert factory.config.temperature == 0.5

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'})
    def test_try_openai_langchain_success(self):
        """Test successful OpenAI LangChain initialization."""
        config = LLMConfig(use_openai=True)
        factory = LLMFactory(config)

        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm

            backend, llm = factory._try_openai_langchain()

            assert backend == "lc_openai"
            assert llm == mock_llm
            mock_chat_openai.assert_called_once_with(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=None
            )

    def test_try_openai_langchain_disabled(self):
        """Test OpenAI LangChain when disabled."""
        config = LLMConfig(use_openai=False)
        factory = LLMFactory(config)

        with pytest.raises(ImportError, match="OpenAI LangChain backend disabled"):
            factory._try_openai_langchain()

    def test_try_openai_langchain_no_api_key(self):
        """Test OpenAI LangChain without API key."""
        config = LLMConfig(use_openai=True, openai_api_key=None)
        factory = LLMFactory(config)

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                factory._try_openai_langchain()

    def test_try_openai_langchain_import_error(self):
        """Test OpenAI LangChain import error."""
        config = LLMConfig(use_openai=True)
        factory = LLMFactory(config)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('langchain_openai.ChatOpenAI', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="OpenAI package not available"):
                    factory._try_openai_langchain()

    def test_try_ollama_success(self):
        """Test successful Ollama initialization."""
        config = LLMConfig(use_ollama=True)
        factory = LLMFactory(config)

        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_chat_ollama.return_value = mock_llm

            backend, llm = factory._try_ollama()

            assert backend == "ollama"
            assert llm == mock_llm
            mock_chat_ollama.assert_called_once_with(
                model="llama3.1:8b",
                temperature=0.0
            )

    def test_try_ollama_disabled(self):
        """Test Ollama when disabled."""
        config = LLMConfig(use_ollama=False)
        factory = LLMFactory(config)

        with pytest.raises(ImportError, match="Ollama backend disabled"):
            factory._try_ollama()

    def test_try_ollama_import_error(self):
        """Test Ollama import error."""
        config = LLMConfig(use_ollama=True)
        factory = LLMFactory(config)

        with patch('langchain_community.chat_models.ChatOllama', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="Ollama package not available"):
                factory._try_ollama()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'})
    def test_try_raw_openai_success(self):
        """Test successful raw OpenAI initialization."""
        config = LLMConfig(use_raw_openai=True)
        factory = LLMFactory(config)

        with patch('openai.OpenAI') as mock_openai_client:
            mock_client = Mock()
            mock_openai_client.return_value = mock_client

            backend, client = factory._try_raw_openai()

            assert backend == "raw_openai"
            assert client == mock_client
            mock_openai_client.assert_called_once_with(api_key='sk-test123')

    def test_try_raw_openai_disabled(self):
        """Test raw OpenAI when disabled."""
        config = LLMConfig(use_raw_openai=False)
        factory = LLMFactory(config)

        with pytest.raises(ImportError, match="Raw OpenAI backend disabled"):
            factory._try_raw_openai()

    def test_try_raw_openai_no_api_key(self):
        """Test raw OpenAI without API key."""
        config = LLMConfig(use_raw_openai=True, openai_api_key=None)
        factory = LLMFactory(config)

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                factory._try_raw_openai()

    def test_try_raw_openai_import_error(self):
        """Test raw OpenAI import error."""
        config = LLMConfig(use_raw_openai=True)
        factory = LLMFactory(config)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="OpenAI package not available"):
                    factory._try_raw_openai()

    @patch('builtins.print')
    def test_create_llm_success_first_backend(self, mock_print):
        """Test successful LLM creation with first available backend."""
        factory = LLMFactory()

        # Mock successful first backend
        with patch.object(factory, '_try_openai_langchain', return_value=("lc_openai", Mock())):
            backend, llm = factory.create_llm()

            assert backend == "lc_openai"
            assert llm is not None

    @patch('builtins.print')
    def test_create_llm_success_fallback_backend(self, mock_print):
        """Test successful LLM creation with fallback backend."""
        factory = LLMFactory()

        # Mock first backend failure, second success
        with patch.object(factory, '_try_openai_langchain', side_effect=Exception("Failed")):
            with patch.object(factory, '_try_ollama', return_value=("ollama", Mock())):
                with patch.object(factory, '_try_raw_openai', side_effect=Exception("Failed")):
                    backend, llm = factory.create_llm()

                    assert backend == "ollama"
                    assert llm is not None

    @patch('builtins.print')
    def test_create_llm_all_backends_fail(self, mock_print):
        """Test LLM creation when all backends fail."""
        factory = LLMFactory()

        # Mock all backends failing
        with patch.object(factory, '_try_openai_langchain', side_effect=Exception("OpenAI failed")):
            with patch.object(factory, '_try_ollama', side_effect=Exception("Ollama failed")):
                with patch.object(factory, '_try_raw_openai', side_effect=Exception("Raw OpenAI failed")):
                    with pytest.raises(RuntimeError, match="No usable LLM backend found"):
                        factory.create_llm()

    def test_get_available_backends_all_available(self):
        """Test checking available backends when all are available."""
        factory = LLMFactory()

        with patch.object(factory, '_try_openai_langchain', return_value=("lc_openai", Mock())):
            with patch.object(factory, '_try_ollama', return_value=("ollama", Mock())):
                with patch.object(factory, '_try_raw_openai', return_value=("raw_openai", Mock())):
                    backends = factory.get_available_backends()

                    assert backends["openai_langchain"] is True
                    assert backends["ollama"] is True
                    assert backends["raw_openai"] is True

    def test_get_available_backends_none_available(self):
        """Test checking available backends when none are available."""
        factory = LLMFactory()

        with patch.object(factory, '_try_openai_langchain', side_effect=Exception("Failed")):
            with patch.object(factory, '_try_ollama', side_effect=Exception("Failed")):
                with patch.object(factory, '_try_raw_openai', side_effect=Exception("Failed")):
                    backends = factory.get_available_backends()

                    assert backends["openai_langchain"] is False
                    assert backends["ollama"] is False
                    assert backends["raw_openai"] is False

    def test_get_available_backends_partial_available(self):
        """Test checking available backends when some are available."""
        factory = LLMFactory()

        with patch.object(factory, '_try_openai_langchain', return_value=("lc_openai", Mock())):
            with patch.object(factory, '_try_ollama', side_effect=Exception("Failed")):
                with patch.object(factory, '_try_raw_openai', side_effect=Exception("Failed")):
                    backends = factory.get_available_backends()

                    assert backends["openai_langchain"] is True
                    assert backends["ollama"] is False
                    assert backends["raw_openai"] is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('src.core.llm.LLMFactory')
    def test_create_llm_convenience_function(self, mock_factory_class):
        """Test create_llm convenience function."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        mock_factory.create_llm.return_value = ("test_backend", Mock())

        backend, llm = create_llm("gpt-4", 0.5)

        mock_factory_class.assert_called_once()
        args, kwargs = mock_factory_class.call_args
        config = args[0]
        assert config.model == "gpt-4"
        assert config.temperature == 0.5

        mock_factory.create_llm.assert_called_once()
        assert backend == "test_backend"

    @patch('src.core.llm.LLMFactory')
    def test_get_available_backends_convenience_function(self, mock_factory_class):
        """Test get_available_backends convenience function."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        mock_factory.get_available_backends.return_value = {"openai": True, "ollama": False}

        result = get_available_backends()

        mock_factory_class.assert_called_once()
        mock_factory.get_available_backends.assert_called_once()
        assert result == {"openai": True, "ollama": False}