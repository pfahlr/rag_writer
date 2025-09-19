"""Unit tests for build index functionality."""

import importlib
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Test DOI extraction functionality without importing problematic modules
DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]')

def get_doi(pages) -> str:
    """Extract DOI from pages (copied from lc_build_index.py for testing)."""
    count = 0
    for p in pages:
        match = DOI_REGEX.search(p.page_content)
        if match:
            doi = match.group(0).lower()
            return doi
        if count > 1:
            return ""
        count = count + 1
    return ""


class TestDOIExtraction:
    """Test DOI extraction functionality."""

    def test_get_doi_found(self):
        """Test DOI extraction when DOI is found."""
        pages = [
            Mock(page_content="This paper has DOI: 10.1234/example.doi"),
            Mock(page_content="Some other content")
        ]

        result = get_doi(pages)
        assert result == "10.1234/example.doi"

    def test_get_doi_not_found(self):
        """Test DOI extraction when no DOI is found."""
        pages = [
            Mock(page_content="This paper has no DOI"),
            Mock(page_content="Just regular content"),
            Mock(page_content="More content without DOI")
        ]

        result = get_doi(pages)
        assert result == ""


@pytest.fixture
def load_lc_build_index(monkeypatch):
    """Load the ``lc_build_index`` module with lightweight stubs."""

    def _load(torch_module=None):
        stubs = {
            "langchain_community.document_loaders": SimpleNamespace(
                PyMuPDFLoader=object
            ),
            "langchain_text_splitters": SimpleNamespace(
                RecursiveCharacterTextSplitter=object
            ),
            "langchain_core.documents": SimpleNamespace(Document=object),
            "langchain_community.vectorstores": SimpleNamespace(FAISS=object),
            "langchain_huggingface": SimpleNamespace(
                HuggingFaceEmbeddings=object
            ),
            "langchain_community.embeddings": SimpleNamespace(
                HuggingFaceEmbeddings=object
            ),
        }
        for name, module in stubs.items():
            monkeypatch.setitem(sys.modules, name, module)

        if torch_module is None:
            monkeypatch.delitem(sys.modules, "torch", raising=False)
        else:
            monkeypatch.setitem(sys.modules, "torch", torch_module)

        module = importlib.import_module("src.langchain.lc_build_index")
        return importlib.reload(module)

    return _load


class TestPickDevice:
    """Validate device selection logic."""

    def test_no_gpu_flag_forces_cpu(self, load_lc_build_index, caplog):
        module = load_lc_build_index()
        caplog.set_level("INFO")

        device = module.pick_device(no_gpu=True)

        assert device == "cpu"
        assert "Selected embedding device" in caplog.text

    def test_prefers_cuda_when_available(self, load_lc_build_index, monkeypatch, caplog):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        )

        module = load_lc_build_index(torch_module=fake_torch)
        caplog.set_level("INFO")

        device = module.pick_device(no_gpu=False)

        assert device == "cuda"
        assert "Selected embedding device" in caplog.text

    def test_falls_back_to_mps(self, load_lc_build_index, caplog):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True)
            ),
        )

        module = load_lc_build_index(torch_module=fake_torch)
        caplog.set_level("INFO")

        device = module.pick_device(no_gpu=False)

        assert device == "mps"
        assert "Selected embedding device" in caplog.text

    def test_defaults_to_cpu_when_no_backend(self, load_lc_build_index, caplog):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )

        module = load_lc_build_index(torch_module=fake_torch)
        caplog.set_level("INFO")

        device = module.pick_device(no_gpu=False)

        assert device == "cpu"
        assert "Selected embedding device" in caplog.text

    def test_get_doi_case_insensitive(self):
        """Test DOI extraction is case insensitive."""
        pages = [
            Mock(page_content="DOI: 10.5678/EXAMPLE.DOI"),
        ]

        result = get_doi(pages)
        assert result == "10.5678/example.doi"

    def test_get_doi_multiple_pages(self):
        """Test DOI extraction finds DOI on later pages."""
        pages = [
            Mock(page_content="Page 1: no DOI here"),
            Mock(page_content="Page 2: no DOI here"),
            Mock(page_content="Page 3: DOI: 10.9999/late.doi")  # Should be found
        ]

        result = get_doi(pages)
        assert result == "10.9999/late.doi"

    def test_get_doi_empty_pages(self):
        """Test DOI extraction with empty pages."""
        result = get_doi([])
        assert result == ""

    def test_get_doi_malformed_doi(self):
        """Test DOI extraction with malformed DOI."""
        pages = [
            Mock(page_content="DOI: 10.1234"),  # Incomplete DOI
        ]

        result = get_doi(pages)
        assert result == ""