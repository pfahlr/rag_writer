import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def cleanup_faiss_module(monkeypatch):
    original = sys.modules.get("faiss")
    yield
    if "faiss" in sys.modules:
        del sys.modules["faiss"]
    if original is not None:
        sys.modules["faiss"] = original


def test_is_faiss_gpu_available_false_when_module_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    faiss_utils = importlib.import_module("src.core.faiss_utils")
    importlib.reload(faiss_utils)

    assert faiss_utils.is_faiss_gpu_available() is False


def test_is_faiss_gpu_available_true_when_functions_present(monkeypatch):
    fake_faiss = SimpleNamespace(
        StandardGpuResources=object,
        index_cpu_to_all_gpus=lambda index: f"gpu:{index}",
        index_gpu_to_cpu=lambda index: f"cpu:{index}",
        get_num_gpus=lambda: 2,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    faiss_utils = importlib.import_module("src.core.faiss_utils")
    importlib.reload(faiss_utils)

    assert faiss_utils.is_faiss_gpu_available() is True

    gpu_index = faiss_utils.clone_index_to_gpu("idx")
    assert gpu_index == "gpu:idx"

    cpu_index = faiss_utils.clone_index_to_cpu(gpu_index)
    assert cpu_index == "cpu:gpu:idx"


def test_clone_index_to_gpu_returns_none_when_unavailable(monkeypatch):
    fake_faiss = SimpleNamespace(get_num_gpus=lambda: 0)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    faiss_utils = importlib.import_module("src.core.faiss_utils")
    importlib.reload(faiss_utils)

    assert faiss_utils.is_faiss_gpu_available() is False
    assert faiss_utils.clone_index_to_gpu("idx") is None

    # Without GPU helpers, clone_index_to_cpu should return the input unchanged
    assert faiss_utils.clone_index_to_cpu("idx") == "idx"
