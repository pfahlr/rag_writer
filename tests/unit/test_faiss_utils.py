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


def _reload_utils():
    faiss_utils = importlib.import_module("src.core.faiss_utils")
    return importlib.reload(faiss_utils)


def test_gpu_helpers_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    faiss_utils = _reload_utils()

    assert faiss_utils.is_faiss_gpu_available() is False
    assert faiss_utils.try_index_cpu_to_gpu("idx") is None
    assert faiss_utils.ensure_cpu_index("idx") == "idx"

    # Even without FAISS, setting threads should be a no-op
    faiss_utils.set_faiss_threads(4)


def test_gpu_helpers_present(monkeypatch):
    calls: list[tuple[str, int]] = []

    class FakeFaiss(SimpleNamespace):
        def get_num_gpus(self):
            return 2

        def index_cpu_to_all_gpus(self, index):
            return f"gpu:{index}"

        def index_gpu_to_cpu(self, index):
            return f"cpu:{index}"

        def omp_set_num_threads(self, value: int):
            calls.append(("omp", value))

        def set_num_threads(self, value: int):
            calls.append(("set", value))

    fake_faiss = FakeFaiss(StandardGpuResources=object)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    faiss_utils = _reload_utils()

    assert faiss_utils.is_faiss_gpu_available() is True

    faiss_utils.set_faiss_threads(8)
    assert ("omp", 8) in calls

    gpu_index = faiss_utils.try_index_cpu_to_gpu("idx")
    assert gpu_index == "gpu:idx"

    cpu_index = faiss_utils.ensure_cpu_index(gpu_index)
    assert cpu_index == "cpu:gpu:idx"


def test_set_threads_falls_back(monkeypatch):
    calls: list[tuple[str, int]] = []

    class FakeFaiss(SimpleNamespace):
        def get_num_gpus(self):
            return 1

        def index_cpu_to_all_gpus(self, index):
            return f"gpu:{index}"

        def index_gpu_to_cpu(self, index):
            return f"cpu:{index}"

        def set_num_threads(self, value: int):
            calls.append(("set", value))

    fake_faiss = FakeFaiss(StandardGpuResources=object)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    faiss_utils = _reload_utils()

    faiss_utils.set_faiss_threads(6)
    assert calls == [("set", 6)]


def test_try_index_cpu_to_gpu_returns_none_when_not_supported(monkeypatch):
    fake_faiss = SimpleNamespace(get_num_gpus=lambda: 0)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    faiss_utils = _reload_utils()

    assert faiss_utils.is_faiss_gpu_available() is False
    assert faiss_utils.try_index_cpu_to_gpu("idx") is None
    assert faiss_utils.ensure_cpu_index("idx") == "idx"


def test_ensure_cpu_index_handles_missing_import(monkeypatch):
    """Regression: missing importlib import should not raise."""

    monkeypatch.delitem(sys.modules, "faiss", raising=False)

    faiss_utils = _reload_utils()

    sentinel = object()
    assert faiss_utils.ensure_cpu_index(sentinel) is sentinel
