"""Regression tests for ``src.core.faiss_merge_helpers``."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_import_succeeds_when_faiss_missing(monkeypatch):
    """The helper module should not hard-crash if FAISS is absent."""

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faiss":
            raise ModuleNotFoundError("faiss not available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "src.core.faiss_merge_helpers"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    with pytest.raises(ModuleNotFoundError):
        module.merge_faiss_vectorstores_cpu(object(), object())
