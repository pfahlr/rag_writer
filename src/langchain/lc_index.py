#!/usr/bin/env python3
"""Backward-compatible wrapper for lc_build_index."""

try:
    from .lc_build_index import main
except ImportError:  # pragma: no cover - fallback for direct script execution
    from lc_build_index import main

if __name__ == "__main__":
    main()
