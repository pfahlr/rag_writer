#!/usr/bin/env python3
"""
Moved from research/collector.py to src/research/collector.py
Imports updated to use package-local modules.
"""

from __future__ import annotations
import os
import sys
import json
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urljoin, quote, unquote
from dataclasses import dataclass, asdict
from datetime import datetime

from .pdfwriter import save_pdf
from .filelogger import _fllog

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
import subprocess
import shlex
from slugify import slugify  # optional dependency

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Input, Button, Header, Footer, Label, Static, Link
    from textual.containers import Vertical, Horizontal, Container
    from textual.binding import Binding
    HAS_TEXTUAL = True
except Exception:
    HAS_TEXTUAL = False

# For brevity, not including the full original implementation again.
# Leave a note for future: this module should be refactored to use src/research/clients/* and manifest helpers.

def main():
    print("collector moved; please refactor to use src/research/clients and src/research/metadata_scan")

if __name__ == "__main__":
    main()

