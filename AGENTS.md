# AGENTS.md

> A briefing packet for coding agents (e.g., Codex) working in this repository. Treat this as **source-of-truth** for build, test, style, and project rules. Follow these instructions before editing files or running commands.

---



## 1) Project Overview

**Name:** RAG Writer — Retrieval-Augmented Writing & Analysis

**Purpose:** Automate literature reviews, meta-analysis, and derivative content generation using a multi-agent, RAG-centric pipeline. Outputs include structured notes, reports, and publishable artifacts.

**Core pillars:**

* Modularity (swappable backends and steps)
* Reproducibility (deterministic builds, pinned deps, logged runs)
* Auditability (citations, provenance, config capture)

**Primary language:** Python (backend/CLIs).
**Key frameworks:** FastAPI, SQLModel, Alembic, Rich (logging/UI), Textual (TUI research tools).

---

## 2) Repository Map (current)

```
/                    # repo root
├─ src/              # python app + packages
│  ├─ models/        # SQLModel models & Pydantic schemas
│  ├─ payloads/      # request/response payload schemas (mirrors /schemas/payloads)
│  ├─ services/      # service classes (e.g., messaging components)
│  ├─ api/           # versioned FastAPI routes (e.g., /api/v1)
│  └─ ...
├─ migrations/       # Alembic migration scripts (parallel to /src)
├─ schemas/          # shared schema definitions
│  └─ payloads/      # payload schemas imported by /src/payloads
├─ research/         # collectors, experiments, evaluation templates
├─ eval/             # evaluation configs and gold template files
├─ scripts/          # helper CLI utilities
├─ tests/            # unit/integration tests
├─ .env.example      # sample environment
├─ pyproject.toml    # python build metadata
├─ requirements.txt  # pinned dependencies (if used)
├─ Makefile          # common developer commands
└─ README.md         # human-facing intro
```

---

## 3) Local Setup & Environment

**Python version:** 3.11+ recommended.

### Quick start

```bash
python -m venv .venv
. .venv/bin/activate

pip install -U pip
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
if [ -f pyproject.toml ]; then pip install -e .; fi
```

### Environment variables

Copy `.env.example` → `.env` and fill in as needed. **Never commit secrets.**

Required (typical):

* `DATABASE_URL` — SQLModel/Alembic connection string
* `DATA_DIR` — base path for processed corpora
* `LOG_LEVEL` — default `INFO`

---

## 4) Build, Run, Test

### Make targets

```bash
make setup     # install Python deps
make fmt       # format with ruff + black
make lint      # static checks (ruff, mypy if configured)
make test      # run pytest suite
make dev       # run FastAPI dev server with reload
```

### Direct commands (fallback)

```bash
ruff check --fix . || true
black .
pytest -q
uvicorn src.api.main:app --reload --port 8000
```

---

## 4.1) TDD Protocol (MANDATORY)

> Codex and other coding agents **must follow Test‑Driven Development**. Write/modify tests **before** implementing or changing code. Treat failing tests as your guidance loop.

### Red → Green → Refactor

1. **Red:**

   * For a *new feature*: write failing tests in `tests/` that specify the behavior.
   * For a *bug*: reproduce with a failing **regression test** (name like `test_issue_<id>_regression`).
   * Run `pytest -q` and confirm failures.
2. **Green:**

   * Implement the minimal code to pass the tests.
   * Run `make test` (or `pytest -q`) until green.
3. **Refactor:**

   * Improve code structure, keep tests green.
   * Run `make fmt lint test` to ensure style and static checks.

### Test authoring rules

* **Mirroring:** tests mirror source paths (e.g., `src/services/retrieval/foo.py` → `tests/services/retrieval/test_foo.py`).
* **Coverage gate:** target ≥ **80%** for changed files. If under target, add tests. Command:

  ```bash
  pytest -q --cov=src --cov-report=term-missing
  ```
* **Property tests (where useful):** use `hypothesis` for pure functions and text transforms.
* **Snapshot tests:** for deterministic RAG outputs (store under `tests/snapshots/`).
* **Network isolation:** mock providers and HTTP; mark external calls with `-m external`.
* **Fixtures:** place shared fixtures in `tests/conftest.py` and `tests/fixtures/`.
* **Regression first:** any discovered bug must first land as a failing test.

### Self‑feedback loop for agents

* If tests fail, **read the traceback**, locate the function, and iterate.
* If `ruff`/`mypy` fail, fix style/types **before** pushing.
* Commit tests and implementation together; PR description must note new/changed tests.

### Optional but recommended

* **Mutation tests:** (e.g., `mutmut`) for critical modules.
* **Pre‑commit hooks:** add `pre-commit` with `ruff`, `black`, and `pytest -q` (fast subset) on commit.

---

## 5) RAG Pipeline Contracts

1. **Ingestion → Chunking → Embedding → Indexing** interfaces: do not change function signatures without updating adapters & tests.
2. **Provenance fields** (`doc_id`, `chunk_id`, `source_path`, `page`, `span`) must survive transformations.
3. **Retrieval** must return ranked JSON results consistently.
4. **Synthesis** must include citations or `references[]`.
5. **Evaluation** configs in `eval/` must not break; add metrics incrementally.

---

## 6) Coding Standards

* **Python style:** ruff + black defaults, 88 cols, f-strings, type hints required for public functions.
* **FastAPI:** dependency-injected services; no global mutable state.
* **SQLModel/Alembic:** one migration per schema change.
* **Logging:** use Rich logging factory; no raw `print()`.
* **Errors:** raise typed exceptions with context.
* **Directory discipline:** core logic in `src/`; side effects isolated in `services/`.

Docstrings: Google or NumPy style.

---

## 7) Testing Policy

We practice **Test Driven Development (TDD)**. Codex and other coding agents must always:

1. **Write or update tests first** that describe the desired behavior or bug fix.
2. Run `pytest -q` and observe failures.
3. Implement the minimal code changes to make the new tests pass.
4. Re-run all tests and ensure the full suite is green.

Guidelines:

* For every new function/class, create a corresponding test file in `tests/` mirroring the source path.
* For bug fixes, add regression tests that fail under the old code.
* Do not submit PRs without matching tests.

Run:

```bash
pytest -q
pytest -q -m "not external"  # default in CI
```

**Feedback loop:** Agents must interpret failing test output as self‑feedback, iterating until all tests pass locally before proposing changes.

---

## 8) Data & Security

* Do **not** commit raw PDFs or proprietary datasets.
* Redact PII in stored chunks.
* Secrets in `.env`, not code.

---

## 9) Git Hygiene

* Branches: `feat/…`, `fix/…`, `docs/…`, `chore/…`.
* Conventional commits enforced.
* PRs must:

  * Pass `make fmt lint test`
  * Document migrations or breaking changes

---

## 10) Agent Operating Instructions (Codex, Cursor, etc.)

**You may:**

* Run commands in Sections 3–4.
* Edit code under `src/`, `scripts/`, `tests/`.

**You must:**

* **Follow the TDD protocol in §4.1** (write/modify tests first; ensure red→green→refactor).
* Pass `make fmt lint test` before proposing patches.
* Preserve contracts in Section 5.
* Update this file if commands/structure/env vars change.

**You must NOT:**

* Commit secrets or binaries.
* Strip provenance fields from outputs.
  \*\*
* Commit secrets or binaries.
* Strip provenance fields from outputs.

---

## 11) CI Expectations

* Lint + tests run on PRs.
* CI green is required before merge.

---

## 12) Developer Recipes

These are common **development playbooks** for adding or modifying functionality.

**Add a new ingestion loader**

1. Implement `src/services/ingestion/<name>_loader.py` with `load()` → `Iterable[Doc]`.
2. Register in ingestion factory.
3. Add tests under `tests/ingestion/test_<name>_loader.py`.

**Add a retriever strategy**

1. Implement `src/services/retrieval/<name>.py` with `retrieve(query, k, …)`.
2. Wire into strategy registry.
3. Add tests under `tests/retrieval/`.

**Add an evaluation metric**

1. Create `eval/metrics/<metric>.py`.
2. Add gold examples in `eval/data/`.
3. Write pytest assertions for correctness.

---

## 13) YAML Playbooks (multi-stage operations)

The project also defines **YAML playbooks** that orchestrate multi-stage RAG workflows.

**Format example:**

```yaml
- section: "1B1"
  task: "Define AI literacy for teachers"
  instruction: "Write a 3–4 paragraph overview..."
```

**Fields:**

* `section` — identifier (ties to book chapter/outline)
* `task` — description of what the agent should do
* `instruction` — detailed prompt for generation

Playbooks live in `eval/` or `research/` and are consumed by orchestration scripts.  Agents must preserve field names and structure.

---

## 14) Research CLI Tools

The `research/` directory contains experimental CLIs and TUIs (Textual-based).

**Example: `collector.py`**

* A Textual TUI app for capturing article metadata and notes.
* Key class: `ArticleFormApp(App)`.
* Common issues: ensure correct `LinkClicked` import from Textual (`textual.widgets.Link` events may differ by version).

**To run:**

```bash
python research/collector.py
```

Agents extending these tools should:

* Follow Textual 0.5+ API conventions.
* Keep event handler signatures aligned with framework imports.
* Provide minimal fixtures for tests under `tests/research/`.

---

## 15) File & Naming Conventions

* Python: `snake_case.py`; classes: `PascalCase`; functions/vars: `snake_case`.
* Test files: `tests/<pkg>/test_<module>.py`.
* Configs: YAML/JSON under `config/` or alongside feature.

---

## 16) Contact / Ownership

* Primary owner: @pfahlr

---

## 14) Developer Recipes

**Add a new ingestion loader**

1. Create `src/services/ingestion/<name>_loader.py` exposing `load(path_or_url, **kwargs) -> Iterable[Doc]`.
2. Register it in the ingestion factory/registry.
3. Add tests under `tests/ingestion/test_<name>_loader.py` with small fixtures.

**Add a retriever strategy**

1. Implement `src/services/retrieval/<name>.py` with `retrieve(query: str, k: int = 10, **kwargs) -> list[Hit]`.
2. Wire it into the strategy registry and config.
3. Add tests + a small benchmark in `tests/retrieval/`.

**Add an evaluation metric**

1. Create `eval/metrics/<metric>.py` and register it.
2. Add golden examples in `eval/data/` with expected JSON outputs.
3. Write `pytest` for metric correctness.

---

## 15) YAML Playbooks (Executable Jobs)

> Your YAML playbooks orchestrate multi‑stage operations (e.g., chunk → embed → retrieve → synthesize). Codex should **generate/modify** these files rather than hard‑coding pipelines.

### Minimal job schema

```yaml
# eval/jobs/1b1.yaml (example)
section: "1B1"
task: "Define AI literacy for teachers"
instruction: |
  Write a 3–4 paragraph overview covering core skills and understandings.
```

### Extended job schema (recommended)

```yaml
id: "job-001"
section: "1.B.1"
track: "education"
stages:
  - name: chunk
    params: { strategy: recursive, max_tokens: 1200 }
  - name: embed
    params: { provider: openai, model: text-embedding-3-large }
  - name: retrieve
    params: { k: 12, hybrid: true }
  - name: synthesize
    params: { style: "teacher-friendly", cite: inline }
outputs:
  - type: markdown
    path: "out/chapters/ch1/1b1.md"
metadata:
  sourceset: "edu-core-2025-08"
  references: []
```

### Conventions

* Place jobs under `eval/jobs/` (or the directory you choose and document here).
* File names: lowercase with dots/dashes, e.g., `1a1a.yaml`, `1b1.yaml`.
* Each stage must map to a registered pipeline step in `src/services/`.
* Parameters are **validated**; add schema tests in `tests/eval/test_job_schema.py`.
* Job outputs should include provenance (citations or `references[]`).

### Common commands

```bash
# run a single job
python -m scripts.run_job eval/jobs/1b1.yaml

# run all jobs in a directory
python -m scripts.run_job --glob 'eval/jobs/*.yaml'
```

---

## 16) Research CLI Tools (Textual collector.py, etc.)

Location: `research/collector.py` (Textual 5.x UI for manual article capture/annotation).

### Run

```bash
# from repo root
python research/collector.py
# or
python -m research.collector
```

### Expected environment

* Python 3.11+
* `textual` (v5.x), `rich`, and any parsers you enable

### Notes on link events (Textual 5.x)

* Event class names and handlers changed across Textual versions.
* If you see `NameError: LinkClicked`, ensure you import the correct event symbol for **your installed version** and use the current handler pattern (e.g., decorators like `@on(...)` or the appropriate `on_*` method signature).
* Prefer consulting the versioned API docs for 5.x and updating the handler to match the widget/event you use.

### Developer tasks

* Keep UI state handling isolated; avoid global state.
* Add fixtures for CLI/UI behaviors under `tests/research/`.
* Provide small sample inputs in `research/fixtures/`.

---

> **Reminder for agents:** If you change any command or directory here, reflect that change in this file as part of the patch. This file is authoritative for future runs.

## GLOSSARY
MCP - MCP (Model Context Protocol) is an open-source standard for connecting AI applications to external systems.
Using MCP, AI applications like Claude or ChatGPT can connect to data sources (e.g. local files, databases), tools (e.g. search engines, calculators) and workflows (e.g. specialized prompts)—enabling them to access key information and perform tasks.
Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems. [MODEL CONTEXT PROTOCOL WEBSITE](http://modelcontextprotocol.io/docs)[MODEL CONTEXT PROTOCOL WIKIPEDIA](https://en.wikipedia.org/wiki/Model_Context_Protocol)
