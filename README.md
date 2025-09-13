# Advanced LangChain Content Processing Suite

A sophisticated suite of scripts for AI-powered content generation, processing, and merging with multi-stage editorial pipelines.

## 🎯 Overview

This suite provides a complete workflow for content creation and processing using LangChain and large language models. From initial content generation through intelligent merging and editorial refinement, it supports both simple workflows and complex multi-stage pipelines.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Makefile Usage](#makefile-usage)
- [Scripts Overview](#scripts-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Pipeline Types](#pipeline-types)
- [API Reference](#api-reference)
- [Tool Agent Schema](#tool-agent-schema)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   lc_ask.py     │    │   lc_batch.py   │    │ lc_merge_runner │
│                 │    │                 │    │                 │
│ • Core LLM      │    │ • Batch Jobs    │    │ • Multi-stage   │
│ • Single Query  │    │ • Parallel Proc │    │ • Critique      │
│ • JSON Output   │    │ • Result Storage│    │ • Merge         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ lc_build_index │
                    │                 │
                    │ • Vector Index  │
                    │ • RAG Support   │
                    │ • Embeddings    │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (if needed)
cp env.json.template env.json
# Edit env.json with your API keys and settings
```

### Basic Usage


#### Option 1: Makefile (Recommended)
```bash
# Complete setup and workflow
make init                # Set up environment
make lc-index KEY=default  # Build FAISS index
make cli-ask "What is machine learning?"  # Ask questions via Typer CLI
make cli-shell           # Interactive shell

# Complete book generation workflow
make book-from-outline OUTLINE="examples/sample_outline_text.txt" TITLE="My Book"
```

#### Option 2: Interactive Examples
```bash
# See all available example files
make examples

# Quick RAG query with custom parameters
make quick-ask "What is machine learning?" KEY="science" CONTENT_TYPE="technical_manual_writer"

# Batch processing with parallel execution
make batch-workflow FILE="examples/sample_jobs_1A1.jsonl" PARALLEL=4
```

#### Option 3: Direct Script Execution
```bash
# 1. Build knowledge index (optional, for RAG)
python src/langchain/lc_build_index.py

# 2. Generate content variations
python src/langchain/lc_batch.py

# 3. Merge and refine content
python src/langchain/lc_merge_runner.py
```


## 📋 Makefile Usage

The enhanced Makefile provides a comprehensive command center for the entire LangChain RAG Writer pipeline. It includes all command-line options from the scripts, workflow automation, and quality tools.

### Getting Help

```bash
make help          # Show comprehensive help with all targets
make examples      # List all available example files
```

### Core Workflow Targets

#### Core CLI Targets
```bash
make init                                # Initialize environment and install dependencies
make lc-index KEY=foo SHARD_SIZE=2000 RESUME=1  # Build sharded FAISS index
make cli-ask "question"                  # RAG query via Typer CLI
make cli-shell                           # Interactive shell
```

#### LangChain Content Generation
```bash
make lc-ask INSTR="instruction" [TASK="task"]   # RAG query with custom parameters
make lc-batch FILE="jobs.jsonl" [PARALLEL=4]    # Batch processing
make lc-merge-runner [SUB=1A1]                  # Content merging
make lc-outline-converter OUTLINE="file.txt"    # Convert outlines to book structure
make lc-book-runner BOOK="book.json"            # Complete book generation
```

#### Quality and Development
```bash
make test                 # Run test suite
make test-coverage        # Run tests with coverage reporting
make format              # Format code with black
make lint                # Lint code with flake8
make quality             # Run full quality check
make show-config         # Display current configuration
make check-setup         # Validate project setup
```

### Advanced Makefile Features

#### Complete Workflows

```bash
# Generate book from outline (complete pipeline)
make book-from-outline OUTLINE="examples/sample_outline_text.txt" TITLE="My Book"

# Quick RAG query with all options
make quick-ask "What is machine learning?" KEY="science" CONTENT_TYPE="technical_manual_writer" K=20

# Batch processing workflow
make batch-workflow FILE="jobs.jsonl" KEY="biology" PARALLEL=4
```


### Makefile Benefits

- **Complete Option Coverage**: All command-line options available as variables
- **Smart Defaults**: Sensible defaults for all parameters
- **Workflow Automation**: Multi-step processes in single commands
- **Error Prevention**: Parameter validation and help messages
- **Quality Tools**: Integrated testing, formatting, and linting
- **Example Discovery**: Easy access to sample files and usage patterns

---
---

## 🖥️ CLI Usage

The project provides multiple CLI interfaces for different use cases:

### 📜 Scripts Overview

Note on FAISS index paths:
- The multi-model builder writes FAISS directories like `storage/faiss_<key>__<embed_model>`.
- The Typer CLI (`python -m src.cli.commands`) looks for `storage/faiss_<key>` by default.
- If you use the multi-model builder and the Typer CLI, copy or symlink your chosen embedding index to the generic path, e.g.:
  - `ln -s storage/faiss_science__BAAI-bge-small-en-v1.5 storage/faiss_science`
  - `lc_ask.py` will automatically use a `...__<embed_model>_repacked` directory if present.

If you upgraded LangChain and your old FAISS index fails to load, repack it without re-embedding:

```bash
# Derive paths from KEY and EMBED_MODEL
make repack-faiss KEY=science EMBED_MODEL=BAAI/bge-small-en-v1.5

# Or specify explicit directories
make repack-faiss FAISS_DIR=storage/faiss_science__BAAI-bge-small-en-v1.5 OUT=storage/faiss_science__BAAI-bge-small-en-v1.5_repacked
```

#### ⏯️ `lc_ask.py` - Core LLM Interface

**Purpose**: Direct interface to language models for single queries.

**Key Features**:
- Flexible prompt engineering
- Multiple content types
- JSON output support
- Retrieval-augmented generation (RAG)

**Options:**
- `--key`: string specifying the faiss index to query
- `--k`: number of results to return from vector database
- `--embed-model`: the model index to query (default:`BAAI/bge-small-en-v1.5`)
- `--ce-model`: cross encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)

**Usage**:

```bash
# Basic query
python src/langchain/lc_ask.py ask "What is machine learning?"

# Advanced query with options
python src/langchain/lc_ask.py ask "Explain neural networks" --content-type technical_manual_writer --key science --k 20

# Query from JSON file
python src/langchain/lc_ask.py ask --file query.json --key biology
```

**Makefile Usage**:

**`make lc-ask`** - Complete RAG query options:
- `INSTR`: Instruction for retrieval (what to search for)
- `TASK`: Task prefix for LLM (how to answer)
- `FILE`: JSON file containing query parameters
- `KEY`: Collection key (default: default)
- `CONTENT_TYPE`: Writing style (default: pure_research)
- `K`: Number of documents to retrieve (default: 30)

```bash
# Simple RAG query
make lc-ask "Explain neural networks"

# Advanced RAG query with all options
make lc-ask INSTR="Explain neural networks" TASK="Write for beginners" KEY="science" CONTENT_TYPE="technical_manual_writer" K=20
```

---

#### ⏯️ `lc_batch.py` - Batch Processing

**Purpose**: Process multiple content generation jobs in parallel.

**Key Features**:
- JSONL job file processing
- Parallel execution
- Result aggregation
- Progress tracking

**Options:**
- `--key`: string specifying the faiss index to query
- `--k`: number of results to return from vector database
- `--parallel`: number of parallel workers (default: `1`)
- `--output-dir`: specify output directory
- `--jobs`: JSON or JSONL file containing job definitions

**Usage**:
```bash
# Process jobs from JSONL file
python src/langchain/lc_batch.py --jobs data_jobs/example.jsonl --key science

# Parallel processing
python src/langchain/lc_batch.py --jobs jobs.jsonl --parallel 4 --k 30

# Custom output directory
python src/langchain/lc_batch.py --jobs jobs.jsonl --output-dir ./custom_output
```

**Makefile Usage**:

**`make lc-batch`** - Batch processing with full options:
- `FILE`: JSON or JSONL file containing job definitions
- `KEY`: Collection key (default: default)
- `CONTENT_TYPE`: Writing style (default: pure_research)
- `K`: Retriever top-k (default: 30)
- `PARALLEL`: Number of parallel workers (default: 1)
- `OUTPUT_DIR`: Custom output directory

```bash
# Batch processing with parallel execution
make lc-batch FILE="examples/sample_jobs_1A1.jsonl" KEY="biology" PARALLEL=4
```

---

#### ⏯️ `lc_build_index.py` - Index Builder

**Purpose**: Create vector indexes for retrieval-augmented generation.

**Key Features**:
- Document ingestion
- Vector embeddings
- Index optimization
- Multiple data sources

**Options:**
- `--shard-size`: number of chunks per shard (default:`1000`)
- `--resume`: Skip shards already built (default: `False`)
- `--keep-shards`: Do not delete shard directories after merge (default: `False`)
- `<argument>`:  string specifying the faiss index to query (same as key in other operations)

**Usage**:
```bash

#simple
python src/langchain/lc_build_index.py --source data/ --index my_index

#specify shard size of 200, check for existing shards and resume a previously unfinished index operation, and do not delete shards after merge
python src/langchain/lc_build_index.py --source data/ --index --resume --shard_size 200 --key_shards my_index

```

---

#### ⏯️ `lc_merge_runner.py` - Advanced Merge System

**Purpose**: Intelligent content merging with multi-stage editorial pipelines.

**Key Features**:
- Multi-stage processing (critique → merge → style → images)
- AI-powered content scoring
- Jaccard similarity de-duplication
- YAML-driven configuration
- Command-line and interactive modes

**Options:**
- `--sub`: Subsection ID (e.g., 1A1) for job file processing"
- `--jobs`: Path to JSONL jobs file
- `--key`: Collection key for lc_ask
- `--k`: Retriever top-k for lc_ask (default: `10`)
- `--batch-only`: Force use of batch results only (skip job file prompts) (default: `False`)
- `--chapter`: Chapter title for context
- `--section`: Section title for context
- `--subsection`: Subsection title for context
  
**Usage**:

```bash
# Interactive mode
python src/langchain/lc_merge_runner.py

# Process specific subsection
python src/langchain/lc_merge_runner.py --sub 1A1 --key science

# Custom job file
python src/langchain/lc_merge_runner.py --jobs data_jobs/1A1.jsonl --chapter "Chapter 1"
```

**Makefile Usage**:

**`make lc-merge-runner`** - Intelligent content merging:
- `SUB`: Subsection ID (e.g., 1A1)
- `JOBS`: Path to JSONL jobs file
- `KEY`: Collection key for lc_ask
- `K`: Retriever top-k for lc_ask
- `BATCH_ONLY`: Force use of batch results only
- `CHAPTER/SECTION/SUBSECTION`: Hierarchical context titles

---

#### ⏯️ `lc_outline_generator.py` - Interactive Outline Creator

**Purpose**: Generate intelligent book outlines using LangChain's indexed knowledge.

**Key Features**:
- Interactive book detail collection
- Outline depth selection (3-5 levels)
- AI-powered outline generation using indexed content
- Automatic conversion to book runner format
- Comprehensive outline validation and summary

**Options:**
- `--output`: Output JSON file path
- `--non-interactive`: not yet implemented (user will be able to supply json file containing answers to all prompts)
  
**Usage**:
```bash
# Interactive outline generation
python src/langchain/lc_outline_generator.py

# Save to specific location
python src/langchain/lc_outline_generator.py --output my_book_outline.json
```

---

#### ⏯️ `lc_outline_converter.py` - Outline to Book Structure Converter

**Purpose**: Convert existing outlines into book structure and job files.

**Key Features**:
- Multiple input format support (JSON, Markdown, Text)
- Automatic hierarchical context generation
- Job file generation for each subsection
- Dependency relationship detection
- Format validation and conversion

**Options:**
- `--outline`: Input outline file (JSON, Markdown, or Text)
- `--output`: Output book structure JSON file"
- `--title`:  Override book title
- `--topic`: Override book topic
- `--audience`: Override target audience
- `--wordcount`: Override word count target
- `--num-prompts`: Number of prompts to generate per section
- `--content-type`: Content type for job generation (default: technical_manual_writer)
  
**Supported Input Formats**:
- **JSON**: Structured outline format (from lc_outline_generator.py)
- **Markdown**: Header-based outline (# ## ### ####)
- **Text**: Numbered/lettered outline (1. 2. A. B. etc.)

```bash
# Convert text outline
python src/langchain/lc_outline_converter.py --outline examples/sample_outline_text.txt

# Convert JSON outline
python src/langchain/lc_outline_converter.py --outline my_outline.json

# Convert with custom metadata
python src/langchain/lc_outline_converter.py --outline outline.md --title "My Book" --topic "AI" --audience "developers"

# Convert markdown outline
python src/langchain/lc_outline_converter.py --outline examples/sample_outline_markdown.md --output book.json
```

**Makefile Usage**:

**`make lc-outline-converter`** - Outline conversion:
- `OUTLINE`: Input outline file (JSON, Markdown, or Text)
- `OUTPUT`: Output book structure JSON file
- `TITLE/TOPIC/AUDIENCE`: Override metadata
- `WORDCOUNT`: Override word count target
- `NUM_PROMPTS`: Number of prompts to generate per section
- `CONTENT_TYPE`: Content type for job generation

```bash
# Interactive outline generation
make lc-outline-generator

# Convert outline with custom output
make lc-outline-converter examples/sample_outline_markdown.md OUTPUT=my_book.json

# Convert with metadata overrides
make lc-outline-converter examples/sample_outline_text.txt \
  TITLE="My Book" \
  TOPIC="Machine Learning" \
  AUDIENCE="Data Scientists" \
  WORDCOUNT=75000

# Run complete book generation
make lc-book-runner examples/book_structure_example.json

# Force regeneration of all content
make lc-book-runner examples/book_structure_example.json FORCE=1

# Skip merge step (batch only)
make lc-book-runner examples/book_structure_example.json SKIP_MERGE=1

# Convert outline with custom metadata
make lc-outline-converter OUTLINE="examples/sample_outline_text.txt" TITLE="My Book" TOPIC="AI" AUDIENCE="developers"

```

---

#### ⏯️ `lc_book_runner.py` - Book Orchestration

**Purpose**: High-level orchestration for entire books and chapters.

**Key Features**:
- Hierarchical book structure processing (4 levels deep)
- Automatic job file generation
- Batch and merge pipeline orchestration
- Final document aggregation
- Progress tracking and error recovery
- Section dependency management

**Options:**
- `--book`: JSON file defining book structure
- `--output`: Output markdown file path
- `--force`: Force regeneration of all content (default: `False`)
- `--skip-merge`: Skip merge processing, only run batch (default: `False`)
- `--use-rag`: Use RAG for additional context when generating job prompts (default: `False`)
- `--rag-key`: Collection key for RAG retrieval (required if --use-rag is specified)
- `--num-prompts`: Number of prompts to generate per section (default: 4)
  
**Usage**:

```bash
# Process complete book structure
python src/langchain/lc_book_runner.py --book examples/book_structure_example.json

# Custom output location
python src/langchain/lc_book_runner.py --book book.json --output /path/to/final_book.md

# Force regeneration
python src/langchain/lc_book_runner.py --book book.json --force

# Skip merge step (batch only)
python src/langchain/lc_book_runner.py --book book.json --skip-merge
```

**Makefile Usage**:

**`make lc-book-runner`** - Complete book orchestration:
- `BOOK`: JSON file defining book structure
- `OUTPUT`: Output markdown file path
- `FORCE`: Force regeneration of all content
- `SKIP_MERGE`: Skip merge processing, only run batch
- `USE_RAG`: Use RAG for additional context
- `RAG_KEY`: Collection key for RAG retrieval
- `NUM_PROMPTS`: Number of prompts to generate per section

```bash
# Complete book generation
make lc-book-runner BOOK="examples/book_structure_example.json" OUTPUT="my_book.md"

# Force regeneration of all content
make lc-book-runner BOOK="book.json" FORCE=1

# Skip merge step (batch only)
make lc-book-runner BOOK="book.json" SKIP_MERGE=1
```

---

####  ⏯️ `research/metadata_scan.py`

**Purpose**: To add metadata (doi/isbn/author/title/date) to pdfs prior to indexing

**Key Features**:
- Scans PDFs filename, content, metadata for DOI/ISBN
- fetches metadata (Crossref/OpenLibrary),
- writes a manifest, and updates PDF metadata (Info + XMP/DC/Prism).
- renames files using a consistent format ([slugified title]-[year].pdf)

**Options**:
- `--dir`: Root to scan (default `data_raw`)
- `--glob`: File pattern (default `**/*.pdf`)
- `--write`: Write manifest and PDF metadata (default off)
- `--quickscan`: Skip files requiring input from user (default off)
- `--manifest`: Manifest path (default `research/out/manifest.json`)
- `--rename`: `yes|no` to rename files by slugified title/year (default `yes`)
- `--skip-existing`: Skip already processed files in manifest (default off)
 
**Usage**:

```bash
# Preview (no writes)
python research/metadata_scan.py --dir data_raw

# Process pdf files in data_raw, adding metadata and renaming them where possible from information in files 
python research/metadata_scan.py --dir data_raw  --write --quickscan

# Process pdf files in data_raw, prompting user to provide the isbn or doi where this is not found or ambiguous
python research/metadata_scan.py --dir data_raw  --write 
```

**Makefile Usage**:

```bash
make scan-metadata DIR=data_raw WRITE=1 RENAME=yes SKIP_EXISTING=1
```

---

####  ⏯️ `research/collector_ui.py`

**Purpose**: to aid in the gathering and metadata population of journal articles and books in pdf format which serve as the basis for the RAG index

**Key Features**:
- TUI user interface
- Import html source via textarea
- Recalls state between runs via manifest.json
- Export list of PDF URLs for download using any download manager
- Download PDF directly from UI
- Load PDF files and URLs, and scholar details URLs with the click of a button
- Fuzzy search for DOI/ISBN on click

**Options**:
- `--file`: HTML file with Google Scholar results
- `-xml`: XML-like markup file with entries
- `--skip-existing`: Skip entries with processed=true in manifest
- `--allow-delete`: Enable delete actions
  
**Usage**:
```bash
# run without filr input or already populated manifest
python research/collector_ui.py

# load entries from html source
python research/collector_ui.py --file ../research/out/research.html

# load entries from simple xml format
python research/collector_ui.py --xml ../research/out/research.xml
```

---

### CLI Commands (src/cli/commands.py) 

The CLI commands module provides a streamlined interface using Typer:

```bash
# Basic RAG query
python -m src.cli.commands ask "What is machine learning?"

# Advanced query with options
python -m src.cli.commands ask "Explain neural networks" --key science --k 20 --task "Write for beginners"

# Query from JSON file
python -m src.cli.commands ask --file query.json --key biology
```

**CLI Command Options:**
- `--question, -q`: Your question or instruction (required)
- `--key, -k`: Collection key (default: from config)
- `--k`: Top-k results for retrieval (default: 15)
- `--task, -t`: Optional task prefix (excluded from retrieval)
- `--file, -f`: JSON file containing prompt parameters

### Interactive Shell (src/cli/shell.py)

The interactive shell provides an advanced REPL interface with presets and multi-step workflows:

```bash
# Start interactive shell
python -m src.cli.shell

# Start with specific collection
RAG_KEY=science python -m src.cli.shell
```

**Shell Commands:**
- `ask <question>`: General RAG answer with citations
- `compare <topic>`: Contrast positions/methods/results across sources
- `summarize <topic>`: High-level summary with quotes
- `outline <topic>`: Book/essay outline with evidence bullets
- `presets`: List dynamic presets from playbooks.yaml
- `preset <name> [topic]`: Run guided multi-step preset
- `sources`: Show sources from last answer
- `help`: Show available commands
- `quit`: Exit shell

**Example Shell Session:**
```bash
$ python -m src.cli.shell
RAG Tool Shell
ROOT: /path/to/project
KEY: default
Index: /path/to/storage/faiss_default

rag> ask What is machine learning?
[AI generates response with citations]

rag> sources
- Machine Learning Basics (p.15) :: ml_basics.pdf
- Neural Networks Explained (p.42) :: nn_guide.pdf

rag> quit
```

### Comparison of Interfaces

| Interface | Best For | Features |
|-----------|----------|----------|
| **Makefile** | Complete workflows, automation | All options as variables, error handling, examples |
| **Direct Scripts** | Direct control, scripting | Full command-line options, programmatic use |
| **CLI Commands** | Simple queries, automation | Streamlined interface, JSON file support |
| **Interactive Shell** | Exploration, complex queries | Presets, multi-step workflows, source inspection |

### Configuration for CLI

All CLI interfaces use the same configuration system:

1. **Environment Variables**: `OPENAI_API_KEY`, `RAG_KEY`
2. **Configuration Files**: `env.json`, YAML config files
3. **Command-line Options**: Override defaults per command

**Example Configuration:**
```json
{
  "openai_api_key": "your-key-here",
  "default_model": "gpt-4",
  "rag_key": "science",
  "embedding_model": "text-embedding-ada-002"
}
```

### Models & Backends

- OpenAI via LangChain (preferred): requires `OPENAI_API_KEY` and `langchain-openai`.
- OpenAI (raw client) fallback: requires `openai` package.
- Ollama (local): requires `langchain-ollama` or `langchain-community` ChatOllama and a running Ollama daemon; set `OLLAMA_MODEL`.

The factory auto-selects an available backend in this order: OpenAI (LangChain) → Ollama → OpenAI (raw). See `src/core/llm.py`.

### Version Compatibility

This project targets LangChain 0.2.x with the split provider packages:
- `langchain>=0.2.13,<0.3`
- `langchain-community>=0.2.12,<0.3`
- `langchain-text-splitters>=0.2.2,<0.3`
- `langchain-openai>=0.1.7,<0.2`
- Optional: `langchain-huggingface>=0.0.3`, `langchain-ollama>=0.1.0`

These versions ensure stable imports for retrievers (e.g., EnsembleRetriever) and LLM integrations. If you upgrade beyond these ranges, prefer the Typer CLI (`python -m src.cli.commands`) which already includes robust fallbacks.

---
---

## 🧩 Tool Agent Schema

Defines the JSON contract for tool-enabled agents. Each assistant message must be either a tool invocation:

```json
{"tool": "<tool_name>", "args": { /* ... */ }}
```

or a final response:

```json
{"final": "<answer text>"}
```

See [docs/tool_agent_schema.md](docs/tool_agent_schema.md) for the full specification and transcript example.

### Multi-agent CLI

Run a tool-enabled agent that combines local RAG retrieval with tools served
over MCP:

```bash
python -m src.cli.multi_agent "Find papers on transformers and call the time tool" \
  --key default --mcp ./tools/mcp_server.py
```

The command registers the `rag_retrieve` tool for vector search and loads any
tools exposed by the MCP server so the agent can invoke them during the
conversation.

### MCP Tool Server

Start the tool server to expose registered tools via the Model Context Protocol:

```bash
python -m src.tool.mcp_server  # or: make tool-shell
uvicorn src.tool.mcp_app:app --host 127.0.0.1 --port 3333
```

---
---

## 💾 Classes and Function Libraries

### 📄 `research/functions/pdf-io.py`
Handles writing PDF files with metadata in dublin core and prism formats. 

####  `write_pdf_info(src_pdf: Path, dest_pdf: Path, metadata: Dict[str, str]) -> Path`
Write standard PDF metadata fields with pypdf

#### `write_pdf_xmp(path: Path, dc: Dict[str, str], prism: Dict[str, str]) -> None:`

Write XMP (Dublin Core + Prism) metadata in-place using pikepdf. If pikepdf is not installed, this function silently returns

### 📄 `research/functions/filelogger.py` 
Provides debout log output to a file for cases where it is not possible to access the standard error and standard out streams directly.

#### `_fllog(s: str) -> void:`
log str to file

---
---

## 💽 External Libraries
### Python Argparse 
[Documentation](https://docs.python.org/3/library/argparse.html): The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages. The module will also issue errors when users give the program invalid arguments.

**Example Usage**:
```python
parser = argparse.ArgumentParser()
parser.add_argument("--json", dest="json_path", help="JSON job file containing 'question'")
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()
```

### Python Magic
[Documentation](https://pypi.org/project/python-magic/): python-magic is a Python interface to the libmagic file type identification library. libmagic identifies file types by checking their headers according to a predefined list of file types. This functionality is exposed to the command line by the Unix command file. 
**Example Usage**:
```python
# returns a value such as image/png, application/pdf, text/html
mime_type = magic.from_file(filepath, mime=True)
```

### FastAPI
[Documentation](https://fastapi.tiangolo.com/): FastAPI is a modern, high-performance web framework for building APIs with Python.

**Example Usage**:
```python
from fastapi import FastAPI

app = FastAPI()
```

### Uvicorn
[Documentation](https://www.uvicorn.org/): Uvicorn is a lightning-fast ASGI server implementation, perfect for serving FastAPI apps.

**Example Usage**:
```bash
uvicorn src.tool.mcp_app:app --host 127.0.0.1 --port 3333
```

### Pydantic
[Documentation](https://docs.pydantic.dev/): Pydantic provides data validation and settings management using Python type annotations.

**Example Usage**:
```python
from pydantic import BaseModel

class Meta(BaseModel):
    traceId: str
```


---
---

## 😵‍💫 Miscellaneous


---
---


## ⚙️ Environment Variables

| name | description | secret | default | 
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | API key for OpenAI backends | 🔐 | N/A | 
| `RAG_KEY` | Default collection key | 📖 | None | 
| `OPENAI_MODEL` | Override OpenAI chat model | 📖 | gpt-4o-mini |
| `OLLAMA_MODEL` | Override local Ollama model | 📖 | llama3.1:8b |
| `DEBUG` |  Set to 1/true to enable debug mode in config | 📖 | `0/False` |

use `sops-edit env.json` to add/edit new environment variables... append `_pt` (for plaintext) to names of non-secret values. Load sops values into environment using `eval(make sops-env-export) or ``make sops-env-export`` ` operation as this handles removing the suffix and loading them properly 

---
---

## 🛠️ YAML Configuration Files

The system uses several YAML configuration files located in `src/config/content/prompts/` to define content types, merge pipelines, and interactive presets.

### Content Types Configuration

**File**: `src/config/content/prompts/content_types.yaml` (main index)
**Individual Files**: `src/config/content/prompts/content_types/*.yaml`

Content types define different writing styles and system prompts for content generation. Each content type is stored in its own YAML file.

**Available Content Types:**
- `pure_research` - Academic research with citations
- `technical_manual_writer` - Technical documentation
- `science_journalism_article_writer` - Science journalism
- `folklore_adaptation_and_anthology_editor` - Creative writing adaptations

**Example Content Type Structure** (`pure_research.yaml`):
```yaml
description: "Pure research assistant focused on citations and evidence"
system_prompt:
  - "You are a careful research assistant.\n"
  - "Use ONLY the provided context\n."
  - "Every claim MUST include inline citations like ([filename], p.X) or ([filename], pp.X–Y)."
  - "If the context is insufficient or conflicting, state what is missing and stop."
  - "Current date: {{current_date}} (for temporal context if needed)\n"
job_generation_prompt: |
  You are a research-focused content strategist. Generate {{num_prompts}} research-oriented writing prompts...
job_generation_rag_context: |
  **Additional Research Context from RAG:**
  Use the following relevant research information from the knowledge base...
```

**Template Variables Available:**
- `{{book_title}}` - Full book title
- `{{chapter_title}}` - Chapter title
- `{{section_title_hierarchy}}` - Hierarchical section path
- `{{subsection_title}}` - Subsection title
- `{{subsection_id}}` - Hierarchical ID (e.g., "1A1")
- `{{target_audience}}` - Target audience
- `{{topic}}` - Book topic
- `{{num_prompts}}` - Number of prompts to generate
- `{{rag_context}}` - Additional RAG context
- `{{current_date}}` - Current date

**RAG Context Query Templates:**
Each content type now includes a `rag_context_query` template for retrieving relevant context:

```yaml
rag_context_query: |
  Find relevant information about: {{section_title}}

  Context: This is for creating educational content for a book titled "{{book_title}}"
  for {{target_audience}}.

  Please provide any relevant background information, examples, or context that would be
  helpful for writing educational content about this topic.
```

**Template Fallback System:**
The template engine now supports a robust fallback system:
1. **First**: Try to load template from the specific content type file (e.g., `pure_research.yaml`)
2. **Fallback**: If not found, try to load from `default.yaml`
3. **Error**: If still not found, raise an informative error message

This ensures that new content types can inherit templates from the default configuration while still allowing for customization when needed.

### Merge Types Configuration

**File**: `src/config/content/prompts/merge_types.yaml`

Defines different merge pipeline configurations for content consolidation and editing.

**Available Merge Types:**
- `generic_editor` - Basic single-stage merging
- `advanced_pipeline` - Multi-stage critique → merge → style → images
- `educator_handbook` - Specialized for educational content

**Example Merge Type Structure**:
```yaml
generic_editor:
  system_prompt:
    - "You are a senior editor for a publisher..."
    - "It is your job to merge these together so that the final resulting text..."

advanced_pipeline:
  description: "Multi-stage pipeline with critique, merge, and style harmonization"
  parameters:
    top_n_variations: 3
    similarity_threshold: 0.85
  stages:
    critique:
      system_prompt:
        - "You are a senior editor evaluating content quality..."
      output_format: "json"
      scoring_instruction: "Return only JSON: {\"score\": <0-10>, ...}"
    merge:
      system_prompt:
        - "You are a consolidating editor..."
      output_format: "markdown"
    style:
      system_prompt:
        - "You are a line editor harmonizing tone..."
      output_format: "markdown"
```

**Stage Configuration Options:**
- `system_prompt` - Array of prompt strings
- `output_format` - "json" or "markdown"
- `scoring_instruction` - For critique stages
- `parameters` - Pipeline-level settings (top_n_variations, similarity_threshold)

### Playbooks Configuration

**File**: `src/config/content/prompts/playbooks.yaml`

Defines interactive presets for complex multi-step workflows used in the CLI shell.

**Example Playbook Structure**:
```yaml
literature_review:
  label: Literature Review
  description: Structured synthesis with methods appraisal and evidence-backed themes
  inputs: []  # preset-level interactive inputs
  system_prompt: |
    You are a meticulous literature-review assistant...
  stitch_final: true  # Combine step outputs
  final_prompt: |
    Combine the step outputs into a cohesive literature review...
  steps:
    - name: scope
      prompt: |
        Clarify the research question(s) and implicit inclusion/exclusion criteria...
    - name: themes
      prompt: |
        Extract major themes/claims with evidence...
    - name: methods
      prompt: |
        Critically appraise methods for each study...
    - name: synthesis
      prompt: |
        Summarize agreements and disagreements...
```

**Playbook Features:**
- **Interactive Inputs**: User prompts for dynamic content
- **Multi-step Workflows**: Sequential processing stages
- **Template Variables**: Jinja2 templating support
- **Flexible Output**: JSON arrays or final synthesis

**Input Types:**
```yaml
inputs:
  - name: audience
    prompt: Primary audience?
    default: general
    choices: [general, policy, practitioners]
  - name: styles
    prompt: List 3 styles (comma-separated)
    default: modern retelling, mythic high-fantasy
    multi: true  # Allow multiple values
  - name: target_length
    prompt: Target length (words)
    default: 450
    type: int  # Type validation
```

### Output Templates

**File**: `src/config/content/prompts/templates.md`

Provides structural templates for different output formats:

**Literature Review Template:**
```
- **Research Question**: ...
- **Scope/Inclusion**: ...
- **Themes**
  - Theme A — evidence (pages)
  - Theme B — evidence (pages)
- **Methods Appraisal**
  - Source — strengths/limitations
- **Synthesis**
  - Agreements / Disagreements
  - Gaps & Future Work
```

**Science Journalism Template:**
```
- **Headline**: ...
- **Dek**: ...
- **What's New**
- **Why It Matters**
- **Evidence (plain-language)**
- **Caveats**
- **Quote(s)** (with page cites)
- **How Solid Is The Evidence?** (1–5)
```

### Customizing Configuration

#### Adding New Content Types

1. Create new YAML file in `src/config/content/prompts/content_types/`
2. Define system prompt and job generation templates
3. Use template variables for dynamic content
4. Test with `make lc-batch CONTENT_TYPE=your_type`

#### Adding New Merge Types

1. Add new entry to `src/config/content/prompts/merge_types.yaml`
2. Define stages with system prompts and output formats
3. Configure parameters for advanced pipelines
4. Test with `python src/langchain/lc_merge_runner.py`

#### Creating Custom Playbooks

1. Add new entry to `src/config/content/prompts/playbooks.yaml`
2. Define interactive inputs and step workflows
3. Use Jinja2 templating for dynamic content
4. Test with `python -m src.cli.shell` → `preset your_preset`

---
---


## 📋 Book Structure Format

### JSON Schema

```json
{
  "title": "Book Title",
  "metadata": {
    "author": "Author Name",
    "version": "1.0",
    "target_audience": "Target audience",
    "word_count_target": 100000,
    "created_date": "2025-08-28",
    "description": "Book description"
  },
  "sections": [
    {
      "subsection_id": "1A1",
      "title": "Section Title",
      "job_file": "data_jobs/1A1.jsonl",
      "batch_params": {
        "key": "collection_name",
        "k": 5
      },
      "merge_params": {
        "key": "collection_name",
        "k": 3
      },
      "dependencies": ["parent_section_id"]
    }
  ]
}
```

### Section ID Convention

Use hierarchical naming for 4-level deep structure:
- **Level 1**: Chapter (1, 2, 3...)
- **Level 2**: Major section (A, B, C...)
- **Level 3**: Subsection (1, 2, 3...)
- **Level 4**: Sub-subsection (a, b, c...)

Examples: `1A1`, `2B3`, `3C2a`

### Job File Format

Each line is a JSON object with hierarchical context:
```json
{
  "task": "system prompt with book context",
  "instruction": "specific instruction with hierarchical positioning",
  "context": {
    "book_title": "Book Title",
    "chapter": "Chapter X",
    "section": "Section Y",
    "subsection": "Subsection Z",
    "subsection_id": "XYZ",
    "target_audience": "target audience"
  }
}
```

**Context Fields:**
- `book_title`: Full book title for context
- `chapter`: Chapter identifier (e.g., "Chapter 1")
- `section`: Section identifier (e.g., "Section A")
- `subsection`: Subsection identifier (e.g., "Subsection 1")
- `subsection_id`: Hierarchical ID (e.g., "1A1")
- `target_audience`: Who the content is for

See `examples/sample_jobs_1A1.jsonl`, `examples/sample_jobs_1B1.jsonl`, and `examples/sample_jobs_2A1.jsonl` for complete examples showing different hierarchical contexts.

### content_viewer.py - Content Viewer

**Purpose**: View and analyze generated content.

**Key Features**:
- Content browsing
- Quality assessment
- Export functionality

### cleanup_sources.py - Source Cleanup

**Purpose**: Clean and preprocess source documents.

**Key Features**:
- Document normalization
- Metadata extraction
- Quality filtering

---
---

## 📝 Usage Examples

### Example 1: Simple Content Generation

```bash
# Generate a single piece of content
python src/langchain/lc_ask.py ask \
  --content-type pure_research \
  --task "Explain quantum computing to a beginner"
```

### Example 2: Batch Processing

Create `jobs.jsonl`:
```json
{"task": "system prompt", "instruction": "Generate introduction"}
{"task": "system prompt", "instruction": "Generate examples"}
{"task": "system prompt", "instruction": "Generate conclusion"}
```

```bash
python src/langchain/lc_batch.py --jobs jobs.jsonl
```

### Example 3: Advanced Merging

```bash
# Use advanced pipeline for educational content
python src/langchain/lc_merge_runner.py --sub 1A1
```

### Example 4: Custom Pipeline

Add to `merge_types.yaml`:
```yaml
technical_docs:
  description: "Optimized for technical documentation"
  stages:
    critique:
      system_prompt: "You are a technical editor..."
    merge:
      system_prompt: "You are a technical writer consolidating docs..."
```

### Example 5: Complete Book Generation Workflow

#### Option A: AI-Generated Outline
**Step 1: Generate Intelligent Outline**
```bash
python src/langchain/lc_outline_generator.py
```
*Interactively collects book details and generates AI-powered outline*

**Step 2: Generate Complete Book**
```bash
python src/langchain/lc_book_runner.py --book outlines/book_structures/my_book_outline.json
```
*Orchestrates the complete content generation pipeline*

#### Option B: Convert Existing Outline
**Step 1: Convert Outline to Book Structure**
```bash
# Convert markdown outline
python src/langchain/lc_outline_converter.py --outline examples/sample_outline_markdown.md

# Convert text outline
python src/langchain/lc_outline_converter.py --outline examples/sample_outline_text.txt

# Convert with custom metadata
python src/langchain/lc_outline_converter.py --outline my_outline.md \
  --title "My Custom Book Title" \
  --topic "Data Science" \
  --audience "Data Scientists" \
  --wordcount 75000
```

**Step 2: Generate Complete Book**
```bash
python src/langchain/lc_book_runner.py --book outlines/converted_structures/converted_book_structure.json
```

#### Option C: Manual Book Structure
Create a book structure file:
```json
{
  "title": "Professional Development Handbook",
  "metadata": {
    "author": "AI Content Generator",
    "target_audience": "Primary school teachers",
    "word_count_target": 100000
  },
  "sections": [
    {
      "subsection_id": "1A1",
      "title": "Understanding Modern Learning Theories",
      "job_file": "data_jobs/1A1.jsonl",
      "batch_params": {"key": "education", "k": 5},
      "merge_params": {"key": "education", "k": 3},
      "dependencies": []
    }
  ]
}
```

Generate the complete book:
```bash
python src/langchain/lc_book_runner.py --book book_structure.json
```

---
---

## 🔗 Hierarchical Context System

### Automatic Context Embedding

The book runner automatically embeds hierarchical context into every job file, eliminating the need for manual context entry in merge scripts:

**Generated Job Structure:**
```json
{
  "task": "Contextual system prompt with book title and audience",
  "instruction": "Instruction with hierarchical positioning",
  "context": {
    "book_title": "Professional Development Handbook",
    "chapter": "Chapter 1",
    "section": "Section A",
    "subsection": "Subsection 1",
    "subsection_id": "1A1",
    "target_audience": "primary school teachers"
  }
}
```

### Benefits:

- **No Manual Input**: Merge scripts get context automatically
- **Consistent Positioning**: All content knows its place in the hierarchy
- **Audience Awareness**: Content tailored to target readers
- **Scalable**: Works for books of any size and complexity
- **Automated**: Context generation happens during job file creation

### Context Flow:

1. **Book Runner** parses subsection ID (e.g., "1A1")
2. **Automatically generates** hierarchical context
3. **Embeds context** in job file during generation
4. **Batch processing** uses contextualized jobs
5. **Merge processing** receives properly positioned content
6. **Final aggregation** maintains hierarchical structure

---
---

## 🔧 Pipeline Types

### 1. Generic Editor
- Simple single-stage merging
- Basic content consolidation
- Fast processing

### 2. Advanced Pipeline
- Multi-stage processing
- AI-powered critique and scoring
- Intelligent de-duplication
- Style harmonization
- Optional image suggestions

### 3. Educator Handbook
- Specialized for educational content
- Teacher-focused language
- Classroom utility emphasis
- PD handbook optimization

### 4. Custom Pipelines
- YAML-driven configuration
- Domain-specific prompts
- Custom processing stages
- Flexible parameters

---
---

## 🐳 Docker

Run the full system in a Docker container without needing a local Python setup.

### Build

```bash
docker build -t rag-writer:latest .
```

Or with Docker Compose:

```bash
docker compose build
```

### Run (ad-hoc)

Examples below assume you’re in the project root. Mounting `./` into `/app` keeps your data and outputs on the host.

```bash
# Show CLI help (default CMD)
docker run --rm -it \
  -v "$PWD":/app \
  -e OPENAI_API_KEY=sk-... \
  rag-writer:latest --help

# Build FAISS index from PDFs in ./data_raw (mount this dir with your PDFs)
docker run --rm -it \
  -v "$PWD":/app \
  -e RAG_KEY=science \
  rag-writer:latest python src/langchain/lc_build_index.py

# Ask a question using the Typer CLI
docker run --rm -it \
  -v "$PWD":/app \
  -e OPENAI_API_KEY=sk-... \
  -e RAG_KEY=science \
  rag-writer:latest ask "What is machine learning?"

# Interactive shell (optional)
docker run --rm -it \
  -v "$PWD":/app \
  -e OPENAI_API_KEY=sk-... \
  rag-writer:latest shell
```

Tip: To persist model downloads across runs, mount a cache:

```bash
docker run --rm -it \
  -v "$PWD":/app \
  -v hf-cache:/root/.cache/huggingface \
  -e OPENAI_API_KEY=sk-... \
  rag-writer:latest ask "Explain neural networks"
```

### Run (Compose)

`compose.yaml` ships with sensible defaults:

```bash
# With OPENAI_API_KEY exported in your shell
export OPENAI_API_KEY=sk-...

# Show help
docker compose run --rm rag-writer --help

# Build index
docker compose run --rm rag-writer python src/langchain/lc_build_index.py

# Ask
docker compose run --rm rag-writer ask "What is machine learning?"
```

### Notes

- Data and outputs are in project subfolders: `data_raw`, `data_processed`, `storage`, `output`, `exports`.
- Set `RAG_KEY` to switch collections (defaults to `default`).
- If you prefer the Makefile workflows, run them inside the container shell and call the Python scripts directly (the Makefile’s venv targets are designed for host use).
- Some first-time runs will download models (HuggingFace). Use the provided cache volume to avoid repeated downloads.
- Entrypoint shortcuts: `ask` runs the Typer CLI; `shell` starts the interactive REPL; any other command is executed verbatim (e.g., `python -m src.cli.shell`).

### Directory Layout

- `data_raw/`: Place source PDFs and documents to ingest.
- `data_processed/`: Extracted chunks and intermediate artifacts.
- `storage/`: Vector stores (e.g., FAISS), per collection key.
- `output/`, `exports/`: Generated content and final artifacts.
- `outlines/`, `data_jobs/`: Outline and job files for the book pipeline.

### SOPS in Docker

- The image includes `sops` and `jq`. If `/app/env.json` exists and is decryptable (via AWS KMS, GCP KMS, or PGP), the entrypoint auto-loads its values into the environment before running your command.
- Keep your existing PGP recipient for local; add cloud KMS recipients to `.sops.yaml` for production. See `docs/sops_kms_examples.md`.

#### SOPS Makefile Helpers

```bash
# Rewrap env.json with current recipients from .sops.yaml
make sops-updatekeys [FILE=env.json]

# Decrypt to stdout (or redirect)
make sops-decrypt [FILE=env.json] > /tmp/env.json

# Print export lines for env.json (use with eval in your shell)
make sops-env-export [FILE=env.json] | source /dev/stdin
```

See also: `docs/ci_sops_rewrap_example.yml` for a GitHub Actions example to automate rewrapping.

### Makefile Helpers

```bash
# Build image
make docker-build [DOCKER_IMAGE=rag-writer:latest]

# Ask via Docker
make docker-ask "What is machine learning?" KEY=science

# Index via Docker
make docker-index KEY=science

# Compose variants
make compose-build
make compose-ask "What is machine learning?" KEY=science
make compose-index KEY=science
make compose-shell

# Run full book pipeline
make docker-book-runner BOOK=outlines/converted_structures/my_book.json OUTPUT=exports/books/my_book.md
make compose-book-runner BOOK=outlines/converted_structures/my_book.json OUTPUT=exports/books/my_book.md

# Index maintenance
make clean-faiss KEY=your_key         # remove FAISS dirs for key
make clean-shards KEY=your_key EMB=BAAI/bge-small-en-v1.5  # remove FAISS shards for model
make reindex KEY=your_key             # clean + rebuild FAISS for key
make repack-faiss KEY=your_key EMBED_MODEL=BAAI/bge-small-en-v1.5  # salvage old index

# Metadata scanning (pre-alpha)
make scan-metadata DIR=data_raw WRITE=1 RENAME=yes SKIP_EXISTING=1

# Collector UI (import HTML/XML, export links)
make collector-ui
```


### Image Structure and Faster Rebuilds

The Docker image uses a multi-stage build to speed up development rebuilds:

- base-sys: OS deps (build tools, curl, jq, ca-certificates, libgomp1) + sops binary
- py-deps: Python dependencies from `requirements.txt`
- runner: App source + entrypoint

Only the runner layer changes when you edit source files, so rebuilds are much faster.

Common workflows:

```bash
# Seed base layers (do this after changing system or Python deps)
make docker-build-base

# Regular dev cycle (code changes only)
make docker-build             # or: docker compose build

# Compose variant for base layers
make compose-build-base
```

To use prebuilt base layers across machines/CI, you can tag and push the base stages to a registry and update `Dockerfile` FROM references if you want to pin them.

### CI Cache Tips

- Use Docker BuildKit + Buildx to push/pull cache to a registry for fast CI builds.
- Example GitHub Actions workflow is provided at `docs/ci_build_cache_example.yml`.
- Typical flow:
  - Cache base-sys and py-deps stages to registry
  - Build the runner stage with `--cache-from` pointing at those cache refs
  - Push final image to your registry (e.g., GHCR, ECR, GCR)

If you use Podman in CI, layer caching is local by default; pushing prebuilt base stage images to your registry can still improve cold-start builds.

---
---




## 🔍 Troubleshooting

### Common Issues

#### 1. No Batch Results Found
```bash
# Run batch processing first
python src/langchain/lc_batch.py --jobs your_jobs.jsonl
```

#### 2. Job File Not Found
```bash
# Check file path and permissions
ls -la data_jobs/
python src/langchain/lc_merge_runner.py --jobs data_jobs/your_file.jsonl
```

#### 3. YAML Configuration Errors
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('src/config/content/prompts/merge_types.yaml'))"
```

#### 4. API Key Issues
```bash
# Check environment configuration
cat env.json
# Ensure API keys are set and valid
```

#### 5. Relative Import Errors
If you see "attempted relative import with no known parent package", run modules in package mode:
```bash
python -m src.cli.commands ask "What is machine learning?"
python -m src.cli.shell
```
Or use the Docker/Compose entrypoint shortcuts:
```bash
docker compose run --rm rag-writer ask "What is machine learning?"
docker compose run --rm rag-writer shell
```

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=src
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python src/langchain/lc_merge_runner.py --sub 1A1
```

### Performance Optimization

- Use `--parallel` in batch processing for faster execution
- Adjust `similarity_threshold` in pipeline config for different de-duplication levels
- Configure appropriate `top_n_variations` based on content complexity

---
---

## 🤝 Contributing

### Adding New Pipeline Types

1. Add configuration to `src/config/content/prompts/merge_types.yaml`
2. Test with sample content
3. Update documentation

### Adding New Content Types

1. Add configuration to `src/config/content/prompts/content_types.yaml`
2. Test with lc_ask.py
3. Document usage examples

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include error handling
- Update tests for new functionality

---
---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
---

## 🙏 Acknowledgments

- Built with LangChain for LLM integration
- Uses Rich for beautiful terminal interfaces
- Inspired by advanced content processing workflows
- Designed for educational and professional content creation
