# ===== Makefile =====
# Project root via env var
ROOT ?= $(RAG_ROOT)
ifeq ($(strip $(ROOT)),)
ROOT := /var/srv/IOMEGA_EXTERNAL/rag
endif
PY_CMD := $(shell command -v python3.11 || command -v python3.10 || command -v python3)
#PY := $(ROOT)/venv/bin/python
PY := python
#PIP := $(ROOT)/venv/bin/pip
PIP := pip
# FAISS backend selector (cpu or gpu)
FAISS_BACKEND ?= cpu
# Docker
DOCKER_IMAGE ?= rag-writer:latest

# ---- Gold data validation ----
SCHEMAS_DIR := eval/schemas
DATA_DIR    := eval/data

RETRIEVAL_SCHEMA := $(SCHEMAS_DIR)/retrieval.schema.json
SCREENING_SCHEMA := $(SCHEMAS_DIR)/screening.schema.json
EXTRACTION_SCHEMA:= $(SCHEMAS_DIR)/extraction.schema.json
SYNTHESIS_SCHEMA := $(SCHEMAS_DIR)/synthesis.schema.json
MANUALS_SCHEMA   := $(SCHEMAS_DIR)/manuals.schema.json

RETRIEVAL_DATA := $(DATA_DIR)/retrieval/queries.jsonl
SCREENING_DATA := $(DATA_DIR)/screening/abstracts.jsonl
EXTRACTION_DATA:= $(DATA_DIR)/extraction/studies.jsonl
SYNTHESIS_DATA := $(DATA_DIR)/synthesis/questions.jsonl
MANUALS_DATA   := $(DATA_DIR)/manuals/tasks.jsonl

.PHONY: all init lc-index lc-ask lc-batch content-viewer cleanup-sources lc-merge-runner lc-outline-generator lc-outline-converter lc-book-runner cli-ask cli-shell tool-shell clean clean-all help show-config check-setup test test-coverage format lint quality book-from-outline quick-ask batch-workflow examples validate-gold validate-gold-retrieval validate-gold-screening validate-gold-extraction validate-gold-synthesis validate-gold-manuals docker-build docker-ask docker-index docker-shell compose-build compose-ask compose-index compose-shell sops-updatekeys sops-decrypt sops-env-export docker-build-base compose-build-base


# ===== HELP =====
help:
	@echo "LangChain RAG Writer - Complete Content Generation Pipeline"
	@echo ""
	@echo "QUICK START:"
	@echo "  make init          # Set up environment and install dependencies"
	@echo "  make lc-index KEY=default  # Build FAISS index for retrieval"
	@echo "  make cli-ask \"question\"   # Ask questions using Typer CLI"
	@echo "  make cli-shell        # Interactive RAG shell"
	@echo ""
	@echo "CONTENT GENERATION WORKFLOW:"
	@echo "  1. make lc-outline-converter OUTLINE=path/to/outline.txt"
	@echo "  2. make lc-book-runner BOOK=path/to/book_structure.json"
	@echo "  3. Review generated content in exports/books/"
	@echo ""
	@echo "AVAILABLE TARGETS:"
	@awk '/^[a-zA-Z_-]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 1, index($$1, "\:")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  %-20s %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "For detailed options, use: make <target> --help"

## Initialize project environment
all: help

## Initialize project with virtual environment and dependencies
init:
	mkdir -p $(ROOT)/data_raw $(ROOT)/data_processed $(ROOT)/storage/lancedb_default $(ROOT)/storage/faiss_default $(ROOT)/src/llamaindex $(ROOT)/src/langchain $(ROOT)/src/tool $(ROOT)/src/config/content/prompts
	$(PY_CMD) -m venv $(ROOT)/venv
	$(PIP) install -U pip wheel setuptools
        $(PIP) install -r $(ROOT)/requirements.txt
        if [ -f "$(ROOT)/requirements-faiss-$(FAISS_BACKEND).txt" ]; then \
                $(PIP) install -r $(ROOT)/requirements-faiss-$(FAISS_BACKEND).txt; \
        else \
                echo "Unknown FAISS_BACKEND=$(FAISS_BACKEND); expected cpu or gpu"; \
                exit 1; \
        fi
	@echo "Init complete. Put PDFs into $(ROOT)/data_raw/"

# ----- LangChain -----

## Build FAISS index for LangChain retrieval [KEY=key_name] [SHARD_SIZE=n] [RESUME=1]
lc-index:
	@k="$(filter-out $@,$(MAKECMDGOALS))"; \
	shard_size="$(SHARD_SIZE)"; \
	resume="$(RESUME)"; \
	if [ -z "$$k" ]; then k="$(KEY)"; fi; \
	if [ -z "$$k" ]; then k=default; fi; \
	cmd="$(PY) $(ROOT)/src/langchain/lc_build_index.py \"$$k\""; \
	if [ -n "$$shard_size" ]; then cmd="$$cmd --shard-size \"$$shard_size\""; fi; \
	if [ -n "$$resume" ]; then cmd="$$cmd --resume \"$$resume\""; fi; \
	eval $$cmd

## Ask questions using LangChain RAG system
## Usage:
##   make lc-ask "What is machine learning?"
##   make lc-ask INSTR="Explain neural networks" TASK="Write for beginners"
##   make lc-ask FILE="path/to/query.json" KEY="science" CONTENT_TYPE="technical_manual_writer"
##   make lc-ask KEY="biology" K=10 "What are enzymes?"
## Options:
##   INSTR: Instruction for retrieval (what to search for)
##   TASK: Task prefix for LLM (how to answer)
##   FILE: JSON file containing query parameters
##   KEY: Collection key (default: default)
##   CONTENT_TYPE: Writing style (default: pure_research)
##   K: Number of documents to retrieve (default: 30)
lc-ask:
	@instr="$(INSTR)"; task="$(TASK)"; file="$(FILE)"; key="$(KEY)"; content_type="$(CONTENT_TYPE)"; k="$(K)"; \
	if [ -z "$$instr" -a -z "$$file" ]; then instr="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$instr" -a -z "$$file" ]; then echo "Usage: make lc-ask INSTR=\"instruction\" [TASK=\"task prefix\"] [KEY=\"collection_key\"] [CONTENT_TYPE=\"content_type\"] [K=30] OR make lc-ask \"instruction\" OR make lc-ask FILE=\"path/to/json\""; exit 1; fi; \
	if [ -z "$$key" ]; then key=default; fi; \
	if [ -z "$$content_type" ]; then content_type=pure_research; fi; \
	if [ -z "$$k" ]; then k=30; fi; \
	if [ -n "$$file" ]; then \
	  $(PY) $(ROOT)/src/langchain/lc_ask.py ask --file "$$file" --key "$$key" --content-type "$$content_type" --k "$$k"; \
	elif [ -n "$$task" ]; then \
	  $(PY) $(ROOT)/src/langchain/lc_ask.py ask "$$instr" --task "$$task" --key "$$key" --content-type "$$content_type" --k "$$k"; \
	else \
	  $(PY) $(ROOT)/src/langchain/lc_ask.py ask "$$instr" --key "$$key" --content-type "$$content_type" --k "$$k"; \
	fi

## Process multiple RAG queries from JSON/JSONL file
## Usage:
##   make lc-batch FILE="jobs.jsonl" KEY="science" CONTENT_TYPE="technical_manual_writer"
##   make lc-batch FILE="jobs.json" PARALLEL=4 K=20
##   cat jobs.json | make lc-batch KEY="biology"
## Options:
##   FILE: JSON or JSONL file containing job definitions
##   KEY: Collection key (default: default)
##   CONTENT_TYPE: Writing style (default: pure_research)
##   K: Retriever top-k (default: 30)
##   PARALLEL: Number of parallel workers (default: 1)
##   OUTPUT_DIR: Custom output directory
lc-batch:
        @file="$(FILE)"; key="$(KEY)"; content_type="$(CONTENT_TYPE)"; k="$(K)"; parallel="$(PARALLEL)"; output_dir="$(OUTPUT_DIR)"; \
	if [ -z "$$key" ]; then key=default; fi; \
	if [ -z "$$content_type" ]; then content_type=pure_research; fi; \
	if [ -z "$$k" ]; then k=30; fi; \
	if [ -z "$$parallel" ]; then parallel=1; fi; \
	if [ -n "$$file" ]; then \
	  $(PY) $(ROOT)/src/langchain/lc_batch.py --jobs "$$file" --key "$$key" --content-type "$$content_type" --k "$$k" --parallel "$$parallel" $(if $(output_dir),--output-dir "$$output_dir",); \
        else \
          $(PY) $(ROOT)/src/langchain/lc_batch.py --key "$$key" --content-type "$$content_type" --k "$$k" --parallel "$$parallel" $(if $(output_dir),--output-dir "$$output_dir",); \
        fi

## Ask questions via Typer CLI
cli-ask:
        @q="$(QUESTION)"; key="$(KEY)"; k="$(K)"; task="$(TASK)"; file="$(FILE)"; \
        if [ -z "$$q" ]; then q="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
        if [ -z "$$q" -a -z "$$file" ]; then echo "Usage: make cli-ask \"question\" [KEY=key] [K=15] [TASK=task] [FILE=path]"; exit 1; fi; \
        if [ -n "$$file" ]; then \
          $(PY) -m src.cli.commands ask --file "$$file" $(if $(KEY),--key "$$key",) $(if $(K),--k "$$k",) $(if $(TASK),--task "$$task",); \
        else \
          $(PY) -m src.cli.commands ask "$$q" $(if $(KEY),--key "$$key",) $(if $(K),--k "$$k",) $(if $(TASK),--task "$$task",); \
        fi

## Interactive Typer shell
cli-shell:
	$(PY) -m src.cli.shell

## Interactive viewer for batch-generated content
content-viewer:
	$(PY) $(ROOT)/src/langchain/content_viewer.py

## Merge batch-generated content variations into cohesive subsections
## Usage:
##   make lc-merge-runner  # Interactive mode
##   make lc-merge-runner SUB=1A1 JOBS="data_jobs/1A1.jsonl" KEY="science"
##   make lc-merge-runner BATCH_ONLY=1 CHAPTER="Chapter 1" SECTION="Introduction"
## Options:
##   SUB: Subsection ID (e.g., 1A1)
##   JOBS: Path to JSONL jobs file
##   KEY: Collection key for lc_ask
##   K: Retriever top-k for lc_ask
##   BATCH_ONLY: Force use of batch results only
##   CHAPTER: Chapter title for context
##   SECTION: Section title for context
##   SUBSECTION: Subsection title for context
lc-merge-runner:
	@sub="$(SUB)"; jobs="$(JOBS)"; key="$(KEY)"; k="$(K)"; batch_only="$(BATCH_ONLY)"; \
	chapter="$(CHAPTER)"; section="$(SECTION)"; subsection="$(SUBSECTION)"; \
	if [ -n "$$sub" ]; then \
	  cmd="$(PY) $(ROOT)/src/langchain/lc_merge_runner.py --sub \"$$sub\""; \
	  if [ -n "$$jobs" ]; then cmd="$$cmd --jobs \"$$jobs\""; fi; \
	  if [ -n "$$key" ]; then cmd="$$cmd --key \"$$key\""; fi; \
	  if [ -n "$$k" ]; then cmd="$$cmd --k \"$$k\""; fi; \
	  if [ -n "$$chapter" ]; then cmd="$$cmd --chapter \"$$chapter\""; fi; \
	  if [ -n "$$section" ]; then cmd="$$cmd --section \"$$section\""; fi; \
	  if [ -n "$$subsection" ]; then cmd="$$cmd --subsection \"$$subsection\""; fi; \
	  eval $$cmd; \
	elif [ -n "$$batch_only" ]; then \
	  cmd="$(PY) $(ROOT)/src/langchain/lc_merge_runner.py --batch-only"; \
	  if [ -n "$$chapter" ]; then cmd="$$cmd --chapter \"$$chapter\""; fi; \
	  if [ -n "$$section" ]; then cmd="$$cmd --section \"$$section\""; fi; \
	  if [ -n "$$subsection" ]; then cmd="$$cmd --subsection \"$$subsection\""; fi; \
	  eval $$cmd; \
	else \
	  $(PY) $(ROOT)/src/langchain/lc_merge_runner.py; \
	fi

## Interactive outline generation using LangChain index
## Usage:
##   make lc-outline-generator  # Interactive mode
##   make lc-outline-generator OUTPUT="my_outline.json"
## Options:
##   OUTPUT: Output file path for generated outline
lc-outline-generator:
	@output="$(OUTPUT)"; \
	if [ -n "$$output" ]; then \
	  $(PY) $(ROOT)/src/langchain/lc_outline_generator.py --output "$$output"; \
	else \
	  $(PY) $(ROOT)/src/langchain/lc_outline_generator.py; \
	fi

## Convert outlines to book structure and job files
## Usage:
##   make lc-outline-converter OUTLINE="examples/sample_outline_text.txt"
##   make lc-outline-converter OUTLINE="outline.md" OUTPUT="book.json" TITLE="My Book"
##   make lc-outline-converter OUTLINE="outline.json" TOPIC="science" AUDIENCE="students"
## Options:
##   OUTLINE: Input outline file (JSON, Markdown, or Text)
##   OUTPUT: Output book structure JSON file
##   TITLE: Override book title
##   TOPIC: Override book topic
##   AUDIENCE: Override target audience
##   WORDCOUNT: Override word count target
##   NUM_PROMPTS: Number of prompts to generate per section
##   CONTENT_TYPE: Content type for job generation
lc-outline-converter:
	@outline="$(filter-out $@,$(MAKECMDGOALS))"; if [ -z "$$outline" ]; then outline="$(OUTLINE)"; fi; \
	output="$(OUTPUT)"; title="$(TITLE)"; topic="$(TOPIC)"; \
	audience="$(AUDIENCE)"; wordcount="$(WORDCOUNT)"; num_prompts="$(NUM_PROMPTS)"; content_type="$(CONTENT_TYPE)"; \
	if [ -z "$$outline" ]; then echo "Usage: make lc-outline-converter OUTLINE=\"path/to/outline\" [OUTPUT=\"output.json\"] [TITLE=\"Book Title\"] [TOPIC=\"topic\"] [AUDIENCE=\"audience\"] [WORDCOUNT=50000] [NUM_PROMPTS=4] [CONTENT_TYPE=\"technical_manual_writer\"]"; exit 1; fi; \
	cmd="$(PY) $(ROOT)/src/langchain/lc_outline_converter.py --outline \"$$outline\""; \
	if [ -n "$$output" ]; then cmd="$$cmd --output \"$$output\""; fi; \
	if [ -n "$$title" ]; then cmd="$$cmd --title \"$$title\""; fi; \
	if [ -n "$$topic" ]; then cmd="$$cmd --topic \"$$topic\""; fi; \
	if [ -n "$$audience" ]; then cmd="$$cmd --audience \"$$audience\""; fi; \
	if [ -n "$$wordcount" ]; then cmd="$$cmd --wordcount $$wordcount"; fi; \
	if [ -n "$$num_prompts" ]; then cmd="$$cmd --num-prompts $$num_prompts"; fi; \
	if [ -n "$$content_type" ]; then cmd="$$cmd --content-type \"$$content_type\""; fi; \
	eval $$cmd

## Orchestrate complete book generation pipeline
## Usage:
##   make lc-book-runner BOOK="outlines/converted_structures/my_book_structure.json"
##   make lc-book-runner BOOK="book.json" OUTPUT="my_book.md" FORCE=1
##   make lc-book-runner BOOK="book.json" SKIP_MERGE=1 USE_RAG=1 RAG_KEY="science"
## Options:
##   BOOK: JSON file defining book structure
##   OUTPUT: Output markdown file path
##   FORCE: Force regeneration of all content
##   SKIP_MERGE: Skip merge processing, only run batch
##   USE_RAG: Use RAG for additional context when generating job prompts
##   RAG_KEY: Collection key for RAG retrieval
##   NUM_PROMPTS: Number of prompts to generate per section
lc-book-runner:
	@book="$(filter-out $@,$(MAKECMDGOALS))"; if [ -z "$$book" ]; then book="$(BOOK)"; fi; \
	output="$(OUTPUT)"; force="$(FORCE)"; skip_merge="$(SKIP_MERGE)"; \
	use_rag="$(USE_RAG)"; rag_key="$(RAG_KEY)"; num_prompts="$(NUM_PROMPTS)"; \
	if [ -z "$$book" ]; then echo "Usage: make lc-book-runner BOOK=\"path/to/book_structure.json\" [OUTPUT=\"output.md\"] [FORCE=1] [SKIP_MERGE=1] [USE_RAG=1] [RAG_KEY=\"collection_key\"] [NUM_PROMPTS=4]"; exit 1; fi; \
	cmd="$(PY) $(ROOT)/src/langchain/lc_book_runner.py --book \"$$book\""; \
	if [ -n "$$output" ]; then cmd="$$cmd --output \"$$output\""; fi; \
	if [ -n "$$force" ]; then cmd="$$cmd --force"; fi; \
	if [ -n "$$skip_merge" ]; then cmd="$$cmd --skip-merge"; fi; \
	if [ -n "$$use_rag" ]; then cmd="$$cmd --use-rag"; fi; \
	if [ -n "$$rag_key" ]; then cmd="$$cmd --rag-key \"$$rag_key\""; fi; \
	if [ -n "$$num_prompts" ]; then cmd="$$cmd --num-prompts $$num_prompts"; fi; \
	eval $$cmd

## Clean up sources in existing batch files
cleanup-sources:
	$(PY) $(ROOT)/src/langchain/cleanup_sources.py

## List all available content types for lc-ask
list-content-types:
	$(PY) $(ROOT)/src/langchain/lc_ask.py list-types

## Show current configuration settings
show-config:
	$(PY) -c "from src.config.settings import get_config; import json; print(json.dumps(get_config().model_dump(), indent=2))"

## Validate project setup and dependencies
check-setup:
	@echo "Checking Python environment..."
	@$(PY) --version
	@echo "Checking key imports..."
	@$(PY) -c "import langchain; print('✓ LangChain available')"
	@$(PY) -c "import faiss; print('✓ FAISS available')"
	@$(PY) -c "import rich; print('✓ Rich available')"
	@echo "Checking data directories..."
	@if [ -d "$(ROOT)/data_raw" ]; then echo "✓ data_raw directory exists"; else echo "✗ data_raw directory missing"; fi
	@if [ -d "$(ROOT)/storage" ]; then echo "✓ storage directory exists"; else echo "✗ storage directory missing"; fi
	@echo "Setup check complete!"

## Run test suite
test:
	$(PY) -m pytest tests/ -v

## Run test suite with coverage
test-coverage:
	$(PY) -m pytest tests/ --cov=src --cov-report=html --cov-report=term

## Format code with black
format:
	$(PY) -m black src/ tests/

## Lint code with flake8
lint:
	$(PY) -m flake8 src/ tests/

## Run full quality check (format, lint, test)
quality: format lint test

# ----- Unified Tool -----

## Start unified tool shell for advanced operations
tool-shell:
	@k="$(KEY)"; \
	if [ -n "$$k" ]; then RAG_KEY="$$k" $(PY) -m src.tool.mcp_server; \
	else $(PY) -m src.tool.mcp_server; fi

# ===== EXAMPLE WORKFLOWS =====

## Complete book generation workflow from outline
## Usage: make book-from-outline OUTLINE="examples/sample_outline_text.txt" TITLE="My Book"
book-from-outline:
	@outline="$(OUTLINE)"; title="$(TITLE)"; topic="$(TOPIC)"; audience="$(AUDIENCE)"; \
	if [ -z "$$outline" ]; then echo "Usage: make book-from-outline OUTLINE=\"path/to/outline\" [TITLE=\"Book Title\"] [TOPIC=\"topic\"] [AUDIENCE=\"audience\"]"; exit 1; fi; \
	echo "Step 1: Converting outline to book structure..."; \
	make lc-outline-converter OUTLINE="$$outline" $(if $(title),TITLE="$$title",) $(if $(topic),TOPIC="$$topic",) $(if $(audience),AUDIENCE="$$audience",); \
	book_file="$$(ls -t outlines/converted_structures/*.json | head -1)"; \
	echo "Step 2: Generating book content..."; \
	make lc-book-runner BOOK="$$book_file"; \
	echo "Book generation complete! Check exports/books/ for output."

## Quick RAG query with custom parameters
## Usage: make quick-ask "What is machine learning?" KEY="science" CONTENT_TYPE="technical_manual_writer"
quick-ask:
	@query="$(filter-out $@,$(MAKECMDGOALS))"; key="$(KEY)"; content_type="$(CONTENT_TYPE)"; k="$(K)"; \
	if [ -z "$$query" ]; then echo "Usage: make quick-ask \"Your question\" [KEY=key] [CONTENT_TYPE=type] [K=30]"; exit 1; fi; \
	make lc-ask INSTR="$$query" $(if $(key),KEY="$$key",) $(if $(content_type),CONTENT_TYPE="$$content_type",) $(if $(k),K="$$k",)

## Batch processing workflow
## Usage: make batch-workflow FILE="jobs.jsonl" KEY="science" PARALLEL=4
batch-workflow:
	@file="$(FILE)"; key="$(KEY)"; content_type="$(CONTENT_TYPE)"; parallel="$(PARALLEL)"; \
	if [ -z "$$file" ]; then echo "Usage: make batch-workflow FILE=\"jobs.jsonl\" [KEY=key] [CONTENT_TYPE=type] [PARALLEL=4]"; exit 1; fi; \
	echo "Running batch processing..."; \
	make lc-batch FILE="$$file" $(if $(key),KEY="$$key",) $(if $(content_type),CONTENT_TYPE="$$content_type",) $(if $(parallel),PARALLEL="$$parallel",); \
	echo "Batch processing complete! Check output/batch/ for results."; \
	echo "To merge results: make lc-merge-runner BATCH_ONLY=1"

## Show available example files
examples:
	@echo "Available example files:"
	@echo ""
	@echo "OUTLINES:"
	@find examples/ -name "*outline*" -type f | sort
	@echo ""
	@echo "JOB FILES:"
	@find examples/ -name "*job*" -type f | sort
	@echo ""
	@echo "BOOK STRUCTURES:"
	@find examples/ -name "*book*" -type f | sort
	@echo ""
	@echo "Usage examples:"
	@echo "  make lc-outline-converter OUTLINE=\"examples/sample_outline_text.txt\""
	@echo "  make lc-batch FILE=\"examples/sample_jobs_1A1.jsonl\""
	@echo "  make lc-book-runner BOOK=\"examples/book_structure_example.json\""

%: ;

## Clean project data and indexes
clean:
	rm -rf $(ROOT)/data_processed/*
	find $(ROOT)/storage -maxdepth 1 -type d -name 'lancedb_*' -o -name 'faiss_*' | xargs -r rm -rf
	mkdir -p $(ROOT)/storage/lancedb_default $(ROOT)/storage/faiss_default
	@echo "✓ Cleaned processed data and indexes."

## Remove FAISS index for a specific KEY (inside container)
## Usage: make clean-faiss KEY=your_key
clean-faiss:
	@key="$(KEY)"; \
	if [ -z "$$key" ]; then echo "Usage: make clean-faiss KEY=your_key"; exit 1; fi; \
	rm -rf storage/faiss_"$$key" storage/faiss_"$$key"__*
	@echo "✓ Removed FAISS index(es) for key: $(KEY)"

## Remove FAISS shard directories for a KEY and embedding model
## Usage: make clean-shards KEY=your_key EMB=BAAI/bge-small-en-v1.5
clean-shards:
	@key="$(KEY)"; emb="$(EMB)"; \
	if [ -z "$$key" ] || [ -z "$$emb" ]; then echo "Usage: make clean-shards KEY=your_key EMB=embed_model"; exit 1; fi; \
	python src/langchain/cleanup_shards.py "$$key" "$$emb"

## Rebuild FAISS index for a KEY (inside container)
## Usage: make reindex KEY=your_key
reindex:
	@key="$(KEY)"; \
	if [ -z "$$key" ]; then echo "Usage: make reindex KEY=your_key"; exit 1; fi; \
	$(MAKE) clean-faiss KEY="$$key"; \
	python src/langchain/lc_build_index.py "$$key"

## Repack an existing FAISS index to current LangChain format (no re-embedding)
## Usage:
##   make repack-faiss KEY=your_key [EMBED_MODEL=BAAI/bge-small-en-v1.5] [OUT=dir] [CHUNKS=path]
##   make repack-faiss FAISS_DIR=storage/faiss_key__BAAI-bge-small-en-v1.5 [OUT=dir] [CHUNKS=path]
repack-faiss:
	@key="$(KEY)"; embed="$(EMBED_MODEL)"; faiss_dir="$(FAISS_DIR)"; out="$(OUT)"; chunks="$(CHUNKS)"; \
	if [ -n "$$faiss_dir" ]; then \
	  cmd="python tools/repack_faiss_index.py --faiss-dir \"$$faiss_dir\""; \
	  if [ -n "$$chunks" ]; then cmd="$$cmd --chunks \"$$chunks\""; fi; \
	else \
	  if [ -z "$$key" ]; then echo "Usage: make repack-faiss KEY=your_key [EMBED_MODEL=...] [OUT=dir] [CHUNKS=path]"; exit 1; fi; \
	  cmd="python tools/repack_faiss_index.py --key \"$$key\""; \
	  if [ -n "$$embed" ]; then cmd="$$cmd --embed-model \"$$embed\""; fi; \
	  if [ -n "$$chunks" ]; then cmd="$$cmd --chunks \"$$chunks\""; fi; \
	fi; \
	if [ -n "$$out" ]; then cmd="$$cmd --out-dir \"$$out\""; fi; \
	echo "Running: $$cmd"; \
	eval $$cmd

## Scan PDFs for DOI/ISBN and write manifest entries (pre-alpha)
## Usage:
##   make scan-metadata [DIR=data_raw] [WRITE=1] [RENAME=yes] [SKIP_EXISTING=1]
scan-metadata:
	@dir="$(DIR)"; write="$(WRITE)"; rename="$(RENAME)"; skip="$(SKIP_EXISTING)"; \
	if [ -z "$$dir" ]; then dir=data_raw; fi; \
	cmd="python -m src.research.metadata_scan scan --dir \"$$dir\""; \
	if [ -n "$$write" ]; then cmd="$$cmd --write"; fi; \
	if [ -n "$$rename" ]; then cmd="$$cmd --rename \"$$rename\""; fi; \
	if [ -n "$$skip" ]; then cmd="$$cmd --skip-existing"; fi; \
	echo "Running: $$cmd"; \
	eval $$cmd

## Collector UI (import HTML/XML; export known PDF links)
collector-ui:
	python -m src.research.collector_ui

## Deep clean including generated content
clean-all: clean
	@echo "Deep cleaning all generated content..."
	rm -rf $(ROOT)/output/*
	rm -rf $(ROOT)/exports/*
	rm -rf $(ROOT)/generated/*
	find $(ROOT)/data_jobs -name "*.jsonl" | xargs -r rm -f
	@echo "✓ Cleaned all generated content."

.PHONY: lc-ask-faiss lc-ask-bm25 lc-ask-hybrid lc-ask-hybrid-ce

# Legacy direct wrappers around lc_ask.py with safer optional args
lc-ask-faiss:
	@instr="$(INSTR)"; \
	if [ -z "$$instr" ]; then instr="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$instr" -a -z "$(JSON)" ]; then echo "Usage: make $@ INSTR=\"question\" [KEY=key] [K=10] [JSON=path]"; exit 1; fi; \
	cmd="python src/langchain/lc_ask.py --key $(KEY) --mode faiss --embed-model BAAI/bge-small-en-v1.5"; \
	if [ -n "$(K)" ]; then cmd="$$cmd --k $(K)"; fi; \
	if [ -n "$(JSON)" ]; then cmd="$$cmd --json $(JSON)"; fi; \
	if [ -n "$$instr" ]; then cmd="$$cmd \"$$instr\""; fi; \
	eval $$cmd

lc-ask-bm25:
	@instr="$(INSTR)"; \
	if [ -z "$$instr" ]; then instr="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$instr" -a -z "$(JSON)" ]; then echo "Usage: make $@ INSTR=\"question\" [KEY=key] [K=10] [JSON=path]"; exit 1; fi; \
	cmd="python src/langchain/lc_ask.py --key $(KEY) --mode bm25"; \
	if [ -n "$(K)" ]; then cmd="$$cmd --k $(K)"; fi; \
	if [ -n "$(JSON)" ]; then cmd="$$cmd --json $(JSON)"; fi; \
	if [ -n "$$instr" ]; then cmd="$$cmd \"$$instr\""; fi; \
	eval $$cmd

lc-ask-hybrid:
	@instr="$(INSTR)"; \
	if [ -z "$$instr" ]; then instr="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$instr" -a -z "$(JSON)" ]; then echo "Usage: make $@ INSTR=\"question\" [KEY=key] [K=10] [JSON=path]"; exit 1; fi; \
	cmd="python src/langchain/lc_ask.py --key $(KEY) --mode hybrid --embed-model BAAI/bge-small-en-v1.5"; \
	if [ -n "$(K)" ]; then cmd="$$cmd --k $(K)"; fi; \
	if [ -n "$(JSON)" ]; then cmd="$$cmd --json $(JSON)"; fi; \
	if [ -n "$$instr" ]; then cmd="$$cmd \"$$instr\""; fi; \
	eval $$cmd

lc-ask-hybrid-ce:
	@instr="$(INSTR)"; \
	if [ -z "$$instr" ]; then instr="$(filter-out $@,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$instr" -a -z "$(JSON)" ]; then echo "Usage: make $@ INSTR=\"question\" [KEY=key] [K=10] [JSON=path]"; exit 1; fi; \
	cmd="python src/langchain/lc_ask.py --key $(KEY) --mode hybrid --rerank ce --ce-model cross-encoder/ms-marco-MiniLM-L-6-v2 --embed-model BAAI/bge-small-en-v1.5"; \
	if [ -n "$(K)" ]; then cmd="$$cmd --k $(K)"; fi; \
	if [ -n "$(JSON)" ]; then cmd="$$cmd --json $(JSON)"; fi; \
	if [ -n "$$instr" ]; then cmd="$$cmd \"$$instr\""; fi; \
	eval $$cmd

validate-gold: \
	validate-gold-retrieval \
	validate-gold-screening \
	validate-gold-extraction \
	validate-gold-synthesis \
	validate-gold-manuals
	@echo "All gold files validated."

validate-gold-retrieval:
	@python tools/validate_jsonl.py $(RETRIEVAL_SCHEMA) $(RETRIEVAL_DATA)

validate-gold-screening:
	@python tools/validate_jsonl.py $(SCREENING_SCHEMA) $(SCREENING_DATA)

validate-gold-extraction:
	@python tools/validate_jsonl.py $(EXTRACTION_SCHEMA) $(EXTRACTION_DATA)

validate-gold-synthesis:
	@python tools/validate_jsonl.py $(SYNTHESIS_SCHEMA) $(SYNTHESIS_DATA)

validate-gold-manuals:
	@python tools/validate_jsonl.py $(MANUALS_SCHEMA) $(MANUALS_DATA)

# ----- Docker -----

## Build Docker image (DOCKER_IMAGE=rag-writer:latest)
docker-build:
	docker build -t $(DOCKER_IMAGE) --target runner .

## Build base layers (system + python deps) for faster subsequent builds
docker-build-base:
	docker build -t rag-writer-base:py311-slim --target base-sys . \
	  && docker build -t rag-writer-deps:py311-slim --target py-deps .

## Ask a question via Docker
## Usage: make docker-ask "What is ML?" [KEY=science]
docker-ask:
	@query="$(filter-out $@,$(MAKECMDGOALS))"; key="$(KEY)"; \
	if [ -z "$$query" ]; then echo "Usage: make docker-ask \"Your question\" [KEY=key]"; exit 1; fi; \
	if [ -n "$$key" ]; then rk="-e RAG_KEY=$$key"; else rk="-e RAG_KEY"; fi; \
	docker run --rm -it \
	  -v "$$PWD":/app \
	  -e OPENAI_API_KEY \
	  $$rk \
	  $(DOCKER_IMAGE) ask "$$query"

## Build FAISS index via Docker (uses PDFs in ./data_raw)
## Usage: make docker-index [KEY=science]
docker-index:
	@key="$(KEY)"; \
	if [ -n "$$key" ]; then rk="-e RAG_KEY=$$key"; else rk="-e RAG_KEY"; fi; \
	docker run --rm -it \
	  -v "$$PWD":/app \
	  $$rk \
	  $(DOCKER_IMAGE) python src/langchain/lc_build_index.py

## Interactive shell in the container working directory
docker-shell:
	docker run --rm -it \
	  -v "$$PWD":/app \
	  -e OPENAI_API_KEY \
	  -e RAG_KEY \
	  $(DOCKER_IMAGE) bash

## Run full book pipeline via Docker
## Usage:
##   make docker-book-runner BOOK=outlines/converted_structures/my_book.json [OUTPUT=exports/books/my_book.md]
##   make docker-book-runner BOOK=book.json FORCE=1 SKIP_MERGE=1 USE_RAG=1 RAG_KEY=science NUM_PROMPTS=4
docker-book-runner:
	@book="$(BOOK)"; output="$(OUTPUT)"; force="$(FORCE)"; skip_merge="$(SKIP_MERGE)"; \
	use_rag="$(USE_RAG)"; rag_key="$(RAG_KEY)"; num_prompts="$(NUM_PROMPTS)"; \
	if [ -z "$$book" ]; then echo "Usage: make docker-book-runner BOOK=\\\"path/to/book_structure.json\\\" [OUTPUT=\\\"output.md\\\"] [FORCE=1] [SKIP_MERGE=1] [USE_RAG=1] [RAG_KEY=key] [NUM_PROMPTS=4]"; exit 1; fi; \
	cmd="python src/langchain/lc_book_runner.py --book \"$$book\""; \
	if [ -n "$$output" ]; then cmd="$$cmd --output \"$$output\""; fi; \
	if [ -n "$$force" ]; then cmd="$$cmd --force"; fi; \
	if [ -n "$$skip_merge" ]; then cmd="$$cmd --skip-merge"; fi; \
	if [ -n "$$use_rag" ]; then cmd="$$cmd --use-rag"; fi; \
	if [ -n "$$rag_key" ]; then cmd="$$cmd --rag-key \"$$rag_key\""; fi; \
	if [ -n "$$num_prompts" ]; then cmd="$$cmd --num-prompts $$num_prompts"; fi; \
	docker run --rm -it \
	  -v "$$PWD":/app \
	  -e OPENAI_API_KEY \
	  -e RAG_KEY \
	  $(DOCKER_IMAGE) sh -lc "$$cmd"

## Build images via docker compose
compose-build:
	docker compose build

## Build only base layers with compose’s builder
compose-build-base:
	docker build -t rag-writer-base:py311-slim --target base-sys . \
	  && docker build -t rag-writer-deps:py311-slim --target py-deps .

## Ask a question via docker compose
## Usage: make compose-ask "What is ML?" [KEY=science]
compose-ask:
	@query="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$query" ]; then echo "Usage: make compose-ask \"Your question\" [KEY=key]"; exit 1; fi; \
	docker compose run --rm \
	  -e OPENAI_API_KEY \
	  $(if $(KEY),-e RAG_KEY="$(KEY)",-e RAG_KEY) \
	  rag-writer ask "$$query"

## Build FAISS index via docker compose (uses PDFs in ./data_raw)
## Usage: make compose-index [KEY=science]
compose-index:
	docker compose run --rm \
	  $(if $(KEY),-e RAG_KEY="$(KEY)",-e RAG_KEY) \
	  rag-writer python src/langchain/lc_build_index.py

## Interactive shell in the compose service container
compose-shell:
	docker compose run --rm \
	  -e OPENAI_API_KEY \
	  $(if $(KEY),-e RAG_KEY="$(KEY)",-e RAG_KEY) \
	  rag-writer bash

## Run full book pipeline via docker compose
## Usage:
##   make compose-book-runner BOOK=outlines/converted_structures/my_book.json [OUTPUT=exports/books/my_book.md]
##   make compose-book-runner BOOK=book.json FORCE=1 SKIP_MERGE=1 USE_RAG=1 RAG_KEY=science NUM_PROMPTS=4
compose-book-runner:
	@book="$(BOOK)"; output="$(OUTPUT)"; force="$(FORCE)"; skip_merge="$(SKIP_MERGE)"; \
	use_rag="$(USE_RAG)"; rag_key="$(RAG_KEY)"; num_prompts="$(NUM_PROMPTS)"; \
	if [ -z "$$book" ]; then echo "Usage: make compose-book-runner BOOK=\\\"path/to/book_structure.json\\\" [OUTPUT=\\\"output.md\\\"] [FORCE=1] [SKIP_MERGE=1] [USE_RAG=1] [RAG_KEY=key] [NUM_PROMPTS=4]"; exit 1; fi; \
	cmd="python src/langchain/lc_book_runner.py --book \"$$book\""; \
	if [ -n "$$output" ]; then cmd="$$cmd --output \"$$output\""; fi; \
	if [ -n "$$force" ]; then cmd="$$cmd --force"; fi; \
	if [ -n "$$skip_merge" ]; then cmd="$$cmd --skip-merge"; fi; \
	if [ -n "$$use_rag" ]; then cmd="$$cmd --use-rag"; fi; \
	if [ -n "$$rag_key" ]; then cmd="$$cmd --rag-key \"$$rag_key\""; fi; \
	if [ -n "$$num_prompts" ]; then cmd="$$cmd --num-prompts $$num_prompts"; fi; \
	docker compose run --rm \
	  -e OPENAI_API_KEY \
	  $(if $(RAG_KEY),-e RAG_KEY="$(RAG_KEY)",) \
	  rag-writer sh -lc "$$cmd"

# ----- SOPS helpers -----

## Rewrap SOPS file with new recipients from .sops.yaml
## Usage: make sops-updatekeys [FILE=env.json]
sops-updatekeys:
	@file="$(FILE)"; if [ -z "$$file" ]; then file=env.json; fi; \
	if ! command -v sops >/dev/null 2>&1; then echo "sops not found; install sops first"; exit 1; fi; \
	if [ ! -f "$$file" ]; then echo "File not found: $$file"; exit 1; fi; \
	sops updatekeys "$$file"

## Decrypt a SOPS file to stdout (or redirect to a file)
## Usage: make sops-decrypt [FILE=env.json]
sops-decrypt:
	@file="$(FILE)"; if [ -z "$$file" ]; then file=env.json; fi; \
	if ! command -v sops >/dev/null 2>&1; then echo "sops not found; install sops first"; exit 1; fi; \
	if [ ! -f "$$file" ]; then echo "File not found: $$file"; exit 1; fi; \
	sops -d "$$file"

## Print export lines for env.json
## Usage: make sops-env-export [FILE=env.json]
sops-env-export:
	@file="$(FILE)"; if [ -z "$$file" ]; then file=env.json; fi; \
	if ! command -v sops >/dev/null 2>&1; then echo "sops not found; install sops first"; exit 1; fi; \
	if ! command -v jq >/dev/null 2>&1; then echo "jq not found; install jq first"; exit 1; fi; \
	if [ ! -f "$$file" ]; then echo "File not found: $$file"; exit 1; fi; \
	sops -d --output-type json "$$file" | sed -e "s|_pt||g;s|_unencrypted||g" | jq -r 'to_entries[] | "export \(.key)=\(.value|@sh)"'
