# ===== Makefile =====
# Project root via env var
ROOT ?= $(RAG_ROOT)
ifeq ($(strip $(ROOT)),)
ROOT := /var/srv/IOMEGA_EXTERNAL/rag
endif
PY_CMD := $(shell command -v python3.11 || command -v python3.10 || command -v python3)
PY := $(ROOT)/venv/bin/python
PIP := $(ROOT)/venv/bin/pip

.PHONY: all init ingest index ask lc-index lc-ask tool-shell clean

all: init ingest index

init:
	mkdir -p $(ROOT)/data_raw $(ROOT)/data_processed $(ROOT)/storage/lancedb_default $(ROOT)/storage/faiss_default $(ROOT)/src/llamaindex $(ROOT)/src/langchain $(ROOT)/src/tool $(ROOT)/src/tool/prompts
	$(PY_CMD) -m venv $(ROOT)/venv
	$(PIP) install -U pip wheel setuptools
	$(PIP) install -r $(ROOT)/requirements.txt
	@echo "Init complete. Put PDFs into $(ROOT)/data_raw/"

# ----- LlamaIndex -----

ingest:
	$(PY) $(ROOT)/src/llamaindex/parse_pdf.py

# index [key]; default key if omitted
index:
	@k="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$k" ]; then k=default; fi; \
	$(PY) $(ROOT)/src/llamaindex/build_index.py "$$k"

ask:
	@q="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$q" ]; then echo "Usage: make ask \"Your question\""; exit 1; fi; \
	$(PY) $(ROOT)/src/llamaindex/ask.py "$$q"

# ----- LangChain -----

# lc-index [key]; default if omitted
lc-index:
	@k="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$k" ]; then k=default; fi; \
	$(PY) $(ROOT)/src/langchain/lc_build_index.py "$$k"

lc-ask:
	@q="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$q" ]; then echo "Usage: make lc-ask \"Your question\""; exit 1; fi; \
	$(PY) $(ROOT)/src/langchain/lc_ask.py "$$q"

# ----- Unified Tool -----

tool-shell:
	@k="$(KEY)"; \
	if [ -n "$$k" ]; then RAG_KEY="$$k" $(PY) $(ROOT)/src/tool/cli.py shell; \
	else $(PY) $(ROOT)/src/tool/cli.py shell; fi

%: ;

clean:
	rm -rf $(ROOT)/data_processed/*
	find $(ROOT)/storage -maxdepth 1 -type d -name 'lancedb_*' -o -name 'faiss_*' | xargs -r rm -rf
	mkdir -p $(ROOT)/storage/lancedb_default $(ROOT)/storage/faiss_default
	@echo "Cleaned processed data and indexes."
