## README.md
```md
# RAG Starter (Multi-Collection)

Set your ROOT once:
```bash
export RAG_ROOT=/var/srv/IOMEGA_EXTERNAL/rag
```

## Build separate collections (keys)
1) Put the desired PDFs into `$RAG_ROOT/data_raw/`.
2) Build LangChain FAISS index with a **key**:
```bash
make lc-index llms_education      # -> data_processed/lc_chunks_llms_education.jsonl, storage/faiss_llms_education/
```
Or build LlamaIndex LanceDB index:
```bash
make index llms_education         # -> data_processed/chunks_llms_education.jsonl, storage/lancedb_llms_education/
```
Repeat with different keys by swapping the PDFs in `data_raw/` before each build.

## Use a collection in the interactive shell
```bash
make tool-shell KEY=llms_education
# inside shell, run: ask / compare / preset literature_review "your topic"
```

## Defaults
- If you omit a key, `default` is used → `faiss_default/`, `lancedb_default/`, `lc_chunks_default.jsonl`, `chunks_default.jsonl`.

## Notes
- You can also pass the key via env var: `RAG_KEY=llms_education make tool-shell`.
- To rebuild with a different embedding model, set `EMBED_MODEL` env var and rerun `lc-index`/`index`.
```
bash
   make init
   ```
3. **LlamaIndex ingest & index:**
   ```bash
   make ingest   # writes data_processed/chunks.jsonl
   make index
   ```
4. **LlamaIndex query:**
   ```bash
   make ask "What is the consensus across these books and papers about X, and where do they disagree? Include page-cited evidence."
   ```
5. **LangChain index (cache-first):**
   - Uses `data_processed/lc_chunks.jsonl` if present; otherwise parses/splits PDFs and writes it.
   ```bash
   make lc-index
   ```
6. **LangChain query:**
   ```bash
   make lc-ask "Decompose and compare claims, methods, and results with page-cited quotes."
   ```

## Notes
- If embeddings feel slow on CPU, switch `BAAI/bge-m3` → `BAAI/bge-small-en` in both pipelines.
- `chunks.jsonl` (LlamaIndex) and `lc_chunks.jsonl` (LangChain) keep artifacts separate.
- You can safely delete FAISS/LanceDB indexes to rebuild with a new embedding model while keeping the chunk artifacts.
