# Features TODO List

## High Priority Features

### [x] 1. Improved Metadata Collection (Supplement or Even Replacement to collector.py):
Step through a directory of pdfs; and for each file: 
  - scan content for DOI or ISBN 
  - if found 
    1. call API (crossref.py for DOI, not yet implemented for ISBN) to get complete metadata field set. 
    2. Present data to user along with filename for verification, display prompt: 
      - [w|W]rite this information to metadata 
        - w|W: write metadata, update manifest.json adding or editing entry for current file, continue to next file.  

      - provide new [d|D]OI and requery API
        - d|D|i|I: prompt user for DOI, get complete information from API using provided DOI, return to step (2) and disply write/new doi/new isbn prompt, continue 
      
      - provide new [i|I]SBN and requery API 
        - i|I: prompt user for ISBN, get complete information from API using provided ISBN, return to step (2) and disply write/new doi/new isbn prompt, continue 
      
      - [s|S]kip this file 
        - if metadata not exists, write empty metadata (can include 'processed by'), else do nothing continue to next file

      - [r|R]emove this file
        - delete this file, continue to next file

    - if record NOT found, assume ISBN/DOI is invalid:
      - prompt with [d|D],[i|I],[s|S], and [r|R]


  **Additional Details:**
  - add (pdfpark?) module implementation to write dublin core / prism versions of extended metadata AND
  - add read extended metadata from dublin core/prism to indexing script. 
    {priority: medium}

  - update content generation to use DOI/ISBN token to generated content. 
    {priority: high}

  - add step prior to final formatting that replaces DOI/ISBN token with in-text citation and generates works cited using playbook style configuration 
    {priority: medium}

  - does scribbr have an API? if so, add integration with this or similar service.    
    {priority: low}

#### Proposed CLI and Flow (v1 – low friction)

- New module: `research/metadata_scan.py`
- Goal: Walk `data_raw/` (or a provided folder), detect DOI/ISBN for each PDF, fetch full metadata, confirm with user, then write:
  - PDF Info dictionary (via `pypdf.PdfWriter.add_metadata`)
  - Sidecar `manifest.json` (aggregated record of all files)

##### Flow
- For each `*.pdf` (depth configurable):
  1) Detect identifiers: DOI (regex already in `lc_build_index.py`), ISBN‑10/13 (regex + checksum).
  2) Query metadata:
     - DOI → Crossref (primary)
     - ISBN → OpenLibrary (primary), Google Books (fallback)
  3) Present summary (filename + proposed metadata) with actions:
     - [W] Write: Save metadata → PDF Info + `manifest.json`, optionally rename PDF using slugified title + year.
     - [D] New DOI: prompt; requery; return to confirmation.
     - [I] New ISBN: prompt; requery; return to confirmation.
     - [S] Skip: mark as processed=false in manifest (or no entry if `--no-manifest`), continue.
     - [R] Remove: delete file (guarded by `--allow-delete`), continue.
  4) On not found/invalid IDs: offer [D]/[I]/[S]/[R] menu.

##### CLI Options (initial)
- `--dir DIR` (default `data_raw`): root to scan; `--glob "**/*.pdf"` to control pattern.
- `--write/--dry-run`: actually write files vs. preview.
- `--manifest PATH` (default `research/out/manifest.json`): location for manifest aggregation.
- `--rename yes|no` (default yes): rename file to `slugified_title[_YEAR].pdf` on write.
- `--interactive tui|cli` (default cli): TUI via Textual if available; falls back to prompts.
- `--skip-existing`: skip files already present in manifest with `processed=true`.
- `--allow-delete`: enable [R] remove.
- `--rescan`: ignore cached results; re-detect IDs and re-query remote APIs.
- `--depth N`: recursion depth limit; `--jobs N`: parallel metadata lookups (rate‑limited).
- `--file` : file containing listing html from google scholar search
- `--xml`: file containing rough 'xml' markup for manual addition <book || article><author/><title/><isbn/><doi/><publisher/><date/><pdf_link/><scholar_url/></book || article>

- on load: if --file or --xml contains listings or $root/research/out/manifest.json
contains listings display the edit form just like the original collector.py, with the same functionality. Below the row of prev, save, complete, open url, open pdf, delete, next buttons (edit mode actions), display the 'import/edit mode toggle', 'save to file', 'quit' (program level actions)  If no listings found, display the textarea to paste listings markup to parse listings from. Below the text area and above the program level action buttons, display a 'run import' button  (import mode actions)

##### Data Sources and Rate Limits
- Crossref for DOI (JSON API), OpenLibrary for ISBN, fallback Google Books.
- Exponential back‑off on HTTP 429/5xx; user‑friendly error messages; offline mode if `--offline`.

##### Manifest Schema (v1)
```json
{
  "version": 1,
  "entries": [
    {
      "id": "<stable-id or checksum>",
      "filename": "<current filename>",
      "title": "",
      "authors": [""],
      "publication": "",
      "date": "YYYY-MM-DD" ,
      "year": 2023,
      "doi": "",
      "isbn": "",
      "pdf_url": "",
      "source_url": "",
      "processed": true,
      "retrieved_at": "ISO8601",
      "notes": "",
      "tags": []
    }
  ]
}
```

##### PDF Metadata Writing (v1)
- Use `pypdf.PdfWriter.add_metadata` to set `/Title`, `/Author` (comma‑separated), `/Subject` (publication), and custom fields in the Info dict (e.g., `/doi`, `/isbn`).
- XMP/DC/Prism embedding is deferred to v2 (likely via `pikepdf`).

##### Idempotency and Safety
- Compute file checksum (e.g., SHA‑256) to create a stable `id` for manifest deduplication.
- Respect `--skip-existing` to avoid re‑prompting processed files.
- `--dry-run` shows proposed changes (rename path, metadata) without writing.
- Deletion guarded by `--allow-delete` and confirmation prompt.

##### File Renaming
- Slugify `title` and append `_YEAR` if available; ensure uniqueness by appending numeric suffix on collision.
- Optionally move renamed files into `data_raw/` (in‑place by default); emit mapping in manifest.

##### Makefile Integration
- `make scan-metadata [DIR=data_raw] [WRITE=1] [RENAME=yes] [SKIP_EXISTING=1]`
- `make repair-metadata FILE=...` (open in single‑file mode for quick edits).

##### Indexing Integration (v1)
- `lc_build_index.py` reads `manifest.json` (if present) and merges fields (doi, isbn, title, authors, publication, year) into chunk metadata.
- Downstream: allow “DOI/ISBN token” replacement at formatting time (see separate feature bullet) to render inline citations + works cited from manifest.

##### Extensibility
- Source adapters: pluggable client interface so we can add Crossref+, OpenAlex, PubMed, ArXiv.
- Output adapters: JSON manifest today, add CSV/NDJSON if requested.

##### Acceptance Criteria (v1)
- Runs over a folder, detects IDs, fetches metadata, confirms with user, writes Info + manifest.
- Re‑running with `--skip-existing` and `--dry-run` behaves predictably.
- `lc_build_index.py` includes manifest fields in chunk metadata when available.

### [ ] 2. Multi-Shot / Iterative / Agentic / Self-Ask/ Chain-of Query Interaction with LLMs:

Goal: enable an agentic “self-ask” loop where the model can iteratively: (a) decompose a task into sub‑questions, (b) fetch/expand evidence via RAG or other tools, (c) evaluate sufficiency, and (d) either continue or finalize a grounded answer with citations — while keeping internal scratchpads private.

#### Objectives
- Iterative retrieval and reasoning with strict grounding and citations.
- Configurable stop criteria (iterations, token/cost budget, confidence threshold).
- Pluggable tool interface (RAG, web search, calculator, data loaders), with domain allowlists and sandboxing.
- Deterministic/resumable runs via transcript logging and seeds.
- Playbook integration so any step can opt into agentic iteration with per‑step controls.

#### High-Level Architecture
- Components:
  - Planner: proposes sub‑questions, tool calls, and stop conditions (internal scratchpad not exposed).
  - Tool Runner: executes declared tools with validated inputs and returns structured outputs.
  - Evidence Store: merges contexts from tools (RAG docs, web snippets, calc results) with provenance.
  - Decision Loop: state machine that repeats (analyze → act → observe) until stop.
  - Finalizer: composes a user-facing answer with citations, no chain-of-thought.
- State Machine (v1):
  1) init → 2) analyze_task → 3) plan_or_refine → 4) run_tool(s) → 5) assess_sufficiency →
     6a) continue (go to 3) or 6b) finalize → 7) emit_result.

#### Message & Trace Model (internal)
- Transcript item schema (JSONL per run):
  - {"ts": iso8601, "iter": n, "role": "planner|tool|system", "event": "plan|tool_call|tool_result|finalize|error", "content": {...}}
- Planner content:
  - plan: { subquestions: [..], rationale: <string, optional>, requested_tools: [ {name, input} ], stop_if: {confidence>=x | iter>=n | tokens>=m} }
  - Note: rationale is internal; never emitted in user outputs.
- Tool result content:
  - { name, ok: bool, output, provenance: {source_ids|urls|calc}, usage: {latency_ms, tokens}, error?: string }
- Finalizer content:
  - { answer_text, citations: [ {label, source_id|url, pages?} ], limitations: [..] }

#### Tool Interface (spec)
- Tool registration: Python entrypoints under `src/tools/` or registry mapping.
- Tool spec (YAML or code):
  ```yaml
  name: web_search
  version: 1
  description: Restricted domain web search and snippet fetcher
  input_schema:
    type: object
    properties:
      query: {type: string}
      allow: {type: array, items: {type: string}, description: domain allowlist patterns}
      top_k: {type: integer, minimum: 1, maximum: 20, default: 5}
    required: [query]
  output_schema:
    type: object
    properties:
      results: {
        type: array,
        items: {type: object, properties: {title: {type: string}, url: {type: string}, snippet: {type: string}, source_id: {type: string}}}
      }
  sandbox: {network: true}
  side_effects: false
  auth: none
  rate_limits: {rpm: 30}
  ```
- Runtime contract:
  - Input validation against `input_schema`; reject unknown fields.
  - Return strictly `output_schema` on success; include `ok=false` and `error` on failure.
  - Capture `provenance` per item (e.g., `source_id`, `url`, `pages`).
  - Enforce allowlists: `allow` from CLI/config/playbook is intersected with tool defaults.

#### RAG Tool (built-in)
- Name: `rag_retrieve`
- Input: {query: string, k: int, retriever_profile?: enum(vector|hybrid|bm25), rerank?: bool}
- Output: {docs: [ {source_id, title, text, pages?, score, metadata} ]}
- Implementation: wraps `RetrieverFactory.create_hybrid_retriever()` with options aligned to `src/core/retriever.py`.

#### CLI Additions (v1)
- `ask` command flags:
  - `--agent on|off` (default: off)
  - `--iterations N` (default: 3)
  - `--expand N` (query expansions per iter; default: 0)
  - `--tools name1,name2` (default: rag_retrieve)
  - `--domains pattern1,pattern2` (for tools that access network; default: none)
  - `--budget.tokens N` and `--budget.cost USD` (default: disabled)
  - `--confidence float` (0–1 threshold to stop; optional)
  - `--transcript PATH` (write JSONL trace; `--resume PATH` to continue)
  - `--seed INT` (optional deterministic behavior where supported)
  - `--k INT` (top‑k per `rag_retrieve`)
  - `--profile vector|hybrid|bm25` (retriever profile)

#### Config Schema (YAML)
```yaml
agent:
  enabled: false
  iterations: 3
  expand: 0
  confidence: null
  budgets:
    tokens: null
    cost: null
  tools:
    allow: [rag_retrieve]
    domains: []   # e.g., ['*.wikipedia.org', 'arxiv.org']
  retriever:
    profile: hybrid   # vector|hybrid|bm25
    k: 15
    rerank: true
  transcripts:
    dir: logs/agent_runs
    keep: true
```

#### Iteration Flow (concrete)
1) Initialize context with user `question` and optional `task` (kept distinct from retrieval query as in `src/cli/commands.py`).
2) Planner proposes sub‑questions and tool calls. If `expand>0`, also yields `expanded_queries` variants.
3) Run tools in sequence or parallel (configurable later; serial in v1). Always include `rag_retrieve` unless disabled.
4) Evidence Store merges results, deduplicates by `source_id`, tracks pages/metadata.
5) Assess sufficiency:
   - Heuristics: at least X distinct sources; coverage of sub‑questions; confidence threshold via LLM scoring prompt.
   - Stop if: `iter>=iterations` OR `tokens/cost` budgets exceeded OR `confidence>=threshold`.
6) Finalizer composes answer:
   - Include inline citations and a source list; no chain-of-thought.
   - Add a “limitations/next steps” section if evidence sparse.
7) Emit result, write transcript JSONL, include summary metrics (iters, tokens, tools used).

#### Playbook Integration (v1)
- Any step may opt into agentic mode:
  ```yaml
  steps:
    - name: themes
      agent:
        enabled: true
        iterations: 4
        tools: [rag_retrieve, web_search]
        domains: ['*.nih.gov', '*.wikipedia.org']
        retriever: {profile: hybrid, k: 20, rerank: true}
        expand: 1
      prompt: |
        Extract major themes/claims with evidence. Return a table:
        | Source | Claim | Evidence (quote) | Pages | Notes |
  ```
- If a playbook sets `agent.enabled`, CLI flags can override or complement (CLI has precedence when provided).

#### Persistence, Resume, and Metrics
- Write transcripts to `logs/agent_runs/<timestamp>_<slug>.jsonl` with the trace schema above.
- `--resume` loads the last transcript line, restores Evidence Store and iteration counters, continues until stop.
- Metrics per iteration: tokens in/out, tool latency, docs added, confidence score; summarized to a final `metrics.json`.

#### Safety, Privacy, and Sandboxing
- Never emit planner rationales or chain-of-thought in user outputs.
- Domain allowlist enforced for network tools; no file system side‑effects by default.
- Configurable max pages/bytes per fetch; redact secrets from logs; honor `OPENAI_API_KEY` only for LLM calls.

#### Acceptance Criteria (v1)
- `ask --agent --iterations 2 --tools rag_retrieve` performs two iterations of retrieval + assessment and returns a citation‑rich answer.
- Transcript file contains planner steps, tool calls, and results per iteration; final output omits internal rationales.
- Playbook step with `agent.enabled: true` runs iteratively and respects tool/domain/retriever settings.
- Stop conditions trigger correctly (iterations, budgets, confidence).
- Unit tests include: state transition logic, tool input validation, and finalizer citation formatting.

#### Implementation Notes
- Reuse `RetrieverFactory` and `LLMFactory` for consistent setup.
- Prefer OpenAI tool/function-calling via LangChain when available; otherwise implement a thin local tool runner enforcing the spec.
- Start with `rag_retrieve` only; add `web_search` behind a domain allowlist in a later PR.

#### Open Questions / Future Work
- Parallel tool execution and evidence ranking fusion across tools.
- Advanced query planning (learning to stop, reward models) and multi‑agent roles (planner vs. solver).
- Web archiving of fetched pages and PDF citation extraction.


### [ ] 3. Revisit Playbooks Functionality
  - Operation of system should be abstracted enough such that a yaml file can represent a multi-stage processess resulting in a finished product. Something like
  
  ```yaml
  #playbooks/technical_manual.yaml
  config: {(path supplied as a cli argument, or content piped in through stdin, format:json)}
  step: 1
    name: 'generate outline' 
    model: {config:outline-model|gpt-5-mini} (after | indicates default value, need to extend token system to handle this)
    verbosity: {config:step-1-verbosity}
    system_prompt: 'you are an managing editor for a publishing company producing technical manuals... etc... etc..'
    multiple_parts: false
    max_requery_iterations: {config:outline_requery_iterations|1} 
    instruction_user_prompt: 'generate an outline for a book titled {config:title}, about {config:subject}, {config:depth} level deep with {config:sections-count[0]} chapters at level 1, {config:sections-count[1]} sections per chapter at level 2, {config:sections-count[2]} sections each at level 3, {config:sections-count[3]} sections each at level 4. {config:outline-additional-instructions}, return the outline in json format.'
    output: ./exports/outlines/outline_{config:title_slug}.json
    output_format: [{'section':{input:this::section},'title':{config:title},'parent_section':{input:parent::section or null}},...]
  step: 2
    name: 'generate prompts'
    model: {config:prompt-model|gpt-5-mini}
    verbosity: {config:step-4-verbosity}
    system_prompt: 'you are a writing assistant ai helping to produce a technical manual for {target-audience} about {config:subject}, titled {config:title}. with a target word count of {config:target-word-count}. you create prompts optimized for consumption by the llm model {config:writing-model|gpt-5} for the generation of content for the book using the sections from a detailed outline, the prompts you generate are specifically designed to result in the generation of content that fits within the context of the book without generating content that bleeds outside of the scope of the section targeted. this is important since the model generating each section will see this as an isolated task. provide as much content/context as needed, and use up to {config:max-iterations} iterations querying the vector database to determine the content of the prompt and any context/primary sources that should be provided to support any claims of fact to be written about.'
    multiple_parts: true (input file is an array at top level, so it knows to split job there without further instruction, but I'll define a syntax anyway)
    input_division: file + array (only one level of looping even if input were split over multiple files designated by the +,if there were multiple files - denoted by a * in the input value - it would load the entire list of files and concatenate the top level arrays and the outermost loop would be on that resulting array. If instead here we listed 'file > array', this would mean outer loop is list of files and the inner loop each array within files. Since 1) there are not multiple files AND 2) even if there were, we have specified +, there will only be one loop, no nested loops for this task input)
    input:./exports/outlines/outline_{config:title_slug}.json
    instruction_user_prompt: 'generate {config:variations} prompts for the section titled {config:section_title}, these should contain unique content that can be merged together to create a larger section. For example if 2 resulting sections of 1000 words each are merged together, the final result should be approximately 1500 words. It is ok for some replication to occur so that options are available to select the best composition, but this is only half of the reason for the variations. target word count for each section is {config:section-word-count:1000}.'
    max_requery_iterations: {config:prompts_requery_iterations|3} (a value of 0 here disables requerying)
    output: ./exports/jobs/job_{config:title_slug}.json
    output_file_format: '{(input file fields)...,[{'variation':this::%i, 'prompt':{this::output}}]}' (make 1 big array but flush after each prompt is written in case script crashes partway through we don't loose work )
  step 3: 
    name: 'generate sections'
    multiple_parts: true (input is multiple files with array at top level)
    input_division: file + array (same as the last example, even though in this case the jobs represent nested sections, it is not necessary to nest the jobs since the jobs file contains 'parent' to indicate the placement of the section in the content hierarcy for merging later on)
    system_prompt: 'you are a technical writer, you are very detail oriented, you are producing a book for {target-audience} about {config:subject}, titled {config:title}. with a target word count of {config:target-word-count}'
    instruction_user_prompt: {input:this::prompt}
    input:./exports/jobs/job_{title_slug}.json
    verbosity: {config:step-3-verbosity}
    max_requery_iterations: {config:sections_requery_iterations|1} 
    model: {config:writing-model|gpt-5}
    multiple_parts: true
    output: ./exports/content/sections_{config:title_slug}.json
    output_file_format: '[{(input item fields)...,variations:[{'variation':this::%i, 'generated_content':{this::ouput}}...] ...]'
  step 4: 
    name: 'merge sections'
    model: {config:merge-model|gpt-5 }
    verbosity: {config:step-4-verbosity}
    max_requery_iterations: {config:merge_requery_iterations|0} 
    system_prompt: 'you are a managing editor for a publishing company, you receive multiple sections of text for each part in the outline of the book titled {config:title}, about {config:subject}, with a target word count of {config:target-word-count} you read through the variations and select the best version or an amalgamation thereof of any redundant content, and intellegently merge the results into {config:draft_count|1} final draft version(s). '
    instruction_user_prompt: 'merge the following sections {input:variations[0]} ...{input:variations[n]}'
    output: ./exports/content/sections_merged_{config:title_slug}.json
    output_file_format: '[{(input item fields)...,draft:[{'draft':this::%i, 'generated_content':{this::ouput}}...] ...]'
  ```

  ### and a config file

  ```yaml
  #job/config/ai_in_primary_education_a_technical_manual.yaml
  playbook: 'playbooks/technical_manual.yaml'
  step-1-verbosity: 5
  step-2-verbosity: 3
  step-3-verbosity: 8
  step-4-verbosity: 8
  outline-additional-instructions: 'lorem ipsum dolor sit amet...'
  section-word-count: 1200
  outline_requery_iterations: 2
  draft_count: 2
  title: 'AI Use in Primary School Education'
  title_slug: 'ai-use-primary-school-education'
  rag_key: 'chatgptedu2'
  subject: 'AI Use in Education'
  depth: 4
  sections-count:
    - '10-15'
    - '4'
    - '2-4'
    - '3-6'
  target-audience: "primary school teachers and administrators with a master's degree"
  target-word-count: '150000-300000'
  writing-model: gpt-5
```


### [ ] 4. Make Tools Available to Models:
In addition to being able to query RAG for information necessary to perform an analysis, it there are other sources of information that would certainly be beneficial to an LLM capable of performing reasoning tasks for scientific analysis or any other sort of 'cognitive' or 'thinking' sort of task. Of course the LLM can perform simple aritmetic and algebra, but I'm not so sure about complex statistical analyses. For that reason, it might be useful to provide it access to a mathematical package, like R? (along with some recipies for performing stuff like z-tests, t-tests, chi-square, anova etc, this would be especially useful in the literature review/meta analysis research type work:

  a) R/Julia/Spark/KNIME/Numpy
  b) Pandas/Polars/Tidytable/dtplyr
  c) SageMath/Spyder/Maxima/SymPy/OpenAxiom/Octave/Scilab/
  d) XCos/QUCS/Qucs-S/Pspice/SPICE/
  e) ELKI/WEKA/AdvancedMiner/Altair RapidMiner/Orange

Some other data sources we might find useful to expose:
  a) GIS Data files
  b) Scraped Data
  c) Maltego Projects
  d) OSINT sources
  e) Various APIs 
  f) Wikipedia
  g) any suggestions?

It seems like the creation of a standardized connector module is in order for these sorts of (external) data sources. 

So we have 3 sorts of tools we should handle 1) external data sources 2) existing data analysis tools, and finally 3) information display/formatting tools such as:
  a) graphing/plotting
    -mathplotlib
    -plotly
    -d3.js
    -Bokeh
    -Google Charts
    -Apache Echarts
    -Vis.js
    -Scichart
  
  b) mapping (GIS etc)
    -ArcGis
    -QGis
    -GrassGIS
    -Placemark
    -Plonk

  c) Advanced Text Formatting tools like TEX/LaTEX
    -TeX
    


