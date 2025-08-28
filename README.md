# Advanced LangChain Content Processing Suite

A sophisticated suite of scripts for AI-powered content generation, processing, and merging with multi-stage editorial pipelines.

## 🎯 Overview

This suite provides a complete workflow for content creation and processing using LangChain and large language models. From initial content generation through intelligent merging and editorial refinement, it supports both simple workflows and complex multi-stage pipelines.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Scripts Overview](#scripts-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Pipeline Types](#pipeline-types)
- [API Reference](#api-reference)
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

```bash
# 1. Build knowledge index (optional, for RAG)
python src/langchain/lc_build_index.py

# 2. Generate content variations
python src/langchain/lc_batch.py

# 3. Merge and refine content
python src/langchain/lc_merge_runner.py
```

## 📜 Scripts Overview

### lc_ask.py - Core LLM Interface

**Purpose**: Direct interface to language models for single queries.

**Key Features**:
- Flexible prompt engineering
- Multiple content types
- JSON output support
- Retrieval-augmented generation (RAG)

**Usage**:
```bash
python src/langchain/lc_ask.py ask --content-type research --task "Your prompt here"
```

### lc_batch.py - Batch Processing

**Purpose**: Process multiple content generation jobs in parallel.

**Key Features**:
- JSONL job file processing
- Parallel execution
- Result aggregation
- Progress tracking

**Usage**:
```bash
python src/langchain/lc_batch.py --jobs data_jobs/example.jsonl
```

### lc_build_index.py - Index Builder

**Purpose**: Create vector indexes for retrieval-augmented generation.

**Key Features**:
- Document ingestion
- Vector embeddings
- Index optimization
- Multiple data sources

**Usage**:
```bash
python src/langchain/lc_build_index.py --source data/ --index my_index
```

### lc_merge_runner.py - Advanced Merge System

**Purpose**: Intelligent content merging with multi-stage editorial pipelines.

**Key Features**:
- Multi-stage processing (critique → merge → style → images)
- AI-powered content scoring
- Jaccard similarity de-duplication
- YAML-driven configuration
- Command-line and interactive modes

**Usage**:
```bash
# Interactive mode
python src/langchain/lc_merge_runner.py

# Job file processing
python src/langchain/lc_merge_runner.py --sub 1A1

# Custom job file
python src/langchain/lc_merge_runner.py --jobs /path/to/jobs.jsonl
```

### lc_book_runner.py - Book Orchestrator

**Purpose**: High-level orchestration for entire books and chapters.

**Key Features**:
- Hierarchical book structure processing (4 levels deep)
- Automatic job file generation
- Batch and merge pipeline orchestration
- Final document aggregation
- Progress tracking and error recovery
- Section dependency management

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

### lc_outline_generator.py - Interactive Outline Creator

**Purpose**: Generate intelligent book outlines using LangChain's indexed knowledge.

**Key Features**:
- Interactive book detail collection
- Outline depth selection (3-5 levels)
- AI-powered outline generation using indexed content
- Automatic conversion to book runner format
- Comprehensive outline validation and summary

**Usage**:
```bash
# Interactive outline generation
python src/langchain/lc_outline_generator.py

# Save to specific location
python src/langchain/lc_outline_generator.py --output my_book_outline.json
```

### lc_outline_converter.py - Outline to Book Structure Converter

**Purpose**: Convert existing outlines into book structure and job files.

**Key Features**:
- Multiple input format support (JSON, Markdown, Text)
- Automatic hierarchical context generation
- Job file generation for each subsection
- Dependency relationship detection
- Format validation and conversion

**Supported Input Formats**:
- **JSON**: Structured outline format (from lc_outline_generator.py)
- **Markdown**: Header-based outline (# ## ### ####)
- **Text**: Numbered/lettered outline (1. 2. A. B. etc.)

**Usage**:
```bash
# Convert JSON outline
python src/langchain/lc_outline_converter.py --outline my_outline.json

# Convert markdown outline
python src/langchain/lc_outline_converter.py --outline outline.md --output book_structure.json

# Convert with metadata overrides
python src/langchain/lc_outline_converter.py --outline outline.txt \
  --title "My Book Title" \
  --topic "Machine Learning" \
  --audience "Data Scientists" \
  --wordcount 75000
```

**Makefile Usage**:
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
```

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

## ⚙️ Configuration

### YAML Configuration Files

#### merge_types.yaml

Defines different merge pipeline configurations:

```yaml
generic_editor:
  description: "Basic editor merge"
  system_prompt:
    - "You are a senior editor for a publisher..."

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

#### content_types.yaml

Defines content type configurations for lc_ask.py:

```yaml
pure_research:
  system_prompt: "You are a research assistant..."
  temperature: 0.7

creative_writing:
  system_prompt: "You are a creative writer..."
  temperature: 0.9
```

### Environment Configuration

Create `env.json` with your API keys and settings:

```json
{
  "openai_api_key": "your-key-here",
  "anthropic_api_key": "your-key-here",
  "default_model": "gpt-4",
  "embedding_model": "text-embedding-ada-002"
}
```

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

## 📚 API Reference

### lc_ask.py

```bash
python src/langchain/lc_ask.py ask [OPTIONS]

Options:
  --content-type TEXT    Content type from content_types.yaml
  --task TEXT           The task/prompt for the LLM
  --json FILE          JSON file with job specification
  --key TEXT           Collection key for RAG
  --k INT             Top-k results for retrieval
  --output FILE        Output file path
```

### lc_batch.py

```bash
python src/langchain/lc_batch.py [OPTIONS]

Options:
  --jobs FILE          JSONL file with jobs
  --output DIR         Output directory
  --parallel INT       Number of parallel processes
  --key TEXT          Collection key for RAG
  --k INT            Top-k results for retrieval
```

### lc_merge_runner.py

```bash
python src/langchain/lc_merge_runner.py [OPTIONS]

Options:
  --sub TEXT          Subsection ID for job file
  --jobs FILE         Custom job file path
  --key TEXT         Collection key for RAG
  --k INT           Top-k results for retrieval
  --batch-only       Force batch results only
```

### lc_book_runner.py

```bash
python src/langchain/lc_book_runner.py [OPTIONS]

Options:
  --book FILE         JSON file defining book structure (required)
  --output FILE       Output markdown file path
  --force             Force regeneration of all content
  --skip-merge        Skip merge processing, only run batch
```

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
python -c "import yaml; yaml.safe_load(open('src/tool/prompts/merge_types.yaml'))"
```

#### 4. API Key Issues
```bash
# Check environment configuration
cat env.json
# Ensure API keys are set and valid
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

## 🤝 Contributing

### Adding New Pipeline Types

1. Add configuration to `src/tool/prompts/merge_types.yaml`
2. Test with sample content
3. Update documentation

### Adding New Content Types

1. Add configuration to `src/tool/prompts/content_types.yaml`
2. Test with lc_ask.py
3. Document usage examples

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include error handling
- Update tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain for LLM integration
- Uses Rich for beautiful terminal interfaces
- Inspired by advanced content processing workflows
- Designed for educational and professional content creation

