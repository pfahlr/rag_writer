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

---

**Version**: 2.0.0
**Last Updated**: 2025-08-28
**Maintained by**: Kilo Code Assistant
