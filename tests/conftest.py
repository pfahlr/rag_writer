"""
Pytest configuration and shared fixtures for LangChain content generation tests.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Also add current directory src for alternative import path
sys.path.insert(0, "src")

from langchain.lc_outline_converter import OutlineSection, BookMetadata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_outline():
    """Sample text outline for testing."""
    return """Professional Development Handbook for Primary School Educators

1. Foundations of Modern Education
  1A. Learning Theories
    1A1. Constructivist Learning Theory
    1A2. Social Learning and Connectivism
  1B. Classroom Management Fundamentals
    1B1. Establishing Classroom Rules

2. Implementing Modern Teaching Strategies
  2A. Technology Integration
    2A1. Digital Tools for Learning
"""


@pytest.fixture
def sample_markdown_outline():
    """Sample markdown outline for testing."""
    return """# Professional Development Handbook for Primary School Educators

## Chapter 1: Foundations of Modern Education

### Section 1A: Learning Theories

#### Subsection 1A1: Constructivist Learning Theory
#### Subsection 1A2: Social Learning and Connectivism

### Section 1B: Classroom Management Fundamentals

#### Subsection 1B1: Establishing Classroom Rules

## Chapter 2: Implementing Modern Teaching Strategies

### Section 2A: Technology Integration

#### Subsection 2A1: Digital Tools for Learning
"""


@pytest.fixture
def sample_json_outline():
    """Sample JSON outline for testing."""
    return {
        "title": "Professional Development Handbook for Primary School Educators",
        "topic": "Education",
        "target_audience": "Primary School Teachers",
        "chapters": [
            {
                "number": 1,
                "title": "Foundations of Modern Education",
                "sections": [
                    {
                        "letter": "A",
                        "title": "Learning Theories",
                        "subsections": [
                            {"number": 1, "title": "Constructivist Learning Theory"},
                            {"number": 2, "title": "Social Learning and Connectivism"}
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_outline_sections():
    """Sample parsed outline sections."""
    return [
        OutlineSection(
            id="1",
            title="Foundations of Modern Education",
            level=1,
            description="Level 1: Foundations of Modern Education"
        ),
        OutlineSection(
            id="1A",
            title="Learning Theories",
            level=2,
            parent_id="1",
            description="Level 2: Learning Theories"
        ),
        OutlineSection(
            id="1A1",
            title="Constructivist Learning Theory",
            level=3,
            parent_id="1A",
            description="Level 3: Constructivist Learning Theory"
        ),
        OutlineSection(
            id="1A2",
            title="Social Learning and Connectivism",
            level=3,
            parent_id="1A",
            description="Level 3: Social Learning and Connectivism"
        )
    ]


@pytest.fixture
def sample_book_metadata():
    """Sample book metadata."""
    return BookMetadata(
        title="Professional Development Handbook for Primary School Educators",
        topic="Education",
        target_audience="Primary School Teachers",
        author_expertise="Intermediate",
        word_count_target=50000,
        description="A comprehensive guide for primary school educators"
    )


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return [
        {
            "task": "You are a content writer creating educational material.",
            "instruction": "Write an engaging introduction to the topic.",
            "context": {
                "book_title": "Test Book",
                "subsection_id": "1A1",
                "target_audience": "General readers"
            }
        },
        {
            "task": "You are a content writer creating educational material.",
            "instruction": "Provide detailed explanations and examples.",
            "context": {
                "book_title": "Test Book",
                "subsection_id": "1A1",
                "target_audience": "General readers"
            }
        }
    ]


@pytest.fixture
def sample_batch_results():
    """Sample batch processing results."""
    return [
        {
            "section": "1A1",
            "task": "Content writing task",
            "instruction": "Write introduction",
            "generated_content": "This is a sample introduction about constructivist learning theory.",
            "sources": [],
            "status": "success"
        },
        {
            "section": "1A1",
            "task": "Content writing task",
            "instruction": "Write detailed explanation",
            "generated_content": "This is a detailed explanation of constructivist learning theory.",
            "sources": [],
            "status": "success"
        }
    ]


@pytest.fixture
def mock_lc_ask_response():
    """Mock response from lc_ask.py."""
    return {
        "generated_content": "This is a mock response from the language model.",
        "sources": ["source1", "source2"],
        "status": "success"
    }


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing."""
    def mock_run(cmd, **kwargs):
        # Create a mock result that looks like a successful subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "generated_content": "Mock generated content",
            "sources": [],
            "status": "success"
        })
        mock_result.stderr = ""
        return mock_result

    return mock_run


@pytest.fixture(autouse=True)
def mock_root_directory(temp_dir):
    """Mock the ROOT directory for testing."""
    with patch('langchain.lc_outline_converter.ROOT', temp_dir), \
         patch('langchain.lc_batch.ROOT', temp_dir), \
         patch('langchain.lc_merge_runner.ROOT', temp_dir), \
         patch('langchain.lc_book_runner.ROOT', temp_dir):
        yield temp_dir


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    with patch('langchain.lc_outline_converter.console') as mock_console, \
         patch('langchain.lc_batch.console') as mock_batch_console, \
         patch('langchain.lc_merge_runner.console') as mock_merge_console, \
         patch('langchain.lc_book_runner.console') as mock_book_console:

        # Configure all console mocks to do nothing
        for console_mock in [mock_console, mock_batch_console, mock_merge_console, mock_book_console]:
            console_mock.print = Mock()
            console_mock.rule = Mock()
            console_mock.status = Mock()

        yield mock_console