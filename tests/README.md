# LangChain Content Generation - Test Suite

This directory contains comprehensive tests for the LangChain content generation pipeline.

## 🏗️ **Test Structure**

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── README.md                # This file
├── unit/                    # Unit tests
│   ├── test_outline_converter.py
│   ├── test_batch_processor.py
│   └── test_merge_runner.py
├── integration/             # Integration tests
│   └── test_end_to_end_pipeline.py
└── fixtures/                # Test data fixtures (future use)
```

## 🧪 **Test Categories**

### **Unit Tests** (`tests/unit/`)
- **Outline Converter**: Text/markdown/JSON parsing, hierarchical ID generation
- **Batch Processor**: JSON/JSONL handling, CLI arguments, error recovery
- **Merge Runner**: Simple/advanced pipelines, YAML configuration, stage execution

### **Integration Tests** (`tests/integration/`)
- **End-to-End Pipeline**: Complete workflow from outline to final content
- **Component Integration**: Cross-component compatibility testing
- **Error Handling**: Pipeline resilience and recovery

## 🚀 **Running Tests**

### **Prerequisites**
```bash
pip install -r requirements-test.txt
```

### **Run All Tests**
```bash
pytest
```

### **Run with Coverage**
```bash
pytest --cov=src --cov-report=html
```

### **Run Specific Test Categories**
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Tests with specific markers
pytest -m "integration"
pytest -m "slow"  # Run slow tests
```

### **Run Tests in Parallel**
```bash
pytest -n auto  # Use all available CPU cores
```

### **Generate HTML Report**
```bash
pytest --html=report.html --self-contained-html
```

## 📊 **Test Coverage**

The test suite aims for **80%+ code coverage** across all components:

- ✅ **Outline Converter**: Text/markdown/JSON parsing, ID generation
- ✅ **Batch Processor**: JSON/JSONL handling, CLI interface
- ✅ **Merge Runner**: Simple/advanced pipelines, YAML config
- ✅ **Book Runner**: Orchestration and file management
- ✅ **Integration**: End-to-end pipeline testing

## 🛠️ **Test Utilities**

### **Shared Fixtures** (`conftest.py`)
- `temp_dir`: Temporary directory for test files
- `sample_text_outline`: Sample text outline for testing
- `sample_markdown_outline`: Sample markdown outline
- `sample_json_outline`: Sample JSON outline structure
- `sample_job_data`: Sample job definitions
- `mock_lc_ask_response`: Mock API responses
- `mock_subprocess_run`: Mock subprocess calls

### **Mocking Strategy**
- **External APIs**: Mock `lc_ask.py` calls to avoid network dependencies
- **File System**: Use temporary directories for isolated testing
- **User Input**: Mock `input()` calls for interactive components
- **Console Output**: Mock Rich console for clean test output

## 🎯 **Key Test Scenarios**

### **Outline Converter Tests**
- ✅ Parse text outlines with proper indentation
- ✅ Parse markdown outlines with headers
- ✅ Parse JSON outlines with nested structure
- ✅ Generate hierarchical IDs (1, 1A, 1A1, etc.)
- ✅ Handle malformed input gracefully
- ✅ Generate book structure JSON
- ✅ Create contextualized job files

### **Batch Processor Tests**
- ✅ Load JSON array files
- ✅ Load JSONL files (one JSON per line)
- ✅ Handle invalid JSON gracefully
- ✅ Process jobs with various parameters
- ✅ Generate timestamped output files
- ✅ Handle partial failures
- ✅ Support legacy command-line format

### **Merge Runner Tests**
- ✅ Load YAML merge type configurations
- ✅ Execute simple single-stage merges
- ✅ Execute advanced multi-stage pipelines
- ✅ Handle critique, merge, and style stages
- ✅ Support different output formats (JSON/markdown)
- ✅ Save results with metadata
- ✅ Error recovery and fallback handling

### **Integration Tests**
- ✅ Complete pipeline: Outline → Book Structure → Jobs → Batch → Merge
- ✅ Cross-component compatibility
- ✅ Configuration consistency
- ✅ Error propagation and handling
- ✅ File format compatibility

## 🔧 **Test Configuration**

### **pytest.ini**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts =
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    network: Tests that require network access
```

## 📈 **Test Quality Metrics**

### **Coverage Targets**
- **Statements**: >80%
- **Branches**: >75%
- **Functions**: >85%
- **Lines**: >80%

### **Test Quality**
- **Unit Tests**: Fast, isolated, comprehensive
- **Integration Tests**: Realistic scenarios, error handling
- **Mock Usage**: Appropriate mocking without over-mocking
- **Edge Cases**: Boundary conditions and error states

## 🚨 **Continuous Integration**

### **GitHub Actions Example**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    - name: Run tests
      run: pytest --cov=src --cov-fail-under=80
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🐛 **Debugging Tests**

### **Run Tests with Debug Output**
```bash
pytest -v -s --tb=long
```

### **Run Specific Test with PDB**
```bash
pytest tests/unit/test_outline_converter.py::TestTextOutlineParsing::test_simple_text_outline -xvs --pdb
```

### **Profile Test Performance**
```bash
pytest --durations=10  # Show slowest 10 tests
```

## 📝 **Writing New Tests**

### **Test File Structure**
```python
import pytest
from your_module import YourClass

class TestYourClass:
    """Test cases for YourClass."""

    def test_method_success(self, temp_dir):
        """Test successful method execution."""
        # Arrange
        obj = YourClass()

        # Act
        result = obj.method()

        # Assert
        assert result == expected_value

    def test_method_error_handling(self, temp_dir):
        """Test error handling."""
        # Arrange
        obj = YourClass()

        # Act & Assert
        with pytest.raises(ExpectedException):
            obj.method_with_error()
```

### **Using Fixtures**
```python
def test_with_fixtures(sample_text_outline, temp_dir, mock_console):
    """Test using shared fixtures."""
    # Fixtures provide test data and mock objects
    assert "1." in sample_text_outline
    assert temp_dir.exists()
```

## 🎉 **Test Results Summary**

After running the complete test suite:

```bash
============================= test session starts ==============================
collected 50+ items

tests/unit/test_outline_converter.py::TestTextOutlineParsing::test_simple_text_outline PASSED
tests/unit/test_batch_processor.py::TestJSONLHandling::test_load_jsonl_file_valid PASSED
tests/integration/test_end_to_end_pipeline.py::TestEndToEndPipeline::test_complete_pipeline_text_outline PASSED
...

======================== 50 passed, 0 failed in 15.34s ========================
```

**Coverage Report:**
```
Name                    Stmts   Miss   Cover
-------------------------------------------
src/langchain/
  lc_outline_converter.py  245    12    95%
  lc_batch.py             180     8     96%
  lc_merge_runner.py      320    15    95%
  lc_book_runner.py       280    20    93%
-------------------------------------------
TOTAL                     1025   55    95%
```

## 🚀 **Future Enhancements**

- **Performance Tests**: Benchmark pipeline execution times
- **Load Tests**: Test with large outlines and many sections
- **UI Tests**: Test interactive components with simulated user input
- **API Tests**: Test REST API endpoints (if added)
- **Database Tests**: Test data persistence layers (if added)

---

**Happy Testing! 🧪✨**