# Research Collector

A powerful tool for collecting and managing academic sources from Google Scholar search results.

## Features

- 📚 **Extract article metadata** from Google Scholar search results
- 🔍 **Parse HTML** from URLs or saved files
- 📝 **Interactive form** for editing metadata (Textual UI with fallback)
- 📥 **PDF download** with automatic slugified naming
- 💾 **Metadata storage** in PDF files and manifest.json
- 🎯 **Duplicate detection** to avoid collecting the same articles twice

## Installation

### Required Dependencies

```bash
pip install beautifulsoup4 requests typer rich textual
```

### Optional Dependencies

- `textual` - For the interactive TUI (highly recommended)
- If Textual is not available, the script falls back to a console-based interface

## Usage

### Basic Usage

```bash
# From a Google Scholar search URL
python research/collector.py --url "https://scholar.google.com/scholar?q=artificial+intelligence+education"

# From a saved HTML file
python research/collector.py --file "path/to/saved_search.html"

# Custom output directory
python research/collector.py --url "https://scholar.google.com/scholar?q=..." --output "my_research"
```

### Command Line Options

- `--url, -u`: Google Scholar search URL
- `--file, -f`: Path to saved HTML file
- `--output, -o`: Output directory (default: `research/out`)

## How It Works

1. **Input Processing**: Accepts either a Google Scholar URL or saved HTML file
2. **HTML Parsing**: Uses BeautifulSoup to extract article metadata from Google Scholar's HTML structure
3. **Metadata Extraction**: Parses title, authors, publication, date, DOI, and PDF links
4. **Interactive Review**: Displays a form for each article to review and edit metadata
5. **PDF Download**: Downloads PDFs with automatic filename generation
6. **Data Storage**: Saves metadata in both PDF files and a `manifest.json`

## Interactive Interface

The script provides an interactive terminal interface with:

### Textual UI (Recommended)
- **Fuchsia background** with black input fields
- **White text** on black backgrounds for fields
- **Blue buttons** with black text
- **Two-column layout** for efficient data entry
- **Keyboard shortcuts** and clickable buttons

### Fallback Console UI
- Clean table-based interface using Rich
- Keyboard-driven navigation
- All functionality available through text commands

## Controls

### Keyboard Shortcuts
- `d` - Download PDF (if available)
- `s` - Save changes
- `n` - Next article
- `p` - Previous article
- `q` - Quit and save

### Field Editing
Type any field name (`title`, `authors`, `date`, `publication`, `doi`) to edit it.

## Output Structure

```
research/out/
├── manifest.json          # Complete metadata for all articles
├── understanding_artificial_intelligence_2023.pdf
├── machine_learning_applications_2022.pdf
└── future_of_ai_2024.pdf
```

### manifest.json Format

```json
[
  {
    "title": "Understanding Artificial Intelligence in Education",
    "authors": ["Smith J", "Johnson A", "Brown R"],
    "date": "2023",
    "publication": "Journal of Educational Technology",
    "doi": "10.1000/example1",
    "pdf_url": "https://example.com/paper1.pdf",
    "scholar_url": "https://scholar.google.com/...",
    "pdf_filename": "understanding_artificial_intelligence_2023.pdf",
    "downloaded": true
  }
]
```

## Google Scholar HTML Structure

The script parses Google Scholar's HTML structure, specifically looking for:

- **Article containers**: `div` elements with classes like `gs_r`, `gs_ri`, `gs_scl`
- **Titles**: `h3` elements with `gs_rt` class containing links
- **Authors/Publication**: `div` elements with `gs_a` class
- **PDF links**: Any `a` elements with `.pdf` in the href
- **DOI links**: Links containing `doi.org` or DOI patterns

## Tips for Best Results

1. **Save Google Scholar pages** for offline processing to avoid rate limiting
2. **Use specific search terms** to get more relevant results
3. **Review metadata** carefully before downloading PDFs
4. **Check the manifest.json** to see all collected articles
5. **Resume collection** - the script detects duplicates automatically

## Troubleshooting

### No articles found
- Check that the HTML contains Google Scholar search results
- Verify the HTML structure hasn't changed
- Try a different search or save a fresh page

### PDF download fails
- Some PDFs may require authentication or have access restrictions
- Check the PDF URL in the metadata
- Try downloading manually from the Google Scholar link

### Textual UI not working
- Install Textual: `pip install textual`
- The script will automatically fall back to console mode

## Integration with RAG Writer

The collected PDFs and metadata can be used directly with the RAG Writer system:

1. Place PDFs in your data source directory
2. Use the manifest.json for bulk metadata import
3. The slugified filenames work perfectly with the RAG indexing system

## Examples

### Collect AI Education Research
```bash
python research/collector.py --url "https://scholar.google.com/scholar?q=artificial+intelligence+education+2020-2024"
```

### Process Saved Search Results
```bash
python research/collector.py --file "downloads/scholar_ai_education.html"
```

### Custom Output Location
```bash
python research/collector.py --url "https://scholar.google.com/scholar?q=machine+learning" --output "research/ai_papers"
```

## File Structure

```
research/
├── collector.py          # Main script
├── styles.css           # Textual UI styles
├── test_scholar.html    # Test HTML file
├── README.md            # This documentation
└── out/                 # Default output directory
    ├── manifest.json    # Article metadata
    └── *.pdf           # Downloaded PDFs
```

---

**Happy researching! 📚🔬**