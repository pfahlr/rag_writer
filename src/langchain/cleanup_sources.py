#!/usr/bin/env python3
"""
Source Cleanup Script for Batch Results

Cleans up sources in existing batch result files by:
1. Filtering to only include cited sources
2. Deduplicating by article (removing page-level duplicates)
3. Updating the files with cleaned sources
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

DOI_RE = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-z0-9]*[a-z0-9]\b', re.I)

def _norm(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r'\W+', ' ', (s or '')).strip().lower()

def _parse_citation_identifier(citation_text: str) -> str:
    """Parse citation text to extract the identifier (removes page numbers and punctuation)."""
    # Remove page references like ", p.10", " p. 5", etc.
    citation_text = re.sub(r',\s*p\.\s*\d+', '', citation_text, flags=re.IGNORECASE)
    citation_text = re.sub(r'\s+p\.\s*\d+', '', citation_text, flags=re.IGNORECASE)

    # Remove surrounding punctuation
    citation_text = citation_text.strip('()[]"')

    return citation_text.strip()

def _extract_cited_docs(content: str, all_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract cited documents from content based on substring matching for citations."""
    if not content or not all_sources:
        return []

    content_lower = content.lower()
    cited_docs = []

    for source in all_sources:
        title = source.get("title", "")
        source_path = source.get("source", "")

        # Use title or source path as identifier
        identifier = title if title else source_path

        if not identifier:
            continue

        identifier_lower = identifier.lower()
        hit = False

        # 1) DOI match (if available) - keep this as it's precise
        doi = source.get("doi", "")
        if doi and (doi.lower() in content_lower or any(m.group(0).lower() == doi.lower() for m in DOI_RE.finditer(content_lower))):
            hit = True

        # 2) Citation pattern matching with substring check
        if not hit:
            # Look for citation patterns like (identifier, p.10) or "identifier"
            citation_pattern = r'["(]([^")]+?)["),]'
            matches = re.findall(citation_pattern, content_lower)

            for match in matches:
                citation_identifier = _parse_citation_identifier(match)

                # Check if citation identifier is a substring of the source identifier
                if citation_identifier and citation_identifier in identifier_lower:
                    # Additional check: citation should be reasonably long (not just single words)
                    if len(citation_identifier) > 10:  # Avoid very short matches
                        hit = True
                        break

        # 3) Direct substring match (for cases without citation formatting)
        if not hit and identifier_lower in content_lower:
            hit = True

        # 4) Fallback: exact title match (for cases where title appears in full)
        if not hit and title and title.lower() in content_lower:
            hit = True

        if hit:
            cited_docs.append(source)

    return cited_docs

def deduplicate_sources(cited_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate sources by article, keeping only one entry per unique article."""
    seen_articles = set()
    deduplicated = []

    for source in cited_sources:
        title = source.get("title", "")
        source_path = source.get("source", "")

        # Use title or source path as the unique identifier
        article_id = title if title else source_path

        # Only add if we haven't seen this article before
        if article_id and article_id not in seen_articles:
            seen_articles.add(article_id)
            # Create clean source entry without page info
            clean_source = {
                "title": title,
                "source": source_path
            }
            deduplicated.append(clean_source)

    return deduplicated

def cleanup_batch_file(file_path: Path, verbose: bool = False) -> bool:
    """Clean up sources in a single batch file."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list. Skipping.")
            return False

        cleaned_count = 0
        total_sources_before = 0
        total_sources_after = 0

        # Process each result
        for result in data:
            if not isinstance(result, dict):
                continue

            content = result.get("generated_content", "")
            current_sources = result.get("sources", [])

            if not content or not current_sources:
                continue

            total_sources_before += len(current_sources)

            # Find cited sources
            cited_sources = _extract_cited_docs(content, current_sources)

            # Deduplicate by article
            cleaned_sources = deduplicate_sources(cited_sources)

            total_sources_after += len(cleaned_sources)

            # Update the result
            if len(cleaned_sources) != len(current_sources):
                if verbose:
                    print(f"  Entry '{result.get('section', 'unknown')}': {len(current_sources)} → {len(cleaned_sources)} sources")
                result["sources"] = cleaned_sources
                cleaned_count += 1

        # Save the cleaned file
        if cleaned_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            reduction = total_sources_before - total_sources_after
            print(f"Cleaned {cleaned_count} entries in {file_path} (removed {reduction} duplicate/unused sources)")
            return True
        else:
            print(f"No changes needed for {file_path}")
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main cleanup function."""
    batch_dir = ROOT / "output" / "batch"

    if not batch_dir.exists():
        print(f"Batch directory not found: {batch_dir}")
        print("Make sure you have run lc-batch and have batch results to clean.")
        return 1

    # Find all batch result files
    batch_files = list(batch_dir.glob("batch_results_*.json"))

    if not batch_files:
        print(f"No batch result files found in {batch_dir}")
        return 1

    print(f"Found {len(batch_files)} batch files to process")

    success_count = 0
    total_cleaned = 0

    for file_path in sorted(batch_files):
        print(f"\nProcessing {file_path.name}...")
        if cleanup_batch_file(file_path):
            success_count += 1

    print(f"\nCleanup complete! Successfully processed {success_count}/{len(batch_files)} files.")

    return 0

if __name__ == "__main__":
    exit(main())