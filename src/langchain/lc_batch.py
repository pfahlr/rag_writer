#!/usr/bin/env python3
"""
LangChain batch processor for multiple lc-ask calls

Reads a JSON array of objects with 'task', 'instruction', and 'section' fields,
calls lc-ask for each item, and writes results to a timestamped JSON file.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# --- ROOT relative to repo ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = Path(root_dir)

def run_lc_ask(task: str, instruction: str, key: str = "default"):
    """Run lc-ask with given parameters and return parsed JSON result."""
    cmd = [
        sys.executable, str(ROOT / "src/langchain/lc_ask.py"),
        instruction,
        "--task", task,
        "--key", key
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running lc-ask: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return {"error": str(e), "generated_content": "", "sources": []}
    except json.JSONDecodeError as e:
        print(f"Error parsing lc-ask output: {e}", file=sys.stderr)
        return {"error": f"JSON decode error: {e}", "generated_content": "", "sources": []}

def main():
    if len(sys.argv) > 1:
        # Read from file
        input_file = sys.argv[1]
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print("Error: Input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    # Get optional key parameter
    key = sys.argv[2] if len(sys.argv) > 2 else "default"

    results = []
    for item in data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-object item: {item}", file=sys.stderr)
            continue

        task = item.get('task', '')
        instruction = item.get('instruction', '')
        section = item.get('section', '')

        if not instruction:
            print(f"Warning: Skipping item without instruction: {item}", file=sys.stderr)
            continue

        print(f"Processing section: {section}", file=sys.stderr)

        # Run lc-ask
        result = run_lc_ask(task, instruction, key)

        # Add section identifier to result
        result['section'] = section
        result['task'] = task
        result['instruction'] = instruction

        results.append(result)

    # Write results to timestamped file in output directory
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    output_file = output_dir / f"batch_results_{timestamp}.json"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to: {output_file}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()