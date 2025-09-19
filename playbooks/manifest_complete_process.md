# Codex Directives — `manifest_complete_process.py`

## Objectives

1. Add a **`retry`** subcommand that parses an **aria2c** summary log and regenerates an aria2 batch file containing only the failed items.
2. Harden Crossref lookups with **polite User-Agent, longer timeout, and exponential backoff**.
3. Fix metadata writing to always pass a **`Path`** to `pdf_io.write_pdf_metadata`.
4. Keep everything **idempotent** and backwards-compatible with your current manifest format.

---

## Files to Edit

* `research/manifest_complete_process.py` (the script pasted above)

---

## New/Changed CLI

* Add subcommand:

  ```
  retry --manifest <path> --aria2-log <path> --script-out <path> --downloads-dir <dir> [--no-manifest-update]
  ```
* Add optional flags to tune Crossref:

  ```
  --crossref-timeout <float> (default 25.0)
  --crossref-tries <int>     (default 4)
  --crossref-backoff <float> (default 1.5)
  --crossref-ua <str>        (default "Content-Expanse/1.0 (mailto:pfahlr@gmail.com)")
  ```

  (These can be accepted by `process` only. Keep preprocess unchanged.)

---

## Tasks (exact steps)

### A) Add `retry` subcommand

1. **Helpers (near other utils):**

   * `parse_aria2_summary_lines(path: Path) -> List[Tuple[str, str]]`

     * Parse lines like:
       `ff786a|ERR |       0B/s|../research/out/data/pdfs_inbox/DL-00142-e68f4f.pdf`
     * Return list of `(status_upper, dest_basename)`.
   * `build_retry_batch_from_manifest(entries: List[Dict], failed_basenames: set, downloads_dir: Path) -> Tuple[List[str], int, List[str]]`

     * Map `entry['temp_filename']` → entry.
     * For each basename in `failed_basenames`, if found and `pdf_url` exists, append aria2c stanza:

       ```
       <URL>
         out=<basename>
         dir=<downloads_dir>
       ```

       * Increment `matched` count.
       * Optionally set `entry['download_status']="retry"` and clear `failure_reason`.
     * Return `(lines, matched_count, missing_basenames)`.

2. **Command function:**

   * `retry_from_aria2_log(manifest_path: Path, aria2_log: Path, script_out: Path, downloads_dir: Path, update_manifest: bool=True)`

     * Load manifest with your existing `load_manifest`.
     * Call the two helpers.
     * Write the retry batch to `script_out`.
     * If `update_manifest`, persist with `save_manifest`.
     * Print a summary: total failed, matched, unmatched.

3. **Wire into CLI (`main()`):**

   * Add parser:

     ```python
     sr = sub.add_parser("retry", help="Parse aria2 log; regenerate aria2 batch for failed items.")
     sr.add_argument("--manifest", required=True, type=Path)
     sr.add_argument("--aria2-log", required=True, type=Path)
     sr.add_argument("--script-out", required=True, type=Path)
     sr.add_argument("--downloads-dir", required=True, type=Path)
     sr.add_argument("--no-manifest-update", action="store_true")
     ```
   * Dispatch:

     ```python
     elif args.cmd == "retry":
         retry_from_aria2_log(
             args.manifest,
             args.aria2_log,
             args.script_out,
             args.downloads_dir,
             update_manifest=not args.no_manifest_update,
         )
     ```

**Acceptance criteria**

* Running:

  ```
  aria2c -i data/downloads.aria2c.txt | tee data/aria2.summary.log
  python manifest_complete_process.py retry \
    --manifest data/manifest.json \
    --aria2-log data/aria2.summary.log \
    --script-out data/retry.aria2c.txt \
    --downloads-dir data/pdfs_inbox
  ```

  produces a valid `data/retry.aria2c.txt` with only failed `ERR` items and correct `out=` basenames pulled from `temp_filename`.

---

### B) Crossref hardening (timeouts, backoff, UA)

1. **Add CLI options to `process` parser:**

   ```python
   sp2.add_argument("--crossref-timeout", type=float, default=25.0)
   sp2.add_argument("--crossref-tries", type=int, default=4)
   sp2.add_argument("--crossref-backoff", type=float, default=1.5)
   sp2.add_argument("--crossref-ua", type=str, default="Content-Expanse/1.0 (mailto:pfahlr@gmail.com)")
   ```
2. **Create wrapper near top-level utils:**

   ```python
   def safe_enrich_crossref(enrich_fn, doi=None, title=None, author=None, timeout=25.0, tries=4, backoff=1.5, ua="Content-Expanse/1.0 (mailto:pfahlr@gmail.com)"):
       last = None
       for i in range(tries):
           try:
               return enrich_fn(doi=doi, title=title, author=author,
                                headers={"User-Agent": ua}, timeout=timeout)
           except Exception as e:
               last = e
               time.sleep(backoff * (2 ** i))
       print(f"[WARN] Crossref enrichment failed after {tries} attempts: {last}")
       return {}
   ```
3. **Use wrapper in `process_pipeline`** (replace direct call):

   ```python
   cr = safe_enrich_crossref(
       enrich_via_crossref,
       doi=meta.get("doi"),
       title=meta.get("title"),
       author=(meta.get("authors")[0] if isinstance(meta.get("authors"), list) and meta.get("authors") else ""),
       timeout=args.crossref_timeout,
       tries=args.crossref_tries,
       backoff=args.crossref_backoff,
       ua=args.crossref_ua,
   )
   if cr:
       for k, v in cr.items():
           if v and not meta.get(k):
               meta[k] = v
   ```

**Acceptance criteria**

* Under load or transient network slowness, Crossref calls no longer hard-fail immediately; logs show retries and eventual success/final warning.

---

### C) Metadata write bug (Path vs str)

* In `write_pdf_metadata`, ensure `pdf_path` is **always a `Path`** before passing into `pdfio`:

  ```python
  def write_pdf_metadata(pdf_path: Path, meta: Dict) -> None:
      if not isinstance(pdf_path, Path):
          pdf_path = Path(pdf_path)
      ...
      if HAVE_PDFIO and hasattr(pdfio, "write_pdf_metadata"):
          try:
              pdfio.write_pdf_metadata(pdf_path, core=core, dc=dc, prism=prism, dcterms=dcterms, full_meta_for_subject=meta)
              return
          except Exception as e:
              print(f"[WARN] pdf_io.write_pdf_metadata failed, falling back to pypdf: {e}")
      ...
  ```

**Acceptance criteria**

* No occurrence of `'str' object has no attribute 'with_suffix'` during runs.

---

## Edge Cases / Policies

* `retry` must **only** include items with statuses in `{ERR, ERROR, NG, FAILED}`; treat `OK` as success.
* If an aria2 log entry’s basename **isn’t** in the manifest (`temp_filename` mismatch), list it under “unmatched” in stdout; do not crash.
* `retry` must **not** duplicate items already present in the generated file for this run (de-dup by basename).
* Manifest updates by `retry` (setting `download_status="retry"`) are **optional** via `--no-manifest-update`. Default: update.
* `process` should **skip** entries whose `temp_filename` file is missing in `--inbox` (leave `download_status` as is).
* Maintain existing behavior for JDownloader in `preprocess` (no changes required).

---

## Logging & Observability

* On `retry` completion, print:

  * total failed found in log
  * matched count
  * unmatched basenames (one per line)
  * path to `script_out`
* On Crossref retry, log only one line per failure attempt set:

  * final `[WARN] Crossref enrichment failed after X attempts: <err>`

---

## Quick Manual Test Plan

1. Run `preprocess` to generate an aria2c list; download a subset; mock a log with a couple `ERR` lines.
2. Run `retry` and confirm the new batch contains exactly the `ERR` basenames and correct `out=` names.
3. Drop corresponding PDFs (named as `temp_filename`) into `--inbox`; run `process` with small Crossref timeouts (e.g., `--crossref-timeout 1.0`) to observe backoff behavior and ensure it doesn’t crash.
4. Verify metadata sidecars are written and final filenames follow `[title-slug]-[year].pdf`.

---

## Snippets Codex Can Paste (minimal)

**Add parser + dispatch (in `main`)**

```python
sr = sub.add_parser("retry", help="Parse aria2 log; regenerate aria2 batch for failed items.")
sr.add_argument("--manifest", required=True, type=Path)
sr.add_argument("--aria2-log", required=True, type=Path)
sr.add_argument("--script-out", required=True, type=Path)
sr.add_argument("--downloads-dir", required=True, type=Path)
sr.add_argument("--no-manifest-update", action="store_true")
```

```python
elif args.cmd == "retry":
    retry_from_aria2_log(
        args.manifest, args.aria2_log, args.script_out, args.downloads_dir,
        update_manifest=not args.no_manifest_update
    )
```

**Add Crossref tuning flags to `process` parser**

```python
sp2.add_argument("--crossref-timeout", type=float, default=25.0)
sp2.add_argument("--crossref-tries", type=int, default=4)
sp2.add_argument("--crossref-backoff", type=float, default=1.5)
sp2.add_argument("--crossref-ua", type=str, default="Content-Expanse/1.0 (mailto:pfahlr@gmail.com)")
```

**Safe Crossref wrapper**

```python
def safe_enrich_crossref(enrich_fn, doi=None, title=None, author=None, timeout=25.0, tries=4, backoff=1.5, ua="Content-Expanse/1.0 (mailto:pfahlr@gmail.com)"):
    last = None
    for i in range(tries):
        try:
            return enrich_fn(doi=doi, title=title, author=author, headers={"User-Agent": ua}, timeout=timeout)
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    print(f"[WARN] Crossref enrichment failed after {tries} attempts: {last}")
    return {}
```

**aria2 log parser**

```python
def parse_aria2_summary_lines(path: Path):
    results = []
    for raw in path.read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 4: continue
        status, target = parts[1], parts[3]
        if not target: continue
        results.append((status.upper(), Path(target).name))
    return results
```

**retry builder**

```python
def build_retry_batch_from_manifest(entries, failed_basenames: set, downloads_dir: Path):
    by_temp = { (e.get("temp_filename") or "").strip(): e for e in entries if e.get("temp_filename") }
    lines, matched, missing = [], 0, []
    for base in sorted(failed_basenames):
        e = by_temp.get(base)
        if e and (e.get("pdf_url") or "").strip():
            url = e["pdf_url"].strip()
            lines += [url, f"  out={base}", f"  dir={str(downloads_dir)}"]
            matched += 1
            e["download_status"] = "retry"
            e.pop("failure_reason", None)
        else:
            missing.append(base)
    return lines, matched, missing
```

**retry command**

```python
def retry_from_aria2_log(manifest_path: Path, aria2_log: Path, script_out: Path, downloads_dir: Path, update_manifest: bool = True):
    entries, wrapped = load_manifest(manifest_path)
    rows = parse_aria2_summary_lines(aria2_log)
    FAIL = {"ERR", "ERROR", "NG", "FAILED"}
    failed_basenames = {base for status, base in rows if status in FAIL}
    if not failed_basenames:
        print("[retry] No failures found."); return
    lines, matched, missing = build_retry_batch_from_manifest(entries, failed_basenames, downloads_dir)
    if not lines:
        print("[retry] No failed items matched manifest."); 
        if missing: print("[retry] Unmatched:", *missing, sep="\n  - ")
        return
    script_out.parent.mkdir(parents=True, exist_ok=True)
    script_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if update_manifest: save_manifest(manifest_path, entries, wrapped)
    print(f"[retry] Wrote: {script_out}")
    print(f"[retry] Matched: {matched} | Unmatched: {len(missing)}")
```

**Path guard in metadata**

```python
def write_pdf_metadata(pdf_path: Path, meta: Dict) -> None:
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)
    ...
```
