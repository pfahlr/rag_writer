# RAG Verification Harness

`run_rag_verification.py` drives the end-to-end verification loop by executing
the existing CLI scripts (`lc_build_index.py`, `lc_ask.py`, optional
`multi_agent.py`) against a gold corpus and a set of reference questions. The
script never imports those tools directly; instead it shells out to the CLIs so
that the exact same entry points used in production are exercised.

## Questions file schema

`--questions` must point to a YAML file structured either as a top-level list or
as a mapping with a `questions:` array. Each entry is validated and normalised
into the following fields:

| Field         | Type              | Required | Notes |
| ------------- | ----------------- | -------- | ----- |
| `id`          | `str`             | ✅       | Unique question identifier. |
| `type`        | `str`             | ✅       | One of `direct`, `ambiguous`, `synthesis`, `conflict`, `freshness`, `multiturn`. |
| `question`    | `str`             | ✅       | Natural language prompt passed to the QA script. |
| `gold_doc`    | `str`             | ➖ (see notes) | Exactly one of `gold_doc` **or** `gold_docs` must be supplied. |
| `gold_docs`   | `list[str]`       | ➖ (see notes) | Hints for the expected supporting documents. |
| `answer`      | `str`             | Optional | Gold answer for scoring. `--require-answers` enforces presence. |
| `clarify`     | `str`             | Optional | Clarification prompt forwarded to multi-turn scripts. |
| `followups`   | `list[str]`       | Optional | Follow-up prompts for multi-turn flows. |
| `note`        | `str`             | Optional | Free-form metadata (not used in scoring). |

If `answer` is omitted the question is still executed, but it is marked as
`pass: null` in the JSONL output and excluded from pass/fail tallies.

## Example invocation

```bash
python run_rag_verification.py \
  --corpus path/to/corpus_dir \
  --questions eval/questions.yaml \
  --workdir .rag_tmp \
  --jsonl-out logs/verification_results.jsonl \
  --save-outputs logs/verification/answers \
  --require-answers
```

Key directories are created inside `--workdir` unless overridden:

* `index_dir/` – destination passed to `lc_build_index.py`
* `logs/` – stdout/stderr captures for the builder and per-question runs
* `outputs/` – model answers per question (overridden by `--save-outputs`)

## Document hints and retriever configuration

When the asker CLI advertises a document hint flag (detected from `--help`), the
verification harness forwards the comma-joined list of `gold_doc(s)` via that
flag. This enables askers that support restricted retrieval to focus on the
expected sources while still falling back gracefully for CLIs that lack such an
option.

`--topk` is similarly forwarded when the asker exposes a depth flag (e.g.
`--topk` or `--k`). If you use a custom asker/builder pair with different flag
names you can override them with the environment variables
`RAG_VERIFICATION_BUILDER_CORPUS_FLAG` and
`RAG_VERIFICATION_BUILDER_OUT_FLAG`.

## Outputs

Each question yields a JSON object written to `--jsonl-out` with the keys:

* `qid`, `type`, `question`
* `gold_answer`, `model_answer`
* `pass` (boolean or `null`) and `score` (`float` or `null`)
* `route` – `"asker"` or `"multi"` indicating which CLI handled the query
* Optional `note` (e.g., `"no gold answer"` or `"command exit code ..."`)

A summary table is printed at the end and the process exits non-zero when any
graded item fails unless `--no-fail-on-error` is supplied.
