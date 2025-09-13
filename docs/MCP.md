# MCP Tool & Prompt Framework

This guide explains how to add tools and prompts without code changes.
Dropping a `*.tool.yaml` file (and optional script) under `tools/` or a
prompt pack under `prompts/` makes it automatically discoverable.

## Running Servers

### HTTP server
```bash
uvicorn src.tool.mcp_app:app --host 127.0.0.1 --port 3333
```

### STDIO server
```bash
python -m src.tool.mcp_stdio
```

## Toolpack YAML Reference

| field | type | description |
| ----- | ---- | ----------- |
| `id` | string | Canonical tool identifier |
| `kind` | enum | `python`, `cli`, `node`, `php`, or `http` |
| `entry` | string &#124; list | Module path, CLI/Node argv, or HTTP URL |
| `php` | string &#124; list | PHP script path (for `php` kind) |
| `phpBinary` | string | PHP interpreter override |
| `schema` | object | `$ref` to input/output JSON Schemas |
| `timeoutMs` | int | Execution timeout in milliseconds |
| `limits` | object | `input` and `output` byte caps |
| `env` | list | Environment variables to pass through |
| `headers` | map | HTTP headers (templated) |
| `templating.cacheKey` | string | Jinja2 template overriding cache key |
| `deterministic` | bool | Enables idempotent caching |

### Env passthrough
Only variables listed under `env` are forwarded from the host environment.

### Templating
`entry`, `headers`, and `templating.cacheKey` support Jinja2 with the
request JSON available as `input`.

## Subprocess JSON Contract

For `python`, `cli`, `node`, and `php` kinds the tool receives the JSON
payload on **stdin** and must return JSON on **stdout**.

```json
// stdin
{"path": "/tmp"}

// stdout
{"data": {"ok": true}}
```

Errors should use canonical codes:

```json
{"error": {"code": "EINVAL", "message": "bad input"}}
```

## Prompt Registry

`prompts/REGISTRY.yaml` maps `<domain>.<name>` to allowed versions. Each
version has:

- Markdown body: `prompts/packs/<domain>/<name>.v<major>.md`
- JSON Schema spec: `prompts/packs/<domain>/<name>.spec.yaml`

Increment the version when inputs or behaviour change; older versions
remain available for reproducibility.

## Determinism & Limits

- `deterministic: true` enables cache reuse when inputs and cache key match.
- `timeoutMs` terminates long‑running tools.
- `limits.input`/`limits.output` cap payload sizes. Exceeding them should
  return `ECAP`.
- Canonical error codes: `EINVAL`, `ETIMEOUT`, `ECAP`, `EINTERNAL`.

## Examples

### Python tool
```yaml
id: markdown
kind: python
entry: tools.example.markdown:run
deterministic: true
timeoutMs: 1000
schema:
  input:
    $ref: ./markdown.input.schema.json
  output:
    $ref: ./markdown.output.schema.json
```

### HTTP tool
```yaml
id: http_echo
kind: http
entry: https://example.com/{{input.path}}
headers:
  X-Token: "{{input.token}}"
deterministic: true
templating:
  cacheKey: "{{input.path}}"
schema:
  input:
    $ref: ./http_echo.input.schema.json
  output:
    $ref: ./http_echo.output.schema.json
```

### Node tool
```yaml
id: node_echo
kind: node
entry:
  - tools/example/node_echo.mjs
  - "{{input.prefix}}"
deterministic: true
schema:
  input:
    $ref: ./node_echo.input.schema.json
  output:
    $ref: ./node_echo.output.schema.json
```

### Prompt pack (`writing/sectioned_draft@3`)
`prompts/REGISTRY.yaml`
```yaml
writing:
  sectioned_draft:
    - 3
```
`prompts/packs/writing/sectioned_draft.v3.md`
```markdown
# Sectioned Draft v3

Write a sectioned draft about the provided topic.
```

