#!/usr/bin/env bash
set -euo pipefail

# Optionally decrypt env.json into environment variables via SOPS
# Supports AWS KMS/GCP KMS/PGP recipients, depending on runtime credentials
if [ -f "/app/env.json" ] && command -v sops >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
  if sops -d --output-type json /app/env.json >/dev/null 2>&1; then
    # Export all key/values with safe shell quoting
    eval "$(sops -d --output-type json /app/env.json | jq -r 'to_entries[] | "export \(.key)=\(.value|@sh)"')"
    echo "[sops] Loaded environment from env.json"
  else
    echo "[sops] Warning: env.json present but decryption failed; continuing without it" >&2
  fi
fi

# Default behavior: if first arg looks like a CLI flag or known command, run the Typer CLI
if [ "$#" -eq 0 ]; then
  exec python src/cli/commands.py --help
fi

case "${1-}" in
  --*|-h|--help|ask|list-types)
    exec python -m src.cli.commands "$@"
    ;;
  shell)
    exec python -m src.cli.shell
    ;;
  python|bash|sh)
    exec "$@"
    ;;
  *)
    # If user passes a script path, run it directly; otherwise forward as-is
    exec "$@"
    ;;
esac
