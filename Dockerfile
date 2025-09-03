FROM python:3.11-slim AS base-sys

# System deps (lean but sufficient for faiss-cpu, transformers, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    jq \
    ca-certificates \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install sops from official release (apt package may not exist on slim images)
ARG SOPS_VERSION=3.8.1
RUN set -eux; \
  arch="$(dpkg --print-architecture)"; \
  case "$arch" in \
    amd64) SOPS_ARCH=amd64 ;; \
    arm64) SOPS_ARCH=arm64 ;; \
    *) echo "Unsupported arch: $arch"; exit 1 ;; \
  esac; \
  curl -fsSL -o /usr/local/bin/sops \
    "https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.${SOPS_ARCH}"; \
  chmod +x /usr/local/bin/sops; \
  sops --version

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_ROOT=/app \
    LANG=C.UTF-8

WORKDIR /app

# Common data dirs (also created at runtime if missing)
RUN mkdir -p data_raw data_processed storage output exports outlines data_jobs

# --- Python dependencies layer ---
FROM base-sys AS py-deps
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# --- Final runtime image ---
FROM py-deps AS runner
WORKDIR /app
COPY . .
COPY docker/entrypoint.sh docker/entrypoint.sh
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["--help"]
