FROM python:3.11-slim

# System deps (lean but sufficient for faiss-cpu, transformers, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_ROOT=/app \
    LANG=C.UTF-8

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy source (exclude heavy dirs via .dockerignore)
COPY . .

# Pre-create common data dirs (also created at runtime if missing)
RUN mkdir -p data_raw data_processed storage output exports outlines data_jobs

# Default to the Typer CLI; override command for other scripts
ENTRYPOINT ["python", "src/cli/commands.py"]
CMD ["--help"]

