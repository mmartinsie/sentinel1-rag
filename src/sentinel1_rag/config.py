"""Paths, services, models, project identity and the list of source pages."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
SOURCES_FILE = ROOT / "sources.yaml"

USER_AGENT = "sentinel1-rag/0.1 (+https://github.com/mmartinsie/sentinel1-rag; learning project)"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Matches the defaults in docker-compose.yml; set it if you override them in .env.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

EMBED_MODEL = "bge-m3"


def load_sources(path: Path = SOURCES_FILE) -> list[str]:
    """Return the page URLs listed in sources.yaml."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["pages"]
