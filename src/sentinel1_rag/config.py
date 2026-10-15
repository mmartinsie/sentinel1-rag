"""Paths, services, models, project identity and the list of source pages."""

import os
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
SOURCES_FILE = ROOT / "sources.yaml"
GEOCODE_CACHE = DATA_DIR / "geocode.json"
QUESTIONS_FILE = ROOT / "eval" / "questions.yaml"
AGENT_QUESTIONS_FILE = ROOT / "eval" / "agent_questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"

USER_AGENT = f"sentinel1-rag/{version('sentinel1-rag')} (+https://github.com/mmartinsie/sentinel1-rag; learning project)"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Matches the defaults in docker-compose.yml; set it if you override them in .env.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
# The Copernicus Data Space Ecosystem's STAC catalogue, searched by the agent (D-027).
STAC_URL = os.environ.get("STAC_URL", "https://stac.dataspace.copernicus.eu/v1")
# OpenStreetMap's geocoder. Its usage policy asks that the server can be switched without
# changing the code (D-028).
NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
# OpenSearch, the second search backend compared with pgvector (D-034), and its index.
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = "sentinel1-chunks"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "gemma4:e4b-it-qat"
# Reorders the top 20 of the search (D-037). A GGUF from Hugging Face, pulled by Ollama itself.
RERANK_MODEL = "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0"
# Context size sent with every request to the chat model (D-002). Ollama's default on this GPU is
# 4096, and a request with a different size than the loaded model makes Ollama load it again
# (about 60 s here), so answering and translating must use the same value.
CHAT_NUM_CTX = 8192


def load_sources(path: Path = SOURCES_FILE) -> list[str]:
    """Return the page URLs listed in sources.yaml."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["pages"]


def load_pdfs(path: Path = SOURCES_FILE) -> list[dict[str, str]]:
    """Return the PDFs listed in sources.yaml, each with its `url` and `title`."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("pdfs", [])
