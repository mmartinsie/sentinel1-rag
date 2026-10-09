"""Find the chunks closest to a question: the retrieval half of RAG, shared by `ask` and the eval."""

from dataclasses import dataclass

import psycopg
from pgvector import Vector

from sentinel1_rag.config import EMBED_MODEL
from sentinel1_rag.ollama import Ollama

TOP_K = 5


@dataclass
class Hit:
    section: str  # heading path, starting with the page title
    url: str  # page URL, plus #anchor when the section has one
    content: str
    distance: float  # cosine distance to the question: lower is closer


def retrieve(conn: psycopg.Connection, ollama: Ollama, question: str, k: int = TOP_K) -> list[Hit]:
    """The k chunks closest to the question, closest first."""
    # The question is embedded as it is: bge-m3 needs no query prefix (D-003).
    vector = Vector(ollama.embed(EMBED_MODEL, [question])[0])
    # The top k come from chunks alone, and documents is joined afterwards: with the join
    # inside, the planner never considers the HNSW index (checked with EXPLAIN, D-015).
    # An HNSW scan returns at most hnsw.ef_search (40) rows, so k must stay below that.
    rows = conn.execute(
        "SELECT c.section, d.url, c.anchor, c.content, c.distance"
        " FROM (SELECT document_id, section, anchor, content, embedding <=> %(vector)s AS distance"
        "       FROM chunks ORDER BY distance LIMIT %(k)s) c"
        " JOIN documents d ON d.id = c.document_id"
        " ORDER BY c.distance",
        {"vector": vector, "k": k},
    ).fetchall()
    return [
        Hit(section, f"{url}#{anchor}" if anchor else url, content, distance)
        for section, url, anchor, content, distance in rows
    ]
