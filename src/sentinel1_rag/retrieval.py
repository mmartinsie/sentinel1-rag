"""Find the chunks closest to a question: the retrieval half of RAG, shared by `ask` and the eval.

Three methods:
- vector (the default): cosine distance between bge-m3 embeddings (D-015);
- lexical: Postgres full-text search, ranking chunks by the question's words they contain (D-019);
- hybrid: both lists merged with reciprocal rank fusion (D-020).
"""

from dataclasses import dataclass, replace

import psycopg
from pgvector import Vector

from sentinel1_rag.config import EMBED_MODEL
from sentinel1_rag.ollama import Ollama

# Raised from 5 after the step-6 eval: the sections that explain GRD and SLC best ranked 9th
# and 11th for the project's example question (D-018).
TOP_K = 10

METHODS = ("vector", "lexical", "hybrid")
# What Hit.score means for each method.
SCORE_LABEL = {
    "vector": "cosine distance: lower is closer",
    "lexical": "ts_rank: higher is closer",
    "hybrid": "RRF score: higher is closer",
}

# Hybrid search fuses this many chunks from each list. An HNSW scan returns at most
# hnsw.ef_search (40) rows, so the vector list cannot be longer.
CANDIDATES = 40
# The constant of reciprocal rank fusion, as in the paper that introduced it (Cormack, Clarke
# and Büttcher, 2009), where 60 was "near-optimal, but the choice was not critical".
RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    section: str  # heading path, starting with the page title
    chunk_index: int  # position within the page
    url: str  # page URL, plus #anchor when the section has one
    content: str
    score: float  # depends on the method: see SCORE_LABEL

    def tie_break(self) -> tuple[str, int]:
        """Orders chunks with equal scores. Chunk ids change whenever a page is indexed again,
        so ordering ties by id would make results depend on the database's history (D-020)."""
        return self.section, self.chunk_index


def retrieve(
    conn: psycopg.Connection, ollama: Ollama, question: str, k: int = TOP_K, method: str = "vector"
) -> list[Hit]:
    """The k chunks that best match the question, best first."""
    if method == "vector":
        return vector_search(conn, ollama, question, k)
    if method == "lexical":
        return lexical_search(conn, question, k)
    if method == "hybrid":
        return hybrid_search(conn, ollama, question, k)
    raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")


def vector_search(conn: psycopg.Connection, ollama: Ollama, question: str, k: int) -> list[Hit]:
    # The question is embedded as it is: bge-m3 needs no query prefix (D-003).
    vector = Vector(ollama.embed(EMBED_MODEL, [question])[0])
    # The top k come from chunks alone, and documents is joined afterwards: with the join
    # inside, the planner never considers the HNSW index (checked with EXPLAIN, D-015).
    return _fetch(
        conn,
        "SELECT id, document_id, section, chunk_index, anchor, content, embedding <=> %(vector)s AS score"
        " FROM chunks ORDER BY score LIMIT %(k)s",
        {"vector": vector, "k": k},
        best_first="ASC",
    )


def lexical_search(conn: psycopg.Connection, question: str, k: int) -> list[Hit]:
    # plainto_tsquery normalizes the question like the chunks (English stemming, stop words
    # removed) but joins its words with AND, and almost no chunk contains every word of a
    # question. Turning the ANDs into ORs matches any of them; ts_rank then favours the
    # chunks where they occur most often (D-019). It often gives several chunks exactly the
    # same score, so ties are ordered by section and position, as in Hit.tie_break.
    return _fetch(
        conn,
        "SELECT id, document_id, section, chunk_index, anchor, content, ts_rank(tsv, q.query) AS score"
        " FROM chunks CROSS JOIN"
        "   (SELECT replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | ')::tsquery AS query) q"
        " WHERE tsv @@ q.query ORDER BY score DESC, section, chunk_index LIMIT %(k)s",
        {"question": question, "k": k},
        best_first="DESC",
    )


def hybrid_search(conn: psycopg.Connection, ollama: Ollama, question: str, k: int) -> list[Hit]:
    """Reciprocal rank fusion: each list adds 1 / (RRF_K + rank) to a chunk's score (D-020).

    Only ranks are used, so the two scores (a distance and ts_rank) never have to be comparable.
    """
    lists = [vector_search(conn, ollama, question, CANDIDATES), lexical_search(conn, question, CANDIDATES)]
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranked in lists:
        for rank, hit in zip(shared_ranks(ranked), ranked):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (RRF_K + rank)
            hits.setdefault(hit.chunk_id, hit)
    best = sorted(hits.values(), key=lambda hit: (-scores[hit.chunk_id], hit.tie_break()))[:k]
    return [replace(hit, score=scores[hit.chunk_id]) for hit in best]


def shared_ranks(hits: list[Hit]) -> list[int]:
    """Rank of each hit in a list sorted best first, where equal scores share the best rank
    (1, 2, 2, 4). Otherwise the arbitrary order inside a tie would change the fused scores."""
    ranks: list[int] = []
    for position, hit in enumerate(hits, start=1):
        tied = position > 1 and hit.score == hits[position - 2].score
        ranks.append(ranks[-1] if tied else position)
    return ranks


def _fetch(conn: psycopg.Connection, top_k_query: str, params: dict, best_first: str) -> list[Hit]:
    """Run a top-k query on chunks, then join documents for the citation URL."""
    rows = conn.execute(
        "SELECT c.id, c.section, c.chunk_index, d.url, c.anchor, c.content, c.score"
        f" FROM ({top_k_query}) c JOIN documents d ON d.id = c.document_id"
        f" ORDER BY c.score {best_first}, c.section, c.chunk_index",
        params,
    ).fetchall()
    return [
        Hit(chunk_id, section, chunk_index, f"{url}#{anchor}" if anchor else url, content, score)
        for chunk_id, section, chunk_index, url, anchor, content, score in rows
    ]
