"""Find the chunks closest to a question: the retrieval half of RAG, shared by `ask` and the eval.

Three methods:
- vector: cosine distance between bge-m3 embeddings (D-015);
- lexical: Postgres full-text search, ranking chunks by the question's words they contain with
  BM25 (D-023) or Postgres's ts_rank (D-019);
- hybrid (the default): both lists merged with reciprocal rank fusion (D-020, D-022).

The lexical search can take the question translated into English instead (translate.py, D-021).
"""

from dataclasses import dataclass, replace

import psycopg
from pgvector import Vector

from sentinel1_rag.config import EMBED_MODEL
from sentinel1_rag.ollama import Ollama

# Hybrid search with the lexical half in English finds as many answers in its top 5 as vector
# search did in its top 10, with prompts about 40 % shorter (D-022; it was 10 in D-018).
TOP_K = 5

METHODS = ("vector", "lexical", "hybrid")
DEFAULT_METHOD = "hybrid"
# How the lexical search ranks the chunks that share words with the question.
RANKINGS = ("bm25", "ts_rank")
DEFAULT_RANKING = "bm25"

# BM25's parameters, at the defaults of Lucene and Elasticsearch: k1 sets how quickly further
# occurrences of a word stop adding to the score, and b how much longer chunks are discounted.
BM25_K1 = 1.2
BM25_B = 0.75

# BM25 in SQL (D-023). Postgres has no BM25 of its own, and its ts_rank ignores how common a word
# is. The corpus statistics are computed on every query, which reads every chunk's tsv: about
# 16 ms here, against 4 for ts_rank. A large corpus would keep them in a table, or use an index
# built for BM25.
BM25_QUERY = """
WITH terms AS (
    -- The question's words, normalized like the chunks (English stemming, no stop words).
    SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(question)s))
),
lengths AS (
    -- Each chunk's length in words, stop words left out, as the full-text index counts them.
    SELECT id, (SELECT sum(cardinality(positions)) FROM unnest(tsv)) AS len FROM chunks
),
corpus AS (
    SELECT count(*) AS n, avg(len) AS avg_len FROM lengths
),
matches AS (
    -- How often each question word occurs in each chunk that contains it.
    SELECT c.id, t.lexeme, cardinality(t.positions) AS tf
    FROM chunks c CROSS JOIN unnest(c.tsv) t JOIN terms USING (lexeme)
),
doc_freq AS (
    -- In how many chunks each question word occurs.
    SELECT lexeme, count(*) AS df FROM matches GROUP BY lexeme
),
scores AS (
    -- Sum over the question's words: rarity (IDF) x saturated, length-normalized frequency.
    SELECT m.id, sum(
        ln(1 + (corpus.n - doc_freq.df + 0.5) / (doc_freq.df + 0.5))
        * m.tf * (%(k1)s + 1) / (m.tf + %(k1)s * (1 - %(b)s + %(b)s * lengths.len / corpus.avg_len))
    ) AS score
    FROM matches m JOIN doc_freq USING (lexeme) JOIN lengths USING (id) CROSS JOIN corpus
    GROUP BY m.id
)
SELECT c.id, c.document_id, c.section, c.chunk_index, c.anchor, c.content, s.score
FROM scores s JOIN chunks c USING (id)
ORDER BY s.score DESC, c.section, c.chunk_index
LIMIT %(k)s
"""


def score_label(method: str, ranking: str = DEFAULT_RANKING) -> str:
    """What Hit.score means for a method."""
    return {
        "vector": "cosine distance: lower is closer",
        "lexical": f"{ranking}: higher is closer",
        "hybrid": "RRF score: higher is closer",
    }[method]

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
    score: float  # depends on the method: see score_label

    def tie_break(self) -> tuple[str, int]:
        """Orders chunks with equal scores. Chunk ids change whenever a page is indexed again,
        so ordering ties by id would make results depend on the database's history (D-020)."""
        return self.section, self.chunk_index


def retrieve(
    conn: psycopg.Connection,
    ollama: Ollama,
    question: str,
    k: int = TOP_K,
    method: str = DEFAULT_METHOD,
    english: str | None = None,
    ranking: str = DEFAULT_RANKING,
) -> list[Hit]:
    """The k chunks that best match the question, best first.

    `english`, the question translated into English, replaces it in the lexical search only,
    and `ranking` picks how that search ranks its chunks.
    """
    if method == "vector":
        if english is not None:
            raise ValueError("the English translation only applies to lexical and hybrid search")
        return vector_search(conn, ollama, question, k)
    if method == "lexical":
        return lexical_search(conn, english or question, k, ranking)
    if method == "hybrid":
        return hybrid_search(conn, ollama, question, k, english, ranking)
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


def lexical_search(conn: psycopg.Connection, question: str, k: int, ranking: str = DEFAULT_RANKING) -> list[Hit]:
    if ranking == "bm25":
        params = {"question": question, "k": k, "k1": BM25_K1, "b": BM25_B}
        return _fetch(conn, BM25_QUERY, params, best_first="DESC")
    if ranking != "ts_rank":
        raise ValueError(f"unknown ranking {ranking!r}; expected one of {RANKINGS}")
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


def hybrid_search(
    conn: psycopg.Connection,
    ollama: Ollama,
    question: str,
    k: int,
    english: str | None = None,
    ranking: str = DEFAULT_RANKING,
) -> list[Hit]:
    """Reciprocal rank fusion: each list adds 1 / (RRF_K + rank) to a chunk's score (D-020).

    Only ranks are used, so the two scores (a distance and ts_rank) never have to be comparable.
    The vector half always embeds the original question: translating it as well scored lower (D-021).
    """
    lists = [
        vector_search(conn, ollama, question, CANDIDATES),
        lexical_search(conn, english or question, CANDIDATES, ranking),
    ]
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
