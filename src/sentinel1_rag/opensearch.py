"""OpenSearch with its k-NN plugin as a second search backend, to compare with pgvector (D-034).

`make opensearch-index` copies every chunk from Postgres into one OpenSearch index: the same
text, the same section and URL, and the same bge-m3 vector, so the two backends search identical
data and any difference in the results comes from the engines:

- vector search: an HNSW graph built by Faiss (OpenSearch's default k-NN engine), with pgvector's
  graph parameters, against pgvector's exact scan (D-035);
- full-text search: OpenSearch's own BM25 on a field analysed in English with Postgres's stop
  words, against the BM25 written in SQL over Postgres's English tsvector (D-023, D-036);
- hybrid search: either the same reciprocal rank fusion in Python over OpenSearch's two lists, or
  OpenSearch's own hybrid query fused by a search pipeline (method `hybrid-native`).

Raw REST with httpx, as for Ollama (D-007): every request is visible here.
"""

import json
import time
from typing import Any

import httpx
import psycopg

from sentinel1_rag.config import OPENSEARCH_INDEX, OPENSEARCH_URL

# pgvector's HNSW defaults (m 16, ef_construction 64, hnsw.ef_search 40), so that both graphs
# are built and searched alike. OpenSearch's own default ef_search is 100 (D-035).
HNSW = {"m": 16, "ef_construction": 64, "ef_search": 40}
# The search pipeline that fuses the hybrid query's two lists by reciprocal rank fusion, with
# the same constant as retrieval.RRF_K.
RRF_PIPELINE = "sentinel1-rrf"
BULK_BATCH = 100

# The 127 English stop words of Postgres's english_stem dictionary (tsearch_data/english.stop, the
# Snowball list). OpenSearch's built-in `english` analyser drops only Lucene's 33, so question
# words such as "what", "how", "does" or "which" became search terms, and BM25 found 19 of 25
# answers in its top 5 instead of 21 (D-036).
POSTGRES_STOP_WORDS = """
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she
her hers herself it its itself they them their theirs themselves what which who whom this that
these those am is are was were be been being have has had having do does did doing a an the and
but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here
there when where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now
""".split()

# OpenSearch's `english` analyser (standard tokenizer, possessives, lowercase, stop words, Porter
# stemming), rebuilt as a custom analyser to swap in Postgres's stop words.
ANALYSIS: dict[str, Any] = {
    "filter": {
        "english_possessive": {"type": "stemmer", "language": "possessive_english"},
        "postgres_stop": {"type": "stop", "stopwords": POSTGRES_STOP_WORDS},
    },
    "analyzer": {
        "english_postgres_stop": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["english_possessive", "lowercase", "postgres_stop", "porter_stem"],
        }
    },
}

MAPPINGS: dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        "chunk_id": {"type": "integer"},  # chunks.id in Postgres, so a hit maps back to its row
        "section": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "url": {"type": "keyword", "index": False},
        "content": {"type": "text", "index": False},
        # The words of the same text that is embedded, as Postgres's tsv column (D-019), for BM25.
        "text": {"type": "text", "analyzer": "english_postgres_stop"},
        "embedding": {
            "type": "knn_vector",
            "dimension": 1024,
            "space_type": "cosinesimil",
            "method": {
                "name": "hnsw",
                "engine": "faiss",
                "parameters": {"m": HNSW["m"], "ef_construction": HNSW["ef_construction"]},
            },
        },
    },
}


class OpenSearchError(RuntimeError):
    pass


class OpenSearchIndex:
    def __init__(self, base_url: str = OPENSEARCH_URL, index: str = OPENSEARCH_INDEX) -> None:
        self._base_url = base_url
        self.index = index
        self._http = httpx.Client(base_url=base_url, timeout=120)

    def request(self, method: str, path: str, body: Any = None, **params: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, json=body, params=params or None)
        except httpx.ConnectError as exc:
            raise OpenSearchError(
                f"cannot reach OpenSearch at {self._base_url}; is it running? (make opensearch-up)"
            ) from exc
        if response.is_error:
            raise OpenSearchError(f"{method} {path} returned {response.status_code}: {response.text[:500]}")
        return response.json()

    # --- Building the index ---

    def rebuild(self, conn: psycopg.Connection) -> dict[str, float]:
        """Delete the index and copy every chunk from Postgres into a new one. With 568 chunks
        this takes seconds, so the index is always rebuilt whole instead of updated."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        rows = conn.execute(
            "SELECT c.id, c.section, c.chunk_index, d.url || coalesce('#' || c.anchor, ''), c.content, c.embedding"
            " FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.id"
        ).fetchall()
        if not rows:
            raise OpenSearchError("the Postgres index is empty; run `make index` first")
        timings["read_postgres"] = time.perf_counter() - started

        if self._http.head(f"/{self.index}").status_code == 200:
            self.request("DELETE", f"/{self.index}")
        settings = {
            "index.knn": True,
            "index.knn.algo_param.ef_search": HNSW["ef_search"],
            "number_of_shards": 1,  # one shard: BM25 statistics over the whole corpus, as in Postgres
            "number_of_replicas": 0,  # a single node has nowhere to put a replica
            "refresh_interval": "-1",  # no refresh while loading
            "analysis": ANALYSIS,
        }
        mappings = MAPPINGS | {"_meta": {"postgres": fingerprint(conn)}}
        self.request("PUT", f"/{self.index}", {"settings": settings, "mappings": mappings})

        started = time.perf_counter()
        for i in range(0, len(rows), BULK_BATCH):
            lines: list[dict[str, Any]] = []
            for chunk_id, section, chunk_index, url, content, embedding in rows[i : i + BULK_BATCH]:
                lines.append({"index": {"_index": self.index, "_id": str(chunk_id)}})
                lines.append(
                    {
                        "chunk_id": chunk_id,
                        "section": section,
                        "chunk_index": chunk_index,
                        "url": url,
                        "content": content,
                        # The same text as the embedding and as Postgres's tsv (index.EMBED_TEMPLATE).
                        "text": f"{section}\n\n{content}",
                        "embedding": embedding.to_list(),
                    }
                )
            self._bulk(lines)
        timings["bulk"] = time.perf_counter() - started

        # One segment, so one HNSW graph and stable results from one rebuild to the next.
        started = time.perf_counter()
        self.request("POST", f"/{self.index}/_forcemerge", max_num_segments=1)
        self.request("PUT", f"/{self.index}/_settings", {"index": {"refresh_interval": "1s"}})
        self.request("POST", f"/{self.index}/_refresh")
        timings["merge_refresh"] = time.perf_counter() - started
        timings["documents"] = len(rows)
        self.ensure_rrf_pipeline()
        return timings

    def _bulk(self, lines: list[dict[str, Any]]) -> None:
        payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
        try:
            response = self._http.post("/_bulk", content=payload, headers={"Content-Type": "application/x-ndjson"})
        except httpx.ConnectError as exc:
            raise OpenSearchError(f"cannot reach OpenSearch at {self._base_url}") from exc
        data = response.json()
        if response.is_error or data.get("errors"):
            failed = next((item for item in data.get("items", []) if "error" in item.get("index", {})), data)
            raise OpenSearchError(f"bulk indexing failed: {str(failed)[:500]}")

    def ensure_rrf_pipeline(self) -> None:
        """A search pipeline that fuses the lists of a hybrid query by reciprocal rank fusion."""
        from sentinel1_rag.retrieval import RRF_K

        self.request(
            "PUT",
            f"/_search/pipeline/{RRF_PIPELINE}",
            {
                "description": "Reciprocal rank fusion for sentinel1-rag's hybrid search",
                "phase_results_processors": [
                    {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": RRF_K}}}
                ],
            },
        )

    def check_in_sync(self, conn: psycopg.Connection) -> None:
        """Stop if the OpenSearch copy is not of the chunks now in Postgres: re-indexing a page in
        Postgres gives its chunks new ids, and then hits would point at rows that no longer exist."""
        try:
            exists = self._http.head(f"/{self.index}").status_code == 200
        except httpx.ConnectError as exc:
            raise OpenSearchError(
                f"cannot reach OpenSearch at {self._base_url}; is it running? (make opensearch-up)"
            ) from exc
        if not exists:
            raise OpenSearchError(f"OpenSearch has no index {self.index!r}; build it with `make opensearch-index`")
        meta = self.request("GET", f"/{self.index}/_mapping")[self.index]["mappings"].get("_meta", {})
        if meta.get("postgres") != fingerprint(conn):
            raise OpenSearchError("the OpenSearch index is out of date with Postgres; run `make opensearch-index`")

    # --- Searching ---

    def search(self, body: dict[str, Any], **params: Any) -> list[dict[str, Any]]:
        body = {"_source": {"excludes": ["embedding", "text"]}} | body
        return self.request("POST", f"/{self.index}/_search", body, **params)["hits"]["hits"]

    def vector_search(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        """The k nearest chunks in the HNSW graph, by cosine similarity."""
        return self.search(
            {
                "size": k,
                "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
                # Equal scores are ordered as in Postgres (retrieval.Hit.tie_break), not by Lucene's doc id.
                "sort": ["_score", {"section": "asc"}, {"chunk_index": "asc"}],
                "track_scores": True,
            }
        )

    def exact_vector_search(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        """Brute-force cosine similarity over every chunk, to measure what HNSW misses."""
        return self.search(
            {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {"field": "embedding", "query_value": vector, "space_type": "cosinesimil"},
                        },
                    }
                },
                "sort": ["_score", {"section": "asc"}, {"chunk_index": "asc"}],
                "track_scores": True,
            }
        )

    def lexical_search(self, text: str, k: int) -> list[dict[str, Any]]:
        """BM25 over the English-analysed text. A match query joins the words with OR, as the
        lexical search in Postgres does (D-019)."""
        return self.search(
            {
                "size": k,
                "query": {"match": {"text": text}},
                "sort": ["_score", {"section": "asc"}, {"chunk_index": "asc"}],
                "track_scores": True,
            }
        )

    def hybrid_search(self, vector: list[float], text: str, k: int, candidates: int) -> list[dict[str, Any]]:
        """OpenSearch's own hybrid query: a BM25 list and a k-NN list, fused by the RRF pipeline."""
        return self.search(
            {
                "size": candidates,
                "query": {
                    "hybrid": {
                        "queries": [
                            {"match": {"text": text}},
                            {"knn": {"embedding": {"vector": vector, "k": candidates}}},
                        ]
                    }
                },
            },
            search_pipeline=RRF_PIPELINE,
        )[:k]


def fingerprint(conn: psycopg.Connection) -> str:
    """Which chunks Postgres holds: their number and a hash of their ids, which change whenever
    a page is indexed again (index.py deletes and re-inserts its rows)."""
    count, digest = conn.execute("SELECT count(*), md5(string_agg(id::text, ',' ORDER BY id)) FROM chunks").fetchone()
    return f"{count}:{digest}"


def main() -> None:
    """`make opensearch-index`: copy the chunks from Postgres into a new OpenSearch index."""
    from sentinel1_rag.db import connect

    index = OpenSearchIndex()
    with connect() as conn:
        timings = index.rebuild(conn)
    stats = index.request("GET", f"/{index.index}/_stats/docs,store,segments")["indices"][index.index]["primaries"]
    print(
        f"Copied {timings['documents']:.0f} chunks from Postgres into OpenSearch index {index.index!r}:"
        f" read {timings['read_postgres']:.1f}s, bulk {timings['bulk']:.1f}s,"
        f" merge and refresh {timings['merge_refresh']:.1f}s"
    )
    print(
        f"{stats['docs']['count']} documents, {stats['segments']['count']} segment,"
        f" {stats['store']['size_in_bytes'] / 2**20:.1f} MiB on disk"
    )


if __name__ == "__main__":
    main()
