"""`make index`: embed the chunks in data/chunks/ and store them in Postgres.

Every page is written in its own transaction: its previous rows are deleted and the
new chunks inserted with their embeddings, so a search never sees a half-indexed page.
A page is skipped when both its text (content_hash) and its index_config are unchanged,
so running `make index` twice does nothing the second time.
"""

import json
import time
from datetime import datetime

from pgvector import Vector

from sentinel1_rag.config import CHUNKS_DIR, EMBED_MODEL
from sentinel1_rag.db import connect
from sentinel1_rag.ollama import Ollama

# Texts per /api/embed request. Going from 1 to 8 saved ~25% of the time; larger
# batches were not faster on this setup (see docs/DECISIONS.md, D-014).
BATCH_SIZE = 32
# The heading path goes in front of the chunk text, so that the vector knows its context.
EMBED_TEMPLATE = "{section}\n\n{content}"


def embedding_input(chunk: dict) -> str:
    return EMBED_TEMPLATE.format(section=chunk["section"], content=chunk["content"])


def index_config(page: dict, model_digest: str) -> str:
    """Everything apart from the page text that shapes the stored vectors."""
    return json.dumps(
        {
            "chunking": page["chunking"],
            "embed_model": EMBED_MODEL,
            "embed_model_digest": model_digest,
            "embed_template": EMBED_TEMPLATE,
        },
        sort_keys=True,
    )


def main() -> None:
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CHUNKS_DIR.glob("*.json"))]
    if not pages:
        raise SystemExit("No chunks in data/chunks/. Run `make ingest` first.")

    ollama = Ollama()
    digest = ollama.digest(EMBED_MODEL)
    started = time.perf_counter()
    embed_seconds = 0.0
    indexed: list[str] = []
    skipped = 0
    embedded = 0

    with connect() as conn:
        stored = {
            url: (content_hash, config)
            for url, content_hash, config in conn.execute(
                "SELECT url, content_hash, index_config FROM documents"
            )
        }
        for page in pages:
            config = index_config(page, digest)
            if stored.get(page["url"]) == (page["content_hash"], config):
                skipped += 1
                continue

            # Embed before opening the transaction: no transaction waits on the network.
            t0 = time.perf_counter()
            texts = [embedding_input(c) for c in page["chunks"]]
            vectors = [
                vector
                for i in range(0, len(texts), BATCH_SIZE)
                for vector in ollama.embed(EMBED_MODEL, texts[i : i + BATCH_SIZE])
            ]
            embed_seconds += time.perf_counter() - t0

            with conn.transaction():
                # Deleting the document also deletes its chunks (ON DELETE CASCADE).
                conn.execute("DELETE FROM documents WHERE url = %s", (page["url"],))
                doc_id = conn.execute(
                    "INSERT INTO documents (url, title, content_hash, index_config, fetched_at)"
                    " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (page["url"], page["title"], page["content_hash"], config,
                     datetime.fromisoformat(page["fetched_at"])),
                ).fetchone()[0]
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (document_id, chunk_index, section, anchor, content, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (doc_id, c["chunk_index"], c["section"], c["anchor"], c["content"], Vector(v))
                            for c, v in zip(page["chunks"], vectors, strict=True)
                        ],
                    )
            indexed.append(page["url"].rsplit("/", 1)[-1])
            embedded += len(vectors)

        # Pages that are no longer in data/chunks/ (removed from sources.yaml) go away too.
        removed = conn.execute(
            "DELETE FROM documents WHERE NOT (url = ANY(%s)) RETURNING url",
            ([p["url"] for p in pages],),
        ).fetchall()
        documents, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        ).fetchone()

    names = f" ({', '.join(indexed)})" if indexed else ""
    print(f"Indexed {len(indexed)} pages{names} | skipped {skipped} unchanged"
          f" | removed {len(removed)} no longer in data/chunks/")
    if embedded:
        print(f"Embeddings: {embedded} chunks in {embed_seconds:.1f}s"
              f" ({1000 * embed_seconds / embedded:.0f} ms each, batches of {BATCH_SIZE})")
    print(f"Database: {documents} documents, {chunks} chunks | total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
