"""`make ingest`: download the source pages, split them into chunks and save them.

Output: one JSON file per page in data/chunks/, read later by `make index`.
Nothing is written to the database in this step.
"""

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict

from sentinel1_rag.chunking import SETTINGS, TOKENS_PER_WORD, Chunk, chunk_page
from sentinel1_rag.config import CHUNKS_DIR, load_sources
from sentinel1_rag.extract import extract
from sentinel1_rag.fetch import PoliteClient, fetch, slug


def main() -> None:
    client = PoliteClient()
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CHUNKS_DIR.glob("*.json"):  # pages removed from sources.yaml disappear too
        stale.unlink()

    pages, all_chunks = [], []
    for url in load_sources():
        raw = fetch(url, client)
        page = extract(raw.html, url)
        chunks = chunk_page(page)
        record = {
            "url": url,
            "title": page.title,
            "content_hash": hashlib.sha256(page.text.encode()).hexdigest(),
            "fetched_at": raw.fetched_at,
            "chunking": {**asdict(SETTINGS), "tokens_per_word": TOKENS_PER_WORD},
            "chunks": [asdict(c) for c in chunks],
        }
        path = CHUNKS_DIR / f"{slug(url)}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        source = "cache" if raw.from_cache else "downloaded"
        pages.append((slug(url), source, len(page.sections), len(chunks), len(page.text.split())))
        all_chunks.extend(chunks)

    print_stats(pages, all_chunks, client.requests)


def print_stats(pages: list[tuple], chunks: list[Chunk], requests: int) -> None:
    print(f"Pages ({requests} HTTP requests made)")
    print(f"  {'page':<18} {'source':<11} {'sections':>8} {'chunks':>7} {'words':>7}")
    for name, source, sections, n_chunks, words in pages:
        print(f"  {name:<18} {source:<11} {sections:>8} {n_chunks:>7} {words:>7}")

    tokens = [c.tokens for c in chunks]
    words = [len(c.content.split()) for c in chunks]
    print(f"\nChunks: {len(chunks)} in {CHUNKS_DIR}")
    for label, values in (("tokens (estimated)", tokens), ("words", words)):
        print(
            f"  {label:<19} mean {statistics.mean(values):5.0f} | median "
            f"{statistics.median(values):5.0f} | min {min(values):4} | max {max(values):4}"
        )
    per_section = Counter((c.section, c.anchor) for c in chunks)
    split = sum(1 for n in per_section.values() if n > 1)
    print(f"  sections: {len(per_section)} | split into several chunks: {split}")


if __name__ == "__main__":
    main()
