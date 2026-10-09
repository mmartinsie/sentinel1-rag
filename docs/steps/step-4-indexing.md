# Step 4: Indexing

*Finished on 2026-10-09.*

## What this step was about

Indexing turns the chunks from step 3 into rows in Postgres, each one stored with its embedding. Three ideas matter here.

**Batching.** Every request to Ollama has a fixed cost on top of the actual work: the network round trip, queueing and setup. Sending several texts in one request pays that cost once for all of them.

**Idempotency.** An operation is idempotent when running it twice leaves things exactly as running it once. For every page, our indexer remembers two things:
- a fingerprint of the page's text;
- the "recipe" used to process it: the chunking settings, the embedding model with its exact version, and how the text is presented to the model.

If both match what is stored, there is nothing to do.

**Transactions.** When a page does need re-indexing, its old rows are deleted and the new ones inserted inside a single transaction, so either all of it happens or none of it does. A search running at the same time sees either the old page or the new one, never a mix, and a crash halfway through leaves the old version intact.

## What we built

- **`ollama.py`:** a tiny client for Ollama's API, which gives a clear error when the service isn't running.
- **`db.py`:** opens the database connection and registers pgvector's vector type with it.
- **`index.py`** (`make index`): for each page, it compares the fingerprints, embeds the chunks in batches of 32 and writes the page in one transaction. Pages that are no longer in the corpus are deleted.

Each chunk is embedded together with its heading path: "S1 Mission > Acquisition Modes > Interferometric Wide Swath", a blank line, then the text. That way the vector knows what the paragraph is about.

## How we checked it

**We measured the batch size.**
- Embedding all 195 chunks one by one took 20.8 seconds. In batches of 8 or more it took 15.8 seconds, about 25 % less, and bigger batches didn't help any further. The saving is the fixed cost of each request, about 26 ms.
- The vectors were exactly identical whatever the batch size.
- We chose 32: it is as fast as sending everything at once, and a failed request only loses 32 embeddings.

**The first run** indexed all 5 pages and 195 chunks in about 17 seconds.

**The second run** skipped every page and computed no embeddings. A fingerprint of the whole table (every id, every text and every vector) was identical before and after. That is the idempotency check the plan asked for.

**Changes are detected, and only where they happen.**
- We changed one page's text fingerprint, and only that page was re-indexed.
- We changed another page's stored recipe, and again only that page was re-indexed.
- We removed a page's file, and its document and its 50 chunks were deleted.

Putting everything back restored the 5 pages and 195 chunks, and one more run did nothing.

**The HNSW index works, but Postgres doesn't need it yet.**
- With only 195 rows, the query planner prefers to read the whole table and sort it, which takes about one millisecond. Forcing it to use the index cuts that to half a millisecond.
- For all 195 test queries, the index found exactly the same top 5 as the exact search. That held even when we turned its search effort (`ef_search`) down from 40 to 5. In such a small graph the approximate search ends up visiting nearly everything; approximation errors only show up with many more vectors.

**Retrieval already looks right.** We asked the brief's example question in Spanish: "¿Qué diferencia hay entre un producto GRD y uno SLC?" The five chunks returned all talk about GRD and SLC products. The closest one is the Level-1 section that lists exactly those two product types.

## What it means for the next steps

Step 5 builds the question-answering command. It embeds the question, fetches the five closest chunks, and hands them to Gemma with instructions to answer only from them and to cite them.

## Check your understanding

**What makes the indexer idempotent, and why store a "recipe" next to the text fingerprint?**
For each page we store a hash of its text and a description of how its vectors were made: the chunking settings, the embedding model and its digest, and the text template. If both match, the page is skipped; otherwise it is rebuilt inside one transaction. The recipe matters because vectors depend on more than the text. If you change the chunk size or the model, the stored vectors no longer match what a fresh run would produce. Comparing only the text would leave stale vectors behind, possibly from two different models mixed in the same index.

**Why didn't Postgres use the HNSW index, and is that a problem?**
The planner estimates the cost of each possible plan and picks the cheapest. Reading 195 rows and keeping the best 5 is so cheap that it beats the index. That is not a problem: both plans take about a millisecond and return the same rows. As the table grows, the full scan gets more expensive and the planner will switch to the index on its own. To see the index at work now, we disabled sequential scans for one query and read the plan with `EXPLAIN`.
