# Step 2: Postgres + pgvector

*Finished on 2026-10-08.*

## What this step was about

This step set up the database that will store the chunks and their embeddings, and that will find the chunks closest to a question.

**pgvector** is a Postgres extension that adds three things:

- A column type, `vector(n)`: a fixed-length list of n numbers. A column declared as `vector(1024)` refuses anything else.
- Distance operators: `<->` for straight-line (Euclidean) distance, `<=>` for cosine distance, and `<#>` for the negative inner product.
- Indexes that speed up the search.

Searching by meaning then becomes ordinary SQL. `ORDER BY embedding <=> :question LIMIT 5` means "the five rows whose vectors point most nearly in the same direction as the question's vector".

Without an index, Postgres computes that distance for every row and sorts the results. That is called *exact search*, and with a few hundred rows it takes milliseconds.

An **HNSW** index (Hierarchical Navigable Small World) is a graph in which each vector is linked to some of its neighbours, organised in layers. The top layers are sparse, like motorways with few exits; the bottom ones are dense, like local streets. A search enters at the top and moves greedily towards the query, dropping a layer each time it can't get any closer. It finds very good neighbours without looking at every row. The price is that the result is *approximate*: occasionally a true neighbour is missed.

An index is built for one distance operator, so ours (`vector_cosine_ops`) only helps queries that use `<=>`.

## What we built

- **`docker-compose.yml`:** one Postgres service using the pinned image `pgvector/pgvector:0.8.7-pg18-trixie`. Its configuration:
  - The port is published on `127.0.0.1` only, so the database is not reachable from the network.
  - The data lives in a named volume, mounted where Postgres 18 expects it (`/var/lib/postgresql`).
  - The schema file is mounted as an *init script*, which Postgres runs automatically the first time it starts with an empty volume.
  - The credentials (`rag` / `rag`) are local-only defaults that can be overridden.
- **`db/schema.sql`:** two tables. `documents` holds one row per page and `chunks` one row per chunk with its vector, plus the HNSW index.
- **`Makefile`:** `make up` starts the database and waits until it is ready, `make down` stops it and keeps the data, and `make clean` also deletes the data.

## How the schema differs from the original plan

Four small changes, each fixing a gap found in step 0:

1. **Each chunk stores its anchor**, the id of its heading in the page. A citation can then link straight to the section, like `…/s1-mission#Interferometric-Wide-Swath`.
2. **Each page stores how it was indexed** (`index_config`): the chunking settings and the embedding model. A page is processed again when its text changes *or* when that recipe changes. Otherwise, after tuning the chunk size in step 6, re-indexing would skip every page. Worse, it would leave vectors from two different recipes mixed in one index, and their distances can't be compared.
3. **A chunk can't exist without its vector.** The embedding column is `NOT NULL`, so every row can be searched. This also fixes how the pipeline works: `make ingest` will write chunks to files, and `make index` will be the only command that writes to the database, one page at a time, inside a transaction.
4. **A chunk position can't be stored twice for the same page**, thanks to a uniqueness rule on (page, chunk number).

## How we checked it

- `make up` took about 11 seconds. Docker reported the container as healthy, and the port is open on localhost only.
- The startup log shows the schema file running and creating the extension, both tables and the index.
- Inside the database, pgvector is version 0.8.7 on PostgreSQL 18.6, and the tables and index look exactly as designed.
- **A small nearest-neighbour test.** Three made-up vectors (e1, e1+e2 and e3) and a query equal to e1 gave cosine distances of 0, 0.293 and 1. That is exactly what geometry predicts: the same direction, 45 degrees apart, and a right angle. The test ran inside a transaction that was then undone, so nothing was left behind.
- **The guard rails work.** A vector with 3 numbers is rejected ("expected 1024 dimensions, not 3"), and so is a chunk with no vector.
- **The data lifecycle works.** Data survives `make down` + `make up`. After `make clean`, the next `make up` starts from an empty database and applies the schema again.
- **Python can talk to it.** The same libraries the project will use (psycopg and pgvector) connected and computed the same 0.293 distance.

One subtle detail is worth knowing. Docker's "healthy" check connects over the network (TCP) on purpose. While it runs the init scripts, Postgres uses a temporary server that only accepts local socket connections; we confirmed this in the image's startup script. A check over the socket could therefore report "ready" before the tables exist, and a script started right after `make up` could fail.

## What it means for the next steps

- **Step 3** (download and chunking) won't touch the database at all: it writes chunks to `data/`.
- **Step 4** (indexing) will:
  - embed each page's chunks;
  - insert them in one transaction;
  - skip pages whose text and recipe are unchanged.

  Citation links are then built as page URL + `#` + anchor.
- The database is running. `make down` stops it whenever it isn't needed.

## Check your understanding

**What does `ORDER BY embedding <=> $1 LIMIT 5` do, and when does Postgres use the HNSW index for it?**
It computes the cosine distance between the question's vector and each chunk's vector, and returns the five closest. Without an index, Postgres scans every row and gets the exact answer. With an HNSW index built on `vector_cosine_ops`, it can walk the graph instead and get an approximate top 5. Three conditions must hold: the query uses the same operator (`<=>`), it orders by that distance in ascending order, and it has a `LIMIT`. With very few rows the planner may still prefer the full scan, because it's cheaper.

**Why store `index_config` next to the page's `content_hash`?**
A chunk's vector depends on three inputs: the page text, how the page was cut into chunks, and which model produced the vector. If only the text were tracked, changing the chunk size or the model would leave stale chunks in place. It could also mix vectors from different models, whose distances are meaningless when compared. Keeping the two values separate also shows *why* a page was indexed again.
