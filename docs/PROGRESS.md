# Progress

_Last updated: 2026-10-09_

## Current step

**Step 5: Query.** Not started. Step 4 was approved on 2026-10-09.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Done |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down` / `make clean`) | Done |
| 3 | Ingest and chunking (`sources.yaml`, cached download, section chunker) | Done |
| 4 | Indexing (batched embeddings, HNSW, idempotency) | Done |
| 5 | Query (`make ask`, prompt, citations) | Next |
| 6 | Retrieval eval (hit@1, hit@5, MRR) | Pending |
| 7 | Documentation (README, final review of the decisions) | Pending |

## Done

### Step 0

- Checked the environment; results are in the baseline below.
- Verified the external facts behind the technical plan: model tags and sizes, embedding dimension, pgvector image tag, Pascal GPU support and the SentiWiki structure. Details are in [DECISIONS.md](DECISIONS.md), D-001 to D-007.
- Created the repo skeleton:
  - `git init` on branch `main` and `.gitignore`.
  - `pyproject.toml` (uv, src layout, minimal dependencies) and `uv.lock`.
  - A `README.md` stub with the extensions list.
  - `docs/DECISIONS.md` and `docs/PROGRESS.md`.
- Reviewed the technical plan; decisions D-001 to D-007 accepted.

### Step 1

- Installed Ollama 0.40.1 in WSL as a systemd service. It detects the GTX 1070 and runs it with its CUDA 12 runner.
- Pulled `gemma4:e4b-it-qat` (6.1 GB) and `bge-m3` (1.2 GB).
- Verified that Gemma runs at 100% GPU with `num_ctx = 8192` and that `bge-m3` returns 1024-dimensional vectors. The measurements are in [DECISIONS.md](DECISIONS.md), D-002 and D-003.
- The checks used a throwaway script, so no code was added in this step.
- New working rule: each finished step gets a plain-English write-up in [steps/](steps/). Steps 0 and 1 already have one.

### Step 2

- `docker-compose.yml`: `pgvector/pgvector:0.8.7-pg18-trixie`, port published on localhost only, volume on `/var/lib/postgresql`, TCP healthcheck, overridable local credentials (D-009).
- `db/schema.sql`: the brief's schema plus a citation anchor, `index_config`, a `NOT NULL` embedding and `UNIQUE (document_id, chunk_index)` (D-008).
- `Makefile` with `make up`, `make down` and `make clean`.
- Verified:
  - The `vector` extension (0.8.7 on PostgreSQL 18.6), the tables and the HNSW index are in place.
  - The `vector(1024)` column rejects other dimensions, and chunks without an embedding are rejected too.
  - A nearest-neighbour smoke test returns the expected cosine distances.
  - Data survives `make down` / `make up`, and `make clean` resets the database.
  - Python (psycopg + pgvector) connects.

### Step 3

- `sources.yaml`: the 5 pages of the Sentinel-1 subtree (D-010).
- `make ingest` (`src/sentinel1_rag/`):
  - `fetch.py` downloads the pages politely into `data/raw/` (D-011).
  - `extract.py` turns each page into sections (D-012).
  - `chunking.py` cuts the sections into chunks (D-013).
  - `ingest.py` writes the result to `data/chunks/<page>.json` and prints the statistics.
- `make clean` now also deletes `data/`.
- Result: 195 chunks from 89 sections. Real bge-m3 size: mean 268 tokens, median 273, minimum 20, maximum 595.
- Fixed during the step:
  - The token estimate was calibrated against real counts (from 1.35 to 1.68 tokens per word).
  - Tables are no longer split when they fit in a chunk.
  - The wiki's `[n]` reference markers are removed.

### Step 4

- `make index` (`index.py`, with the new `ollama.py` and `db.py`) embeds the chunks in batches of 32 and writes each page in its own transaction. It skips pages whose `content_hash` and `index_config` are unchanged, and deletes pages that are gone (D-014).
- Verified:
  - First run: 195 chunks in 16.8 s.
  - Second run: nothing to do, and the database is identical, row by row.
  - Changing the text, changing the recipe and removing a page each touch only the page concerned.
- HNSW (D-006):
  - The planner prefers a sequential scan at this size (1.1 ms). Forcing the index gives 0.5 ms.
  - Recall@5 against exact search is 1.000.
- Retrieval sanity check: the brief's example question, in Spanish, returns 5 chunks about GRD and SLC products.

## Environment baseline (2026-10-08)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS on WSL2 (kernel 6.18), Windows 10 host (build 19045). WSL networking mode: NAT |
| CPU / RAM | 12 logical CPUs. WSL capped at 10 GB: 9.7 GiB visible, 8.6 GiB available, 3 GiB swap |
| GPU | GTX 1070, 8 GiB VRAM (about 7.5 GiB free; the desktop uses about 420 MiB), compute capability 6.1. Windows driver 582.28, CUDA 13.0 |
| Disk | 893 GB free on the WSL root filesystem. The WSL virtual disk lives on a hard drive (HDD), so the first load of a model is slow (about 70 s for Gemma) |
| Docker | Engine 29.8.1 running natively in WSL (not Docker Desktop), Compose v5.5.1 |
| Python | uv 0.12.21, CPython 3.12.3 |
| Tools | git 2.43, GNU Make 4.3. No `psql` on the host: use `docker compose exec db psql -U rag -d rag` |
| Ollama | 0.40.1 as the systemd service `ollama`. Models live in `/usr/share/ollama/.ollama/models` |
| Ports | 5432: the project's Postgres (localhost only). 11434: Ollama |

## Findings that affect later steps

- **Query (step 5):**
  - Embed the raw question with bge-m3, with no instruction prefix (D-003). The stored vectors embed `section + "\n\n" + content`.
  - Retrieve with `ORDER BY embedding <=> %s LIMIT k`. If the planner ever uses the HNSW index, keep `hnsw.ef_search` (40 by default) at or above k, because the index scan only produces about `ef_search` candidates.
  - A citation URL is `documents.url` + `#` + `anchor`, or just `documents.url` when the anchor is `NULL`.
  - The `Ollama` client in `ollama.py` needs a `chat` method for Gemma.
  - Retrieval already looks right for the brief's example question: the top 5 chunks are all about GRD and SLC (cosine distances 0.36–0.46).
- **Experiments for step 6:**
  - Add the cross-mission candidate pages (POD, Glossary; D-010).
  - Merge small sections: 15 chunks have fewer than 100 tokens.
- **Gemma 4 requests (step 5):**
  - Send `think: false`, `num_ctx: 8192` and `temperature: 0.2` with every request. On this GPU Ollama's default context is 4096, so without `num_ctx` a long RAG prompt would be cut.
  - The model's own defaults are temperature 1, top_k 64 and top_p 0.95.
- **Model loading (steps 5 and 6):** Ollama unloads a model after 5 idle minutes (`OLLAMA_KEEP_ALIVE`), and reloading Gemma from the HDD takes about 70 s. For eval runs, consider sending a longer `keep_alive` with the requests.
- **Memory budget:** with both models loaded, the GPU uses 5.4 of 8 GiB and WSL uses 5.5 of 9.7 GiB of RAM. Postgres fits, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Step 5:**
- `make ask Q="..."`:
  1. Embed the question and retrieve the 5 closest chunks.
  2. Build a prompt with the chunks numbered.
  3. Ask Gemma (`/api/chat` with `think: false`, `num_ctx: 8192` and `temperature: 0.2`) to answer only from those chunks, cite them with [n], say so when the answer isn't there, and reply in the language of the question.
- Print the answer followed by its sources: page, section and URL with anchor. `--show-context` also prints the retrieved chunks with their distances.
- Test with 3 questions, two that the corpus answers and one that it doesn't, and review the prompt together.

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has both models: `systemctl is-active ollama && ollama list`.
3. Start the database with `make up`. If `data/` is missing, rebuild everything with `make ingest && make index`; otherwise `make index` alone confirms there is nothing to do.
4. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
