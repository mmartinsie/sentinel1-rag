# Progress

_Last updated: 2026-10-09_

## Current step

**Step 4: Indexing.** Not started. Step 3 was approved on 2026-10-09.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Done |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down` / `make clean`) | Done |
| 3 | Ingest and chunking (`sources.yaml`, cached download, section chunker) | Done |
| 4 | Indexing (batched embeddings, HNSW, idempotency) | Next |
| 5 | Query (`make ask`, prompt, citations) | Pending |
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

- **Indexing input (step 4):**
  - `make index` reads `data/chunks/<page>.json`. Each file holds `url`, `title`, `content_hash`, `fetched_at`, `chunking` (the settings plus `tokens_per_word`) and the chunks (`chunk_index`, `section`, `anchor`, `content`, `tokens`).
  - The text to embed is `section + "\n\n" + content`. The section path already starts with the page title, so it is the "Page > Section" prefix.
  - Pages that disappear from `sources.yaml` have to be deleted from the database.
- **Pipeline shape (step 4, from D-008):** `make index` is the only step that writes rows: one page per transaction, with every chunk inserted together with its embedding.
- **Idempotency (step 4):** skip a page only when both its `content_hash` (page text) and its `index_config` are unchanged. `index_config` must cover the `chunking` settings and the embedding model.
- **Experiments for step 6:**
  - Add the cross-mission candidate pages (POD, Glossary; D-010).
  - Merge small sections: 15 chunks have fewer than 100 tokens.
- **Embedding API (step 4):**
  - Use `/api/embed` with an array `input`.
  - Its `prompt_eval_count` gives the real token count, which can validate the chunker's token estimate.
  - Send `truncate: false` so that long inputs fail instead of being cut silently.
  - The vectors come back L2-normalized (checked in step 1).
- **Gemma 4 requests (step 5):**
  - Send `think: false`, `num_ctx: 8192` and `temperature: 0.2` with every request. On this GPU Ollama's default context is 4096, so without `num_ctx` a long RAG prompt would be cut.
  - The model's own defaults are temperature 1, top_k 64 and top_p 0.95.
- **Model loading (steps 5 and 6):** Ollama unloads a model after 5 idle minutes (`OLLAMA_KEEP_ALIVE`), and reloading Gemma from the HDD takes about 70 s. For eval runs, consider sending a longer `keep_alive` with the requests.
- **Memory budget:** with both models loaded, the GPU uses 5.4 of 8 GiB and WSL uses 5.5 of 9.7 GiB of RAM. Postgres fits, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Step 4:**
- `make index`: embed the chunks in batches with bge-m3 (`/api/embed`, `truncate: false`), then insert each page's chunks in one transaction.
- Make it idempotent with `content_hash` + `index_config`, and check this by running it twice.
- Look at the HNSW index at work (`EXPLAIN`).

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has both models: `systemctl is-active ollama && ollama list`.
3. Start the database with `make up`, and rebuild the chunks with `make ingest` (it uses the cache in `data/raw/`).
4. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
