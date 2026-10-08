# Progress

_Last updated: 2026-10-08_

## Current step

**Step 1: Ollama and models.** Not started. Step 0 was approved on 2026-10-08.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Next |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down`) | Pending |
| 3 | Ingest and chunking (`sources.yaml`, cached download, section chunker) | Pending |
| 4 | Indexing (batched embeddings, HNSW, idempotency) | Pending |
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

## Environment baseline (2026-10-08)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS on WSL2 (kernel 6.18), Windows 10 host (build 19045). WSL networking mode: NAT |
| CPU / RAM | 12 logical CPUs. WSL capped at 10 GB: 9.7 GiB visible, 8.6 GiB available, 3 GiB swap |
| GPU | GTX 1070, 8 GiB VRAM (about 7.5 GiB free; the desktop uses about 420 MiB), compute capability 6.1. Windows driver 582.28, CUDA 13.0 |
| Disk | 893 GB free on the WSL root filesystem |
| Docker | Engine 29.8.1 running natively in WSL (not Docker Desktop), Compose v5.5.1 |
| Python | uv 0.12.21, CPython 3.12.3 |
| Tools | git 2.43, GNU Make 4.3. No `psql` on the host, so use `docker compose exec` |
| Ollama | Not installed |
| Ports | 5432 and 11434 are free |

## Findings that affect later steps

- **Corpus (step 3):**
  - Size: the Sentinel-1 SentiWiki is a landing page plus 4 long chapters (`s1-mission`, `s1-products`, `s1-processing`, `s1-applications`), not 15–20 pages. That is roughly 28k words (rough count) under about 90 h2–h4 headings.
  - Format: server-rendered HTML (Scroll Sites on top of Confluence), and `robots.txt` allows everything.
  - Anchors: every heading has a stable `id`, so citation URLs look like `/web/s1-mission#Interferometric-Wide-Swath`.
  - Chunker: skip the "On this Page" table-of-contents heading, and handle h4 headings (their own chunk, or merged into the parent section).
  - Possible extra sources, linked from the chapters: `/web/safe-format` and `/web/precise-orbit-determination`.
- **Schema (step 2):**
  - The proposed `chunks` table has no column for the anchor or URL used in citations.
  - Consider `UNIQUE (document_id, chunk_index)`.
  - The Postgres 18 image needs the volume mounted at `/var/lib/postgresql`.
- **Idempotency (step 4):** if the skip check only compares the page `content_hash`, a re-run after changing the chunking parameters in step 6 would skip everything. The hash, or an extra column, must also cover the chunking parameters and the embedding model.
- **Embedding API (steps 1 and 4):**
  - Use `/api/embed` with an array `input`.
  - Its `prompt_eval_count` gives the real token count, which can validate the chunker's token estimate.
  - Consider `truncate: false` so that long inputs fail instead of being cut silently.
  - Check whether the returned vectors are L2-normalized.
- **Gemma 4 (steps 1 and 5):**
  - It has a thinking mode: send `think: false` explicitly and check the output.
  - Its default sampling parameters are temperature 1, top_k 64 and top_p 0.95, so set the temperature per request.

## Next

**Step 1:**
- Install Ollama in WSL.
- Pull `gemma4:e4b-it-qat` and `bge-m3`.
- Verify that `ollama ps` shows 100% GPU and that `bge-m3` returns 1024-dimensional vectors.

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Read this file, then [DECISIONS.md](DECISIONS.md).
