# Progress

_Last updated: 2026-10-08_

## Current step

**Step 2: Postgres + pgvector.** Not started. Step 1 was approved on 2026-10-08.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Done |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down`) | Next |
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

### Step 1

- Installed Ollama 0.40.1 in WSL as a systemd service. It detects the GTX 1070 and runs it with its CUDA 12 runner.
- Pulled `gemma4:e4b-it-qat` (6.1 GB) and `bge-m3` (1.2 GB).
- Verified that Gemma runs at 100% GPU with `num_ctx = 8192` and that `bge-m3` returns 1024-dimensional vectors. The measurements are in [DECISIONS.md](DECISIONS.md), D-002 and D-003.
- The checks used a throwaway script, so no code was added in this step.
- New working rule: each finished step gets a plain-English write-up in [steps/](steps/). Steps 0 and 1 already have one.

## Environment baseline (2026-10-08)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS on WSL2 (kernel 6.18), Windows 10 host (build 19045). WSL networking mode: NAT |
| CPU / RAM | 12 logical CPUs. WSL capped at 10 GB: 9.7 GiB visible, 8.6 GiB available, 3 GiB swap |
| GPU | GTX 1070, 8 GiB VRAM (about 7.5 GiB free; the desktop uses about 420 MiB), compute capability 6.1. Windows driver 582.28, CUDA 13.0 |
| Disk | 893 GB free on the WSL root filesystem. The WSL virtual disk lives on a hard drive (HDD), so the first load of a model is slow (about 70 s for Gemma) |
| Docker | Engine 29.8.1 running natively in WSL (not Docker Desktop), Compose v5.5.1 |
| Python | uv 0.12.21, CPython 3.12.3 |
| Tools | git 2.43, GNU Make 4.3. No `psql` on the host, so use `docker compose exec` |
| Ollama | 0.40.1 as the systemd service `ollama`. Models live in `/usr/share/ollama/.ollama/models` |
| Ports | 5432 is free; 11434 is used by Ollama |

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
- **Embedding API (step 4):**
  - Use `/api/embed` with an array `input`.
  - Its `prompt_eval_count` gives the real token count, which can validate the chunker's token estimate.
  - Send `truncate: false` so that long inputs fail instead of being cut silently.
  - The vectors come back L2-normalized (checked in step 1).
- **Gemma 4 requests (step 5):**
  - Send `think: false`, `num_ctx: 8192` and `temperature: 0.2` with every request. On this GPU Ollama's default context is 4096, so without `num_ctx` a long RAG prompt would be cut.
  - The model's own defaults are temperature 1, top_k 64 and top_p 0.95.
- **Model loading (steps 5 and 6):** Ollama unloads a model after 5 idle minutes (`OLLAMA_KEEP_ALIVE`), and reloading Gemma from the HDD takes about 70 s. For eval runs, consider sending a longer `keep_alive` with the requests.
- **Memory budget:** with both models loaded, the GPU uses 5.4 of 8 GiB and WSL uses 5.5 of 9.7 GiB of RAM. Postgres will fit, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Step 2:**
- `docker-compose.yml` with `pgvector/pgvector:0.8.7-pg18-trixie`, and `schema.sql` mounted in `/docker-entrypoint-initdb.d`.
- `make up` and `make down`.
- Verify that the `vector` extension is active and the tables exist.

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has both models: `systemctl is-active ollama && ollama list`.
3. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
