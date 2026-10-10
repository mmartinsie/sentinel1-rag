# Progress

_Last updated: 2026-10-10_

## Current step

**Step 7: Documentation.** Not started. Step 6 was approved on 2026-10-10.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Done |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down` / `make clean`) | Done |
| 3 | Ingest and chunking (`sources.yaml`, cached download, section chunker) | Done |
| 4 | Indexing (batched embeddings, HNSW, idempotency) | Done |
| 5 | Query (`make ask`, prompt, citations) | Done |
| 6 | Retrieval eval (hit@k, MRR) and top-k 10 | Done |
| 7 | Documentation (README, final review of the decisions) | Next |

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

### Step 5

- `make ask Q="..."` (`ask.py`, the new `retrieval.py`, and `chat` in `ollama.py`):
  1. Embed the question and take the 5 closest chunks (D-015).
  2. Send Gemma the instructions in the system turn, and the numbered passages followed by the question in the user turn (D-016).
  3. Print the answer, then only the cited sources, one entry per section with its URL.
- `ARGS=--show-context` also prints the retrieved chunks with their distances.
- Fixed during the step:
  - **Answer language:** the first prompt wording made English questions get Spanish answers 8 times out of 10. With the new wording it was 20 out of 20 in the right language.
  - **HNSW:** with the join inside the top-k query, the planner could never use the index. The top 5 now come from a subquery on `chunks`.
  - **Output order:** progress and statistics went to stderr and came out of order when piped. Everything now goes to stdout, as in the other commands.
- Verified with the 3 test questions, 3 runs each:
  - The two answerable ones are answered in the question's language, with correct citations.
  - The unanswerable one gets "No se encontró información…" every time.
  - With the models loaded, an answer takes 1–3 s.

### Step 6

- `eval/questions.yaml`: 19 questions, each with the sections that answer it under a strict rule (D-017).
  - 5 were written blind by the user, without reading the pages. 14 were drafted by Claude from the indexed text. The `written` field records which.
  - Questions about three or more things are left out, on the user's decision. For "Level-0, Level-1 and Level-2", the three sections ranked 2, 3 and 14, so part of the answer never reached the model.
- `make eval` (`evaluate.py`): runs `retrieval.retrieve` to depth 20 and reports hit@1/3/5/10 and MRR@k, overall and per `written` group. It prints the misses and saves every run in `eval/results/`. `ARGS="--label <name>"` names a run, and `--top-k` scores another k.
- **Baseline with the top 5:** hit@1 0.684, hit@5 0.895, MRR@5 0.761.
  - Blind questions: hit@5 0.600, MRR@5 0.500. Questions drafted from the text: hit@5 1.000, MRR@5 0.854.
  - Misses: `grd-vs-slc` (rank 9) and `slc-content` (rank 16). The `S1 Processing > L1 Algorithms` sections on SLC and GRD rank low for every product question; chunks from the `S1 Products` page win.
  - Capping the chunks per section, simulated on the saved results, changed no metric.
- **Change: top-k from 5 to 10** (D-018), chosen by the user. hit@10 0.947, MRR@10 0.766; blind questions hit@10 0.800.
  - `grd-vs-slc` is now found. In 3 runs out of 3, the answer uses the SLC section (phase, complex samples, slant range), which never reached Gemma before.
  - `slc-content` is still a miss.
- **Generation with 10 passages:** prompts of 2,150–3,804 tokens (worst case 4,288) out of 8,192. The unanswerable cost question still gets "No se encontró información…" and the English question an English answer, 3 runs out of 3 each.
- Not run: the chunking experiments (chunk size, the "Page > Section" prefix, merging small sections, extra pages).

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

- **Numbers for the README (step 7):** with the top 10 over 19 questions, hit@10 0.947 and MRR@10 0.766, from `eval/results/2026-10-10T130918-top-k-10.json`. Report the blind subset too (hit@10 0.800 over 5 questions) and the caveat about questions drafted from the text (D-017).
- **Limitations for the README:**
  - Questions about products rank the `S1 Products` page above the `S1 Processing > L1 Algorithms` sections that explain SLC and GRD. `slc-content` is still missed at rank 16.
  - Distances alone cannot tell an answerable question from an unanswerable one (D-015).
  - A question about three or more things cannot get every part into the top k with one search.
  - **Partial answers:** for "¿Cómo se exportan los resultados a GeoTIFF?", which the corpus does not answer, Gemma states nearby facts (products are saved as GeoTIFF), with correct citations, but does not say that the export itself is not covered. The prompt asks it to say what is missing (D-016). It happens with the top 5 too, so top-k is not the cause. Left as a documented limitation.
- **Ideas not measured:** merging small sections (15 chunks under 100 tokens), another chunk size, embedding without the "Page > Section" prefix, adding the POD or Glossary pages (D-010), a reranker or hybrid search (extensions).
- **Gemma 4 requests:** `ask.py` sends `think: false`, `num_ctx: 8192`, `temperature: 0.2` and `num_predict: 1024` with every request. On this GPU Ollama's default context is 4096. The model's own defaults are temperature 1, top_k 64 and top_p 0.95.
- **Model loading:**
  - Ollama unloads a model after 5 idle minutes (`OLLAMA_KEEP_ALIVE`). The first question after that takes about 70–90 s, almost all of it loading Gemma from the HDD.
  - In a long session (step 5, before a WSL restart), Ollama saw little free RAM and evicted one model to load the other on every question, so each `make ask` paid a full reload. After the restart both models stayed loaded together. If it happens again, check `ollama ps` and the `evicting` lines in `journalctl -u ollama`.
- **GPU detection after a WSL restart:** right after a restart, Ollama's GPU discovery timed out while reading its CUDA libraries from the cold HDD, and Ollama fell back to the CPU (`inference compute ... library=cpu` in the log). `sudo systemctl restart ollama` fixed it once the libraries were in the disk cache. Check after every restart (see "How to resume").
- **Memory budget:** with both models loaded, the GPU uses 5.4 of 8 GiB and WSL uses 5.5 of 9.7 GiB of RAM. Postgres fits, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Step 7: Documentation.**
- `README.md` in English: what it is, an architecture diagram (Mermaid), how to run it on WSL, the eval results with their number, limitations and extensions.
- Final review of `docs/DECISIONS.md`.

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has both models: `systemctl is-active ollama && ollama list`.
3. After a WSL restart, check that Ollama found the GPU: `journalctl -u ollama -b | grep "inference compute"` must show `library=CUDA`. If it shows `library=cpu`, run `sudo systemctl restart ollama` and check again.
4. Start the database with `make up` (it does not start by itself after a restart). If `data/` is missing, rebuild everything with `make ingest && make index`; otherwise `make index` alone confirms there is nothing to do.
5. Try it: `make ask Q="¿Qué diferencia hay entre un producto GRD y uno SLC?"`. The first question after a while takes about a minute and a half while Gemma loads. `make eval` re-measures retrieval in a few seconds without Gemma.
6. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
