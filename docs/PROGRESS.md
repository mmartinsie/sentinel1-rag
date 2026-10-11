# Progress

_Last updated: 2026-10-11_

## Current step

**Step 13: to be chosen by the user.** Step 12 (technical PDFs) was approved on 2026-10-11, without new blind questions (the user's choice). All steps of the brief (0–7) are done, and so are the extension steps 8 to 12.

| Step | Scope | Status |
|---|---|---|
| 0 | Bootstrap and environment | Done |
| 1 | Ollama and models (`gemma4:e4b-it-qat`, `bge-m3`) | Done |
| 2 | Postgres + pgvector (`docker-compose.yml`, `schema.sql`, `make up` / `make down` / `make clean`) | Done |
| 3 | Ingest and chunking (`sources.yaml`, cached download, section chunker) | Done |
| 4 | Indexing (batched embeddings, HNSW, idempotency) | Done |
| 5 | Query (`make ask`, prompt, citations) | Done |
| 6 | Retrieval eval (hit@k, MRR) and top-k 10 | Done |
| 7 | Documentation (README, final review of the decisions) | Done |
| 8 | Extension 1: lexical baseline and hybrid search | Done |
| 9 | Translate the question before the lexical search | Done |
| 10 | Hybrid search with translation and a smaller top-k, judged on the answers | Done |
| 11 | BM25 instead of `ts_rank` for the full-text search | Done |
| 12 | Extension: technical PDFs from the Sentinel-1 document library | Done |
| 13 | To be chosen by the user (see Next) | Next |

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

### Step 7

- `README.md` rewritten: what it is, a real example answer, a Mermaid diagram of the two pipelines (building the index, answering), the eval results with their caveats, how to run it on WSL with measured times, the commands, the project layout, the limitations and the extensions.
- **Clean-clone check:** a fresh clone in a separate Compose project (`s1rag-cleancheck`, port 5433) ran `make up`, `make ingest`, `make index`, `make ask` and `make eval` with no manual step. The pages downloaded again were identical to the cached ones, and the eval matched the cited run exactly. The throwaway project and its volume were removed afterwards.
  - Times: `make up` 13 s, `make ingest` 10 s, `make index` 33 s (bge-m3 loaded cold), first `make ask` 69 s (60 s loading Gemma).
- **Answer time with the top 10**, models loaded: 3.9–7.4 s (prompts of 2,901–3,359 tokens). The 1–3 s measured in step 5 was with the top 5.
- **Final review of [DECISIONS.md](DECISIONS.md):**
  - D-002, D-016: notes that the prompt now carries 10 passages (D-018), with the measured sizes.
  - D-010, D-013: the experiments planned for step 6 (extra pages, merging small chunks) were not run.
  - D-017: title changed from "MRR over the top 5" to "MRR cut at the top k", since the cut follows `TOP_K`.
  - D-018: the clean-clone reproduction and the new answer times.
  - D-003: Ollama can unload bge-m3 when it loads Gemma (seen twice in this step).

### Step 8

- **Lexical search** (D-019): a stored generated `tsv` column on `chunks` (English configuration, same text as the embedding) with a GIN index, queried with `plainto_tsquery` turned from AND into OR and ranked with `ts_rank`.
  - Added to the existing database with `ALTER TABLE` (no re-indexing needed); `db/schema.sql` creates it for new databases. A fresh database built from `schema.sql` gave identical results.
- **Hybrid search** (D-020): reciprocal rank fusion (k = 60) of the top 40 of each method. Tied scores share a rank, and ties are ordered by section and chunk position, not by chunk id.
  - The first version ordered ties by chunk id, and two databases with the same data disagreed (`ground-subsidence` 7th vs 17th), because ids depend on the indexing history.
- `--method vector|lexical|hybrid` in `make eval` and `make ask`; vector stays the default. Eval questions have a `lang` field, and the eval reports metrics per language.
- **Results** (hit@10 / MRR@10, all 19): vector 0.947 / 0.766, lexical 0.737 / 0.500, hybrid 0.842 / 0.633. On the 7 English questions all three reach hit@10 1.000; the 12 Spanish ones (including the 5 blind ones) are where lexical search fails.
  - Variants measured with a throwaway script: `ts_rank` with length normalization, `ts_rank_cd` and a hand-written BM25. BM25 is the best lexical ranking by MRR (0.559) but no hybrid beats vector search.
- README: new "Lexical and hybrid search" section, `--method` in the commands, extension 1 marked as done.

### Step 9

- `translate.py`: Gemma translates the question into English (temperature 0) for the lexical search only; `--translate` in `make ask` and `make eval`, with `--method lexical` or `hybrid` (D-021).
  - Prompt: detect English first (reply ENGLISH, keep the original), then translate with British spelling, as the documentation. Earlier wordings reworded 2 or 3 of the 7 English questions, once with American spelling that broke the lexical match.
  - `CHAT_NUM_CTX` moved to `config.py`: a request with another context size made Ollama reload Gemma (58 s).
- **Results** (hit@10 / MRR@10, all 19): lexical 0.737 / 0.500 → translated 0.947 / 0.746; hybrid 0.842 / 0.633 → translated 0.947 / 0.860 (hit@1 0.789, hit@3 0.947). Vector search alone: 0.947 / 0.766. Translating the vector half too did not help (0.782 alone, 0.833 in the hybrid).
- **Answers:** the translated hybrid loses the SLC section for the example question (shallower answer, the D-018 gain undone) and gains it for `slc-content`; the cost question is still "not found". hit@10 is equal, so `make ask` keeps vector search by default.
- Translations repeated exactly in 4 probe runs and 2 eval runs; a translation takes 0.2–0.7 s.
- New run `eval/results/2026-10-10T223354-vector.json`: the same retrieval as the cited top-10 run, with the per-language groups added in step 8.

### Step 10

- **Answer-level comparison** (throwaway script, 168 answers: 19 eval questions + 2 unanswerable, 2 runs) of vector top 10, and hybrid with translation at top 10, 5 and 3. Every configuration cites a correct section in 36 of 38 answers; the medians of prompt tokens are 2,851 / 3,147 / 1,615 / 1,042 and of generation time 5.4 / 6.2 / 3.7 / 2.7 s. The top 3 cites fewer sections per answer (1.36 against 1.74–2.0).
- **New default (D-022):** hybrid search, question translated for the full-text half, top 5. Eval: hit@5 0.947, MRR@5 0.860 (`eval/results/2026-10-11T011251-hybrid-en-k5.json`). D-018 superseded.
  - Cost: the example question (GRD vs SLC) loses the SLC section; `--method vector --top-k 10` brings it back. One Gemma call per question, also in the eval.
- CLI: `--translate` defaults to on for lexical and hybrid (`--no-translate` turns it off); `make ask` takes `--top-k`. The five cited runs of steps 6, 8 and 9 are reproduced exactly with the new options.
- README: new example (oil spills), diagram with both searches and the fusion, results for the new default, the answer comparison, and the GRD/SLC limitation.
- End to end with the models loaded: 4.2–16.3 s for the new default and 5.4–12.4 s for vector top 10 on 6 questions; answer length dominates.

### Step 11

- **BM25 in SQL** (`BM25_QUERY` in `retrieval.py`, D-023): k1 1.2, b 0.75, Lucene's IDF; corpus statistics computed on every query from `tsv` (15.7 ms against 3.9 ms for `ts_rank`). It matches the throwaway Python BM25 on 38 queries to 4·10⁻¹⁵.
- `--lexical-ranking bm25|ts_rank` in `make ask` and `make eval`, BM25 by default; results record `lexical_ranking`. The six cited runs of steps 6 and 8–10 reproduce with `ts_rank`.
- **Eval, default configuration:** hit@1 0.789 → 0.895, MRR@5 0.860 → 0.908, hit@5 0.947 unchanged, hit@3 0.947 → 0.895 (`slc-content` 3rd → 4th). English questions all at rank 1. Runs at top 10 for lexical and hybrid, with and without translation, saved for the README table.
- **Answers** (84, top 5, `ts_rank` vs BM25): the same 36 of 38 cite a correct section, the same "not found" behaviour, the same prompt size and time. Mixed differences when read: oil spills better, product metadata worse. Proposed anyway for the ranking, the IDF fix and the standard recipe.
- README: new example output (BM25), BM25 in "How it works", results and comparison tables, the BM25 limitation (statistics on every query).

### Step 12

- **Plan approved by the user** (all items as recommended): six PDFs, pdfplumber, sections from numbered headings with page anchors, `pdfs:` in `sources.yaml`, the eval re-labelled plus new questions, the prompt wording. The plan also had the user write 3–5 blind questions about the PDFs; at the review the user chose to keep the 25 questions without them.
- **Corpus** (D-024): Radiometric Calibration (TN-0002 1.0), Thermal Denoising (MPC-0392 2.3), Masking No-value Pixels on GRD (MPC-0243 2.2), TOPS SLC Deramping (TN-14-0025 1.3), Guide to Sentinel-1 Geocoding (UZH 1.12) and the Sentinel-1 Product Definition (2.11). 373 new chunks from 237 sections; 568 in all.
- **Code:**
  - `extract_pdf.py` (D-025): headers and footers by repetition in the margins, headings by numbering plus a larger font, `page=N` anchors, paragraphs by line gaps, lists, tables with row-merged cells repeated, hyphen joining checked against the document's own words, chart and equation debris dropped.
  - `fetch.py` stores bytes and names PDFs by URL; `ingest.py` reads `pdfs:` from `sources.yaml`; `index.py` names documents by slug.
  - `evaluate.py` matches labels on URL and section path (sections on one PDF page share the URL) and reports `wiki-only`, `wiki+pdf` and `pdf-only` groups.
  - `ask.py`: the system turn says "SentiWiki pages and technical documents".
- **Eval** (D-026): 11 of the 19 questions gained PDF labels, and 6 questions were drafted from the PDFs. Default: hit@5 0.880, MRR@5 0.810 over 25 (`eval/results/2026-10-11T113743-pdfs.json`). The 19 earlier questions: hit@5 0.947 as before, MRR@5 0.895 (was 0.908). PDF-only questions: hit@5 0.667. The default is still the best method at 5 passages.
- **Answers** (throwaway script, 108): earlier questions 36/38 citing a correct section before and after; PDF questions 0/12 → 8/12. Two misses: `sigma0-calibration` (vocabulary) and `tops-deramping` (a long section crowds out the introduction).
- **Fixed during the step:** words split by pypdf (switched to pdfplumber), running footers outside the first margin guess (12 % → 15 %), private-use bullet glyphs, chart text, unmapped equation glyphs, hyphenated line breaks in the geocoding guide, and merged table cells, which had made an answer give pixel spacing as resolution.
- **Checks:** every PDF section heading is on the page its URL cites (237 of 237); a fresh database built from `schema.sql` reproduces the cited eval run exactly; the five SentiWiki pages' chunks are byte-identical to before.
- A copy of the index without the PDFs, `rag_wiki`, was made in the same Postgres for the "before" answers, and dropped after the review.

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

## Findings worth keeping

- **Eval numbers, limitations and experiments not run** are now in the [README](../README.md) (step 7): hit@10 0.947 and MRR@10 0.766 over 19 questions, 0.800 hit@10 for the 5 blind ones, from `eval/results/2026-10-10T130918-top-k-10.json`.
- **Gemma 4 requests:** `ask.py` sends `think: false`, `num_ctx: 8192`, `temperature: 0.2` and `num_predict: 1024` with every request. On this GPU Ollama's default context is 4096. The model's own defaults are temperature 1, top_k 64 and top_p 0.95.
- **Model loading:**
  - Ollama unloads a model after 5 idle minutes (`OLLAMA_KEEP_ALIVE`). The first question after that takes about 70–90 s, almost all of it loading Gemma from the HDD.
  - In a long session (step 5, before a WSL restart), Ollama saw little free RAM and evicted one model to load the other on every question, so each `make ask` paid a full reload. After the restart both models stayed loaded together. If it happens again, check `ollama ps` and the `evicting` lines in `journalctl -u ollama`.
  - In step 7, loading Gemma while bge-m3 was loaded unloaded bge-m3 twice (D-003). Loading bge-m3 next to Gemma did not evict Gemma.
- **GPU detection after a WSL restart:** right after a restart, Ollama's GPU discovery timed out while reading its CUDA libraries from the cold HDD, and Ollama fell back to the CPU (`inference compute ... library=cpu` in the log). `sudo systemctl restart ollama` fixed it once the libraries were in the disk cache. Check after every restart (see "How to resume").
- **Stale schema in a running container:** `docker-compose.yml` bind-mounts the single file `db/schema.sql`. A bind mount of a file keeps pointing at the old file when an editor replaces it, so the container started before step 8 still saw the schema without the `tsv` column (found in step 12, building a fresh database inside it). It only matters for databases created inside an old container; `make clean && make up` creates a new one.
- **Memory budget:** with both models loaded, the GPU uses 5.4 of 8 GiB and WSL uses 5.5 of 9.7 GiB of RAM. Postgres fits, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Step 13:** the user chooses. The remaining extensions in the [README](../README.md#extensions-not-implemented) are an agent with a STAC `search_scenes` tool, an MCP server and OpenSearch. A smaller experiment that step 12 motivates: at most two chunks per section in the top 5 (`tops-deramping` lost its introduction to four chunks of one section).

The current pipeline, for reference: translate the question (Gemma, temperature 0), vector search (bge-m3, original question) and BM25 full-text search (English question), reciprocal rank fusion, top 5 passages to Gemma. Corpus: 5 SentiWiki pages and 6 PDFs, 568 chunks. Eval: hit@5 0.880, MRR@5 0.810 over 25 questions (`eval/results/2026-10-11T113743-pdfs.json`).

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has both models: `systemctl is-active ollama && ollama list`.
3. After a WSL restart, check that Ollama found the GPU: `journalctl -u ollama -b | grep "inference compute"` must show `library=CUDA`. If it shows `library=cpu`, run `sudo systemctl restart ollama` and check again.
4. Start the database with `make up` (it does not start by itself after a restart). If `data/` is missing, rebuild everything with `make ingest && make index` (about 80 s and a minute); otherwise `make index` alone confirms there is nothing to do.
5. Try it: `make ask Q="¿Qué diferencia hay entre un producto GRD y uno SLC?"`. The first question after a while takes about a minute and a half while Gemma loads. `make eval` re-measures retrieval in about 15 s once Gemma is loaded (it translates the questions); `ARGS="--method vector"` skips Gemma.
6. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
7. The comparisons of generated answers (steps 10 to 12), the BM25 cross-check and the PDF checks (page anchors, token counts, heading fonts) ran as throwaway scripts in a session scratchpad, so they are not in the repo. What they measured and their numbers are in DECISIONS.md, under "Verified" in D-022 to D-026.
