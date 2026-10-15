# Progress

_Last updated: 2026-10-15_

## Current step

**None: step 16 is done (approved by the user on 2026-10-15) and committed; its push and the `v0.4` tag wait for the user's "súbelo".** All steps of the brief (0–7) are done, and so are the extension steps 8 to 16; steps 8 to 15 cover all five planned extensions. What comes next is open (see "Next").

**Versions:** the annotated tag `v0.1` marks the end of step 12 (commit `2c8d919`, README with hit@5 0.880 from `…113743-pdfs.json`); it was created and pushed on 2026-10-11 with the user's OK, because a public post links to it. Never move or delete it. **v0.2** = the agent (step 13) + the MCP server (step 14): annotated tag on commit `4247db7`, created and pushed on 2026-10-14 with the user's OK. The reranker is not in v0.2. **v0.3** = OpenSearch compared with pgvector (step 15): annotated tag on commit `3c1f359`, created and pushed on 2026-10-14 with the user's OK. **v0.4** = the reranker (step 16): approved by the user on 2026-10-15, as an annotated tag on the step-16 commit, created and pushed together with that commit. Any later tag is created and pushed only on the user's OK.

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
| 13 | Extension: agent with a STAC `search_scenes` tool | Done |
| 14 | Extension: MCP server exposing `search_docs` and `search_scenes` | Done (v0.2) |
| 15 | Extension: OpenSearch k-NN compared with pgvector | Done (v0.3) |
| 16 | Reranker over the top 20 (Qwen3-Reranker-0.6B through Ollama) | Done (v0.4) |

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

### Step 13

- **Approved by the user** on 2026-10-11, every item as recommended (D-027 to D-030 accepted; next: the MCP server).
- **Plan approved by the user** (all items as recommended): `make agent`, two tools, Nominatim inside `search_scenes`, a hand-written loop, `make eval-agent`, one run with thinking. No blind questions; all 18 eval questions were drafted by Claude.
- **Code** (D-027, D-028):
  - `agent.py`: the loop (at most 5 turns, temperature 0, thinking off), a trace of every call, and the output: the answer, the cited sources and a table of the scenes, printed by the code.
  - `tools.py`: `search_docs` (the default retrieval) and `search_scenes` with their JSON Schemas. The loop forces a documentation search when the model answers before calling a tool; `days` counts back from today in code; `mode` and `orbit` take `any`; errors name what is wrong; a place under 1 km across is sent back once; scene results are a summary that says it has no number to cite.
  - `scenes.py` (CDSE STAC search, raw httpx), `geocode.py` (Nominatim with a cache in `data/geocode.json`, 1 request per second, attribution), `evaluate_agent.py` and `eval/agent_questions.yaml`.
  - `ollama.py` sends `use_mmap: true` with every request (D-030).
- **Eval** (D-029), questions passing every check out of 18: first design 10 (held-out 4 of 6); final 13 (held-out 5 of 6), median 5.9 s per question (`eval/results/2026-10-11T152216-agent-final.json`); with thinking also 13, in 15.1 s, so thinking stays off. Seven tuning iterations are listed in D-029. The fixes first traded failures: the stricter prompt made the model cite scene results as "[1]", which only reading the answers revealed.
- **Still failing:** one scene answer citing "[1]", two mixed questions with an ambiguous place (the model reports the error instead of retrying), and two retrieval misses (thermal noise, GRD vs SLC).
- **Memory incident** (D-030): WSL froze for 28 minutes during an eval, because Ollama had turned mmap off and Gemma's runner held 6 GB. Since then every request asks for mmap, and WSL has 12 GB instead of 10 (raised by the user). The step's runs were made with a guard that stopped them if available RAM fell below 2 GiB.
- `make eval` reproduces the cited retrieval run exactly with the new Ollama option.

### Step 14

- **Plan approved by the user** on 2026-10-11, every item as recommended. Before it, the user asked for the `v0.1` tag on the step-12 commit (done, see "Current step") and set the rules for v0.2.
- **Verified first** (D-031): the MCP specification's current revision is 2026-07-28, which removed sessions and the `initialize` handshake; the Python SDK `mcp` 2.3.0 serves both eras on stdio (checked on the wire with a throwaway server); a buffered `print()` reached the protocol channel when the server exited, until `sys.stdout` was pointed at stderr.
- **Code:**
  - `mcp_server.py` (`make mcp`): the low-level SDK `Server` on stdio, the agent's schemas plus `confirm_place`, results as text plus `structuredContent` with an `outputSchema`, `isError` for tool errors and `-32602` for an unknown tool, one call at a time in a worker thread, logs on stderr.
  - `mcp_check.py` (`make mcp-check`): starts the server through the SDK's client, with the handshake and with per-request metadata; 19 checks.
  - `tools.py`: `SceneFinder` (scene searches, moved out of `Toolbox` unchanged), `SmallPlace`, `scene_line`. The agent's tool outputs are identical before and after (18 calls compared).
  - `.mcp.json` registers the server for Claude Code; `pyproject.toml` version 0.2.0, which the server reports and the User-Agent now uses; `mcp>=2.3,<3` added (21 packages).
- **Design** (D-032): nothing is kept between calls (passages numbered per result with section and URL; `confirm_place` for places under 1 km); `search_docs` takes an English query from the host's model and runs without Gemma. Measured before choosing: with Gemma's translations standing in for the host's query, 22 of 25 in the top 5 and MRR@5 0.810, as before, and the same through the protocol.
- **v0.2 runs:** `make eval` (`eval/results/2026-10-11T180312-v0.2.json`) is identical to the cited v0.1 run, chunk by chunk. `make eval-agent` (`…181107-agent-v0.2.json`): the checks counted 14 of 18, but one was an empty answer (`scenes-madrid-week`).
- **Found: the empty answer** (D-033). Gemma now ends its turn without text for the Madrid scenes question, every time, with the code of step 13 too, so this step did not cause it. The eval's citation check passed an answer that cites nothing. Chosen by the user at the review: the eval gains the check *answered*, and `make agent` says when the answer is empty; retrying in the loop is left for later. With the check: 13 of 18, held-out 5 of 6 (`eval/results/2026-10-12T143532-agent-v0.2-answered.json`). The step-13 runs have no empty answer, so their numbers stand.
- **Approved by the user** on 2026-10-12, every item as recommended: D-031 to D-033 accepted, version 0.2.0, `.mcp.json` in the repo, the v0.2 runs committed.
- **Test with a real host** (2026-10-14, D-031): Claude Code started the server from `.mcp.json` and answered four questions in 7 calls. It passes `structuredContent` to its model. For GRD vs SLC it searched twice and found the SLC section that `make ask` misses; for Doñana it searched again with the national park's name. The first connection after a reboot took 4.7 s (Claude Code's probe timed out while the server started from the cold disk, then it reconnected by itself); warm, the server starts in 1.0 s.
- **Pushed and tagged** on 2026-10-14: commit `4247db7` on `main`, annotated tag `v0.2`.

### Step 15

- **Plan approved by the user** on 2026-10-14, every item as recommended.
- **Verified first** (D-034): OpenSearch 3.9.0 is the newest stable image (Lucene 10.5.1); with the security plugin disabled it starts without an admin password; a new k-NN index defaults to Faiss, `ef_search` 100 and graphs even for small segments.
- **Code:**
  - `docker-compose.yml`: the `opensearch` service under a Compose profile (one node, security off, localhost only, 1 GB heap, its own volume). `make opensearch-up`, `make opensearch-index`; `make down` and `make clean` include it.
  - `opensearch.py`: raw REST with httpx; the index rebuilt from Postgres with the stored vectors (Faiss HNSW with pgvector's m 16, ef_construction 64, ef_search 40; BM25 on a custom English analyser with Postgres's stop words); a fingerprint of Postgres's chunk ids in `_meta`; vector, exact, BM25 and native hybrid searches; the RRF search pipeline.
  - `retrieval.py`: `opensearch=` in `retrieve`, method `hybrid-native`, `open_backend` (checks the combination and that the copy is in sync). `--backend pgvector|opensearch` in `make ask` and `make eval`; results record the backend.
- **Results** (D-035, D-036): vector search identical (the same 20 sections for all 25 questions, recall 1.000 against exact search); BM25 hit@5 0.840 / MRR@5 0.687 against 0.800 / 0.663 in SQL (0.760 / 0.605 with the built-in `english` analyser, whose short stop list keeps question words); the default hybrid search 0.880 / 0.810 on both, and the native hybrid query the same; the 4 questions whose top 5 changes get answers just as correct. OpenSearch's BM25 is 9 times faster (4.9 against 44.9 ms), pgvector's vector search 2 times faster (3.6 against 6.5 ms); OpenSearch uses 1.4–1.6 GiB of RAM and 36–64 s to start. pgvector stays the default.
- **Runs:** `eval/results/2026-10-14T214507-os-vector.json`, `…214621-os-lexical-en`, `…214725-os-hybrid`, `…214741-os-hybrid-native`. The analyser variants, recall, latency and answer comparisons ran as throwaway scripts.
- **Approved by the user** on 2026-10-14, every item as recommended: D-034 to D-036 accepted, Postgres's stop words in the OpenSearch analyser, pgvector stays the default, `--method hybrid-native` kept, the 4 runs committed, and the tag `v0.3`. The package version is now 0.3.0, which the MCP server reports.
- **Pushed and tagged** on 2026-10-14: commit `3c1f359` on `main`, annotated tag `v0.3`.

### Step 16

- **Plan approved by the user** on 2026-10-14, every item as recommended, with a criterion for making it the default set before measuring: hit@5 or MRR@5 higher, no question leaving the top 5, under 3 s more per question.
- **Verified first** (D-037): Ollama 0.40.1 has no reranking endpoint but returns up to 20 log-probabilities per token; PyTorch's current CUDA builds no longer support the GTX 1070 (Pascal); the `ggml-org` GGUF of Qwen3-Reranker loads only as an embedding model, the `mradermacher` Q8_0 as a language model.
- **Found: the RAM prompt cache** (D-038). Ollama's `llama-server` keeps every prompt in RAM, up to 8 GiB per model; the reranker's runner reached 8.8 GB and WSL had 1.6 GiB left. The user set `LLAMA_ARG_CACHE_RAM=0` in a systemd drop-in for Ollama; after it, the runner stays around 1 GiB and a question takes 1.2–2.0 s instead of 2–79 s.
- **Code:** `rerank.py` (prompt, score as log P(yes) − log P(no), `reorder`, `check_reranker`); `Ollama.next_token_logprobs`; `retrieve(..., rerank=)` and `DEFAULT_RERANK`; `--rerank` / `--no-rerank` in `make ask` and `make eval` (the results record the model, its digest, the candidates and the instruction); the agent and the MCP server pass the default; `make agent` and `make eval-agent` check that the reranker is pulled.
- **Results** (D-039): with an instruction for the domain, hit@1 0.840, hit@5 0.960 (24 of 25, the most reachable: `sigma0-calibration` has no correct chunk in the top 40), MRR@5 0.890, against 0.760 / 0.880 / 0.810; the card's instruction gave 0.600 / 0.920 / 0.708; the English translation as query 0.800 / 0.960 / 0.880; the top 40 0.840 / 0.920 / 0.880 in 3.3 s; a cap of 1 or 2 chunks per section changed nothing. Answers for the 23 questions whose top 5 changed: 43 of 46 cite a correct section against 40. Search time 1.7 s median (0.09 without). The agent: the same 13 of 18 checks, question by question. Against the reference Transformers code (60 pairs): rank correlation 0.989–1.000, same top 5. The README's thermal noise example lost its "why" section (5th → 8th); the README shows the new answer and says so.
- **Criterion met,** so the reranker is the default (`DEFAULT_RERANK = True`), for the agent and the MCP server too.
- **Runs:** `eval/results/2026-10-14T233134-rerank.json`, `…235006-agent-rerank.json`. The variants, the reference scores, the latency and the answer comparisons ran as throwaway scripts.
- **Approved by the user** on 2026-10-15, every item as recommended: D-037 to D-039 accepted, the reranker as the default for `make ask`, the agent and the MCP server, the README's example replaced by the reranked answer with a note on what it lost, the 2 runs committed, and the tag `v0.4`. The package version is now 0.4.0.

## Environment baseline (2026-10-08)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS on WSL2 (kernel 6.18), Windows 10 host (build 19045). WSL networking mode: NAT |
| CPU / RAM | 12 logical CPUs. WSL capped at 10 GB: 9.7 GiB visible, 8.6 GiB available, 3 GiB swap. Raised to 12 GB (11.7 GiB visible) in step 13 |
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
- **Long runs with Ollama:** watch `MemAvailable` while they run, and stop them below 2 GiB. If WSL freezes, look for "disabling mmap" and "discovery" lines in `journalctl -u ollama` before anything else, and avoid `nvidia-smi` while the system is unresponsive. Step 13 has the details.
- **stdout in an MCP server:** the SDK moves fd 1 to stderr while it serves, but text that a `print()` leaves in Python's buffer is written when the process exits, after fd 1 points at the protocol again. `mcp_server.py` points `sys.stdout` at stderr once the transport is open (D-031).
- **Empty agent answers:** Gemma can end a turn after a tool result without writing anything (`eval_count` 1). It happened for the Madrid scenes question on 2026-10-11, every time, and in none of the 72 earlier agent answers. The agent eval now fails empty answers (D-033).
- **OpenSearch:** the container takes 1.4–1.6 GiB of RAM with a 1 GB heap, and 36–64 s to become healthy from the hard disk. With Gemma, bge-m3, Postgres and OpenSearch all loaded, available RAM stayed at 7 GiB or more of 11.7 (step 15).
- **Ollama's RAM prompt cache:** since Ollama runs models in llama.cpp's `llama-server`, each runner may keep up to 8 GiB of prompt copies in RAM. `LLAMA_ARG_CACHE_RAM=0` in `/etc/systemd/system/ollama.service.d/cache-ram.conf` turns it off (step 16, D-038). After an Ollama upgrade, check that the server's log has no "cache state" lines.
- **Memory budget:** with Gemma, bge-m3 and the reranker loaded, the GPU uses 6.7 of 8 GiB (step 16). Earlier, with two models, WSL used 5.5 of 9.7 GiB of RAM. Postgres fits, but running other heavy workloads at the same time (such as a local Kubernetes cluster) could make RAM tight.

## Next

**Push step 16 and the `v0.4` tag** when the user says "súbelo". Then what to do next is open, and the user chooses. Candidates: new questions written without reading the documents (the 25 have chosen every retrieval change, the reranker's instruction included), the "sigma zero" miss, the smaller experiments listed in the README, asking the model again after an empty agent answer (D-033), retrying a rejected place in mixed agent questions, and new held-out agent questions.

The current `make ask` pipeline, for reference: translate the question (Gemma, temperature 0), vector search (bge-m3, original question) and BM25 full-text search (English question), reciprocal rank fusion, the top 20 reranked by Qwen3-Reranker-0.6B with the original question, top 5 passages to Gemma. Corpus: 5 SentiWiki pages and 6 PDFs, 568 chunks. Eval: hit@5 0.960, MRR@5 0.890 over 25 questions (`eval/results/2026-10-14T233134-rerank.json`); without the reranker 0.880 and 0.810 (`…113743-pdfs`).

## How to resume

1. `cd ~/sentinel1-rag && uv sync`
2. Check that Ollama is running and has the three models (Gemma, bge-m3 and `hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0`): `systemctl is-active ollama && ollama list`. `systemctl show ollama -p Environment` should list `LLAMA_ARG_CACHE_RAM=0`.
3. After a WSL restart, check that Ollama found the GPU: `journalctl -u ollama -b | grep "inference compute"` must show `library=CUDA`. If it shows `library=cpu`, read the CUDA libraries into the disk cache first (`cat /usr/local/lib/ollama/cuda_v12/* > /dev/null`, about 10 s cold), then run `sudo systemctl restart ollama` and check again. Without that read, three restarts in a row fell back to the CPU in step 13.
4. Start the database with `make up` (it does not start by itself after a restart). If `data/` is missing, rebuild everything with `make ingest && make index` (about 80 s and a minute); otherwise `make index` alone confirms there is nothing to do.
5. Try it: `make ask Q="¿Qué diferencia hay entre un producto GRD y uno SLC?"`. The first question after a while takes about a minute and a half while Gemma loads. `make eval` re-measures retrieval in about 55 s once the models are loaded (it translates the questions and reranks); `ARGS="--no-rerank"` takes about 15 s, and `ARGS="--method vector --no-rerank"` skips Gemma and the reranker.
6. Read this file and [DECISIONS.md](DECISIONS.md). The write-ups in [steps/](steps/) tell the story of each finished step in plain English.
7. The comparisons of generated answers (steps 10 to 12), the BM25 cross-check and the PDF checks (page anchors, token counts, heading fonts) ran as throwaway scripts in a session scratchpad, so they are not in the repo. So did the probes of step 14 (the MCP wire tests, the English-query measure and the comparison of the agent's tool outputs) the measurements of step 15 (analyser variants, HNSW recall, latency, answers) and those of step 16 (reranker variants from stored log-probabilities, the reference scores in a throwaway PyTorch environment, latency, answers). What they measured and their numbers are in DECISIONS.md, under "Verified" in D-022 to D-026, D-031, D-032 and D-034 to D-039.
8. `make agent`, `make eval-agent` and `make mcp-check` need the network (the Copernicus STAC API and Nominatim). Geocoded places are cached in `data/geocode.json`. `make mcp-check` takes about 10 s and does not load Gemma.
9. **OpenSearch (optional):** `make opensearch-up` (about a minute), then `make opensearch-index` (seconds) whenever `make index` has changed anything; `ARGS="--backend opensearch"` in `make ask` or `make eval`. `make down` stops it too.
10. **MCP with Claude Code:** `.mcp.json` registers the server; a new Claude Code session in this folder asks to approve it (`/mcp` shows its state). Postgres and Ollama must be running. The first start after a reboot is slow (about 5 s) and Claude Code's first probe may time out; it reconnects by itself.
