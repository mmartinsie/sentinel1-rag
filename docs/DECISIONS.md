# Decisions

One entry per decision: what we chose, the alternatives we weighed, why, and what we give up.
"Verified" lists the external facts checked before deciding (date of the check in brackets).

Status values: **Proposed** (waiting for review) · **Accepted** · **Superseded by D-xxx**.

---

## D-001: Run Ollama natively inside WSL2

- **Status:** Accepted (step 0, 2026-10-08)
- **Decision:** install Ollama inside the WSL2 Ubuntu distro (official Linux installer, systemd service) and use the GPU through WSL's CUDA passthrough. Clients reach it at `http://localhost:11434`.
- **Alternatives considered:**
  - *Ollama on Windows + WSL mirrored networking:* not available, because mirrored mode requires Windows 11 22H2+ and the host runs Windows 10.
  - *Ollama on Windows over the default NAT network:* would need `OLLAMA_HOST=0.0.0.0`, a firewall rule and the host IP from `ip route`. It exposes the API to the LAN, and models would land on a nearly full C: drive.
  - *Ollama in Docker (`ollama/ollama`):* needs the NVIDIA Container Toolkit in the WSL Docker engine (today it only has `runc`). That is one more layer to debug.
- **Why:** `nvidia-smi` inside WSL already sees the GPU, so the simplest option works. Ollama, Docker and Python share one environment and talk over `localhost`.
- **Trade-offs:** installing needs `sudo`. Ollama competes for WSL's 10 GB RAM allowance while a model is loaded, although most weights live in VRAM.
- **Verified [2026-10-08]:**
  - `nvidia-smi` in WSL shows a GTX 1070 with 8192 MiB, compute capability 6.1, Windows driver 582.28 (CUDA 13.0). About 7.5 GiB is free; the desktop uses about 420 MiB.
  - Ollama GPU docs: NVIDIA compute capability 5.0+ is supported, and CC 5.0–6.2 needs driver ≥ 570. The GTX 1070 is listed.
  - Microsoft WSL networking docs: mirrored mode requires Windows 11 22H2 or later.
  - Ollama is not installed (neither in WSL nor on Windows), and port 11434 is free.
- **Verified in step 1 [2026-10-08]:**
  - Ollama 0.40.1 runs as a systemd service. On WSL2 the installer stops before the driver-installation part (lines 256–262 of `install.sh`).
  - The server log shows the GTX 1070 detected as CUDA compute 6.1 and served by the bundled `cuda_v12` runner. The `cuda_v13` runner shipped alongside it no longer supports Pascal.

## D-002: LLM is `gemma4:e4b-it-qat`

- **Status:** Accepted (step 0, 2026-10-08). GPU fit confirmed in step 1.
- **Decision:** `gemma4:e4b-it-qat` with `num_ctx = 8192` and `temperature = 0.2`, both set per request. Fallback: `gemma4:e2b-it-qat`.
- **Alternatives considered:**
  - *`gemma4:e4b` (default tag):* Q4_K_M post-training quantization, 6.58 GB including a speculative-decoding draft model. Bigger, and not QAT.
  - *`gemma4:12b`:* 7.7–8.0 GB according to the Ollama library. It cannot fit in about 7.5 GiB of free VRAM together with its KV cache, so layers would spill to the CPU.
  - *`gemma4:e2b-it-qat`:* 4.34 GB. Kept as the fallback if E4B does not fit fully on the GPU.
- **Why:**
  - It is the largest Gemma 4 variant expected to fit entirely in VRAM. Partial CPU offload slows generation several times.
  - QAT (quantization-aware training) trains the model to tolerate 4-bit weights, so Q4_0 loses less quality than post-training quantization, at a smaller size.
  - 8192 tokens is about 2.5× the expected prompt: 5 chunks × ≤ 500 tokens, plus instructions and the question, is roughly 3k tokens.
  - A low temperature gives more repeatable answers that stay closer to the context.
- **Trade-offs:** a model with about 4.5B effective parameters reasons and writes worse than 12B+ models. The vision and audio encoders (about 1 GB) are loaded onto the GPU even though we only send text. It requires Ollama ≥ 0.30.5.
- **Verified [2026-10-08]** (Ollama registry manifests and GGUF header):
  - `e4b-it-qat`: Q4_0, 7.5B total parameters, 5.16 GB of weights plus a 0.99 GB projector, 6.15 GB in total. Requires Ollama ≥ 0.30.5.
  - `e2b-it-qat`: Q4_0, 4.34 GB in total.
  - Default sampling parameters are temperature 1, top_k 64 and top_p 0.95, so temperature must be overridden.
  - Architecture: 42 layers, 18 of which reuse another layer's KV cache, and sliding-window attention (512 tokens) on most layers, with a maximum context of 128K. The KV cache at 8K context should therefore be small.
- **Verified in step 1 [2026-10-08]** (`ollama ps`, `/api/ps`, llama.cpp load log, `nvidia-smi`):
  - 100% GPU with `num_ctx = 8192`: all 43 layers are offloaded.
  - VRAM: 2,696 MiB of weights, a 168 MiB KV cache at 8192 tokens (small, as expected from the sliding window and the shared KV layers) and a 110 MiB compute buffer. On top of that come the vision and audio encoders (up to about 1.1 GiB). With Gemma loaded, `nvidia-smi` reports 4.6 GiB used out of 8 GiB.
  - System RAM: llama.cpp keeps 2,730 MiB of weights on the CPU, namely the input embedding tables, which are large in E4B. The runner process uses 3.3 GiB.
  - Speed: about 41 tokens/s generating and about 835 tokens/s reading the prompt (a 3.7k-token prompt takes 4.5 s).
  - The first load takes about 70 s, because the WSL virtual disk lives on a hard drive (HDD).
  - With `think: false` the response contains no thinking.
  - On this GPU, Ollama's default context is 4096, which is why `num_ctx` has to be sent with every request.
  - The fallback model was not needed.

## D-003: Embeddings with `bge-m3` via Ollama

- **Status:** Accepted (step 0, 2026-10-08). Dimension confirmed in step 1.
- **Decision:** `bge-m3` dense embeddings (1024 dimensions), requested in batches through Ollama's `/api/embed`.
- **Alternatives considered:**
  - *English-only embedding models* (e.g. `nomic-embed-text`, `mxbai-embed-large`): smaller and faster, but Spanish questions would not match English passages well.
  - *Other multilingual models* (multilingual-e5-large, Qwen3-Embedding).
  - *Hosted embedding APIs:* neither local nor free.
- **Why:**
  - It is multilingual, so a Spanish question lands close to the English passage that answers it (cross-lingual retrieval).
  - The 8192-token context covers any chunk with room to spare.
  - Queries need no instruction prefix.
  - It is a well-known retrieval baseline.
- **Trade-offs:**
  - 567M parameters in F16 (1.16 GB) make it slower than small models. That does not matter for a few hundred chunks.
  - Ollama only returns the dense vector, not BGE-M3's sparse (lexical) or multi-vector (ColBERT) outputs.
- **Verified [2026-10-08]:**
  - GGUF metadata in the Ollama registry: `bert.embedding_length = 1024`, `bert.context_length = 8192`, `bert.pooling_type = 2` (CLS), F16, 567M parameters.
  - Model card: "the BGE-M3 model no longer requires adding instructions to the queries".
  - Ollama API docs: `/api/embed` accepts an array `input` and returns `prompt_eval_count`.
- **Verified in step 1 [2026-10-08]:**
  - `/api/embed` returns 1024-dimensional vectors with an L2 norm of 1.0. Because they are normalized, ranking by cosine distance and by inner product gives the same order.
  - Cross-lingual check: a Spanish question about GRD vs SLC scored 0.631 against an English passage that answers it, 0.271 against an unrelated English sentence and 0.220 against an unrelated Spanish sentence.
  - Ollama loads bge-m3 with its default 4096-token context, far above our chunk size.
  - Gemma and bge-m3 fit in VRAM together (5.4 GiB used out of 8 GiB), so embedding the question does not evict Gemma.

## D-004: Vector store is PostgreSQL 18 + pgvector 0.8.7 in Docker Compose

- **Status:** Accepted (step 0, 2026-10-08)
- **Decision:** image pinned to `pgvector/pgvector:0.8.7-pg18-trixie`, port 5432, and a named volume mounted at `/var/lib/postgresql`.
- **Alternatives considered:**
  - *OpenSearch k-NN:* listed as an extension in the README.
  - *Dedicated vector databases* (Qdrant, Chroma, Milvus).
  - *In-process exact search* (numpy or FAISS) with no database at all.
- **Why:**
  - It reuses existing Postgres knowledge.
  - Document metadata and vectors live in one place, with transactions and joins.
  - pgvector offers exact search and HNSW/IVFFlat indexes with cosine, L2 and inner-product distances.
  - Pinning the pgvector version, the Postgres major and the Debian release keeps the environment reproducible. Trixie is the current Debian stable.
- **Trade-offs:** fewer vector-native features than dedicated databases (built-in hybrid ranking, vector quantization, sharding), and the SQL is ours to write.
- **Verified [2026-10-08]** (Docker Hub tags and image config read from the registry):
  - The latest pgvector release is 0.8.7, and the newest Postgres major with pgvector images is 18 (PG 18.6). There are no `pg19` tags.
  - `0.8.7-pg18` and `0.8.7-pg18-bookworm` are the same image (same digest). `-trixie` is the Debian 13 build.
  - The image sets `PGDATA=/var/lib/postgresql/18/docker` and declares `VOLUME /var/lib/postgresql`. The volume therefore goes on `/var/lib/postgresql`, not on `/var/lib/postgresql/data` as in tutorials written before PG 18.
  - Port 5432 is free on both WSL and Windows.

## D-005: Schema applied by `docker-entrypoint-initdb.d`, no migration tool

- **Status:** Accepted (step 0, 2026-10-08)
- **Decision:** a plain `schema.sql`, mounted read-only into `/docker-entrypoint-initdb.d`.
- **Alternatives considered:** Alembic, yoyo-migrations, or `CREATE … IF NOT EXISTS` run by the application at startup.
- **Why:** the schema is tiny, there is a single developer, and the database can be rebuilt entirely from the corpus.
- **Trade-offs:** init scripts only run when the data volume is empty. Any schema change means `make clean && make up` and indexing again, and there is no migration history.

## D-006: HNSW index, even though exact search would be enough

- **Status:** Accepted (step 0, 2026-10-08)
- **Decision:** `CREATE INDEX … USING hnsw (embedding vector_cosine_ops)`.
- **Alternatives considered:** no index (exact k-NN through a sequential scan), or IVFFlat.
- **Why:** the goal is to learn how an approximate nearest-neighbour index works and be able to explain it. At this scale (a few hundred chunks) exact search takes milliseconds, so **the index is not there for performance**, and the docs say so.
- **Trade-offs:** results are approximate (recall can drop below 100 %), and inserts are slower and use more memory. With so few rows the planner may choose a sequential scan anyway, so showing the index at work needs `EXPLAIN` with `enable_seqscan = off`.
- **Verified in step 4 [2026-10-09]** (195 chunks):
  - With default settings the planner picks a sequential scan plus a top-N sort, which runs in 1.1 ms.
  - With `enable_seqscan = off` the plan becomes `Index Scan using chunks_embedding_hnsw`: 0.5 ms, and 254 buffers read instead of 717. Both times are negligible.
  - Recall@5 of HNSW against exact search, using the 195 stored vectors as queries, is 1.000 with `hnsw.ef_search` set to 5, 10 and 40 (the default). Every query returns the same top 5.
  - The index takes 1.6 MB, built with pgvector's defaults (`m = 16`, `ef_construction = 64`).

## D-007: Python 3.12 + uv, src layout, minimal dependencies, raw HTTP to Ollama

- **Status:** Accepted (step 0, 2026-10-08)
- **Decision:**
  - Python ≥ 3.12 managed with uv: `uv_build` backend, package in `src/sentinel1_rag/`, `uv.lock` committed.
  - Dependencies: `httpx`, `beautifulsoup4`, `psycopg[binary]`, `pgvector`, `pyyaml`.
  - Ollama is called through its REST API with `httpx`, without an SDK.
- **Alternatives considered:** the Ollama Python SDK; LangChain or LlamaIndex; Poetry or pip-tools; a flat layout.
- **Why:**
  - Every request and response sent to Ollama is visible and can be explained.
  - There are few moving parts, and the lockfile makes the environment reproducible.
  - The src layout prevents importing the package by accident from the working directory.
- **Trade-offs:** we write batching, retries and error handling ourselves instead of getting them from a framework.
- **Verified [2026-10-08]:**
  - uv 0.12.21 with CPython 3.12.3.
  - All dependencies resolve and import: httpx 0.28.1, beautifulsoup4 4.15.0, psycopg 3.3.6, pgvector 0.5.0, PyYAML 6.0.3.
  - pgvector-python 0.5.0 does not depend on numpy, and `pgvector.Vector` accepts plain lists (stored as float32).

## D-008: Database schema, adjusted from the brief

- **Status:** Accepted (step 2, 2026-10-08)
- **Decision:** the brief's two tables, with four changes:
  1. `chunks.anchor` holds the heading id used in citation links (`documents.url` + `#` + `anchor`). It is nullable: `NULL` means the top of the page.
  2. `documents.index_config` records the chunking settings and the embedding model used to build the page's chunks. A page is indexed again when its `content_hash` (the page text) or its `index_config` changes.
  3. `chunks.embedding` is `NOT NULL`: a chunk is always inserted together with its embedding.
  4. `UNIQUE (document_id, chunk_index)` guards against storing a page's chunks twice.

  The HNSW index is named (`chunks_embedding_hnsw`) so it is easy to find in `EXPLAIN` output.
- **Alternatives considered:**
  - *Storing the full citation URL in every chunk:* the query gets simpler, but the page URL is repeated in every chunk.
  - *A single hash covering both the text and the settings:* one column fewer, but it hides *why* a page was indexed again.
  - *A nullable embedding with a two-phase pipeline* (ingest inserts the chunks, index fills in the vectors later): a run could resume halfway, but half-built rows would exist and every search would have to filter them out.
- **Why:**
  - Citations need the anchor.
  - Re-indexing must react to every input that shapes the chunks and vectors, not only to the page text. Vectors from two different models or chunkings must never be mixed in one index, because their distances would not be comparable.
  - Every row in `chunks` can be searched.
- **Trade-offs:**
  - This fixes the shape of the pipeline: `make ingest` writes its chunks to `data/` and never touches the database, and `make index` is the only step that writes rows, one page per transaction.
  - If embedding fails halfway through a page, that page is rolled back and retried rather than saved half-indexed.
- **Verified in step 2 [2026-10-08]:**
  - `\d documents` and `\d chunks` show the expected columns and constraints, and the index `chunks_embedding_hnsw` uses `hnsw (embedding vector_cosine_ops)`.
  - A 3-dimensional vector is rejected ("expected 1024 dimensions, not 3"), and so is a chunk without an embedding (not-null violation).
  - Nearest-neighbour smoke test: querying with e1 against e1, e1+e2 and e3 returns cosine distances of 0.000, 0.293 and 1.000, as the maths predicts.

## D-009: Local database setup with Docker Compose

- **Status:** Accepted (step 2, 2026-10-08)
- **Decision:**
  - A single `db` service, with the port published on `127.0.0.1` only.
  - Local credentials (`rag` / `rag`, database `rag`) written as overridable defaults (`${POSTGRES_USER:-rag}`), so a clean clone works without a `.env` file.
  - A `pg_isready` healthcheck over TCP. `make up` runs `docker compose up -d --wait`.
  - An explicit Compose project name, `sentinel1-rag`, so containers and volumes never collide with other projects. `make clean` (`docker compose down -v`) therefore only deletes this project's volume.
- **Alternatives considered:**
  - *A required `.env` file with an `.env.example`:* one more setup step for a local-only database.
  - *Publishing the port on all interfaces:* the database would be reachable from the local network.
  - *A healthcheck over the Unix socket:* it can report "ready" before the schema exists.
  - *A fixed `sleep` in `make up`:* either slow or flaky.
- **Why:** it works from a clean clone with no setup, the defaults are safe, and readiness is deterministic.
- **Trade-offs:** a known default password lives in the repo. That is acceptable only because the port is bound to localhost and the data is public documentation. There is no TLS.
- **Verified in step 2 [2026-10-08]:**
  - `make up` reaches *healthy* in about 11 s, and `ss` shows the listener on `127.0.0.1:5432` only.
  - The image's `docker-entrypoint.sh` (line 297) starts the temporary init server with `listen_addresses=''`, so the TCP check cannot pass during init.
  - Data survives `make down` + `make up`. After `make clean`, the volume is gone and the next `make up` applies `schema.sql` again.
  - From Python, psycopg 3.3.6 and pgvector 0.5.0 connect and compute the same cosine distance (0.293).

## D-010: The corpus is the Sentinel-1 subtree of the SentiWiki (5 pages)

- **Status:** Accepted (step 3, 2026-10-09)
- **Decision:** `sources.yaml` lists the landing page and the 4 chapters (`s1-mission`, `s1-products`, `s1-processing`, `s1-applications`). Those are exactly the pages under "Sentinel-1" in the site's page tree.
- **Alternatives considered:**
  - *"POD in details"* (`/web/precise-orbit-determination`): about 7.3k words, but shared across missions (60 mentions of Sentinel-1 against 47 of S2/S3/S5P), and `s1-mission` already has a section on POD.
  - *The Glossary:* about 6.5k words spread over 184 tiny entries, for every mission. It could help with acronyms.
  - *"SAFE Format":* about 190 words, with no mention of Sentinel-1.
  - *The PDFs in the document library:* an extension (see the README).
- **Why:** the subtree is exactly "the Sentinel-1 SentiWiki". Pages shared across missions add facts about other satellites that retrieval would have to filter out. Whether a candidate helps can be measured in step 6.
- **Trade-offs:** acronyms are only defined where the Sentinel-1 pages define them, and orbit files are only covered by the POD section of `s1-mission`.
- **Verified [2026-10-08]:**
  - `/web/__pagetree.json` lists exactly these 5 pages under Sentinel-1.
  - The sizes and mission mentions of the candidate pages were measured on the live pages.

## D-011: Polite downloader with a local cache

- **Status:** Accepted (step 3, 2026-10-09)
- **Decision:** a small client of our own, built on `httpx`:
  - A User-Agent that names the project and links to the repository.
  - `robots.txt` is read once per site, before the first page: 401/403 disallow everything, 404 allows everything, and 5xx stops the run.
  - At least 1 s between requests, `robots.txt` included.
  - Every page is cached in `data/raw/<page>.html`, with its URL and download time in `<page>.json`.
  - A cached page is never downloaded again. To refresh it, delete the file or run `make clean`.
- **Alternatives considered:** `wget` or `curl` called from the Makefile; Scrapy; conditional requests (ETag / If-Modified-Since) to refresh automatically.
- **Why:** each rule is a few visible lines, and once the pages are cached the ingest runs offline and gives the same output every time.
- **Trade-offs:** changes on the site are only picked up after deleting the cache. There are no retries, so one failing page stops the run.
- **Verified [2026-10-08]:**
  - The first run made 6 requests (`robots.txt` and 5 pages) in about 8 s.
  - The second run made 0 requests and wrote byte-identical chunk files.

## D-012: Text extraction rules

- **Status:** Accepted (step 3, 2026-10-09)
- **Decision:**
  - **Content and title:** the content comes from `article#content section.article-body`, and the page title from its `h1`. The "On this Page" table of contents lives in a separate `nav.toc`, so it is left out.
  - **Noise:** copy-link buttons, icons, images and scripts are removed.
  - **Sections:** every heading from h2 to h6 opens a section. Its path starts with the page title, and its anchor is the heading `id`, the same one the page uses in its own "copy link" URL.
  - **Paragraphs and lists:** one block per paragraph, and lists with one line per item.
  - **Tables:** one line per row, with cells joined by `" | "`. We don't try to detect headers, because some tables mark them with `<th>` and others with plain cells.
  - **Figures and expandable blocks:** figures keep only their caption, and expandable blocks keep their title and body.
  - **Reference markers:** the wiki's numeric markers (`[1]`, `[2, 3]`) are removed so that they don't clash with the `[n]` citations of our answers.
- **Alternatives considered:**
  - Generic boilerplate removers (trafilatura, readability-lxml) or HTML-to-Markdown converters: extra dependencies and less control.
  - Tables as "header: value" pairs: they need header detection, and they are wrong when the first row is not a header.
  - Keeping the reference markers.
- **Why:** the SentiWiki layout is regular, so an extractor of about 100 lines stays predictable and explainable. The 44 tables hold key facts (resolutions, beam parameters, naming conventions) and must survive extraction.
- **Trade-offs:**
  - The extractor is tied to this site's layout; if the layout changes, it fails loudly with "unexpected page layout".
  - Joining table cells with `|` loses which column an empty cell belonged to.
  - The link between a reference marker and its bibliography entry is lost.
- **Verified [2026-10-08]:**
  - No table-of-contents, button or image-file text appears in the chunks, and no numeric reference marker is left.
  - Section paths and anchors match the site, e.g. `S1 Mission > Acquisition Modes > Interferometric Wide Swath` with `#Interferometric-Wide-Swath`.

## D-013: Chunking: sections first, balanced splits, calibrated token estimate

- **Status:** Accepted (step 3, 2026-10-09)
- **Decision:**
  - **Short sections:** a section of up to 500 tokens is one chunk.
  - **Long sections:** a longer section is cut into chunks of similar size, up to 400 tokens each. Cuts fall between sentences, table rows or list items, and a table or list that fits in 400 tokens stays whole.
  - **Overlap:** each chunk after a cut repeats up to 60 tokens (15 %) from the end of the previous one.
  - **Token estimate:** tokens are estimated as words × 1.68, a ratio measured on this corpus with bge-m3.
  - **Output:** each page goes to `data/chunks/<page>.json`, together with its `content_hash`, `fetched_at` and chunking settings.
  - **Prefix:** the section path (which starts with the page title) is the "Page > Section" prefix that step 4 puts in front of each chunk before embedding it.
- **Alternatives considered:**
  - *Fixed-size sliding windows:* they ignore the document structure.
  - *Exact counts with the bge-m3 tokenizer:* an extra dependency plus a model download.
  - *Counting characters instead of words:* worse on this corpus.
  - *Merging small sections, or semantic chunking:* boundaries placed where the embeddings change.
- **Why:**
  - Sections are the natural units of topic.
  - Balanced splits avoid tiny leftover chunks.
  - The estimate is unbiased without adding a dependency.
  - Whole tables keep each number next to its row and column names.
- **Trade-offs:**
  - **Uneven estimate:** it has a spread of 38 tokens, and chunks dominated by tables are under-estimated by up to ~40 %. As a result, 3 chunks exceed 500 real tokens (the largest has 595). That is harmless here: Ollama runs bge-m3 with a 4096-token context.
  - **Gaps in the overlap:** 19 of the 106 cuts carry no overlap, because the previous chunk ends with a whole table or with a sentence longer than 60 tokens.
  - **Small chunks:** short sections stay short, and 15 chunks have fewer than 100 tokens. Merging them is a candidate experiment for step 6.
- **Verified [2026-10-08]** (every chunk sent through bge-m3, reading Ollama's `prompt_eval_count`):
  - **Calibration:**
    - The first guess of 1.35 tokens per word under-estimated chunks by 62 tokens on average, which let real chunks reach 593 tokens.
    - The corpus measures 1.68 tokens per word (prose 1.61, tables 1.86), plus 2 special tokens per input.
    - Counting words predicts the size better than counting characters (error spread 12 % vs 16 %).
  - **Final result:**
    - 195 chunks from 89 sections, of which 32 were split.
    - Real tokens: mean 268, median 273, minimum 20, maximum 595.
    - Distribution: <100: 15, 100–199: 31, 200–299: 71, 300–399: 62, 400–499: 13, ≥500: 3.
    - 43 of the 44 tables stay whole in one chunk.
    - 87 of the 106 cuts carry overlap.
    - The output is deterministic: a second run writes byte-identical files.

## D-014: Indexing: per-page transactions, skip unchanged pages, batches of 32

- **Status:** Accepted (step 4, 2026-10-09)
- **Decision:**
  - **Input:** `make index` reads `data/chunks/*.json`. The text embedded for each chunk is its section path, a blank line and its content (`EMBED_TEMPLATE`).
  - **Embedding requests:** sent to Ollama's `/api/embed` in batches of 32, with `truncate: false`.
  - **Skip check:** a page is skipped when its stored `content_hash` and `index_config` both match. `index_config` is a JSON document with the chunking settings, the embedding model's name and digest, and the template.
  - **Re-indexing a page:**
    1. Compute all its embeddings first, outside any transaction.
    2. Then, in a single transaction, delete the document (its chunks go with it, through `ON DELETE CASCADE`) and insert the document and its chunks with `executemany`.
  - **Removed pages:** documents whose page no longer appears in `data/chunks/` are deleted.
  - **Connection:** it runs in autocommit mode, so that `conn.transaction()` issues a real `BEGIN`/`COMMIT` instead of a savepoint.
- **Alternatives considered:**
  - *Updating chunks in place* (upserts): more SQL, and leftover rows when a page gets shorter.
  - *One transaction for the whole run:* a failure on the last page would undo all the others, and the locks would be held longer.
  - *Embedding inside the transaction:* the transaction would stay open while waiting on Ollama.
  - *Other batch sizes:* one chunk per request is the slowest, and one request for everything is no faster and fails as a whole.
  - *`COPY` instead of `INSERT`:* faster at scale, unnecessary for about 200 rows.
  - *The model name without its digest:* a model pulled again with new weights would go unnoticed.
- **Why:** each page is always either fully indexed with one recipe or not indexed at all. Re-runs cost nothing, and every input that shapes a vector is tracked.
- **Trade-offs:**
  - The unit of change is the page: one changed paragraph re-embeds the whole page (at most 63 chunks, about 5 s).
  - Document ids change on every re-index; nothing outside the database refers to them.
  - Ollama has to be reachable even when every page is skipped, because the run asks it for the model digest.
- **Verified [2026-10-09]:**
  - **Batch size**, timing all 195 chunks:
    - 1 per request: 20.8 s (107 ms per chunk). 8, 32 or all at once: about 15.8 s (81 ms per chunk).
    - The vectors are identical whatever the batch size (largest difference: 0).
  - **First run:** 5 pages and 195 chunks in 16.8 s.
  - **Second run:** all 5 pages skipped and no embeddings computed. A fingerprint of every row (ids, text and vectors) is identical before and after.
  - **Change detection:**
    - An altered `content_hash` re-indexes only that page, and an altered stored `index_config` only that page.
    - A removed page file deletes its document and its 50 chunks.
    - Restoring the files brings the database back to 5 documents and 195 chunks, and one more run skips everything.
  - **Retrieval sanity check:** the brief's example question, in Spanish, returns 5 chunks that all mention GRD and SLC, at cosine distances from 0.356 to 0.455. The first is the Level-1 list "Single Look Complex (SLC) products / Ground Range Detected (GRD) products".
