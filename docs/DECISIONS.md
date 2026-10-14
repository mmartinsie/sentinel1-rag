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
  - 8192 tokens is about 2.5× the expected prompt: 5 chunks × ≤ 500 tokens, plus instructions and the question, is roughly 3k tokens. With the top 10 adopted later (D-018), the measured prompts are 2,150–3,804 tokens, still within budget.
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
- **Seen in step 7 [2026-10-10]:** the reverse is not guaranteed. Twice, loading Gemma while bge-m3 was loaded made Ollama unload bge-m3 first (log: `predicted="5.6 GiB"`, `gpu_free="6.4 GiB"`, `system_free="3.9 GiB"`, "evicting"), and the next question loaded bge-m3 again next to Gemma. The cause was not investigated; the cost is reloading the smaller model.

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

- **Status:** Accepted (step 0, 2026-10-08). Step 12 added `pdfplumber` to read the PDFs (D-025).
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

  The HNSW index is named (`chunks_embedding_hnsw`) so it is easy to find in `EXPLAIN` output. Step 8 added the `tsv` column and its GIN index for full-text search (D-019).
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

- **Status:** Accepted (step 3, 2026-10-09). Since D-024 the corpus also holds six technical PDFs from the Sentinel-1 document library.
- **Decision:** `sources.yaml` lists the landing page and the 4 chapters (`s1-mission`, `s1-products`, `s1-processing`, `s1-applications`). Those are exactly the pages under "Sentinel-1" in the site's page tree.
- **Alternatives considered:**
  - *"POD in details"* (`/web/precise-orbit-determination`): about 7.3k words, but shared across missions (60 mentions of Sentinel-1 against 47 of S2/S3/S5P), and `s1-mission` already has a section on POD.
  - *The Glossary:* about 6.5k words spread over 184 tiny entries, for every mission. It could help with acronyms.
  - *"SAFE Format":* about 190 words, with no mention of Sentinel-1.
  - *The PDFs in the document library:* an extension (see the README).
- **Why:** the subtree is exactly "the Sentinel-1 SentiWiki". Pages shared across missions add facts about other satellites that retrieval would have to filter out. Whether a candidate helps can be measured with `make eval`; step 6 did not try it, because the user chose the top-k change instead (D-018).
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

- **Status:** Accepted (step 3, 2026-10-09). These rules are for the SentiWiki pages; the PDFs have their own extractor (D-025).
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
  - **Small chunks:** short sections stay short, and 15 chunks have fewer than 100 tokens. Merging them was a candidate experiment for step 6 that was not run (D-018); the README lists it among the experiments not tried.
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

## D-015: Retrieval: the raw question, the 5 closest chunks, top-k before the join

- **Status:** Accepted (step 5, 2026-10-09). The top-k is now 10: see D-018.
- **Decision:**
  - **Query vector:** the question is embedded as it is, with bge-m3 and no prefix (D-003).
  - **Top-k:** the 5 chunks with the smallest cosine distance (`<=>`), closest first. There is no distance threshold: deciding that the answer is missing is left to the model.
  - **SQL shape:** a subquery picks the top 5 from `chunks` alone; `documents` is joined afterwards for the URL. The citation URL is `url#anchor`, or the bare URL when the anchor is `NULL`.
  - **Module:** retrieval lives in `retrieval.py`, apart from `ask.py`, so that the step 6 eval measures exactly the retrieval that `make ask` uses.
- **Alternatives considered:**
  - *The join and the `ORDER BY ... LIMIT` in one query:* the planner never uses the HNSW index for it.
  - *A distance threshold* to answer "not found" without calling the model: the distances of answerable and unanswerable questions overlap (see below).
  - *Translating or rewriting the question before embedding it:* one more model call, and bge-m3 already matches Spanish questions to English text. Measured in step 9 (D-021): MRR@10 0.782 instead of 0.766, with the same hit@k.
  - *Another k:* the brief fixes 5; step 6 measures whether it is enough.
- **Why:** the subquery keeps the index usable as the table grows. Five chunks of up to ~600 tokens fit easily in Gemma's context.
- **Trade-offs:**
  - The model always gets 5 chunks, even when none of them is relevant, so it has to recognise that itself.
  - Chunks cut from the same long section can fill several places. For the brief's example question, 3 of the 5 come from `S1 Products > Level-1 Products`, while the sections that explain the difference best (`S1 Processing > L1 Algorithms > Single Look Complex (SLC)` and `> Ground Range Detected (GRD)`) rank 9th and 11th.
- **Verified [2026-10-09]:**
  - **Query plans** (`EXPLAIN ANALYZE` with `enable_seqscan = off`):
    - Join inside: the planner reads `chunks` through its B-tree index, joins and sorts all 195 rows. Writing the distance expression in the `ORDER BY` instead of its alias changes nothing.
    - Subquery: `Index Scan using chunks_embedding_hnsw`, same top 5, 0.3–0.4 ms and 258 buffers once warm, against 1.3–2.0 ms and 692 buffers for the exact search.
    - With default settings the planner still prefers the exact search at this size (D-006).
  - **Distances** of the 5 retrieved chunks:

    | Question | Distances | Answer in the corpus |
    |---|---|---|
    | ¿Qué diferencia hay entre un producto GRD y uno SLC? | 0.356–0.455 | yes |
    | What is the repeat cycle of Sentinel-1…? | 0.324–0.421 | yes |
    | ¿Cuánto costó construir y lanzar la misión Sentinel-1? | 0.424–0.481 | no |

    The unanswerable question's closest chunk (0.424) is closer than the 4th and 5th chunks of the first question, so no threshold separates the two cases.

## D-016: Prompt and answer: instructions in the system turn, numbered passages, citations resolved in code

- **Status:** Accepted (step 5, 2026-10-09). Since D-018 the prompt carried 10 passages instead of 5, and since D-022 it carries 5 again, from hybrid search. Since D-024 the system turn says the passages come from "the Sentinel-1 documentation: SentiWiki pages and technical documents" instead of "the SentiWiki documentation".
- **Decision:**
  - **Messages:** the system turn holds the instructions. The user turn holds `Passages:`, then each chunk as `[n] <section path>` followed by its text (closest first), then `Question: <question>`. URLs are not in the prompt.
  - **Instructions:** use only facts from the passages; cite the supporting passages after each sentence, as `[2]` or `[1][3]`; if the answer is not there, say only that it could not be found, and if it is partly there, answer that part and say what is missing; write in the language of the question, keeping acronyms such as GRD or SLC; be concise.
  - **Generation:** `/api/chat`, not streamed, with `think: false`, `num_ctx: 8192`, `temperature: 0.2` (D-002) and `num_predict: 1024`, a cap for an answer that never ends.
  - **Citations:** the code finds `[n]` and `[n, m]` in the answer and maps each number back to its chunk. The source list shows only the cited chunks, one entry per section (`[1][2][4] S1 Products > Level-1 Products` and its URL). It says "none cited" when there are none, and warns about numbers that match no passage.
  - **Output:** the answer, the sources and a line with the token counts and the time, including how long loading the model took. `--show-context` first prints the retrieved chunks with their distances. `make ask` reads the question from the environment (`"$$Q"`), so quotes and backticks in it are safe.
- **Alternatives considered:**
  - *Everything in the user turn:* what Gemma 3 needed, since it had no system turn. Gemma 4 has one.
  - *Listing all 5 retrieved chunks as sources:* it would show sources that the answer does not use, even next to a "not found" answer.
  - *Letting the model write the source list:* it could invent sections or URLs.
  - *Streaming the answer:* it appears sooner, but needs more code; a warm answer takes 1–3 s.
  - *Few-shot examples or a fixed "not found" sentence:* not needed in the tests.
  - *"Answer in the language of the question, even though the passages are in English":* the first wording, rejected after measuring it (below).
- **Why:**
  - The model only has to write text and numbers. The section and URL of every source come from the database, so a citation can point to a wrong passage but never to a place that doesn't exist.
  - Passages before the question keep the question as the last thing the model reads.
- **Trade-offs:**
  - Answers vary a little between runs (no fixed seed).
  - A 4.5B model decides when the answer is missing; it was tested on one unanswerable question here.
  - Answers are only as good as the 5 passages: the GRD/SLC answer is correct but shallow, because the passages that explain the difference best were not retrieved (D-015). With the top 10 (D-018), the SLC section reaches the model.
  - Small formatting quirks: once `$180^\circ$` (LaTeX) instead of 180°, and `[1, 5]` instead of `[1][5]` (the parser accepts both).
- **Verified [2026-10-09]:**
  - **Template:** Ollama's `_debug_render_only` shows Gemma 4 receiving `<|turn>system ... <turn|>` before the user turn, so the system turn is native.
  - **Language of the answer**, 5 runs per question with the same passages:

    | Wording | Spanish questions (2) | English questions (2) |
    |---|---|---|
    | "…even though the passages are in English" | 10/10 in Spanish | 2/10 in English, 8 in Spanish |
    | "Write the answer in the same language as the question" (chosen) | 10/10 | 10/10 |
    | the same, plus a reminder after the question | 10/10 | 10/10 |

    Before the change, one English question had also been answered in Portuguese.
  - **The 3 test questions**, 3 runs each with the final prompt:
    - GRD vs SLC (Spanish): Spanish, cites [1][2][4] every time, and each cited sentence matches its passage.
    - Repeat cycle (English): English, "12-day repeat cycle, 175 orbits per cycle" and "six-day repeat cycle" with two satellites, citing the Orbit section [1].
    - Mission cost (Spanish, not in the corpus): "No se encontró información…" every time, with no citation and no number.
  - **Sizes and times:** prompts of 1,272–1,792 tokens (of 8,192), answers of 21–108 tokens, 0.6–2.6 s with the model loaded.
  - **CLI:** a question with double quotes, an apostrophe and backticks reaches Python unchanged. An empty question prints the usage line. With Ollama or Postgres down, the run ends with the same clear errors as `make index`.

## D-017: Retrieval eval: section-level labels, the real retrieval, MRR cut at the top k

- **Status:** Accepted (step 6, 2026-10-10). Since D-022 the default retrieval translates each question with Gemma, so the eval loads it. Since D-026 there are 25 questions, labels may name PDF sections, and a label matches on its URL and its section path together.
- **Decision:**
  - **Labels:** each question in `eval/questions.yaml` lists the sections that answer it, by citation URL (`page#anchor`) plus its heading path. `make eval` stops if a listed section is not in the index, so a typo cannot pass as a miss.
  - **Labelling rule (strict, chosen by the user):** a section is listed only if, read on its own, it answers the main part of the question; sections that only name the topic are left out. For a question that compares two things (GRD vs SLC), each section that explains one of them counts, because no single section explains both.
  - **Questions:** 19, of which 5 were written *blind* by the user without reading the pages (including the project's example question) and 14 were drafted *from the text* by Claude. A `written` field records which, and every metric is also reported per group. Three more blind questions are listed in the file but not scored:
    - one duplicates `grd-vs-slc`;
    - one (exporting to GeoTIFF) has no answer in the corpus;
    - one asks about three things (Level-0, Level-1 and Level-2). The user decided to leave such questions out: its three sections ranked 2, 3 and 14, so part of the answer never reaches the model, yet it counted as a hit because one section is enough.
  - **What is measured:** each question goes through `retrieval.retrieve`, the same function `make ask` uses, at depth 20. The result per question is the rank of the first chunk from an expected section.
  - **Metrics:** hit@1, hit@3, hit@5 and hit@10 (the share of questions with that rank or better), and MRR@k, the mean of 1/rank, which counts a rank below k as 0 because `make ask` only passes the top k to the model. k is `TOP_K` (D-018); `--top-k` scores another value on the same retrieval.
  - **Output:** the metrics, a rank per question, and for each miss the expected sections next to the top 5 retrieved. Every run is saved as `eval/results/<timestamp>[-<label>].json`, with the index recipe (`index_config`), the chunk count and the 20 retrieved chunks per question, so runs can be compared later.
- **Alternatives considered:**
  - *Labels per chunk:* chunk ids change whenever the chunking changes, so every chunking experiment would need new labels.
  - *A single expected section per question:* when two sections answer, retrieving the other one would count as a miss.
  - *Evaluating the generated answers* (faithfulness, correctness): needs Gemma or a judge model and is far noisier; the brief limits the eval to retrieval.
  - *MRR over the full depth:* rewards moving an answer from rank 15 to 11, which `make ask` never sees.
  - *Requiring every side of a comparison in the top k:* more faithful for comparisons, but only one question needed it once the three-sided one was dropped.
  - *Precision@k or nDCG:* they need every relevant chunk labelled, not just the first one found.
- **Why:** labels survive changes to the chunking and the embedding, and the numbers describe exactly what `make ask` retrieves.
- **Trade-offs:**
  - With 19 questions, one question moves hit@k by 0.053, and by 0.2 within the 5 blind ones, so only differences of several questions mean anything. The brief asked for 10–15 questions; 19 keeps the blind ones without dropping the rest.
  - The questions drafted from the text share its wording and are measurably easier (see below).
  - Section labels are a judgement call: counting `S1 Products > Level-1 Products` as an answer to the GRD/SLC question would turn that miss into a hit at rank 1.
  - A comparison counts as found when either side is retrieved, so hit@k cannot tell whether both reached the model.
  - Changes are measured on the same 19 questions they are chosen from, with no held-out set, so a change is only kept with a reason beyond the number.
- **Verified [2026-10-10]** (baseline with the top 5, `eval/results/2026-10-10T130917-baseline-k5.json`, 5 s):

  | Questions | hit@1 | hit@3 | hit@5 | hit@10 | MRR@5 |
  |---|---|---|---|---|---|
  | all (19) | 0.684 | 0.789 | 0.895 | 0.947 | 0.761 |
  | blind (5) | 0.400 | 0.600 | 0.600 | 0.800 | 0.500 |
  | from-text (14) | 0.786 | 0.857 | 1.000 | 1.000 | 0.854 |

  - The questions drafted from the text are all found in the top 5; the blind ones are not, which confirms they are easier.
  - Misses: `grd-vs-slc` (rank 9) and `slc-content` (rank 16, behind `ETAD Products`, which keeps saying "standard SLC product"). For `grd-content` the GRD section is not in the top 20 either; it counts as a hit only through `S1 Products > Level-1 Products`.
  - Capping the chunks per section at 1 or 2, simulated on the saved results, changes no metric: `grd-vs-slc` only moves to rank 6.

  - An earlier run of the first 15 questions gave identical numbers on a second run: the eval is deterministic.

## D-018: Top-k raised from 5 to 10

- **Status:** Superseded by D-022 (step 10, 2026-10-11): the default is now hybrid search with the top 5. Accepted in step 6, 2026-10-10.
- **Decision:** `retrieval.TOP_K` is 10, so `make ask` passes Gemma the 10 closest chunks and the eval reports MRR@10.
- **Alternatives considered:**
  - *Keeping 5:* the project's example question keeps missing the sections that explain the difference.
  - *9, the smallest k that finds it:* chosen from one question's rank; 10 is the round value the eval already reported.
  - *16 or more, to also find `slc-content`:* prompts of 5k+ tokens and many more passages for a 4.5B model to sort through, for one question.
  - *Capping the chunks per section:* simulated on the saved eval results, it changes no metric (D-017).
  - *Changing the chunking* (chunk size, the "Page > Section" prefix, merging small sections) or adding pages: not run; the user chose the top-k change.
  - *A reranker or hybrid search:* outside the brief's scope (hybrid search is listed in the README extensions).
- **Why:** the sections that explain GRD and SLC best (ranks 9 and 11) are the ones the example question needs, and with 10 passages they reach the model. The prompt still fits comfortably in the 8,192-token context.
- **Trade-offs:**
  - Prompts are about twice as long (median 2,869 tokens instead of 1,272–1,792), so answers take a little longer and the model has more passages to ignore.
  - The gain on the eval is one question of 19 (hit@10 0.947 against hit@5 0.895), within the noise described in D-017; the reason to keep it is the better answer to the example question.
  - `slc-content` is still a miss (rank 16).
- **Verified [2026-10-10]:**
  - **Eval** (`eval/results/2026-10-10T130918-top-k-10.json`): hit@10 0.947, MRR@10 0.766 over 19 questions. Blind: hit@10 0.800, MRR@10 0.522. Drafted from the text: hit@10 1.000, MRR@10 0.854.
  - **Prompt sizes** with the top 10, from Gemma's `prompt_eval_count`: 2,150–3,804 tokens over the 19 eval questions (median 2,869). The 10 largest chunks of the corpus together give 4,288 tokens, plus up to 1,024 for the answer, against `num_ctx` 8,192.
  - **Answers**, 3 runs each with the top 10:
    - GRD vs SLC: every answer cites the SLC section ([9]) for phase, complex samples and slant range, which never reached Gemma with the top 5. GRD facts still come from `S1 Products > Level-1 Products`.
    - Mission cost (not in the corpus): "No se encontró información…", with no citation.
    - Repeat cycle (English): answered in English, citing the Orbit section; once `$180^\circ$` instead of 180°.
    - Exporting to GeoTIFF (not in the corpus): answered with nearby facts (Level-1 products and S1GBM tiles are GeoTIFF files), correctly cited, without saying that the export itself is not covered. The top 5 gives the same answer, so the top-k is not the cause; it is a limitation of the prompt (D-016).
- **Verified in step 7 [2026-10-10]:**
  - **Clean clone:** a fresh `git clone`, run in a separate Compose project on another port, went through `make up`, `make ingest`, `make index`, `make ask` and `make eval` with no manual step. The pages downloaded again had the same `content_hash` and chunks as on 2026-10-08, and the eval matched the cited run: the same metrics, and the same sections and distances (to 6 decimals) for every question.
  - **Answer time** with the top 10 and the models loaded: 3.9–7.4 s for prompts of 2,901–3,359 tokens, against 0.6–2.6 s with the top 5 (D-016).

## D-019: Lexical retrieval with Postgres full-text search

- **Status:** Accepted (step 8, 2026-10-10). Since D-023 the default ranking is BM25; `ts_rank` remains as `--lexical-ranking ts_rank`.
- **Decision:**
  - **Column and index:** `chunks.tsv` is a stored generated column, `to_tsvector('english', section || E'\n\n' || content)`, built from the same text that is embedded (D-014). A GIN index (`chunks_tsv_gin`) maps each lexeme to the chunks that contain it.
  - **Query:** the question goes through `plainto_tsquery('english', ...)`, which normalizes it like the chunks, and its ANDs are replaced by ORs, so that a chunk matches if it contains any of the question's words.
  - **Ranking:** `ts_rank` with no length normalization. Equal scores are ordered by section path and chunk position (D-020 explains why not by chunk id).
  - **Interface:** `--method lexical` in `make eval` and `make ask`. Each eval question now has a `lang` field (`es` or `en`), and the eval reports every metric per language as well.
- **Alternatives considered:**
  - *The AND query that `plainto_tsquery` builds:* a chunk must contain every word of the question. 18 of the 19 eval questions match no chunk at all.
  - *Other built-in rankings*, measured on the eval: `ts_rank` divided by 1 + log(length) gives hit@10 0.632 and MRR@10 0.435; `ts_rank_cd` (which rewards matched words that are close together) 0.526 and 0.374. Both are worse.
  - *BM25 written by hand* (IDF computed over the chunks, k1 = 1.2, b = 0.75), measured with a throwaway script: hit@10 0.684 and MRR@10 0.559 against 0.737 and 0.500, with every English question at rank 1. It needs corpus statistics on every query (`ts_stat` or a table kept up to date), so it was not implemented.
  - *A BM25 extension* such as ParadeDB's `pg_search`: another image and extension to maintain.
  - *An expression index instead of a stored column:* every query would have to repeat the exact expression to use the index. The column keeps it in one place, and at 195 rows its storage is negligible.
- **Why:** it is built into Postgres, adds no dependency and lives next to the vectors. It is the baseline that the first extension in the README asks for.
- **Trade-offs:**
  - **No IDF:** `ts_rank` ignores how common a word is in the corpus, so `sentinel` (in 128 chunks) and `-1` (in 146) count as much as a rare term.
  - **One language:** the English configuration can only match a Spanish question through the tokens that both languages share: acronyms, numbers and "Sentinel-1". Words such as "diferencia" or "producto" appear in no chunk.
  - **Many ties:** with one or two matching words, many chunks get exactly the same score.
  - **Schema change:** a database created before this step needs the column and the index added, by hand (`ALTER TABLE`, as was done here) or with `make clean && make up && make ingest && make index` (D-005).
- **Verified [2026-10-10]:**
  - **Postgres 18 documentation:** `plainto_tsquery` inserts `&` (AND) between the words; the ranking functions "do not use any global information"; a generated `tsvector` column with a GIN index is the documented pattern, and only the form of `to_tsvector` with an explicit configuration can be indexed.
  - **Generated columns:** in Postgres 18 they are virtual by default (`attgenerated = v`), and indexing one fails with "indexes on virtual generated columns are not supported". The column is therefore declared `STORED`.
  - **Parser:** "Sentinel-1" becomes the word `sentinel` and the number `-1`, the same way in questions and in chunks, so they match.
  - **Query plan:** with `enable_seqscan = off`, `Bitmap Index Scan on chunks_tsv_gin`, about 1 ms. The index takes 240 kB.
  - **Eval** (`eval/results/2026-10-10T213253-lexical.json`; the vector numbers per language are from `…223354-vector.json`, saved in step 9), hit@10 / MRR@10:

    | Questions | vector (D-018) | lexical |
    |---|---|---|
    | all (19) | 0.947 / 0.766 | 0.737 / 0.500 |
    | Spanish (12) | 0.917 / 0.672 | 0.583 / 0.284 |
    | English (7) | 1.000 / 0.929 | 1.000 / 0.871 |
    | blind (5, all Spanish) | 0.800 / 0.522 | 0.600 / 0.158 |

  - **The GRD/SLC question:** "GRD" appears in 43 chunks and "SLC" in 47, so they are too common to single out the two sections that explain them. Lexical search ranks the `S1 Products > Level-1 Products` chunks first, as vector search does, and the expected sections are not in its top 20.
  - **Fresh and altered databases agree:** a database created from the new `schema.sql` and indexed from scratch returns the same chunks and scores, for every method, as the existing database after `ALTER TABLE`.

## D-020: Hybrid search with reciprocal rank fusion; vector search stays the default

- **Status:** Accepted (step 8, 2026-10-10). The default changed in D-022, once the lexical half was translated (D-021).
- **Decision:**
  - **Fusion:** `--method hybrid` takes the top 40 of vector search and the top 40 of lexical search and gives each chunk the sum of 1 / (60 + rank) over the lists it appears in (reciprocal rank fusion, RRF). The 40 is the most an HNSW scan can return with the default `hnsw.ef_search`.
  - **Ties:** chunks with equal scores in a list share its best rank (1, 2, 2, 4), and the remaining ties are ordered by section path and chunk position.
  - **Default:** `make ask` and `make eval` keep vector search. Hybrid search is worse on this eval.
- **Alternatives considered:**
  - *Adding normalized scores* (a cosine distance and a `ts_rank`): the two scales are unrelated, and any normalization is another parameter to tune. RRF uses ranks only.
  - *Ranks by position, ties ordered by chunk id* (the first version): chunk ids change whenever a page is indexed again (D-014), so two databases with the same data gave different results. `ground-subsidence` came 7th in one and 17th in the other.
  - *RRF with the hand-written BM25* (D-019): hit@10 0.789, MRR@10 0.670.
  - *Weighting the lists, or fusing only for English questions:* tuned on these same 19 questions; on the English ones vector search already finds every answer.
  - *Translating the question into English before the lexical search:* one more Gemma call per question. Tried in step 9 (D-021): it closes the language gap, and hybrid search reaches MRR@10 0.860 with the same hit@10 as vector search.
  - *Making hybrid the default:* rejected on the numbers below.
- **Why:** RRF is the standard way to merge rankings whose scores are not comparable, and it needs no calibration. k = 60 comes from the paper that introduced it.
- **Trade-offs:** two more retrieval methods in the code that `make ask` does not use by default, and a GIN index updated on every insert (negligible at this size).
- **Verified [2026-10-10]:**
  - **Paper** (Cormack, Clarke and Büttcher, SIGIR 2009): RRFscore(d) = Σ 1/(k + r(d)), where "k = 60 was fixed during a pilot investigation and not altered during subsequent validation"; the pilot found it "near-optimal, but … the choice was not critical".
  - **Eval** (`eval/results/2026-10-10T213254-hybrid.json`): hit@10 0.842 and MRR@10 0.633 against 0.947 and 0.766 for vector search. Spanish 0.750 / 0.461, English 1.000 / 0.929, blind 0.600 / 0.329.
  - **Per question**, against vector search: better for `slc-content` (16th → 7th) and `orbit-files` (5th → 2nd), worse for six, among them the project's example question `grd-vs-slc` (9th → 16th, which would undo D-018) and `oil-spills` (1st → 15th). A chunk ranked 10th in both lists scores 2/70 = 0.029, more than one ranked 1st in a single list (1/61 = 0.016), so when one list is noise, chunks that are mediocre in both rise above the right one.
  - **Variants** of the lexical side, with the same tie handling: `ts_rank` divided by 1 + log(length) 0.842 / 0.602; `ts_rank_cd` 0.789 / 0.523; BM25 0.789 / 0.670. None reaches vector search alone.
  - **`make ask --method hybrid`:** "What is the repeat cycle of Sentinel-1?" is answered from the Orbit section, with a 3,262-token prompt in 6.6 s.

## D-021: Translate the question into English for the lexical search; vector search stays the default

- **Status:** Accepted (step 9, 2026-10-10). The default changed in D-022: with a smaller top-k, the better ranking pays off.
- **Decision:**
  - **Translation:** `--translate` (in `make ask` and `make eval`, with `--method lexical` or `hybrid`) first asks Gemma to translate the question into English (`translate.py`), and the lexical search uses that translation. The vector search keeps the original question.
  - **Prompt:** detect first ("if the question is written in English, reply with the single word ENGLISH"), then translate with the standard terminology of radar remote sensing and British spelling, keeping acronyms, product names and numbers. For ENGLISH, or a translation cut off at the token limit, the original question is used.
  - **Request:** `temperature: 0` so that the translation is repeatable, `num_predict: 128`, and the same `num_ctx` as the answer (`CHAT_NUM_CTX` in `config.py`), since another value makes Ollama load the model again.
  - **Default:** `make ask` keeps plain vector search. Hybrid search with translation is offered as an option.
- **Alternatives considered:**
  - *Translating for the vector search too:* MRR@10 0.782 instead of 0.766 for vector search alone, with the same hit@k; and 0.833 instead of 0.860 for the hybrid. bge-m3 does not need it, as D-003 and D-015 assumed.
  - *Prompt wordings*, on the 7 English eval questions, which should come back untouched:
    - "If the question is already in English, repeat it unchanged" (first try): 2 were reworded. One changed "polarisations" to "polarizations", which stems to `polar` instead of the corpus's `polaris` (57 chunks), so it would have broken the lexical match.
    - The ENGLISH rule placed after the translation instructions: 3 reworded.
    - The ENGLISH rule first (chosen): 1 changed, "Which" to "What", two stop words that full-text search ignores.
  - *Detecting the language in code* (heuristics or a library): fragile rules or a new dependency, for a step the model already does. Not tried.
  - *BM25 instead of `ts_rank` on the translated question* (throwaway script, D-019): hybrid MRR@10 0.908 and hit@1 0.895, the best ranking measured. Not implemented, for the same reason as in D-019.
  - *Making hybrid search with translation the default:* better ranking, but the same hit@10, and the project's example question loses the section that explains SLC (below).
  - *Hybrid search with translation and a smaller top-k:* its hit@3 (0.947) equals the hit@10 of vector search, so 3 to 5 passages could replace 10 and shorten the prompt about threefold. Not adopted here; it would change D-018 and the answers would need checking again.
- **Why:** the language gap is what sank lexical search (D-019), and a translation closes it with the model that is already loaded. The default stays vector search because `make ask` passes 10 chunks to the model, so what decides the answer is whether the right section is in the top 10. That is hit@10, and there the two are equal.
- **Trade-offs:**
  - One more Gemma call per question (0.2–0.7 s with the model loaded).
  - The eval of `--translate` needs Gemma loaded (about 60 s the first time) and depends on a generative model, although at temperature 0 its output repeated exactly.
  - A wrong translation would silently weaken the lexical half.
- **Verified [2026-10-10]:**
  - **Translations** of the 19 eval questions and the 3 unscored ones: identical in 4 runs with the chosen prompt and in 2 runs of `make eval`. All 12 Spanish questions are translated correctly, with domain terms ("anchura de barrido" as "swath width", "hundimientos del terreno" as "ground subsidence", "millimetres").
  - **Reload:** a request with `num_ctx` 4096 after 8192 took 58 s, loading the model again, and switching back another 56 s.
  - **Eval** (hit@10 / MRR@10), from `eval/results/2026-10-10T223354-vector.json`, `…213253-lexical.json`, `…221424-lexical-en.json`, `…213254-hybrid.json` and `…221448-hybrid-en.json`:

    | Method | all (19) | Spanish (12) | English (7) | blind (5) | hit@1 | hit@3 |
    |---|---|---|---|---|---|---|
    | vector | 0.947 / 0.766 | 0.917 / 0.672 | 1.000 / 0.929 | 0.800 / 0.522 | 0.684 | 0.789 |
    | lexical | 0.737 / 0.500 | 0.583 / 0.284 | 1.000 / 0.871 | 0.600 / 0.158 | 0.421 | 0.579 |
    | lexical + translate | 0.947 / 0.746 | 0.917 / 0.674 | 1.000 / 0.871 | 0.800 / 0.417 | 0.632 | 0.842 |
    | hybrid | 0.842 / 0.633 | 0.750 / 0.461 | 1.000 / 0.929 | 0.600 / 0.329 | 0.526 | 0.684 |
    | hybrid + translate | 0.947 / 0.860 | 0.917 / 0.819 | 1.000 / 0.929 | 0.800 / 0.667 | 0.789 | 0.947 |

  - **Per question**, hybrid with translation against vector search: better for `product-metadata` (2nd → 1st), `slc-content` (16th → 3rd), `orbit-files` (5th → 2nd) and `ground-subsidence` (4th → 1st); worse only for `grd-vs-slc` (9th → 18th).
  - **Answers** (`make ask`):
    - GRD vs SLC: with hybrid search and translation the SLC section no longer reaches the model, and the answer loses the phase, complex samples and slant range that D-018 gained. Only `S1 Products > Level-1 Products` is cited.
    - "¿Qué información contiene un producto Sentinel-1 SLC?": the hybrid answer cites `Level-1 Products Formatting` (a complex measurement dataset in GeoTIFF per polarisation), on point. The vector answer wanders through ETAD and availability dates.
    - Mission cost (not in the corpus): still "No se encontró información…", with no citation.
  - **Eval time:** 9 s with Gemma loaded, against 5–8 s without translation.

## D-022: Default retrieval: hybrid search with a translated question and the top 5

- **Status:** Accepted (step 10, 2026-10-11). Since D-023 the full-text half ranks with BM25 (MRR@5 0.908 instead of 0.860). Measured again with the PDFs in D-026: it is still the best of the methods at the top 5, and stays the default.
- **Decision:**
  - **Default:** `make ask` and `make eval` use hybrid search (D-020) with the question translated into English for the full-text half (D-021), and the model gets the top 5 chunks (`DEFAULT_METHOD = "hybrid"`, `TOP_K = 5`).
  - **Options:** translation is on wherever there is a full-text search; `--no-translate` turns it off, and `--method vector` never translates. `make ask` also takes `--top-k`, so the previous default is `--method vector --top-k 10`.
  - **Eval:** it measures the default, so it now loads Gemma for the translations.
- **Alternatives considered:**
  - *Vector search with the top 10* (D-018, the previous default): the same number of answers reach the model, with prompts about 75 % longer, and the example question keeps its SLC section.
  - *Hybrid with translation, top 3:* the same coverage on the eval and the shortest prompts, but the answers cite fewer sections (1.36 per answer against 1.74 with vector search): the ETAD answer lost one of its two sections, and the answer on how the SAR works lost the acquisition modes. It also leaves no margin for a new question whose answer ranks 4th or 5th.
  - *Hybrid with translation, top 10:* no more answers than the top 5, with the longest prompts.
  - *Vector search with the top 5* (the D-017 baseline): hit@5 0.895, one answer fewer.
- **Why:**
  - With the better ranking, the top 5 holds as many answers as vector search's top 10: 18 of 19 questions.
  - The real answers cite a correct section just as often (36 of 38), with prompts about 43 % shorter and generation about 30 % faster.
  - Some answers are more focused. For oil spills, vector search with 10 passages mixed ship-detection products into the list of products for detecting oil.
- **Trade-offs:**
  - **The example question:** "¿Qué diferencia hay entre un producto GRD y uno SLC?" loses the SLC section (rank 18), and its answer no longer explains the phase information and complex samples, which D-018 raised top-k to obtain. `--method vector --top-k 10` brings it back. In exchange, "¿Qué información contiene un producto Sentinel-1 SLC?" now gets its section (rank 3, from 16).
  - **A model call in retrieval:** every question, and every eval run, needs a Gemma call first (D-021).
  - **Chosen on the eval questions:** like D-018, on the same 19 questions, here backed by the answers to them.
- **Verified [2026-10-11]:**
  - **Eval** (`eval/results/2026-10-11T011251-hybrid-en-k5.json`): hit@1 0.789, hit@3 0.947, hit@5 0.947, MRR@5 0.860. Blind 0.800 / 0.667, drafted from the text 1.000 / 0.929. The top 3 gives the same coverage and MRR@3 0.860.
  - **Answers** (throwaway script, 168 in total: the 19 eval questions and 2 that the corpus does not answer, 4 configurations, 2 runs each):

    | Configuration | Cited a correct section | Prompt tokens (median / max) | Answer tokens (median) | Generation (median) |
    |---|---|---|---|---|
    | vector, top 10 | 36 / 38 | 2,851 / 3,804 | 116 | 5.4 s |
    | hybrid, translated, top 10 | 36 / 38 | 3,147 / 3,563 | 115 | 6.2 s |
    | hybrid, translated, top 5 | 36 / 38 | 1,615 / 2,171 | 90 | 3.7 s |
    | hybrid, translated, top 3 | 36 / 38 | 1,042 / 1,524 | 75 | 2.7 s |

    - In every configuration, an answer cited a correct section whenever one was in the prompt. The two misses are `grd-vs-slc` for the hybrids and `slc-content` for vector search.
    - Mission cost: "not found" in 2 of 2 runs for every configuration. Exporting to GeoTIFF: never said to be missing, in any configuration (the limitation in D-018).
    - Answers read side by side: orbit files (all configurations give the full latency and accuracy table), ETAD, the SAR principle, oil spills, floods, ground subsidence and the repeat cycle.
  - **End to end,** `make ask` with the models loaded on 6 questions: 4.2–16.3 s with the new default, translation included, against 5.4–12.4 s for vector search with the top 10. The answer length dominates (about 40 tokens/s).
  - **The example question,** 2 runs with the new default: both cite only `S1 Products > Level-1 Products` and `SAR Products Overview`; one rendered "Single Look Complex" as "Complejo de Aspectos Únicos".
  - **Earlier behaviour reproducible:** with the new options, `--method vector --top-k 10`, `--method lexical|hybrid --no-translate --top-k 10` and `--method lexical|hybrid --top-k 10` reproduce the five cited runs of steps 6, 8 and 9 exactly.

## D-023: BM25 in SQL ranks the full-text search

- **Status:** Accepted (step 11, 2026-10-11)
- **Decision:**
  - **Ranking:** the full-text search ranks chunks with BM25 instead of `ts_rank`, written as one SQL query (`BM25_QUERY` in `retrieval.py`). For each word of the question that a chunk contains, it adds IDF × tf × (k1 + 1) / (tf + k1 × (1 − b + b × length / average length)), with IDF = ln(1 + (N − df + 0.5) / (df + 0.5)).
  - **Parameters:** k1 = 1.2 and b = 0.75. Lengths count the words in the full-text index, without stop words. Ties are ordered by section and position, as in D-020.
  - **Statistics:** N, the average length and each word's document frequency are computed on every query from the `tsv` column. There is no new table or column.
  - **Options:** `--lexical-ranking ts_rank` keeps the previous ranking. The defaults of D-022 are unchanged otherwise.
- **Alternatives considered:**
  - *Keeping `ts_rank`:* simpler and faster (4 ms against 16), and the answers measured below are just as good. But it gives a word found in two thirds of the chunks the same weight as a rare one (D-019).
  - *Statistics kept in a table or a materialized view,* refreshed by `make index`: faster queries, but one more thing to keep in step with the chunks, for 195 rows.
  - *A BM25 extension* such as ParadeDB's `pg_search`, or a search engine (OpenSearch, which is listed as an extension): another image or service.
  - *Computing BM25 in Python:* every chunk's lexemes would have to cross the connection on every query.
- **Why:** it fixes the defect measured in D-019, ranks better on the eval, and makes the hybrid the standard recipe (BM25 + vectors + RRF), at no visible cost in time.
- **Trade-offs:**
  - **The answers do not improve.** The same 18 of 19 questions reach the model, and the answers cite a correct section just as often (below). The case for BM25 is the ranking, not the answers.
  - **It reads every chunk on every query,** about 16 ms here. That does not scale to a large corpus.
  - **About 40 lines of SQL** to maintain instead of one function call.
- **Verified [2026-10-11]:**
  - **Parameters and IDF:** Lucene's `BM25Similarity` defaults to k1 = 1.2 and b = 0.75, with IDF `log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))` and the average length computed as total term frequency / document count. Elasticsearch documents the same defaults.
  - **Correctness:** the SQL matches the Python BM25 of the throwaway script on all 38 queries (the 19 questions, original and translated): the same top 40 in the same order, with score differences below 4·10⁻¹⁵.
  - **Time:** median 15.7 ms per query for the top 40, against 3.9 ms for `ts_rank` (50 runs).
  - **Eval**, the default configuration of D-022 (hybrid, translated, top 5): hit@1 0.895 instead of 0.789, hit@3 0.895 instead of 0.947, hit@5 0.947 (unchanged), MRR@5 0.908 instead of 0.860; English questions 1.000 / 1.000; blind 0.800 / 0.650 instead of 0.800 / 0.667 (`eval/results/2026-10-11T013216-hybrid-en-bm25-k5.json`). Per question, `centre-frequency` and `orbit-files` move to rank 1 and `slc-content` from 3rd to 4th.
  - **With 10 passages** (hit@10 / MRR@10): lexical 0.684 / 0.559, lexical translated 0.947 / 0.796, hybrid 0.789 / 0.670, hybrid translated 0.947 / 0.908 (`…013217-lexical-bm25.json`, `…013225-lexical-en-bm25.json`, `…013227-hybrid-bm25.json`, `…013236-hybrid-en-bm25.json`).
  - **Answers** (throwaway script, 84: the 19 eval questions and 2 unanswerable ones, 2 runs, top 5 with `ts_rank` and with BM25): both cite a correct section in 36 of 38, both say "not found" for the mission cost and neither does for GeoTIFF, with median prompts of 1,615 and 1,583 tokens and a median generation of 3.7 s each. BM25 cites 1.6 sections per answer against 1.8. Read side by side: oil spills better with BM25 (it names the Level-2 wind products), product metadata worse (it presents ETAD documents as the way to consult any product's metadata), the product name the same.
  - **Earlier runs:** with `--lexical-ranking ts_rank` where they used it, the six cited runs of steps 6 and 8 to 10 are reproduced exactly.

## D-024: Six technical PDFs from the Sentinel-1 document library join the corpus

- **Status:** Accepted (step 12, 2026-10-11)
- **Decision:**
  - **Documents**, the latest version of each, 51.7k words in all:

    | Short title (in citations) | Document | Pages | Chunks |
    |---|---|---|---|
    | Radiometric Calibration of S-1 Level-1 Products | ESA-EOPG-CSCOP-TN-0002, issue 1.0 | 13 | 17 |
    | Thermal Denoising of S-1 Products | MPC-0392, issue 2.3 | 31 | 34 |
    | Masking No-value Pixels on S-1 GRD Products | MPC-0243, issue 2.2 | 15 | 16 |
    | S-1 TOPS SLC Deramping | COPE-GSEG-EOPG-TN-14-0025, issue 1.3 | 15 | 17 |
    | Guide to Sentinel-1 Geocoding | UZH-S1-GC-AD, issue 1.12 | 42 | 80 |
    | Sentinel-1 Product Definition | S1-RS-MDA-52-7440, issue 2.11 | 130 | 209 |

  - **Sources:** `sources.yaml` gets a `pdfs:` list. Each entry has the file's URL, which belongs to that version of the file, and a short title written by hand, because the file names are document codes. The URLs leave out the `?cb=` cache-buster that the library page adds.
  - **Download:** the same polite client as the pages (D-011). A PDF is cached as `data/raw/<slug>.pdf`, its name taken from the URL, and is never committed.
  - **Prompt:** the system turn now says that the passages come from "the Sentinel-1 documentation: SentiWiki pages and technical documents" (D-016).
- **Alternatives considered:**
  - *Only the five short notes* (about 150 chunks): a smaller corpus, but the Product Definition is the reference for the product types, where the eval's hardest question (`grd-vs-slc`) sits.
  - *The Product Specification* (issue 3.16.3, 220 pages) *and the Level 1 Detailed Algorithm Definition* (issue 2.5, 166 pages): mostly format tables and equations, and together they would triple the corpus.
  - *The journal article "GMES Sentinel-1 mission"* (16 pages, the only one with bookmarks): its two-column layout came out with the words glued together ("Inaddition,twounits"), and it repeats the wiki's mission overview.
  - *Every version in the library:* several issues of the same document would compete with each other for the top 5.
  - *Finding the latest version on the library page at every ingest:* the corpus would change by itself whenever ESA uploads a document. Pinned URLs keep ingest reproducible, and the library keeps listing older versions.
- **Why:** the notes explain how to do what the wiki only names (calibrating to σ0, removing thermal noise, masking the GRD borders, deramping TOPS SLC bursts, geocoding), and the Product Definition describes every product type and its characteristics.
- **Trade-offs:**
  - **The PDFs dominate the corpus:** 373 of 568 chunks, 209 from the Product Definition alone. They fill 36 of the 95 top-5 places of the 19 earlier eval questions (D-026).
  - **Slower ingest:** `make ingest` takes about 80 s instead of 10, almost all of it reading the PDFs (D-025). `make index` embeds the 373 new chunks once (about a minute).
  - **Documents of different dates disagree:** for orbit files, the geocoding guide (2022) says restituted orbits arrive about 3 hours after acquisition, and the wiki's POD table says 45 minutes. One answer mixed the two (D-026).
  - **Pinned versions age:** a newer issue of a document is only picked up by editing `sources.yaml`. If ESA removed a file, `make ingest` would stop with a 404.
  - **Terms of use:** the SentiWiki's terms allow downloading and copying for non-commercial use but not redistribution, so the PDFs stay in `data/raw/`, which is not committed, like the pages.
- **Verified [2026-10-11]:**
  - **The library:** `robots.txt` is "Allow: /". The S1 Documents page lists about 130 files: PDFs in several issues (three of the thermal denoising note), zips, orbit-related text files and monthly performance reports. Attachment URLs answer 200 without `?cb=`.
  - **Terms and conditions page:** "ESA grants permission to users to visit the site, and to download and copy information, images, documents and materials"; "ESA does not grant the right to resell or redistribute".
  - **Bookmarks:** 9 of the 10 candidate PDFs inspected have none; only the journal article does.
  - **First ingest:** 7 requests (`robots.txt` and the 6 PDFs). The five pages' chunk files are byte-identical before and after this step, so `make index` skipped them.

## D-025: PDF extraction with pdfplumber: numbered headings in a larger font, page anchors

- **Status:** Accepted (step 12, 2026-10-11)
- **Decision:** `extract_pdf.py` turns a PDF into the same `Page` and `Section` structure as a SentiWiki page, so chunking (D-013), indexing (D-014) and the schema do not change.
  - **Library:** `pdfplumber` 0.11.10 (MIT; it brings `pdfminer.six`, `pypdfium2` and Pillow).
  - **Headers and footers:** a line in the top or bottom 15 % of the page that repeats on at least half the pages, once digits are ignored ("Page 4/13"), is removed. So is rotated text: chart axes and margin notes.
  - **Sections:** a heading is a line that starts with a section number ("3.1", "5.2.1.4", or "A1.4" and "B" in appendices), is set at least 0.5 pt larger than the body text (the size most characters use), and continues the numbering: the first subsection, or the next section at the same or a higher level. A heading that wraps onto a second line in the same font is joined. Everything before section 1 is dropped (title page, change log, table of contents), and the section path starts with the short title from `sources.yaml`.
  - **Citations:** a section's anchor is `page=N`, the physical page of its heading, so the citation URL `url#page=N` opens the PDF at that page.
  - **Text:** lines are joined into paragraphs, which end at a vertical gap larger than half a line. Bulleted lines become list items, nested by indentation. A word broken by a hyphen at the end of a line loses the hyphen only if the joined word appears elsewhere in the document ("Correc-" + "tions"); otherwise the hyphen stays ("cross-section").
  - **Tables:** the tables pdfplumber finds become one line per row, cells joined with `" | "`. A cell merged down several rows is repeated in each row, and an empty cell is kept as an empty field in a column that the first row names, so every row lines up with the header.
  - **Noise:** a line with no word of three letters or more, or made mostly of single characters, is dropped: chart labels, stray subscripts and the remains of equations. Glyphs that a font does not map to any character (`(cid:1830)`) are removed.
- **Alternatives considered:**
  - *pypdf* (BSD, no dependencies): it split words in two ("Faci lity", "dedicate d", "ea sily"), which breaks full-text matching.
  - *pdfminer.six alone:* fewer dependencies and paragraphs for free, but it groups table cells column by column, so rows fall apart.
  - *PyMuPDF:* AGPL-licensed.
  - *Headings from the bookmarks:* only 1 of the 10 PDFs inspected has any.
  - *Headings from the table of contents:* not every note has one, and the Product Definition's leaves out the appendices.
  - *The numbering alone, without the font size:* in the Product Definition a table line "4 Wave Mode (WV)" passed for chapter 4 and threw off the sections after it.
  - *One section per page:* simpler and as precise for citations, but chunks would ignore the topic boundaries and the "Page > Section" prefix (D-014) would say nothing.
  - *An anchor per section:* a PDF link can open a page, not a heading, unless the file has named destinations, and none of these has any. Sections that start on the same page therefore share their URL, so the eval matches on the URL and the section path together (D-026).
  - *Filling every merged cell from the cell above:* wrong for cells merged across columns ("Ground Range Coverage [km] | 251.8" spans three beams), which would get values from the row above.
- **Why:** each rule fixes something seen in these six documents, and every section can be traced back to a page and a heading that a reader can check.
- **Trade-offs:**
  - **Equations are lost.** Their fonts do not say which character each glyph is, so formulas such as σ0 = |DN|² / A²σ disappear from the calibration note, and only the text around them remains. Equations typed as text survive: the denoising one in the thermal noise note did.
  - **Charts leave fragments.** Captions remain, as for the wiki's figures, plus a few stray labels. One vertical table header (Table 5-4 of the Product Definition) comes out as single letters.
  - **The hyphen rule is a guess.** In the geocoding guide, which hyphenates words to justify its lines, 34 of the 112 line-end hyphens stay, some of them breaks ("criti-cal") and some real ("mid-range").
  - **Tuned on six documents.** A PDF with headings in the body font, two columns or unnumbered headings would need new rules. With no numbered heading, the extractor stops with "no numbered sections found".
  - **The token estimate is a little low for PDFs.** The words × 1.68 estimate of D-013 was measured on the wiki. PDF chunks have 1.77 bge-m3 tokens per word: 14 of them exceed 500 real tokens, the largest 801. That is harmless, because Ollama runs bge-m3 with a 4096-token context.
  - **Speed:** reading the six PDFs takes about 65 s on every `make ingest`.
- **Verified [2026-10-11]:**
  - **Fonts** (most common size by characters, against the headings): Product Definition 12 pt body, headings 13–16 pt bold; geocoding guide 11 against 12 and 13.9; calibration note 11.3 against 13.2; thermal noise note 10 against 12.
  - **Sections:** 13, 27, 12, 6, 34 and 145 per document (237 with text), with appendices (A1.4, B2, D1) in the Product Definition. The first heading of every document is its section 1.
  - **Page anchors:** for all 237 sections, the heading's text is on the page that its citation URL points to (checked with a throwaway script). RFC 8118 defines `page=` as the physical page, the first being 1, and pdf.js lists it among its viewer options ("page: page number. Example: page=2").
  - **Hyphens at line ends:** 112 in the geocoding guide (78 joined, because the whole word appears elsewhere in it), 15 in the Product Definition (1 joined), 8 in the thermal noise note (2), 1 each in the masking and deramping notes, and none in the calibration note.
  - **Merged cells:** before the fix, an answer about GRD resolution classes gave the pixel spacing and the number of looks as resolutions, because the rows of Table 5-1 had lost their first two columns. After it, the rows read "SM | GRD | HR | 23 x 23 | 10 x10 | 6 x 6 | 29.7", and both answers gave the right values.
  - **Chunks:** 373 from the PDFs. Real bge-m3 tokens: mean 238, median 236, minimum 12, maximum 801; the estimate is 12 tokens low on average.

## D-026: The eval with the PDFs: labels on PDF sections, matched with the section path; the default retrieval stays

- **Status:** Accepted (step 12, 2026-10-11)
- **Decision:**
  - **Earlier questions:** the 19 questions were checked against the PDFs with the same strict rule (D-017), and 11 gained sections:
    - `grd-vs-slc`: Product Definition 5.2 (the two types side by side), 5.2.1 (SLC) and 5.2.2 (GRD);
    - `grd-content` 5.2.2, `slc-content` 5.2.1, `sar-principle` and `centre-frequency` 3.2 (payload and system parameters), `iw-swath-resolution` and `topsar-vs-scansar` 3.3.2, `wave-vignettes` 3.3.4, `level2-ocean` 6.1, `product-slicing` 5.4;
    - `orbit-files`: geocoding guide 4.3.1 (the three orbit products, their latency and relative quality).
  - **New questions:** 6 drafted from the text, one per document (3 in Spanish, 3 in English): `sigma0-calibration`, `thermal-denoising`, `grd-border-noise`, `tops-deramping`, `dem-height-errors`, `grd-resolution-classes`.
  - **Matching:** a retrieved chunk counts for a label when both its citation URL and its section path match. In the wiki every section has its own anchor, so nothing changes there; in a PDF, sections on the same page share the URL.
  - **Groups:** the eval also reports the questions by where their answer is, `wiki-only` (8), `wiki+pdf` (11) and `pdf-only` (6), worked out from the labels' URLs.
  - **Default:** the retrieval of D-022 and D-023 stays.
  - **No new blind questions:** the user chose to keep the 25 questions. Four of the 5 blind questions of D-017 now have PDF sections among their labels.
- **Alternatives considered:**
  - *Keeping the old labels:* on the new index, the 19 questions would score hit@5 0.895 and MRR@5 0.816. That apparent drop comes from correct PDF sections now ranking first, as the SLC definition does for `slc-content`.
  - *Matching on the URL alone:* a chunk from "2 DOCUMENTATION" of the deramping note would count for "1 INTRODUCTION", which starts on the same page. The label check caught this: five labels did not match the index.
  - *A field for where the answer is:* derived from the labels instead, so there is nothing to keep in step.
  - *Vector search with the top 10:* it reaches hit@10 0.920 and finds `grd-vs-slc` (rank 10) and all the blind questions, with twice the passages. The default reaches the same hit@10, and hit@5 0.880 against 0.800.
- **Why:** labels have to name every section that answers, or the eval punishes finding the right document. Matching the section path keeps a label exact when sections share a page.
- **Trade-offs:**
  - **The new questions were drafted from the text,** so they share its wording. No blind question is about a topic that only the PDFs cover, so how a reader without the documents fares there is not measured.
  - **Judgement calls,** as in D-017: Product Definition 5.1 lists both product types with their resolutions but does not explain the difference, so it is not a label for `grd-vs-slc`, and the per-mode product tables of chapter 7 are not labels for questions about a product type in general.
  - **25 questions:** one question moves hit@k by 0.04 overall, and by 0.167 among the 6 PDF questions.
- **Verified [2026-10-11]:**
  - **Eval** (`eval/results/2026-10-11T113743-pdfs.json`, and `…113745-pdfs-vector`, `…113757-pdfs-lexical-en`, `…113800-pdfs-hybrid`, `…113801-pdfs-vector-k10` for the other methods), hit@5 / MRR@5:

    | Method | all (25) | wiki-only (8) | wiki+pdf (11) | pdf-only (6) | blind (5) | hit@1 |
    |---|---|---|---|---|---|---|
    | hybrid, translated, BM25 (default) | 0.880 / 0.810 | 1.000 / 0.938 | 0.909 / 0.864 | 0.667 / 0.542 | 0.800 / 0.700 | 0.760 |
    | vector | 0.800 / 0.673 | 1.000 / 0.760 | 0.727 / 0.682 | 0.667 / 0.542 | 0.600 / 0.367 | 0.600 |
    | lexical, translated, BM25 | 0.800 / 0.663 | 1.000 / 0.906 | 0.818 / 0.576 | 0.500 / 0.500 | 0.600 / 0.200 | 0.600 |
    | hybrid, not translated | 0.720 / 0.590 | 0.625 / 0.625 | 0.818 / 0.652 | 0.667 / 0.431 | 0.600 / 0.367 | 0.520 |

  - **The 19 earlier questions:** hit@5 0.947 as in step 11 (the same 18 reach the model), MRR@5 0.895 against 0.908, hit@1 0.842 against 0.895. `slc-content` moves from 4th to 1st; `product-metadata` and `orbit-files` from 1st to 2nd, behind Product Definition sections (1.2 Scope, 9.5 Orbit and Attitude Information); `grd-vs-slc` from 16th to 13th.
  - **The 6 PDF questions:** ranks 1, 1, 1, 4, 10 and not in the top 20.
    - `sigma0-calibration` is a vocabulary miss: "sigma cero", translated as "sigma zero", pulls in the sections on the Noise Equivalent **Sigma Zero**, while the calibration note says σ0, DN and calibration LUT. Asked as "How do I convert Sentinel-1 digital numbers into radar backscatter (sigma0) using the calibration LUTs?", the note ranks 1st in vector search and 2nd in BM25.
    - `tops-deramping`: the right document fills the top 7, but with the four chunks of its long section 3, which defines the function; the introduction, which says why it is needed, ranks 10th.
  - **Answers** (throwaway script, 108: the 25 questions and the 2 unanswerable ones, 2 runs each; before = the index without the PDFs, kept in a copy of the database, with the old prompt):

    | | Earlier questions citing a correct section | PDF questions citing a correct section | Median prompt | Median answer | Median time | Sections cited per answer |
    |---|---|---|---|---|---|---|
    | before | 36 / 38 | 0 / 12 | 1,534 tokens | 88 tokens | 3.4 s | 1.6 |
    | after | 36 / 38 | 8 / 12 | 1,570 tokens | 112 tokens | 4.2 s | 2.1 |

    - Every answer cited a correct section whenever one was among its passages. 22 of the 38 answers to earlier questions now cite a PDF.
    - Before, the PDF questions got answers from nearby wiki sections, or "not found" (GRD borders). After, the four found ones are answered from the notes: the denoising equation and why it is not applied by default, the SWST cause of the no-value pixels, the shift caused by DEM height errors, and the resolution classes with the right values.
    - Worse: `orbit-files` takes the three orbit products and their latencies (about 3 hours, 3 weeks) from the geocoding guide and the accuracies from the wiki, whose table gives other latencies (45 minutes for the restituted orbit). Both runs cite "[9]", taken from "Table 9" in the passage, although there were only 5 passages (`make ask` warns about such numbers). `sigma0-calibration` says the calculation is not in the documentation, and `tops-deramping` explains what deramping does but not why.
    - Mission cost: "not found" in every run, before and after. Exporting to GeoTIFF: never said to be missing, as before (D-018).

## D-027: An agent with two tools, `search_docs` and `search_scenes`, in a hand-written loop

- **Status:** Accepted (step 13, 2026-10-11)
- **Decision:**
  - **Command:** `make agent Q="..."` (`agent.py`). `make ask` does not change and stays the RAG baseline.
  - **Loop:** the question goes to Gemma with the two tools' JSON Schemas (`tools.py`). On each turn the model either asks for tool calls or answers. Ollama only returns the calls (`message.tool_calls`); the loop runs them, appends each result as a `tool` message and asks again. At most 5 turns, temperature 0, `think: false`, the same `num_ctx` as `make ask` (D-002). A trace prints every turn and call.
  - **`search_docs(query)`:** the default retrieval of `make ask` (D-022, D-023), run on the model's query, top 5. Passages are numbered across calls, and a chunk found again keeps its number.
  - **`search_scenes(place, start, end, days, product_type, mode, orbit)`:** geocodes the place (D-028) and searches the Copernicus Data Space Ecosystem STAC catalogue (`scenes.py`): collections `sentinel-1-grd` and `sentinel-1-slc`, newest first, up to 10 scenes, with CQL2 filters on mode and orbit direction. Raw HTTP with httpx, as for Ollama (D-007).
  - **What the model reads about scenes:** a summary computed by the code (the place found with its type and size, how many scenes from how many passes, the days, satellites, orbit directions and tracks, modes and polarisations), never the scene names, and a line saying that these results have no passage number to cite. The code prints the full table, with names, under the answer, as it prints the cited sources (D-016).
  - **Guards in the code:**
    - If the model answers before calling any tool, the loop discards that answer, runs `search_docs` with the question itself and asks again. The trace and the eval mark that call as forced.
    - `days` counts back from today in code; `start` and `end` stay required, and `days` replaces them when given.
    - `mode` and `orbit` take `any` as an explicit value and default.
    - A tool error is returned to the model as text, naming what is wrong ("end is missing…"), so it can correct the call.
  - **System prompt:** answer only from the tools; call `search_docs` for every question or part of a question that is not only about finding scenes, even when the documentation may not have the answer; cite passages as `[n]`; how to read relative dates; summarise scenes instead of listing them; answer in the question's language.
- **Alternatives considered:**
  - *An agent framework* (LangChain, LlamaIndex, smolagents): the loop is the thing to learn here, and the brief asks for it written by hand. It is about 50 lines.
  - *A router in code* (keywords decide which search runs): simpler and predictable, but it is not an agent and cannot split a mixed question into two calls.
  - *Prompt rules only, no forced search:* measured. The model skipped `search_docs` for the two questions the documentation cannot answer, and once for "¿Qué es el modo IW?", whatever the prompt said (D-029).
  - *The model lists the scenes:* copying ten 67-character names took about 20 s, risked typos, and kept happening even when the result said not to list them.
  - *The model computes dates:* it turned "the last 10 days" into 20 days.
  - *Optional `start` and `end` once `days` existed:* the model left out `end` for "September 2026" and repeated that same call three times.
  - *`mode` and `orbit` without `any`:* the model filled them in unasked, with IW and once with SM, which found nothing.
  - *Thinking on:* the same score and about 2.6 times the time (D-029).
  - *`pystac-client`:* one more dependency for one POST request.
- **Why:** the documentation answers how Sentinel-1 works, and the catalogue answers what it acquired. An agent lets one question use both, and shows the parts of an agent that matter: tool schemas, a loop that runs the calls, results fed back, and limits. With a 4.5B model, every rule that only lived in the prompt was followed some of the time, so the behaviours that matter (grounding, dates, not inventing filters) are enforced by the code or by the shape of the tools.
- **Trade-offs:**
  - **The loop overrides the model** when it answers without a tool: an off-topic greeting also triggers a documentation search.
  - **At least two Gemma calls per question,** plus one for the translation in `search_docs`. Median 5.9 s per question with the models loaded, about 2,400 prompt tokens in the last turn for a mixed question.
  - **Mixed questions do not retry.** When `search_scenes` fails while `search_docs` succeeds in the same turn, the model answers and reports the error instead of correcting the call (D-028, D-029).
  - **Brittle to wording.** Small prompt changes moved single questions both ways (D-029).
  - **The model sees counts, not scenes,** so it cannot discuss one scene in particular; the table under the answer has them.
  - **Citations can still point at a scene search:** 1 of the 8 answers about scenes only still cites "[1]" (`make agent` warns when the answer cites a number and no passage was read).
  - **Network services:** a question about scenes needs the STAC API and Nominatim (D-028). Results change as the catalogue grows. Downloading a product needs a Copernicus account and is not implemented.
- **Verified [2026-10-11]:**
  - `ollama show` lists the capabilities of `gemma4:e4b-it-qat` as completion, vision, audio, **tools** and thinking (Ollama 0.40.1). Tool calls come back in `message.tool_calls` with the arguments already parsed; a mixed question got two calls in one turn; results are sent back as `{"role": "tool", "tool_name": ..., "content": ...}`.
  - **STAC API** (`https://stac.dataspace.copernicus.eu/v1`): STAC 1.0.0 with item search, CQL2 filters, `sortby` and `fields`. No account is needed to search. It answered in 0.3–2.4 s. It does not return the number of matches, so the code asks for 11 scenes to know whether there are more than 10. An item is about 17 KB of JSON; with `fields`, a search for 4 scenes was 2 KB. Recent GRD products are Cloud Optimized GeoTIFF products (`_COG` in their names).
  - **Probe before coding** (a throwaway loop, 5 questions): Gemma would not invent coordinates and asked the user for a bounding box; it answered "¿Qué es el modo IW?" from memory and wrongly ("IW (Strip Map)"), citing [1] with no passages; it asked to confirm what "last month" meant although it had today's date.
  - **Example** (`make agent`, "Are there Sentinel-1 scenes of Rotterdam from the last week, and what does the IW mode acquire?"): both calls in the first turn, with `days: 7` and a `start` as well (`days` wins); 5 scenes; the answer cites the IW sections of the wiki and of the Product Definition; 2 turns, 9.6 s in the model and 1.1 s in the tools.

## D-028: Places are geocoded with Nominatim inside `search_scenes`

- **Status:** Accepted (step 13, 2026-10-11)
- **Decision:**
  - `search_scenes` takes a place name. `geocode.py` asks OpenStreetMap's Nominatim for it and uses the bounding box of the first result to search the catalogue.
  - **Usage policy kept:** at most 1 request per second, the project's User-Agent, every answer cached in `data/geocode.json` (also "not found"), the attribution "© OpenStreetMap contributors, ODbL" printed whenever a place was used, and the server switchable with `NOMINATIM_URL`. The user chose to use the public server after reading the policy's conditions, including its clause on LLM-generated code (plan of step 13).
  - **The result names the place found,** with its OpenStreetMap type and size, and the scene table under the answer repeats it.
  - **A place under 1 km across is an error, once:** the tool asks the model for a more precise name. A second call that finds the same place goes through, for a real peak or building.
- **Alternatives considered:**
  - *A separate `geocode` tool for the model to chain:* one more step that can fail, for no gain in this project.
  - *The model gives coordinates:* it refused in the probe.
  - *The user gives a bounding box:* accurate, but nobody asks for scenes that way.
  - *The best of Nominatim's top 5 by `importance`:* the right place is not among the five for the names that failed.
  - *An offline gazetteer* (Natural Earth): only large places. *Another service* (GeoNames, which needs an account, or Photon): other terms to follow, and the same ambiguity.
  - *A warning instead of an error for small places:* measured; the model ignored it.
- **Why:** the model cannot place names on a map, and a geocoder can. Nominatim is free, has no account, covers small places, and its policy allows use triggered by a person when it is cached and limited.
- **Trade-offs:**
  - **Only the first result is used.** Short names are ambiguous: "Doñana, Spain" is a peak in Almería and "Delta del Ebro, Spain" a street. The 1 km check catches these two; a wrong place of town size would go through, visible only in the place name printed with the scenes.
  - **In mixed questions the model reports the error instead of retrying,** so the answer says that the place was not found (D-027).
  - **An external service:** it can be slow, change its policy or block heavy use. The cache keeps repeated questions offline.
  - **ODbL data:** the attribution must be shown.
- **Verified [2026-10-11]:**
  - **Usage policy** (operations.osmfoundation.org/policies/nominatim): the requirements above; "Use of the public API is only permitted where the application developer has made a deliberate, informed decision to use it"; forbidden uses include autocomplete and systematic queries. Its `robots.txt` disallows `/search` for crawlers; use as an API is governed by the policy.
  - **Candidates:** "Doñana" alone: a peak in Almería, then hamlets in Gran Canaria and Mexico; the national park only with "National Park" in the name. "Delta del Ebro, Spain": two streets, a sports centre, a restaurant and a hotel. "Madrid", "Villanueva de la Serena", "Svalbard" and "Venice, Italy" found the right place first.
  - **With the error** (D-029): single-tool questions searched again with a better name or the same one; the two mixed questions reported the error.

## D-029: The agent eval checks tool calls, with tuning and held-out questions

- **Status:** Accepted (step 13, 2026-10-11)
- **Decision:**
  - **`make eval-agent`** (`evaluate_agent.py`) runs every question in `eval/agent_questions.yaml` through the agent with the real tools and checks what it did:
    - *tools:* exactly the expected tools were called;
    - *arguments:* a scene search used the expected product type, dates and filters, where a list gives the accepted values and null means "not set";
    - *place:* the place found contains an expected point;
    - *passages:* a passage the model read comes from a section labelled in `eval/questions.yaml` (the labels are reused by id);
    - *citations:* an answerable question gets a cited answer, an unanswerable one no citations, and an answer that read no passages cites none.
    - *answered* (added in step 14, D-033): the answer is not empty.
  - A question passes when every check that applies passes. "Today" is fixed in the file (2026-10-11), so relative dates have one right answer. Scenes are not checked, because the catalogue grows.
  - **18 questions,** all drafted by Claude: 5 about documentation, 8 about scenes, 5 mixed; 10 in Spanish, 8 in English.
    - `tuning` (12): the 5 probe questions, plus 7 that were held out in the first run and then used for the fixes.
    - `held-out` (6): written after the first run and before any fix, and never used to change the prompt or the tools.
  - **Thinking stays off;** `--think` is an option. The rule agreed before the last two runs: off if it scores within one question of thinking on, which it does (13 and 13).
- **Alternatives considered:**
  - *A throwaway script,* as for the answer comparisons of steps 10–12: tool choice is what an agent adds, so its eval belongs in the repo.
  - *Judging the answers with a model:* not reliable with the only local model, which is the one being judged.
  - *Checking the scenes returned:* they change every day.
  - *Exact arguments:* "the last week" can fairly start on the 4th or the 5th, so a list of accepted values is given.
- **Why:** the new failure modes of an agent are choosing the wrong tool, giving it the wrong arguments and answering without a tool; all three are visible in its calls. A held-out set shows whether the fixes generalise: the first run passed 4 of the 5 questions that shaped the first prompt but only 2 of the 7 it had not seen.
- **Trade-offs:**
  - **Small:** one question moves the held-out score by 0.167.
  - **The tuning set measures the fixes as much as the agent.** Only the 6 held-out questions are unbiased, and they were written by the same author as the fixes.
  - **It checks behaviour, not answer quality:** an answer can pass with a weak explanation.
  - **It needs the network,** and Nominatim answers are cached in `data/`.
- **Verified [2026-10-11]:**
  - **Runs** (questions passing every check; the first design rescored from its saved answers with the citation check as it is now):

    | Run | All (18) | Probe questions (5) | Other tuning (7) | Held-out (6) | tools | arguments | place | passages | citations | Median time |
    |---|---|---|---|---|---|---|---|---|---|---|
    | First design (`…130859-agent-first-design`, `…131205-agent-first-design-held-out`) | 10 | 4 | 2 | 4 | 15/18 | 11/12 | 10/12 | 5/8 | 17/18 | 7.3 s |
    | **Final** (`…152216-agent-final`) | **13** | 3 | 5 | **5** | 18/18 | 10/12 | 10/12 | 6/8 | 17/18 | 5.9 s |
    | Final, thinking (`…152720-agent-think`) | 13 | 3 | 5 | 5 | 18/18 | 10/12 | 10/12 | 6/8 | 17/18 | 15.1 s |

  - **First design, what failed:** the place (Doñana and the Ebro delta searched as a peak and a street, and their scenes reported as if right), the dates (20 days for "the last 10 days"), the tools (no `search_docs` for the Lisbon revisit, the mission cost and the operations staff, the last two answered "not in the documentation" without searching), and the passages (thermal noise; GRD vs SLC for Venice).
  - **Tuning iterations** (the 12 tuning questions passing, with the citation check of the time):
    1. A stricter prompt, `days`, the scene summary, and the place's type and size: 6 (Svalbard and Lisbon fixed; "¿Qué es el modo IW?" answered from memory again; `mode: IW` added unasked; the place ignored).
    2. The forced search and a warning for small places: 8 (the cost question fixed; the warning ignored; an unasked `mode: SM` found nothing).
    3. The small place as an error, and `any`: 9 (no more invented filters; single-tool questions retry the place).
    4. A general "correct the call after an error" rule: 9, with the same failures, so it was removed.
    5. Removing an ineffective prompt sentence: 8 (Doñana in English left out `end` three times). A more precise error alone did not help.
    6. `start` and `end` required again: 9 of 12, 14 of 18.
    7. **A citation problem found by reading the answers:** the stricter prompt had made the model cite "[1]" after scene facts, with no passage read, in 3 and then 4 of the 8 answers about scenes only (0 of 7 in the first design). The citation check was extended to those answers, which dropped that run to 10 of 18, and a line was added to every `search_scenes` result saying that it has no number to cite: 13 of 18, with 1 of 8 still citing. The held-out `scenes-garda-fortnight` failed this check before the fix, so it is no longer unbiased for citations.
  - **Final run, what still fails:** `scenes-madrid-week` (cites "[1]"), the two mixed questions with an ambiguous place (reported, not retried), and two retrieval misses (thermal noise 6.1 for the model's query, and GRD vs SLC, the known weak spot of D-022). The two unanswerable questions pass only through the forced search.
  - **Thinking:** the same 13 questions pass. The model called `search_docs` by itself for both unanswerable questions (no forced search), and the median time per question is 15.1 s instead of 5.9.
  - **Repeatability:** a second run of an earlier configuration gave the same checks for all 18 questions and the same passages, but different tool calls for one question (Doñana in English gave `end` at once instead of on its second try) and a differently worded answer for 9. Temperature 0 does not make Ollama's GPU output exactly repeatable here.
  - Lowest available RAM during the final runs: 4.3 GiB (3.4 with thinking), out of 11.7.

## D-030: Every Ollama request asks for mmap

- **Status:** Accepted (step 13, 2026-10-11)
- **Decision:** `ollama.py` sends `use_mmap: true` in the options of every chat and embed request (`LOAD_OPTIONS`). WSL's memory was also raised from 10 to 12 GB in `.wslconfig` (on the Windows side, done by the user).
- **Alternatives considered:**
  - *Only more memory for WSL:* more margin, but Ollama would still turn mmap off whenever free RAM is below its threshold.
  - *One model at a time* (`OLLAMA_MAX_LOADED_MODELS=1`): every switch between Gemma and bge-m3 reloads a model from the hard disk, which happens several times per question.
  - *Leaving Ollama's choice:* WSL froze (below).
- **Why:** when Ollama sees little free RAM it turns mmap off, and llama.cpp then copies the whole model file into memory, which uses more RAM, not less. With mmap, the weights that run on the CPU are file pages that the kernel can drop and read again.
- **Trade-offs:**
  - **It overrides Ollama's heuristic.** If RAM really runs out, mapped pages are dropped and read again from the hard disk, which is slow, instead of the process growing.
  - **Every request must send it:** a request whose load options differ from the loaded model's makes Ollama load it again (as with `num_ctx`, D-002).
- **Verified [2026-10-11]:**
  - **The freeze:** during an agent eval, the Ollama log shows "disabling mmap for llama-server load due to host memory pressure" for Gemma (system_free 5.7 GiB, below a 7.5 GiB headroom), then the same for bge-m3 with 442 MiB free. Gemma's runner held 6.0 GB of RAM, against 3.3 GiB with mmap in step 1. The log has no line for 28 minutes (13:36:46 to 14:04:36) while WSL was unresponsive.
  - **After it,** Ollama's GPU discovery timed out on every load and it kept stale figures (4.2 GiB of free VRAM instead of 7.0) until the service was restarted. A Task Manager check showed no Windows process holding more than 140 MB of GPU memory.
  - **With `use_mmap: true`** and both models loaded at 100% GPU: Gemma's runner 3.1 GB, "used" RAM 2.0 GiB instead of 7.8, available 7.7 GiB instead of 1.9, and no "memory pressure" line in the log.
  - `make eval` reproduces `eval/results/2026-10-11T113743-pdfs.json` exactly with the new option: metrics, ranks, the 20 chunks of every question and the translations.
  - **After raising WSL to 12 GB** (11.7 GiB visible), Ollama fell back to the CPU three times: its GPU discovery has 30 s, and reading its 1.3 GB of CUDA libraries from the cold hard disk took 10.4 s (1.2 s once cached). Restarting Ollama after reading them found the GPU.

## D-031: An MCP server on stdio, built on the official SDK's low-level `Server`

- **Status:** Accepted (step 14, 2026-10-12)
- **Decision:**
  - **`make mcp`** (`mcp_server.py`) serves `search_docs` and `search_scenes` over MCP's stdio transport: a host (Claude Code, Claude Desktop, an IDE) starts it as a subprocess and sends JSON-RPC 2.0 messages on its stdin, one per line, reading the answers on its stdout.
  - **Library:** the official Python SDK, `mcp>=2.3,<3`, with its low-level `Server`. The input schemas are built from the agent's `SCHEMAS` (`tools.py`), so the agent and the server describe the same arguments; the server only changes descriptions and adds `confirm_place` (D-032).
  - **Both protocol eras:** the SDK answers the `initialize` handshake of revisions up to 2025-11-25 and the per-request metadata of 2026-07-28 on the same process, and the client's first message decides which. The server declares the `tools` capability, a title and short `instructions` for the host.
  - **Errors:** a problem the model can fix (a wrong argument, a bad date, a place not found, a service down) is a result with `isError: true` and the agent's wording; an unknown tool is a JSON-RPC error, `-32602`.
  - **Annotations:** both tools are read-only (`readOnlyHint`); `search_scenes` is open-world (`openWorldHint`), because it calls Nominatim and the Copernicus catalogue.
  - **stdout:** after the SDK has taken the protocol channel, `sys.stdout` points at stderr, so a stray `print()` can never reach the protocol. The server logs one line per call to stderr.
  - **One call at a time,** run in a worker thread so that the event loop keeps reading messages. Each call opens its own database connection; the Ollama client, the HTTP client and the geocoder (with its cache and its one-request-per-second limit) are shared.
  - **`make mcp-check`** (`mcp_check.py`) starts the server as a host would and checks it through the SDK's client, in both eras (below).
  - **`.mcp.json`** in the repository root registers the server for Claude Code, which asks before using a project's server.
  - The package version is now 0.2.0; the server reports it, and the User-Agent is built from it.
- **Alternatives considered:**
  - *The SDK's high-level `MCPServer`* (FastMCP before v2): decorators that derive each schema from the function's type hints. The schemas would be written a second time, as types, and could drift from the agent's, which were tuned on the agent eval (D-027).
  - *JSON-RPC written by hand,* as the Ollama client (D-007) and the agent loop were: about 150 lines, but the protocol changed in July 2026, and serving both eras means the handshake, per-request metadata, `server/discover`, version errors and cancellation. The agent loop already shows the mechanics of tool calling; here the point is to interoperate with real hosts.
  - *Streamable HTTP:* it needs a port, `Origin` validation and authentication, for a server that only a local host uses.
  - *Calls in parallel:* the geocoder's cache file and rate limit, and the Ollama client, are not made for threads, and one person asking questions does not need them.
  - *The MCP Inspector* to check the server: it needs Node, which is not installed in WSL.
- **Why:** MCP lets any host use the project's tools without code written for that host. The SDK keeps the server correct across protocol revisions, and the low-level `Server` takes the existing JSON Schemas as they are, which keeps one description of each tool.
- **Trade-offs:**
  - **21 more packages** (pydantic, starlette, uvicorn, jsonschema, …), because the SDK also brings its HTTP transports.
  - **stdio only:** the host must run on the same machine, inside WSL.
  - **Calls wait for each other:** a host that sends both searches at once gets them one after the other (about 1–2 s more).
  - **The host decides,** not this code: the agent's forced documentation search and its system prompt (D-027) do not exist here. The server can only describe its tools and send `instructions`, which a host may ignore.
  - **Nominatim through a host:** the policy allows lookups triggered by a person, cached and at most one per second (D-028); a host's model could make many calls in a loop, which the limit and the cache slow down but do not stop.
- **Verified [2026-10-11]:**
  - **Specification:** the current revision is 2026-07-28 (modelcontextprotocol.io). It removed sessions: there is no `initialize`, every request carries its protocol version and client capabilities in `_meta`, servers must implement `server/discover`, and "an open connection, such as a STDIO process, is not a conversation or session". Revisions up to 2025-11-25 use the handshake. stdio messages are newline-delimited, and the server "MUST NOT write anything to its stdout that is not a valid MCP message". Tool errors the model can act on go in the result with `isError: true`; an unknown tool is a protocol error. A tool with an `outputSchema` must return `structuredContent` that conforms to it.
  - **SDK:** `mcp` 2.3.0 on PyPI (2026-10-02), the v2 line since 2026-07-28; FastMCP was renamed `MCPServer` in v2. The low-level `Server` takes `input_schema` and `output_schema` as plain JSON Schema and does not validate arguments. Its `run` serves both eras (`serve_dual_era_loop`); the SDK knows revisions 2024-11-05 to 2025-11-25 (handshake) and 2026-07-28.
  - **Probe** (a throwaway server with one tool, raw JSON-RPC): a client sending `initialize` with 2025-06-18 got that version back, and `tools/list` and `tools/call` worked; a client sending `server/discover` with 2026-07-28 metadata got `supportedVersions ["2026-07-28"]`; a request without `_meta` got `-32602`. The SDK's client negotiated 2025-11-25 with `mode="legacy"` and 2026-07-28 with `mode="auto"`, received the `instructions` in both, and raised an error for `structuredContent` that did not match the `outputSchema`.
  - **stdout:** the SDK moves fd 1 to stderr while it serves and keeps the protocol on a copy, but a `print()` inside a tool stayed in Python's buffer and reached the client when the server exited, after fd 1 was put back: the client logged "Failed to parse JSONRPC message from server" once per run. With `sys.stdout = sys.stderr` after the transport opens, the line went to stderr and the client logged nothing.
  - **This server, raw JSON-RPC** in both eras: every line on stdout parsed as JSON, nothing followed the last answer, the logs were on stderr, and the server exited by itself when its stdin closed.
  - **`make mcp-check`:** all 19 checks pass, 6 with the handshake in 5.1 s (4.1 s of it the first `search_docs`, loading bge-m3) and 13 with per-request metadata in 2.6 s: the server's name and version, the instructions, the two tools with the agent's arguments plus `confirm_place`, 5 passages with their section and URL, 10 scenes over Rotterdam in September 2026 with their names in the text, `isError` for an unknown argument, a bad date and a small place (twice in a row, as nothing is remembered), the small place searched with `confirm_place`, and `-32602` for an unknown tool.
  - **Claude Code** (code.claude.com/docs/en/mcp): a project's `.mcp.json` lists servers under `mcpServers`, and Claude Code asks before using them in an interactive session; `${CLAUDE_PROJECT_DIR:-.}` is the recommended form for project paths; tool output is limited to 25,000 tokens by default.
  - **With a real host [2026-10-14]:** Claude Code, in a new session in the repository, started the server from `.mcp.json` once the user approved it, and listed both tools with their schemas (`confirm_place` included); the server's `instructions` reached the model. Four questions, 7 calls, all answered:
    - *Connection:* Claude Code's log says that its `server/discover` probe timed out while the server was starting, the server then rejected `initialize` (its first message had already been the modern probe), and a second `server/discover` connected on revision 2026-07-28 without restarting the process, in 4.7 s. That was the first start after a reboot, from the cold hard disk; warm, the server answers `server/discover` 1.0 s after launch (0.73 s of it importing the SDK).
    - *What the model reads:* Claude Code passes the `structuredContent` JSON to its model rather than the text block, and an `isError` result as an error with its message.
    - *"¿Qué diferencia hay entre un producto GRD y uno SLC?":* two `search_docs` calls with English queries. The first returned the product-family sections, the known weak spot (D-022); a second, more specific query found Product Definition 5.2.1 (each SLC pixel is a complex I/Q value with amplitude and phase), which the single search of `make ask` does not reach for this question.
    - *Madrid, last week:* one call with `days: 7` (and start and end, which are required); 4 scenes over the city.
    - *Rotterdam and the IW mode:* both tools called at once, served one after the other; 5 scenes, and the IW sections of the wiki and of the Product Definition.
    - *Doñana, SLC, September 2026:* the small-place error for "Doñana, Spain" (a peak), then a second call with "Parque Nacional de Doñana, Huelva, Spain", which found the national park and its 10 newest SLC scenes (the catalogue has more); `confirm_place` was not needed.
    - *Times:* the first `search_docs` took 15 s (bge-m3 loading after the reboot), the other calls 0.2–0.5 s.
    - The host's model was Claude, which knew how the server was designed, so this shows that the server works in a real host, not how an unknown model uses it.

## D-032: The server's tools stand alone on every call, and the host's model writes the query in English

- **Status:** Accepted (step 14, 2026-10-12)
- **Decision:**
  - **No state across calls,** as revision 2026-07-28 requires: every call gets a new `SceneFinder` and its own database connection.
    - *Passages* are numbered 1 to 5 within each result, and each carries its section path and URL, so the host's model can cite a link. In the agent, numbers run across the whole conversation and the code prints the sources (D-027); the server cannot print anything in the host.
    - *A place under 1 km across* is rejected unless the call says `confirm_place: true`, a new optional argument that only the server has. The agent keeps its rule: the same place asked for twice goes through (D-028). The size check lives once, in `SceneFinder.find`, which raises `SmallPlace`; the agent and the server each add their own way to confirm.
  - **`search_docs` without a language model:** the query description asks for English, the language of the documents, and the server runs the default retrieval (hybrid, BM25, top 5) with that one query for both halves. Translating is left to the host's model, so the server only needs bge-m3 and Postgres, not Gemma.
  - **Results:** text for the host's model plus `structuredContent` that conforms to the tool's `outputSchema`.
    - `search_docs`: the passages with number, section, URL and text.
    - `search_scenes`: the place found (name, OpenStreetMap type, bounding box), the dates and filters searched, the agent's summary, the scenes with their product names and metadata URLs, whether the catalogue has more, and the OpenStreetMap attribution. The text also lists the scenes: hiding their names was for Gemma, which copied them slowly (D-027), and a host's user sees only what the host's model writes.
  - **Shared code:** `tools.py` gains `SceneFinder` (the argument checks, the geocoding, the size check and the catalogue search, moved out of `Toolbox` unchanged), `SmallPlace` and `scene_line`, the table row that `make agent` already printed. The agent's `Toolbox` uses them, with the same results.
- **Alternatives considered:**
  - *One `Toolbox` per server process:* numbers and the "asked twice" rule would carry over between unrelated conversations of the host, and the specification says a server must not treat a connection as a session.
  - *Dropping the size check in the server* and leaving it to the host's model, which sees the place's type and size: it would ask Nominatim's first match for "Doñana, Spain", a peak, without a word.
  - *Translating with Gemma in the server,* as `make ask` does: the same 22 of 25 (below), but every host would load a 6 GB model, taking about a minute from the hard disk the first time, only to translate.
  - *Text results only:* a client could not show the sources or the scenes from data, only from what the model wrote.
  - *Hiding the scene names, as for the agent:* a host's model would have no names to give the user.
- **Why:** the agent's tools assumed one conversation and a model that the project controls; a server talks to unknown hosts and models, one call at a time. What the agent kept in memory now travels in the arguments or in the result. The English query was measured before it was chosen.
- **Trade-offs:**
  - **A non-English query weakens the full-text half.** Vector search crosses languages, BM25 does not: with the Spanish questions as they are, the hybrid search finds 18 of 25 in its top 5 instead of 22 (D-026, "hybrid, not translated"). The schema and the instructions ask for English, but the host's model decides.
  - **The host's English is not measured:** Gemma's translations stood in for it below.
  - **Longer results:** every passage carries its URL (about 200 characters for a PDF), and a scene search lists up to 10 product names.
  - **A small place needs one more call,** and the model has to read the error to learn about `confirm_place`, which is also in the schema.
- **Verified [2026-10-11]:**
  - **English query for both halves** (throwaway script, the 25 eval questions; Gemma's translations from `eval/results/2026-10-11T113743-pdfs.json` stand in for the host's query, and English questions stay as they are): hit@1 0.760, hit@3 0.840, hit@5 0.880 (22 of 25), hit@10 0.920, MRR@5 0.810, the same as the cited run. Three ranks moved: `grd-vs-slc` 13th to 12th, `iw-swath-resolution` 1st to 2nd, `orbit-files` 2nd to 1st. The rule agreed in the plan was to keep translating in the server if hit@5 dropped by more than one question.
  - **Through the protocol:** the same 25 queries sent to the running server as `search_docs` calls (SDK client, revision 2026-07-28) gave the same top-5 ranks for every question: 22 of 25, MRR@5 0.810.
  - **The agent is unchanged:** 18 `Toolbox` calls (scene searches with every kind of argument, the errors, a small place asked for twice, an unknown tool and a wrong argument) returned identical text before and after the refactor, compared as JSON. Its eval on v0.2 is in D-033.

## D-033: The agent eval fails empty answers, and `make agent` says when the answer is empty

- **Status:** Accepted (step 14, 2026-10-12)
- **Decision:**
  - `make eval-agent` gains the check *answered*, for every question: the agent's final answer is not empty. Before it, an empty answer to a question about scenes only passed, because the citation check for those questions asks only that no `[n]` is cited.
  - `make agent` prints "The model ended its turn without writing an answer" instead of an empty line.
  - The loop does not ask the model again after an empty answer: that would change the agent's behaviour, which the eval would have to measure again, and is left for a later step.
- **Alternatives considered:**
  - *Only documenting it:* the eval would keep counting an empty answer as a pass.
  - *Retrying in the loop now:* one more turn after an empty answer. It probably fixes this question, but it needs its own measurement, and v0.2 was meant to add the MCP server, not change the agent.
- **Why:** a check that an empty answer passes measures less than its name says. The step-13 runs had no empty answer, so their numbers stand.
- **Trade-offs:** the Madrid scenes question now fails every run, until the loop handles empty answers.
- **Verified [2026-10-11 and 2026-10-12]:**
  - **First v0.2 run** (`eval/results/2026-10-11T181107-agent-v0.2.json`, before the check): the checks counted 14 of 18, against 13 in the cited step-13 run (`…152216-agent-final`). The extra pass was `scenes-madrid-week` ("¿Hay imágenes Sentinel-1 GRD de Madrid de la última semana?"), whose answer was empty: Gemma ended its second turn at once (`eval_count` 1, `done_reason` "stop"), and an empty answer cites no "[1]". One other question made different calls ("Doñana" instead of "Doñana, Spain", with the same checks) and 7 answers were worded differently.
  - **Not caused by step 14:** the empty answer came back in 5 of 5 runs of `make agent` with the new code, in 2 of 2 with the code of step 13 (`git archive HEAD`), and with Gemma unloaded first. The cause is not known: the search finds the same 3 scenes on the same days that the cited run's answer reports, and temperature 0 is not exactly repeatable here (D-029), so a small difference in the tool result or in Ollama's state may be enough to make the end of the turn the most likely first token.
  - **The step-13 runs:** none of their 72 answers is empty or missing, so *answered* would pass for all of them and their scores do not change.
  - **With the check** (`eval/results/2026-10-12T143532-agent-v0.2-answered.json`): 13 of 18, held-out 5 of 6, *answered* 17 of 18. The failures are those of the cited step-13 run, except that `scenes-madrid-week` now fails *answered* instead of *citations*. Lowest available RAM: 7.9 GiB in both v0.2 runs.

## D-034: OpenSearch as a second search backend, fed from Postgres

- **Status:** Accepted (step 15, 2026-10-14)
- **Decision:**
  - **Service:** `opensearchproject/opensearch:3.9.0` in `docker-compose.yml`, under the Compose profile `opensearch`, so that `make up` still starts Postgres alone. One node (`discovery.type: single-node`), the security plugin disabled (no TLS, no users), the port published on `127.0.0.1` only, a fixed 1 GB heap, and its own volume. `make opensearch-up` starts it; `make down` and `make clean` now include it.
  - **Index:** `make opensearch-index` (`opensearch.py`) copies every chunk from Postgres into the index `sentinel1-chunks`: chunk id, section, position, URL, text, the field `text` with the same words as Postgres's `tsv` (section and content), and the stored bge-m3 vector. Nothing is embedded again, so both backends search identical vectors. The index is rebuilt whole each time (a few seconds), with one shard, no replica and one merged segment.
  - **In sync:** the index's `_meta` keeps a fingerprint of Postgres's chunks (their number and a hash of their ids, which change whenever a page is indexed again). `--backend opensearch` stops with "out of date… run `make opensearch-index`" when it does not match, and with a clear message when OpenSearch is down or the index is missing.
  - **Use:** `--backend pgvector|opensearch` in `make ask` and `make eval`, with the same methods, translation and top-k; results files record the backend. Two combinations stop before any model call: `ts_rank` with OpenSearch, and `hybrid-native` with pgvector.
  - **Client:** raw REST with httpx, as for Ollama, STAC and Nominatim (D-007).
  - **pgvector stays the default:** the default retrieval finds the same answers on both (D-035, D-036), and Postgres is lighter.
- **Alternatives considered:**
  - *`opensearch-py`:* one more dependency, and it hides the requests that this step is about.
  - *Embedding the chunks again for OpenSearch:* about a minute of bge-m3, for the same vectors; and the comparison would also depend on the embedding being repeatable.
  - *Updating the index page by page,* as `make index` does: more code for 568 chunks that load in seconds.
  - *OpenSearch 2.19.6,* the maintained 2.x line: 3.9 is the current release.
  - *The security plugin with the demo configuration:* it needs an admin password and TLS for a service that listens only on localhost.
  - *Replacing pgvector:* the measurements gave no reason to (below).
- **Why:** to compare the engines, everything else must be equal: the same chunks, the same vectors, the same queries and the same eval. Copying from Postgres guarantees it, and the fingerprint keeps the copy honest.
- **Trade-offs:**
  - **Heavier:** a 2.88 GB image (pgvector's: 665 MB), 1.4–1.6 GiB of RAM for the container (Postgres: 30–75 MiB), and 36–64 s to start (Postgres: about 13 s).
  - **A second copy** to rebuild after every `make index` that changes something.
  - **No security:** acceptable only because the port is bound to localhost.
- **Verified [2026-10-14]:**
  - **Image:** 3.9.0 is the newest stable tag on Docker Hub (2026-09-29; 2.19.6 is the 2.x line). It reports Lucene 10.5.1 and JVM 25, a 1 GiB heap, and the plugins `opensearch-knn` and `opensearch-neural-search`. With `DISABLE_SECURITY_PLUGIN` and `DISABLE_INSTALL_DEMO_CONFIG` it starts without `OPENSEARCH_INITIAL_ADMIN_PASSWORD` and answers over plain HTTP. `vm.max_map_count` in WSL is already 1048576.
  - **Defaults of a new k-NN index:** `index.knn.algo_param.ef_search` 100, `index.knn.advanced.approximate_threshold` 0 (graphs are built even for small segments), one replica.
  - **Loading:** 568 chunks in 3.5–7.5 s (reading Postgres 0.2–0.3 s, bulk 1.9–2.9 s, merge and refresh 1.4–1.5 s); 5.1 MiB on disk, one segment.
  - **Checks:** a changed fingerprint, an unreachable URL and a missing index each stop with their message; `make ask --backend opensearch` answers with cited sources; `make eval` on pgvector still reproduces `eval/results/2026-10-11T113743-pdfs.json` exactly; after `make down` and `make opensearch-up` the index and the search pipeline are still there and in sync.

## D-035: k-NN with Faiss's HNSW and pgvector's parameters; the vector results are identical

- **Status:** Accepted (step 15, 2026-10-14)
- **Decision:** the `embedding` field is a `knn_vector` of 1024 dimensions, `space_type: cosinesimil`, method `hnsw` on the `faiss` engine (OpenSearch's default), with pgvector's parameters: `m` 16, `ef_construction` 64, and `ef_search` 40 instead of OpenSearch's 100. OpenSearch's score, (2 − d) / 2 for a cosine distance d, is turned back into the distance, so `--show-context` prints the same numbers as with pgvector. Ties are ordered by section and position, as in Postgres (D-020).
- **Alternatives considered:**
  - *The Lucene engine:* OpenSearch's other HNSW implementation, which ignores `ef_search` and uses k. Faiss is the default and was the first to try; with recall 1.000 there was nothing for another engine to improve.
  - *OpenSearch's default `ef_search` (100):* a larger search than pgvector's, which would mix two differences in one comparison.
  - *Exact search only* (`script_score`): used here to measure recall, not as the search.
- **Why:** with the same vectors and the same graph parameters, any difference would come from the HNSW implementations; there turned out to be none in the results.
- **Trade-offs:** at 568 chunks HNSW is not needed by either engine (D-006); this measures agreement, not speed at scale. Over HTTP, a vector search costs OpenSearch about 3 ms more than Postgres.
- **Verified [2026-10-14]:**
  - **Scores:** for the five nearest chunks of a question, OpenSearch's score equals (2 − d) / 2 for pgvector's distance d, to 4 decimals, with the same chunks in the same order.
  - **Eval, vector search** (`eval/results/2026-10-14T214507-os-vector.json` against `…113745-pdfs-vector`): the same 20 sections in the same order for all 25 questions, scores within 0.0001; hit@5 0.800 and MRR@5 0.673 on both.
  - **Recall against exact search** (25 eval questions): OpenSearch's HNSW 1.000 at 5, 20 and 40 neighbours; pgvector's HNSW (forced with `enable_seqscan = off`, `ef_search` 40) 1.000, 0.998 and 0.993, worst question 0.975 at 40. pgvector's planner uses the exact scan at this size anyway (D-006), which is what the backend comparison runs against.
  - **Memory of the graph:** one graph, 2,356 KB of native memory (outside the JVM heap).
  - **Latency** (throwaway script; the question vectors computed beforehand, 25 questions × 5 runs, from Python): vector search median 3.6 ms on pgvector (exact scan), 6.5 ms on OpenSearch.

## D-036: OpenSearch's BM25 uses Postgres's stop words; its own hybrid query is measured too

- **Status:** Accepted (step 15, 2026-10-14)
- **Decision:**
  - **Analyser:** the field `text` uses a custom analyser that rebuilds OpenSearch's `english` (standard tokenizer, English possessives, lowercase, stop words, Porter stemming) with Postgres's 127 English stop words (the Snowball list of `english.stop`) instead of Lucene's 33. BM25 keeps Lucene's defaults, k1 1.2 and b 0.75, as in D-023.
  - **Hybrid search:** `--method hybrid` fuses OpenSearch's two lists in Python, with the same reciprocal rank fusion as pgvector (D-020), so that only the engines differ. `--method hybrid-native` sends OpenSearch's own `hybrid` query (a BM25 `match` and a `knn` query, 40 candidates each) through a search pipeline with `score-ranker-processor`, technique `rrf`, rank constant 60, which `make opensearch-index` creates.
- **Alternatives considered:**
  - *OpenSearch's built-in `english` analyser:* measured. Its short stop list keeps question words ("what", "how", "does", "which", "why", "can", "from") as search terms, and BM25 found 19 of 25 answers in its top 5.
  - *Snowball's English stemmer (Porter2),* which Postgres's `english_stem` uses, instead of Porter: no change at all on the eval.
  - *Fusing by normalised scores* (`normalization-processor`, min-max or L2), OpenSearch's other hybrid option: a different method from the one measured in steps 8–11.
- **Why:** with the built-in analyser the comparison measured two stop lists rather than two engines. With Postgres's list, BM25 on OpenSearch finds more answers in its top 5 than BM25 in SQL, and the hybrid search stays as good.
- **Trade-offs:**
  - **A custom analyser** is one more thing to know about the index, and differs from what an OpenSearch user would assume.
  - **Not identical to Postgres:** OpenSearch splits "Sentinel-1" into "sentinel" and "1" where Postgres keeps "-1" (and adds compounds such as "ground-rang"), and Lucene stores each field's length approximately, in one byte: one chunk of 169 words in Postgres counts as 152 in OpenSearch's BM25. The lexical top 5 still differs for 12 of the 25 questions.
  - **BM25 alone is better at 5 and worse at 10:** hit@10 0.840 against 0.880 in Postgres.
- **Verified [2026-10-14]:**
  - **Analysis of the 25 translated questions:** all 25 get other terms from the built-in analyser than from Postgres: question words, "1" for "-1", "us" for "use".
  - **BM25 with the question translated** (throwaway script for the analyser variants, then `eval/results/2026-10-14T214621-os-lexical-en.json`), hit@5 / MRR@5 / hit@10, and the questions with the same top 5 as Postgres:

    | Full-text search | hit@5 | MRR@5 | hit@10 | Same top 5 as Postgres |
    |---|---|---|---|---|
    | Postgres, BM25 in SQL (`…113757-pdfs-lexical-en`) | 0.800 | 0.663 | 0.880 | — |
    | OpenSearch, built-in `english` analyser | 0.760 | 0.605 | 0.880 | 2 of 25 |
    | OpenSearch, Postgres's stop words | **0.840** | **0.687** | 0.840 | 13 of 25 |
    | the same, with Snowball's stemmer | 0.840 | 0.687 | 0.840 | 13 of 25 |

    With Postgres's stop words, the introduction of the TOPS deramping note rises from 10th to 2nd; `thermal-denoising` falls from 9th to 15th.
  - **Hybrid search** (`…214725-os-hybrid`, `…214741-os-hybrid-native`, against `…113743-pdfs`): hit@5 0.880 and MRR@5 0.810 for both, as on pgvector; the first correct chunk at the same rank for 23 of 25 questions (`grd-vs-slc` 13th → 15th, `tops-deramping` 10th → 6th); the same top-5 sections in the same order for 18 of 25, and the same sections in another order for 3 more. With the built-in analyser the metrics were the same but the top 5 agreed for only 8 questions. Python RRF and the native query give the same top 5 for all 25 questions; below 5 they differ for 11, where equal ranks are broken differently.
  - **Answers** for the 4 questions whose hybrid top 5 has other sections (`iw-swath-resolution`, `level2-ocean`, `sigma0-calibration`, `thermal-denoising`), 2 runs on each backend: 6 of 8 cite a correct section on both, and `sigma0-calibration` fails on both, as before (D-026). The answers say the same, in other words.
  - **Explain:** for one chunk, OpenSearch's BM25 uses dl 152 and avgdl 112.9 where the SQL uses 169 and 113.9; IDF follows the same formula.
  - **Latency** (as in D-035): BM25 median 44.9 ms in SQL (corpus statistics on every query, D-023) against 4.9 ms in OpenSearch; hybrid 47.0 ms on Postgres, 15.2 ms on OpenSearch with Python RRF (two requests) and 9.0 ms with the native query (one).
  - **Memory:** the OpenSearch container used 1.45–1.65 GiB, Postgres 33–73 MiB; the lowest available RAM during the step's runs, with Gemma, bge-m3, Postgres and OpenSearch loaded, was 7.1 GiB of 11.7.
