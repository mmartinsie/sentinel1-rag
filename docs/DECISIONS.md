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
