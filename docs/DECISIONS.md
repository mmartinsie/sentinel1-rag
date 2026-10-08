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

## D-002: LLM is `gemma4:e4b-it-qat`

- **Status:** Accepted (step 0, 2026-10-08). GPU fit still to be confirmed in step 1.
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
- **Trade-offs:** a model with about 4.5B effective parameters reasons and writes worse than 12B+ models. The download includes a vision/audio projector of about 1 GB that we will not use. It requires Ollama ≥ 0.30.5.
- **Verified [2026-10-08]** (Ollama registry manifests and GGUF header):
  - `e4b-it-qat`: Q4_0, 7.5B total parameters, 5.16 GB of weights plus a 0.99 GB projector, 6.15 GB in total. Requires Ollama ≥ 0.30.5.
  - `e2b-it-qat`: Q4_0, 4.34 GB in total.
  - Default sampling parameters are temperature 1, top_k 64 and top_p 0.95, so temperature must be overridden.
  - Architecture: 42 layers, 18 of which reuse another layer's KV cache, and sliding-window attention (512 tokens) on most layers, with a maximum context of 128K. The KV cache at 8K context should therefore be small.

## D-003: Embeddings with `bge-m3` via Ollama

- **Status:** Accepted (step 0, 2026-10-08). Dimension still to be confirmed with a real call in step 1.
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
