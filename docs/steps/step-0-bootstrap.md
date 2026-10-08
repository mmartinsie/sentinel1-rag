# Step 0: Bootstrap and environment

*Finished on 2026-10-08.*

## What this step was about

Before writing any RAG code, we wanted two answers: can this machine run the plan, and do the plan's assumptions hold up?

A RAG system works in two halves. The offline half downloads the documents, cuts them into chunks, turns each chunk into an *embedding* (a vector of numbers that captures its meaning) and stores those vectors in a database. The online half turns the user's question into an embedding too, fetches the chunks whose vectors are closest to it, and hands them to a language model with the instruction to answer only from them and to cite them. The model knows nothing about Sentinel-1 by itself; it rephrases the context we give it. That is what makes citations, and an honest "I can't find it", possible.

Running everything locally adds one hard constraint: the language model has to fit in the graphics card's memory (VRAM). It needs room for its weights and for a *KV cache*, a working memory that grows with the length of the prompt. If it doesn't fit, part of the model runs on the CPU and answers get several times slower.

## What we checked on the machine

The GPU is visible from inside WSL: a GTX 1070 with 8 GiB, of which about 7.5 GiB are free (the Windows desktop uses the rest). WSL is capped at 10 GB of RAM, there is plenty of disk space, Docker Engine runs natively inside WSL, and uv with Python 3.12 is ready. Ollama was not installed anywhere yet.

## How we checked the plan

Rather than trusting model names and sizes from memory, we asked the sources directly:

- The Ollama registry confirmed that `gemma4:e4b-it-qat` exists and weighs 6.15 GB. Its weights are 4-bit "QAT", meaning the model was trained to cope with 4-bit precision. The fallback, `gemma4:e2b-it-qat`, exists too and weighs 4.34 GB.
- We read only the first few megabytes of the model files (an HTTP *range* request, with no full download). That showed that `bge-m3` produces vectors of exactly 1024 numbers, which is the size the database column must have.
- Docker Hub showed that the newest pgvector image is version 0.8.7 on Postgres 18.
- The SentiWiki allows crawlers in its `robots.txt` and serves plain HTML, so no JavaScript rendering is needed. Every heading has a stable anchor, so citations can link straight to a section.

## What changed from the original plan

- **Ollama runs inside WSL, not on Windows.** The fallback idea of running it on Windows relied on a WSL networking mode ("mirrored") that only exists on Windows 11, and this machine runs Windows 10. Running it inside WSL is simpler anyway, since the GPU is already visible there.
- **The corpus is 5 long pages, not 15–20 short ones.** The Sentinel-1 SentiWiki is a landing page plus four chapters (mission, products, processing and applications). Together they hold about 90 headed sections and roughly 28,000 words. That is the amount of text we expected, packed into fewer pages, so sections rather than pages will be the real unit.
- **The proposed database schema has two gaps.**
  - It had no column for the anchor that citations link to.
  - The plan was to skip re-indexing a page whose content hasn't changed. That check would also skip re-indexing when *we* change how pages are chunked, which is exactly what the evaluation step will do. The page's fingerprint must therefore also cover the chunking settings and the embedding model.
- **Postgres 18 changed where it stores data.** The official image now keeps its data in a different folder, so the storage volume has to be mounted at `/var/lib/postgresql`.

## What we built

We built the project skeleton:

- A Python package managed with uv, with the five minimal dependencies.
- A `.gitignore` that keeps the downloaded corpus and private notes out of the repository.
- A short README, a decision log (`docs/DECISIONS.md`) and a progress tracker (`docs/PROGRESS.md`).

The first commit went to GitHub, authored with GitHub's "noreply" address, so no personal email appears in the public history.

## What it means for the next steps

Step 1 installs Ollama inside WSL and checks the two models for real. The findings about the corpus, the schema and the fingerprint will come back in steps 2 to 4.

## Check your understanding

**How do you know whether a language model fits on your GPU, and what happens if it doesn't?**
Add up three things: the weights (parameters × bits per weight ÷ 8), the KV cache (which grows linearly with the context length) and some working buffers. Then compare the total with the free VRAM. If it doesn't fit, some layers run on the CPU. Generating text is limited by memory bandwidth, so every token gets several times slower. That is why we chose a 4-bit model of about 6 GB for a card with 7.5 GiB free, and why step 1 checks it with `ollama ps`.

**Why Postgres with pgvector instead of a dedicated vector database?**
With a few hundred vectors any option is fast enough, so practical reasons decide. Postgres is one system we already know, and it keeps metadata and vectors in the same transaction, with joins and plain SQL. Dedicated vector databases pay off with tens of millions of vectors, heavy filtering, vector compression or sharding. And pgvector already offers approximate indexes (HNSW and IVFFlat) if we need them.
