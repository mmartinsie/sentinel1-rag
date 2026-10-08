# Step 1: Ollama and the two models

*Finished on 2026-10-08.*

## What this step was about

Ollama is an *inference server*: a program that loads language models, decides how much of each model goes on the GPU, keeps models in memory for a while after use, and answers requests through a local web API (port 11434). Our RAG talks to it over plain HTTP.

We use two models with very different jobs:

- **Gemma** is a *generator*: it writes text one token (a piece of a word) at a time.
- **bge-m3** is an *encoder*: it reads a whole text and returns a single vector of 1024 numbers. Texts with similar meaning get vectors that point in similar directions, and that is what makes searching by meaning possible.

Two things had to be true at the end of the step:

- Gemma had to run entirely on the GPU ("100% GPU").
- bge-m3 had to return vectors of exactly 1024 numbers, because the database column will be declared with that size and will reject vectors of any other size.

## What we did

1. **Read the installer before running it.** Ollama's official install script runs with administrator rights, so we read it first. The key point is that on WSL it stops as soon as it detects the GPU, *before* the part that would install NVIDIA drivers. Installing Linux GPU drivers inside WSL would break WSL's access to the Windows driver.
2. **Installed Ollama 0.40.1** as a background service that starts automatically. Its log shows the GTX 1070 detected (CUDA compute capability 6.1) and served by Ollama's CUDA 12 build. Ollama also ships a CUDA 13 build, but CUDA 13 no longer supports this generation of cards.
3. **Downloaded both models:** about 7.3 GB in total, in a little over a minute.
4. **Ran a throwaway test script** that uses the models the way the RAG will. The script is not part of the repository. It ran Gemma with an 8192-token context and a low temperature, sent it a prompt as large as a real RAG prompt, and ran bge-m3 on a mix of Spanish and English sentences.

## What we found

**Gemma runs fully on the GPU.** `ollama ps` reports "100% GPU" with an 8192-token context, and the loading log confirms that every layer is on the graphics card (43 of 43). The memory split is interesting:

- **On the GPU:**
  - 2.6 GiB of weights, 168 MiB of KV cache and about 110 MiB of working buffers.
  - The image and audio encoders, about 1 GiB more, even though we only send text.
- **In normal RAM:** 2.7 GiB of lookup tables (the input embeddings). Looking up a row in a table needs no heavy maths, so these tables don't need the GPU. This is what the "E" in "E4B" means: about 4.5 billion *effective* parameters do the heavy work, out of 7.5 billion in total.
- **Why the KV cache is so small for an 8K context:** most of Gemma 4's layers only look at the last 512 tokens (a *sliding window*), and 18 of its 42 layers reuse another layer's cache.

**It is fast enough for interactive use.** Gemma reads prompts at about 835 tokens per second and writes at about 41 tokens per second. A RAG-sized prompt (3,700 tokens) takes about 4.5 seconds to read. A typical answer of about 150 tokens should therefore take around 8 seconds once the model is loaded.

**The first load is slow.** Loading Gemma from scratch took about 70 seconds because the WSL disk lives on a mechanical hard drive. After that, Ollama keeps the model in memory for 5 minutes.

**Ollama's default context is too short for us.** On an 8 GB card, Ollama picks a 4096-token context by default. That would cut a RAG prompt with five chunks, so every request must ask for 8192 tokens explicitly.

**bge-m3 behaves as expected.**
- The vectors have 1024 numbers, and each one has length 1.0: they are *normalized*. For normalized vectors, cosine similarity and the plain dot product give the same ranking.
- A quick cross-language test also worked. A Spanish question about GRD vs SLC products scored 0.63 against an English passage that answers it. It scored only 0.27 against an unrelated English sentence and 0.22 against an unrelated Spanish one. Meaning matters more than language, which is the reason we chose this model.

**Both models fit on the GPU at the same time** (5.4 of 8 GiB used). Answering a question needs both: bge-m3 first turns the question into an embedding, then Gemma writes the answer. Because both fit, neither has to be unloaded in between.

**Thinking mode is off.** Gemma 4 can "think" before answering. We turn that off with `think: false`, and the responses came back without any thinking text.

**A first hint of grounding.** In one test, the text we gave Gemma stopped just before the section that lists the acquisition modes. Gemma only repeated what the text said and did not invent the missing mode names.

## What it means for the next steps

- Every Gemma request will send `num_ctx: 8192`, `temperature: 0.2` and `think: false`.
- Embedding requests will send `truncate: false`, so that an oversized chunk fails loudly instead of being cut silently.
- For long evaluation runs, we may ask Ollama to keep the models loaded for longer than 5 minutes, to avoid the 70-second reload.
- Memory is fine. With both models loaded there is still room for Postgres, which comes in step 2.

## Check your understanding

**The embeddings come back normalized. What does that change for search?**
A normalized vector has length 1, so the cosine of the angle between two vectors equals their dot product. Ranking by cosine distance or by inner product therefore gives the same order. We keep cosine anyway, because it stays correct even if a future model returns vectors that are not normalized.

**What does the size of the KV cache depend on, and why is it so small here?**
It grows with the number of layers, the number of key/value heads, the size of each head and the number of tokens in the context. Roughly: 2 × layers × KV heads × head size × tokens × bytes per number. In Gemma 4 E4B it is only 168 MiB at 8,192 tokens, because most layers only attend to a 512-token window and 18 of the 42 layers reuse other layers' caches. A classic model of similar size would need several times more memory.
