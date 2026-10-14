# Step 15: OpenSearch k-NN compared with pgvector

*Built, measured and approved by the user on 2026-10-14; released as v0.3. The last planned extension in the README, chosen by the user.*

## What this step was about

Until now every search ran in Postgres: pgvector for the vectors and a BM25 written in SQL for the words. OpenSearch is a search engine built on Lucene. It keeps documents in an inverted index with BM25 built in, and its k-NN plugin adds vector fields indexed with HNSW, the same graph algorithm as pgvector, in another implementation (here Faiss). The question was whether swapping the engine changes what reaches the model.

A comparison like this is only worth something if nothing else changes. So OpenSearch did not get its own embeddings. `make opensearch-index` copies every chunk from Postgres, its text and its stored bge-m3 vector, into one OpenSearch index. Both engines then answer the same questions over identical data, measured with the same eval. Any difference comes from the engines: the approximate vector search, the way text is cut into words, or the fusion of the two lists.

## What we built

OpenSearch 3.9 runs in Docker next to Postgres, but only when asked (`make opensearch-up`): it needs about 1.5 GB of memory and up to a minute to start, so `make up` stays as light as before. It runs as one node, with its security plugin off and its port open only on this machine. The code talks to it with plain HTTP requests, as with Ollama.

`--backend opensearch` makes `make ask` and `make eval` search OpenSearch instead of Postgres, with the same methods and options. Since the index is a copy, it remembers which chunks it was copied from, and the code refuses to search a copy that no longer matches Postgres. OpenSearch also has a hybrid query of its own, which merges the two lists inside the engine with the same reciprocal rank fusion that our Python code uses; `--method hybrid-native` tries it.

## What we found

**Vector search came out identical.** With the same graph settings as pgvector, OpenSearch returned the same 20 sections, in the same order, for all 25 eval questions, with the same scores to four decimals. Its graph found exactly what a brute-force search finds, even 40 results deep. At this size, an approximate search does not lose anything in either engine.

**Full-text search differed, and the reason was stop words.** BM25 in OpenSearch with its built-in English analyser found fewer answers in its top 5 than our SQL version (19 against 20), and agreed with it on the top 5 for only 2 questions. Comparing the words each engine took from the questions showed why: OpenSearch's English stop list has 33 words, Postgres's has 127, so OpenSearch kept "what", "how", "does" and "which" as search terms. Our questions are full of them. With Postgres's stop words in OpenSearch, its BM25 found 21 answers in its top 5, one more than Postgres, and the introduction of the TOPS deramping note, which had ranked 10th, came 2nd. What still differs comes from details: the two engines split "Sentinel-1" differently, and Lucene stores each chunk's length approximately, so a chunk of 169 words counts as 152.

**The default hybrid search ended up the same.** The vector half is identical and the BM25 half is close, and after the fusion both engines score the same: 22 of 25 in the top 5, and the first correct chunk at the same rank for 23 questions. For the 4 questions whose top 5 changed, the answers were just as correct on both engines. OpenSearch's own hybrid query gave the same top 5 as our fusion in Python.

**The costs are elsewhere.** OpenSearch's BM25 was about 9 times faster than our SQL version, which recomputes the corpus statistics on every query (5 ms against 45). Postgres was faster for vectors (3.6 ms against 6.5), because it scans 568 vectors in place while OpenSearch adds an HTTP round trip. But every search takes milliseconds, and generating the answer takes seconds. Meanwhile OpenSearch needs about 1.5 GB of memory, against well under 100 MB for Postgres, and a 2.9 GB image. So Postgres stays the default.

## What it means for the next steps

With this step, all five planned extensions are done. The comparison taught as much about configuration as about engines: the largest difference came from a list of stop words, not from the vector index or BM25 itself. At a much larger corpus the balance could change. Our SQL BM25 would get slower as the corpus grows, while OpenSearch is built for that. The next steps are open: the smaller experiments in the README (a reranker among them), and the agent's follow-ups.

## Check your understanding

**To compare two search engines fairly, what has to stay the same, and how was that ensured here?**
Everything except the engine: the chunks, their text, the embeddings, the queries and the eval. Here OpenSearch was filled from Postgres, copying each chunk's text and its stored vector, so no embedding was computed twice. The questions and their English translations were the same, and the index remembers which chunks it was copied from. Even so, the first comparison mostly measured a configuration choice, the stop-word list, until the two analysers were aligned.

**Why can BM25 rank differently in two engines that use the same formula?**
The formula scores the words that the analyser produces, so the analyser matters as much as the formula: which words count as stop words, how words are split (hyphens, numbers) and how they are stemmed. The statistics can also differ: Lucene stores each document's length approximately, in one byte, while our SQL counts it exactly. Here the stop words made the largest difference. With the built-in list, question words like "what" and "how" became search terms.
