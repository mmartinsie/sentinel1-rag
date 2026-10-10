# Step 11: BM25 for the full-text search

*Finished on 2026-10-11. A follow-up to step 10, chosen by the user.*

## What this step was about

Full-text search needs a way to rank the chunks that share words with the question. Postgres's built-in `ts_rank` mostly counts how often those words occur. It has no idea how *common* a word is: "sentinel" appears in two thirds of the chunks and "vignettes" in a handful, yet a match on either counts the same. Step 8 found exactly that weakness.

**BM25** is the standard answer, used by default in Lucene, Elasticsearch and OpenSearch. For every question word a chunk contains, it adds three things multiplied together:

- **IDF (inverse document frequency):** how rare the word is across the chunks. A word in almost every chunk adds almost nothing.
- **Saturated term frequency:** more occurrences add more, but with diminishing returns set by **k1** (1.2). The fifth "SLC" in a chunk adds much less than the first, so a chunk cannot win by repetition alone.
- **Length normalization:** set by **b** (0.75). Long chunks contain more words just by being long, so their counts are discounted relative to the average length.

Postgres has no BM25, so this step wrote it in SQL.

## What we built

- **`BM25_QUERY`** in `retrieval.py`: one SQL query that takes the question's words, counts in how many chunks each one appears and how often it appears in each chunk, measures every chunk's length, and adds up the BM25 formula. The statistics are computed from the existing full-text column on every query, so no new table is needed. The parameters and the IDF formula are Lucene's, checked against its documentation.
- **`--lexical-ranking`** for `make ask` and `make eval`, with BM25 as the default and `ts_rank` still available. Every earlier cited run is reproduced exactly with `ts_rank`.

Two checks made sure the SQL is right. It gives the same ranking, with scores that differ by less than 4·10⁻¹⁵, as an independent Python version written earlier, on all 38 queries tried. And it is fast enough: about 16 ms per query, against 4 ms for `ts_rank`, because it reads every chunk. That is negligible next to the translation and the answer, but it would not scale to a corpus a thousand times larger.

## What we found

**BM25 ranks better.** With the default configuration (hybrid search, translated question, top 5), the right section comes first for 17 of the 19 questions instead of 15, and MRR@5 rises from 0.860 to 0.908. Every English question is now answered from rank 1. One question moved down a place (from 3rd to 4th), still within the 5 passages.

**The answers stay the same.** The same 18 questions get their section into the top 5, so the model receives the same evidence. Over 84 generated answers, BM25 and `ts_rank` cited a correct section equally often, said "not found" in the same places, and used the same prompt size and time. Read side by side, the differences cut both ways: the oil-spill answer got better (it names the wind products that help tell real spills from look-alikes), and the product-metadata answer got worse (it presented documents about one product type as the way to consult any product's metadata).

**So the case for BM25 is the ranking, not the answers.** It is proposed as the default because it fixes a measured defect of `ts_rank`, it ranks the right section first more often, which matters if fewer passages are ever used, and it makes the hybrid the standard recipe: BM25 plus vectors, merged with reciprocal rank fusion. The documentation says plainly that the answers did not improve.

## What it means for the next steps

The retrieval now follows the textbook design for hybrid search, and every choice behind it is measured and recorded (D-015 to D-023). The remaining extensions in the README are larger moves: adding the technical PDFs to the corpus, an agent with a tool for searching satellite scenes, an MCP server, or OpenSearch in place of Postgres, where BM25 is built in and the comparison would be direct.

## Check your understanding

**What do the three parts of BM25 do, and which one fixes the problem seen with `ts_rank`?**
IDF weighs each word by how rare it is across the chunks; term-frequency saturation (k1) makes repeated occurrences add less and less; length normalization (b) discounts long chunks for containing more words. IDF is the fix: `ts_rank` gave "sentinel", present in two thirds of the chunks, the same weight as a rare technical term, so chunks that merely repeated common words ranked high.

**BM25 ranked better on the eval but did not improve the answers. How can both be true, and why adopt it anyway?**
`make ask` passes 5 chunks to the model, and BM25 changed the order within those 5 more than which chunks are in them: the same 18 questions got their section in. The model reads all 5 passages, so a better position changes little. It is still worth adopting because it fixes a real weakness that would matter with other questions or fewer passages, at a cost of about 12 ms per query, and because it is the standard design that people expect and can compare against. The decision record says this openly instead of claiming better answers.
