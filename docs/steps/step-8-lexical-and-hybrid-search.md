# Step 8: Lexical baseline and hybrid search

*Finished on 2026-10-10. This is the first extension listed in the README, done after the seven steps of the original plan.*

## What this step was about

Until now, retrieval meant one thing: embed the question and fetch the chunks whose vectors are closest. **Lexical search** is the older alternative. It splits text into *lexemes* (words lower-cased and cut to their stem, so "products" and "product" become the same), finds the chunks that share lexemes with the question, and ranks them by how often those words occur. It knows nothing about meaning, which is both its weakness (no synonyms, no other languages) and its strength: an exact, rare term such as an acronym or a product code can match precisely where an embedding might blur it.

The standard lexical ranking is **BM25**. It weighs each word by how rare it is in the whole collection (its *inverse document frequency*, IDF), so a match on a rare word counts far more than a match on a word that is everywhere. Postgres's built-in `ts_rank` has no such weighting: its documentation says the ranking functions "do not use any global information".

**Hybrid search** runs both searches and merges the two rankings. The usual way is **reciprocal rank fusion** (RRF): each chunk gets 1 / (60 + its rank) from every list it appears in, and the sums decide the final order. Because only ranks are used, it does not matter that one search returns distances and the other word-frequency scores.

The README predicted that lexical search would help with SAR acronyms such as GRD and SLC. This step measured whether it does.

## What we built

- **A full-text column:** each chunk now has a `tsv` column holding the lexemes of the same text that is embedded, computed by Postgres itself, with a GIN index (an *inverted index*: for each lexeme, the list of chunks that contain it). In Postgres 18 such a computed column is virtual by default and cannot be indexed, so it is declared `STORED`; a quick test confirmed the error otherwise.
- **The lexical query:** Postgres's `plainto_tsquery` turns the question into lexemes but requires *all* of them to be present. 18 of the 19 eval questions matched no chunk that way, so the query is rewritten to accept any of them.
- **Hybrid search** with RRF over the top 40 of each list.
- **A `--method` option** (`vector`, `lexical` or `hybrid`) for both `make eval` and `make ask`. Vector search stays the default.
- **Metrics per language:** each eval question now records whether it is in Spanish or English, and the eval reports both groups.

## What we found

| Method (hit@10 / MRR@10) | All (19) | Spanish (12) | English (7) |
|---|---|---|---|
| Vector | 0.947 / 0.766 | 0.917 / 0.672 | 1.000 / 0.929 |
| Lexical | 0.737 / 0.500 | 0.583 / 0.284 | 1.000 / 0.871 |
| Hybrid | 0.842 / 0.633 | 0.750 / 0.461 | 1.000 / 0.929 |

**Lexical search cannot cross languages.** On English questions it is nearly as good as vector search. On Spanish questions it collapses, because Spanish words such as "diferencia" or "producto" appear in no chunk. The only bridges are the tokens both languages share: acronyms, numbers and "Sentinel-1". All five questions written without reading the pages are in Spanish, so the realistic score suffers most.

**The acronyms are too common to help.** "GRD" appears in 43 of the 195 chunks and "SLC" in 47. Lexical search ranks first the chunks that repeat them most, the same "S1 Products > Level-1 Products" chunks that vector search already prefers. The sections that actually explain GRD and SLC never reach its top 20. Acronyms help lexical search when they are rare; in documentation about Sentinel-1 products, the product names are everywhere.

**Fusion hurt more than it helped.** The hybrid improved two questions and worsened six, among them the project's example question (from 9th to 16th, undoing the gain of step 6) and the oil-spill question (from 1st to 15th). RRF rewards agreement: a chunk ranked 10th in both lists scores more than one ranked 1st in only one list. When one list is mostly noise, as lexical search is for Spanish questions, mediocre chunks that happen to appear in both lists push the right answer down.

**Better lexical rankings did not change the conclusion.** A throwaway script also measured `ts_rank` normalized by length, `ts_rank_cd` (which rewards matched words that are close together) and a BM25 written by hand. BM25 was the best lexical ranking and put every English question's answer first, but no hybrid came close to vector search alone.

**A bug that only showed up by checking.** `ts_rank` often gives many chunks exactly the same score. The first version broke those ties by chunk id, and chunk ids depend on how many times each page has been re-indexed. A fresh database built from the schema disagreed with the working one, with one question at rank 7 in one and 17 in the other. The fix has two parts: in the fusion, tied chunks share the same rank (1, 2, 2, 4), and remaining ties are ordered by section and position, which are the same in every database. After the fix, both databases return identical results for every method.

## What it means for the next steps

Vector search stays the default for `make ask`; lexical and hybrid search remain available as options and as a measured baseline. The result is specific to this project: one English corpus, mostly Spanish questions, and acronyms that appear in a quarter of the chunks. On an English-only corpus with rare codes and identifiers, the same comparison could come out the other way.

The most direct follow-up is to translate the question into English before the lexical search, which removes the language gap that sinks it. It costs one more call to the language model per question. The remaining extensions (PDFs, an agent with a STAC search tool, an MCP server, OpenSearch) are independent of this result.

## Check your understanding

**Why does Postgres full-text search do badly on this eval, when lexical search is often recommended for acronyms?**
Two reasons. First, 12 of the 19 questions are in Spanish and the pages are in English, so a Spanish question can only match the documentation through acronyms, numbers and "Sentinel-1". Second, the acronyms in the questions are not rare here: GRD and SLC each appear in about a quarter of the chunks, and `ts_rank` cannot discount common words because it has no IDF. Lexical search wins on rare, exact terms in the same language as the documents. Neither condition holds for most of these questions.

**How does reciprocal rank fusion work, and why did it make results worse here?**
Each list contributes 1 / (k + rank) to every chunk it contains, with k = 60, and the chunks are sorted by the sum. It uses ranks rather than scores, so the two searches do not need comparable scores, and it rewards chunks that both searches agree on. That reward is also its weakness: two middling ranks (10th and 10th, 2/70) beat a single first place (1/61). When one list is mostly noise, chunks that happen to appear in both lists overtake the correct answer that only the good list found. Fusion helps when both lists are informative, and it does not choose between them.
