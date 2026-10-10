# Step 9: Translating the question before the lexical search

*Finished on 2026-10-10. A follow-up to step 8, chosen by the user.*

## What this step was about

Step 8 found that lexical search fails here for one main reason: the pages are in English, most questions are in Spanish, and a search that compares words cannot match "diferencia" with "difference". This step tried the obvious fix, **query translation**: before searching, a language model rewrites the question into English, and the lexical search uses that version.

Query translation is one case of *query rewriting*, a family of techniques where a model transforms the user's question before retrieval: translating it, expanding it with synonyms, splitting a compound question into parts, or writing a hypothetical answer to search with. They all trade an extra model call, and a new way to fail, for a query that matches the documents better.

Vector search did not obviously need this, because bge-m3 is multilingual and was chosen precisely so that Spanish questions land near English passages. Whether that assumption holds was measured too.

## What we built

- **`translate.py`:** one call to Gemma, the model already loaded for answering, with temperature 0 so that the same question always gets the same translation.
- **A `--translate` option** for `make ask` and `make eval`, used with `--method lexical` or `hybrid`. In the hybrid, only the lexical half uses the translation; the vector half keeps the original question.

Getting the prompt right took three tries, all checked on the 7 English questions, which should come back untouched:

- "Translate, or repeat it unchanged if it is already in English" reworded two of them. One became "polarizations" instead of "polarisations". That matters more than it looks: Postgres stems the British spelling to `polaris`, found in 57 chunks, and the American one to `polar`, which those chunks don't contain. The prompt now asks for British spelling, as the documentation uses it.
- Asking the model to reply "ENGLISH" for an English question worked only once that rule came first. Placed last, it was ignored for three of the seven.

One more practical detail: Ollama loads the model again whenever a request asks for a different context size, which costs about a minute on this machine. The translation therefore uses exactly the same context size as the answer, now defined once in `config.py`.

## What we found

| Method (hit@10 / MRR@10) | All (19) | Spanish (12) | hit@1 | hit@3 |
|---|---|---|---|---|
| Vector | 0.947 / 0.766 | 0.917 / 0.672 | 0.684 | 0.789 |
| Lexical | 0.737 / 0.500 | 0.583 / 0.284 | 0.421 | 0.579 |
| Lexical, translated | 0.947 / 0.746 | 0.917 / 0.674 | 0.632 | 0.842 |
| Hybrid | 0.842 / 0.633 | 0.750 / 0.461 | 0.526 | 0.684 |
| Hybrid, lexical half translated | 0.947 / 0.860 | 0.917 / 0.819 | 0.789 | 0.947 |

**Translation closes the language gap.** Lexical search on the translated question finds as many answers as vector search, and the translated hybrid ranks better than anything measured so far: the right section comes first for 15 of the 19 questions, and within the top 3 for 18.

**Vector search did not need it.** Embedding the translated question instead of the original barely changed vector search (MRR@10 0.782 instead of 0.766), which confirms the choice of a multilingual embedding model.

**Better ranking did not mean more answers.** `make ask` gives the model 10 chunks, so what decides whether an answer can be complete is hit@10: is the right section among the 10? There, the translated hybrid and plain vector search are equal, 18 of 19, with one swap. The question about the contents of an SLC product gains its section and gets a more focused answer. The project's own example question, GRD vs SLC, loses the section that explains SLC, and its answer falls back to the shallow version that step 6 had fixed. Ranking metrics such as MRR mostly reorder chunks that the model reads anyway.

So `make ask` keeps vector search by default. The translated hybrid stays available as an option, and the eval points to where it would pay off: with it, 3 chunks reach the same coverage as 10 do with vector search, which would cut the prompt to about a third. That is a separate change to the top-k decision, left for the user.

## What it means for the next steps

The open options are: hybrid search with translation and a smaller top-k, checked on real answers and not only on the eval; BM25 in place of `ts_rank`, which a throwaway script put at MRR@10 0.908 in the translated hybrid; or the remaining extensions in the README (PDFs, an agent with a STAC search tool, an MCP server, OpenSearch).

## Check your understanding

**Why translate only for the lexical search, and not for the vector search too?**
Lexical search compares words, so a Spanish question cannot match English text without translation. Vector search compares meanings, and bge-m3 was trained to place sentences with the same meaning close together across languages. The measurement agreed: translating barely changed vector search (MRR@10 0.782 against 0.766), and in the hybrid the original question did better (0.860 against 0.833 with both halves translated). Translating where it is not needed adds a call and one more way to go wrong.

**The translated hybrid has a much better MRR than vector search. Why does `make ask` still use vector search?**
Because `make ask` passes the top 10 chunks to the model, so the measure that matters for the answer is whether the right section is among those 10, which is hit@10, and there both methods find 18 of 19. MRR rewards putting the right chunk first, but the model reads all 10 passages either way, so a better position within the 10 changes little. Meanwhile the swap is visible in the answers: one question gains its section, and the example question loses it. A better MRR would pay off if fewer chunks were passed to the model, which is a separate decision.
