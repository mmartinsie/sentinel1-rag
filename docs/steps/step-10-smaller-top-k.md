# Step 10: Hybrid search with a smaller top-k, judged on the answers

*Finished on 2026-10-11. A follow-up to step 9, chosen by the user.*

## What this step was about

**Top-k**, the number of passages given to the model, is a trade between recall and focus. More passages make it likelier that the right section is among them. But the prompt grows, answers take longer, and a small model has more irrelevant text to be distracted by, and to cite. With a better ranking, k can shrink without losing recall.

Step 9 found such a ranking. Hybrid search, with the question translated into English for the full-text half, puts the right section in its top 3 as often as vector search does in its top 10. Retrieval metrics cannot say whether the answers stay as good with fewer passages, though. So this step compared the answers themselves.

## What we did

A throwaway script generated 168 real answers: the 19 eval questions plus two the documentation does not answer, under four configurations (vector search with the top 10, which was the default, and the translated hybrid with the top 10, 5 and 3), twice each because answers vary a little between runs. For every answer it recorded whether an expected section reached the prompt, whether the answer cited one, whether it said that something was not found, the prompt size and the time. The answers to the multi-part questions were then read side by side.

| Configuration | Cited a correct section | Median prompt | Median generation |
|---|---|---|---|
| Vector, top 10 | 36 of 38 | 2,851 tokens | 5.4 s |
| Hybrid, translated, top 10 | 36 of 38 | 3,147 tokens | 6.2 s |
| Hybrid, translated, top 5 | 36 of 38 | 1,615 tokens | 3.7 s |
| Hybrid, translated, top 3 | 36 of 38 | 1,042 tokens | 2.7 s |

## What we found

**Fewer passages did not cost answers.** Every configuration cited a correct section in 36 of 38 answers, and in every configuration an answer cited one whenever one was in the prompt. The two misses are the same swap seen in step 9: vector search misses the question about what an SLC product contains, and the hybrids miss the project's example question about GRD vs SLC.

**Some answers got better.** For oil spills, vector search with 10 passages mixed ship-detection products into the list of products for detecting oil; with 5 passages from the hybrid, the answer stayed on the oil-pollution section. Answers on orbit files, floods, ground subsidence and the repeat cycle were as complete with 5 or 3 passages as with 10.

**Three passages are slightly too few.** With 3, answers cited fewer sections (1.4 on average, against 1.7 for vector search and 1.8 with 5): the ETAD answer lost one of its two sections, and the answer on how the SAR works lost the acquisition modes. Five keep the breadth, and leave a margin for a future question whose answer ranks 4th or 5th.

**The prompt is shorter and answers come sooner.** With 5 passages the prompt is about 43 % shorter and generation about 30 % faster. End to end, the time is dominated by the length of the answer, so the gain varies: typical answers take 4–6 s instead of 5–8 s, and long ones up to about 16 s with either.

**The cost is the example question.** Its best sections rank 18th in the hybrid, so its answer explains slant versus ground range and the resolution classes but not the phase information that SLC keeps, which is exactly what step 6 raised the top-k to obtain. The README now shows another example and documents this one as a limitation, with the command that gives the fuller answer.

The proposal is therefore a new default: hybrid search with the translated question and the top 5 (D-022). The eval follows the default, so `make eval` now loads Gemma to translate. Every earlier configuration is still available through `--method`, `--no-translate` and `--top-k`, and reproduces its cited runs exactly.

## What it means for the next steps

If the new default is accepted, the pipeline is: translate, search twice, fuse, answer from 5 passages. The obvious refinement is BM25 in place of `ts_rank` for the full-text half, which a throwaway test put at MRR@10 0.908 against 0.860. The remaining extensions in the README (PDFs, an agent with a STAC search tool, an MCP server, OpenSearch) are independent of this choice.

## Check your understanding

**Why measure the answers, when the eval already showed that hit@3 of the hybrid equals hit@10 of vector search?**
hit@k only says that one correct chunk is among the first k. It cannot say whether the answer stays complete when the other passages are gone, whether the model gets distracted by irrelevant ones, or how long it takes. Reading the answers showed what the metric could not: 3 passages lost some breadth, 5 did not, and 10 sometimes let the model mix in off-topic facts.

**Raising top-k to 10 in step 6 was justified by the example question. Why is lowering it to 5 now justified even though that question gets worse?**
In step 6 there was no alternative that kept that question and improved others; the change cost little and fixed the project's own example. Here the evidence is wider: over 19 questions the new configuration loses the example question but gains another, cites correct sections just as often, keeps answers as broad, and makes prompts much shorter. One question should not outweigh the rest, so the fuller answer stays available as an option and the regression is documented rather than hidden.
