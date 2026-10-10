# Step 6: Retrieval eval

*Finished on 2026-10-10.*

## What this step was about

Until now, the only way to judge retrieval was to ask a question and read the chunks that came back. That works for one question, not for comparing two versions of the system. An **eval** replaces impressions with a number: a fixed set of questions, each labelled with the sections that answer it, and a script that checks where those sections end up in the ranking.

Only retrieval is measured, not the answers. If the right passage never reaches the model, the answer cannot be good. Retrieval can also be measured quickly and deterministically, without running the language model.

Two metrics summarise the ranking:

- **hit@k:** the share of questions where at least one chunk from an expected section is among the first k results. hit@5 answers "did the model get what it needed?" when it is given 5 passages.
- **MRR (mean reciprocal rank):** for each question, 1 divided by the rank of the first relevant chunk (1st gives 1, 2nd gives 0.5, 4th gives 0.25), averaged over the questions. It rewards putting the answer near the top, not just somewhere in the list. We count it as 0 when the first relevant chunk falls below the cut-off k, because the model never sees anything below that.

## What we built

- **`eval/questions.yaml`:** 19 questions in Spanish and English, each with the sections that answer it, written as the citation URL (page and heading anchor) plus the readable heading path.
- **`make eval`** (`evaluate.py`):
  - It runs every question through the same retrieval function that `make ask` uses, but asks for the top 20, so that a miss still shows how far down the answer was.
  - It prints hit@1, hit@3, hit@5, hit@10 and MRR, the rank of every question and, for each miss, what was expected next to what came back.
  - It saves each run, with the index settings, to a dated file in `eval/results/`. A run can be named (`--label`) and scored with another cut-off (`--top-k`).
  - It refuses to run if a labelled section does not exist in the index, so a typo cannot pass as a miss.

The labels follow a **strict rule**: a section counts only if, read on its own, it answers the main part of the question. Sections that merely mention the topic do not count. For a comparison of two things, such as GRD vs SLC, each section that explains one side counts.

## What we found

**Questions written from the text are too easy.** Claude drafted 14 questions after reading the sections that answer them. Five more were written without reading the pages, the way a user would ask. The difference is large:

| Questions | hit@5 | MRR@5 |
|---|---|---|
| Drafted from the text (14) | 1.000 | 0.854 |
| Written without reading the pages (5) | 0.600 | 0.500 |

Questions drafted from the answer reuse its words ("vignettes", "slices", "TOPSAR"), and that makes them easy for the search. An eval made only of such questions would have reported near-perfect retrieval. Each question records how it was written, so the eval reports both groups every time.

**The sections that explain SLC and GRD rank low.** For every question about products, chunks from the "S1 Products" page come first. The sections that actually explain SLC and GRD, under "S1 Processing > L1 Algorithms", rank 9th or lower, sometimes outside the top 20. For "What information does an SLC product contain?", the first correct section is 16th, behind an ETAD section that keeps mentioning "standard SLC product".

**Questions about three things lose information.** For "differences between Level-0, Level-1 and Level-2", the three sections ranked 2nd, 3rd and 14th. The question still counted as a hit, because one section is enough for hit@k, but the model would never see Level-2. We decided to leave questions about three or more things out of the eval and to record this as a limitation.

**An idea that didn't help.** Limiting how many chunks of the same section can enter the top k was the obvious fix for crowding. Simulated on the saved results, it changed no metric, so it was never implemented.

## The change: top-k from 5 to 10

With 5 passages, the project's own example question ("¿Qué diferencia hay entre un producto GRD y uno SLC?") never received the sections that explain the difference. They ranked 9th and 11th. With 10 passages, the SLC section reaches the model. In three runs out of three, the answer now explains the phase information, the complex samples and the slant-range geometry, which it could not do before.

On the eval, hit@10 is 0.947 against hit@5 0.895. That is one question out of 19, so the number alone proves little. The reason to keep the change is the better answer to the example question.

The cost was measured as well. The prompt roughly doubles, to a median of 2,869 tokens and 4,288 in the worst case, which still fits easily in Gemma's 8,192-token context. With 10 passages, the question about the mission's cost still gets "not found", and English questions are still answered in English.

**A limitation that showed up.** "How do I export the results to GeoTIFF?" is not answered anywhere in the documentation. Instead of saying so, Gemma lists nearby facts (the products are stored as GeoTIFF files), with correct citations, and never says that the export itself is not covered. It does the same with 5 passages, so the prompt is the cause, not the top-k. It is recorded for the README.

## What it means for the next steps

Step 7 documents the project. The README will report hit@10 0.947 and MRR@10 0.766 over 19 questions, together with the 0.800 of the questions written without reading the pages. It will also explain why the second number is the more honest one, and list the limitations found here.

Several changes were left unmeasured: other chunk sizes, embedding chunks without the "Page > Section" prefix, merging tiny sections, and adding more pages. `make eval` makes each of them a one-minute experiment if they are ever tried.

## Check your understanding

**What do hit@k and MRR measure, and why is MRR cut off at k?**
hit@k is the share of questions with at least one relevant chunk in the first k results: did the model get what it needs or not. MRR averages 1/rank of the first relevant chunk, so it also rewards ranking that chunk higher: 1st scores 1, 2nd 0.5, 5th 0.2. We cut MRR at k because `make ask` only passes the top k to the model. Moving an answer from 15th to 11th would raise an uncut MRR without changing any answer.

**Why did the questions written without reading the pages score so much lower, and what does that teach about building an eval set?**
Whoever writes a question after reading its answer tends to reuse the answer's words, and an embedding search easily matches shared vocabulary. Real users don't know the documentation's wording, so they ask with other words, and that is harder. An eval built only from questions written that way reports retrieval as better than it is. Include questions written without looking, label how each question was written, and treat the score of those blind questions as the more realistic one.
