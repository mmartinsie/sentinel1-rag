# Step 7: Documentation

*Finished on 2026-10-10.*

## What this step was about

A project is only as useful as its entry point. The README is a contract with someone who has never seen the repository: it has to say what the system does, show how it works inside, give numbers that can be trusted, and get them from a fresh clone to a working answer without help. Anything it promises has to be checked, because a README that only works on the author's machine is worse than none.

The step also closed the decision log. `docs/DECISIONS.md` is a set of *architecture decision records*: each entry says what was decided, what else was considered, why, and what it costs. Records are written when the decision is made, so later steps can quietly make parts of them wrong. A final review looks for those places and adds a forward note, instead of rewriting the old entry, so that the history of how the design changed stays readable.

## What we did

**The README** now covers, in order: a real answer to the project's example question, a diagram of the system, the eval results, how to run it on WSL, the commands, the project layout, the limitations and the extensions.

The diagram, drawn in Mermaid so that GitHub renders it from text, splits a RAG system into its two halves. The first half builds the index once: download the pages, split them into sections and chunks, embed each chunk, store it in Postgres. The second half runs on every question: embed the question, fetch the 10 closest chunks, give them to Gemma with the rules, and turn its `[n]` citations into links. Seeing the two halves side by side also shows why the same embedding model must be used in both: the question and the chunks have to live in the same vector space for their distances to mean anything.

The eval results are reported with their context, not as a single number. The headline is hit@10 0.947 and MRR@10 0.766 over 19 questions, but the table also shows the 5 questions written without reading the pages (hit@10 0.800), which are the realistic score. The README explains why the other 14 are easier, how much one question moves the metrics, and that top-k was chosen on the same questions.

**A clean-clone check** tested the README's promise. A fresh `git clone` went through `make up`, `make ingest`, `make index`, `make ask` and `make eval` with no manual step. To leave the working database untouched, it ran as a separate Compose project on another port, which was removed afterwards. The pages downloaded again were identical to the ones cached two days earlier, and the eval produced exactly the numbers the README cites, with the same chunks at the same distances for every question.

**The final review of the decisions** found five stale places. Most came from the top-k change in step 6: D-002 and D-016 still described a prompt with 5 passages, and the title of D-017 said MRR is cut at 5. D-010 and D-013 promised experiments for step 6 that were never run. Each now has a note pointing to the decision that changed it.

## What we found

**Answers got slower with the top 10.** With the models loaded, an answer now takes 4–7 s instead of the 1–3 s measured in step 5, because the prompt is about twice as long. This is the cost recorded in D-018, now measured.

**Ollama sometimes unloads the embedding model.** Twice, loading Gemma made Ollama unload bge-m3 first, even though the numbers in its own log (5.6 GiB needed, 6.4 GiB of video memory free) suggest that both fit. The next question then loaded bge-m3 again. It costs little, because bge-m3 is the smaller model, but it contradicts what step 1 assumed, so it is recorded in D-003.

**Cold starts dominate the first question.** On this machine the models live on a hard disk, so the first `make ask` took 69 s, 60 of them loading Gemma. The README says so, so that a first-time user does not think it is broken.

## What it means for the next steps

Once this step is approved, every step of the brief is done: the system runs from a clean clone by following the README, `make eval` produces a number, the README reports it, and every decision is recorded with its alternatives. What comes next is a choice among the extensions listed in the README. The first, a lexical and hybrid search measured on the same eval, is the natural continuation, because the misses found in step 6 are product questions full of acronyms, the case where keyword search usually does better than embeddings.

## Check your understanding

**Why test the project from a clean clone, and what did this check prove beyond "it runs"?**
A working copy hides state that a newcomer won't have: cached pages, an installed virtual environment, a filled database, files that were never committed. Cloning into an empty folder removes all of that. Beyond running end to end, this check proved two things. The pages downloaded again had the same hashes as two days earlier, so the corpus has not drifted. And the eval reproduced the cited numbers exactly, so the number in the README is a property of the code and the corpus, not of one lucky run.

**What is an architecture decision record, and why add notes to old entries instead of editing them?**
It is a short entry written when a decision is made: the context, the decision, the alternatives considered and the consequences accepted. Its value is that it records *why*, which the code alone cannot show. Later decisions can override parts of earlier ones, as raising top-k from 5 to 10 did here. Rewriting the old entry would erase the reasoning that held at the time and hide that the design changed. A forward note ("since D-018 the prompt carries 10 passages") keeps both the original reasoning and the change.
