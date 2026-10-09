# Step 5: Query

*Finished on 2026-10-09.*

## What this step was about

This is the "online" half of RAG, the part that runs every time someone asks a question.

**Retrieval.** The question is turned into a vector with the same model that embedded the chunks (bge-m3). Postgres then returns the 5 chunks whose vectors are closest, by cosine distance. bge-m3 is multilingual, so a question in Spanish finds passages written in English.

**Generation.** The 5 chunks are pasted into the prompt, numbered [1] to [5], and Gemma is told to answer only from them and to cite each claim with its number. This is called *grounding*: the answer is tied to passages anyone can check, instead of coming from whatever the model remembers.

**Citations.** The model only writes numbers. The code turns each [n] back into chunk n, whose section and URL come from the database. A citation can therefore point to the wrong passage, but never to a page that doesn't exist.

## What we built

- **`retrieval.py`:** embeds the question and fetches the 5 closest chunks with their section, URL and distance. It is a module of its own because the eval in step 6 must measure exactly this retrieval.
- **`ask.py`** (`make ask Q="..."`): builds the prompt, asks Gemma, and prints three things:
  - the answer;
  - the sources it cited, one entry per section, with its URL;
  - a line with the prompt and answer sizes and the time taken.

  `ARGS=--show-context` also prints the retrieved chunks and their distances, for debugging.
- **`chat` in `ollama.py`:** one request to Ollama's chat API, with thinking turned off.

The prompt has two parts:
- **System turn:** the instructions. Use only the passages. Cite them as [n]. Say so when the answer isn't there. Write in the question's language. Be brief.
- **User turn:** the numbered passages, each headed by its section path, then the question, so that the question is the last thing the model reads.

## How we checked it

**We looked at what Gemma actually receives.** Ollama can render the final prompt without running the model. It showed that Gemma 4 has a real system turn; older Gemma versions didn't.

**The three test questions from the brief:**
- *"¿Qué diferencia hay entre un producto GRD y uno SLC?"*: answered in Spanish. Every sentence carries a citation, and each citation matches what that passage says.
- *"What is the repeat cycle of Sentinel-1, and how does a second satellite change it?"*: answered in English. The facts are right (12 days for one satellite, 6 days for two), and the citation is the Orbit section.
- *"¿Cuánto costó construir y lanzar la misión Sentinel-1?"*: the documentation never mentions the cost. Every time, the answer was that the information could not be found, with no invented number and no citation.

We ran each question three more times with the final prompt and got the same behaviour every time. With the models loaded, an answer takes 1 to 3 seconds.

**A wording problem, measured and fixed.** The first prompt said "answer in the language of the question, even though the passages are in English". English questions then got Spanish answers 8 times out of 10, and once a Portuguese one. The phrase "even though… English" seems to push the model away from English. Replacing it with "write the answer in the same language as the question" gave the right language in 20 out of 20 runs. Adding a reminder after the question didn't improve on that.

**A query that couldn't use the index.** Asking Postgres for the closest chunks and joining the page table in the same query stopped the planner from using the HNSW index, even when forced. Picking the top 5 from the chunks table first and joining the page table afterwards fixed it, and returns the same results. At today's size the plain scan is just as fast, so this only matters as the table grows.

**Distances alone can't spot a missing answer.** The unanswerable question's closest chunk was nearer than some chunks of an answerable question. A fixed distance cut-off would have rejected good chunks or let bad ones through, so that decision stays with the model.

**Edge cases:**
- A question containing quotes, an apostrophe and backticks reaches the program unchanged.
- An empty question prints how to use the command.
- If Ollama or Postgres is down, the error says which one and how to start it.

**Two environment problems.** These are not bugs in our code, but they are worth knowing.
- **Reloads on every question:** during a long session, Ollama believed there wasn't enough free memory for both models. It evicted one to load the other on every question, so each question waited 50 to 90 seconds. After WSL restarted, both models stayed loaded together.
- **No GPU after the restart:** Ollama failed to find the GPU at first, because reading its CUDA libraries from the slow hard drive took too long, and it fell back to the CPU. Restarting the service fixed it. The resume checklist in PROGRESS now includes this check.

## What it means for the next steps

Retrieval is now the weak link. For the GRD/SLC question, three of the five passages come from one long section that lists the product types. The two sections that actually explain the difference rank 9th and 11th. The answer is correct but shallow.

Step 6 builds a set of questions with known answer locations, so that this kind of miss becomes a number (hit@1, hit@5, MRR) that each change can improve or not.

## Check your understanding

**How is the prompt built, and why does the code, not the model, produce the list of sources?**
The instructions go in the system turn. The user turn holds the retrieved chunks, numbered and headed by their section path, followed by the question. The model is asked to cite with [n] only. The program then reads those numbers and looks up each chunk's section and URL in the database. If the model wrote the sources itself, it could invent a section or a URL. This way the worst it can do is cite the wrong one of the five passages, which can be checked with `--show-context`.

**Why not reject unanswerable questions with a distance threshold instead of asking the model?**
Cosine distance measures how similar the topics are, not whether a passage contains the answer. A question about the mission's cost lands close to the mission overview chunks, which talk about Sentinel-1 but never about money. In our tests, the closest chunk for that unanswerable question was nearer than some useful chunks of an answerable one, so no single cut-off separates the two. The model reads the passages and can tell whether the answer is actually there. A threshold could still serve as a cheap first filter for questions that are completely off-topic.
