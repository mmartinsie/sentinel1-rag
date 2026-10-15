# Step 16: A reranker over the top 20

*Built and measured on 2026-10-14 and 2026-10-15, approved by the user on 2026-10-15, and released as v0.4. The reranker was the experiment left pending since v0.2, chosen by the user.*

## What this step was about

The search finds chunks in two ways, by meaning (vectors) and by words (BM25), and merges the two lists. Both ways look at the question and a chunk separately: each becomes a vector, or a set of words, and only those are compared. A reranker is a model that reads the question and the chunk together, in one pass, so it can judge whether this chunk answers this question rather than only whether they are about the same thing. That is more precise and much slower, one model call per chunk, so it only reorders the 20 best chunks of the search, and the 5 it likes best go to Gemma.

The question was whether that reordering puts more answers in front of Gemma, and at what cost.

## What we built

The reranker is Qwen3-Reranker-0.6B, a small language model trained to answer "yes" or "no" to "does this document answer this query?". Its score is how much likelier "yes" is than "no" as its next word. Ollama has no reranking feature, but it can return the probabilities of the next word, so the code sends the model its prompt as plain text, asks for one word, and reads the probabilities of "yes" and "no".

Two other routes did not work here. The usual reranker of the bge family needs PyTorch on the GPU, and current PyTorch no longer supports this GTX 1070; on the processor alone, a model this size took about 12 seconds per chunk. And the most downloaded copy of Qwen3-Reranker is packaged for another program's reranking feature, so Ollama treated it as an embedding model and returned useless vectors. Another upload of the same model, packaged as an ordinary language model, works.

To check that Ollama's copy scores like the original, 60 question–chunk pairs were scored again with the model's official code, on the processor: the orders agreed almost exactly, and the top 5 were the same. `--rerank` and `--no-rerank` switch it in `make ask` and `make eval`.

## What we found

**A memory problem came first.** During the first measurement, Ollama's process for the reranker grew to 8.8 GB of RAM and left WSL with 1.6 GB, close to the freeze of step 13. The program that Ollama now uses to run each model keeps a copy of every prompt it has seen, up to 8 GB per model, in case a later prompt starts the same way. The reranker sends 20 new prompts per question, so the copies piled up. One setting in the Ollama service turns this off; after it, the process stayed around 1 GB and each question took under 2 seconds instead of up to 79.

**The instruction mattered more than anything else.** The reranker's prompt includes a one-line task description. With the generic one from the model card, it found more answers but put fewer of them first, and pushed one from 1st to 13th. With a line that names the Sentinel-1 mission and its documentation, written before measuring, it put the right section first for 21 of 25 questions, against 19 without a reranker, and found the answer in the top 5 for 24 of 25, against 22. The 24th is the most a reranker can do here: the last question's answer is not among the 40 best chunks of the search.

**It fixed the two misses that had a known cause.** For why TOPS bursts are deramped, the note's introduction had been pushed to 10th by four chunks of one long section; the reranker brings it to 1st, and the answers now give the reason. A simpler fix, allowing at most two chunks per section in the top 5, changed nothing at all. For the project's original question, GRD versus SLC, the product definition's GRD section rises from 13th to 1st.

**The answers improved too.** For the 23 questions whose top 5 changed, Gemma answered twice with and twice without the reranker: 43 of 46 answers cited a correct section, against 40, and no question did worse. The search takes about 1.7 seconds more; Gemma's answer takes the same time.

**It is not better everywhere.** Two questions' best chunks moved down within the top 5. And for the example at the top of the README, about thermal noise in GRD products, the section that explains why the noise is not removed by default fell from 5th to 8th, so the answer no longer gives that reason. The README now shows that answer and says so.

Since it met the criterion set before measuring (more answers found, none lost from the top 5, under 3 seconds), the reranker is now the default, for the agent and the MCP server too. The agent passes the same 13 of 18 checks as before, question by question.

## What it means for the next steps

All three models now share the GPU: 6.7 of its 8 GB. The eval questions have done a lot of work by now, and the reranker's instruction was chosen on them as well; new questions, written without reading the documents, would say more about the next change than another run on these 25. The remaining miss ("sigma zero") needs a better search, not a better reordering.

## Check your understanding

**Why does a reranker read only the top 20, instead of all 568 chunks?**
Because it needs one pass of a model for every chunk it judges: about 0.08 s per chunk here, so 20 chunks take under 2 seconds and all 568 would take most of a minute. The two searches are cheap because each chunk's vector and words are prepared in advance; a reranker cannot prepare anything, since it reads the question and the chunk together. So the cheap searches narrow the field and the reranker decides the order at the top. Reading the top 40 was also measured: it took twice as long and did slightly worse.

**The reranker improved the eval, yet the README's own example got a worse answer. How can both be true, and what does it say about the eval?**
The eval has 25 questions, and the example question is not one of them; a change that helps most questions can still hurt others. Here it moved two eval questions down within the top 5 and pushed the example's best section out of it. A small eval shows the direction of a change, not that it is right for every question, which is why the answers were read and the loss was reported rather than hidden.
