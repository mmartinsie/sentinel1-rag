"""Reorder the search's top 20 with a reranker that reads the question and each passage together.

The searches in retrieval.py compare the question and a chunk separately: two vectors, or the
words they share. A reranker reads both in one pass of a model, so it can judge whether the
passage answers the question, at the cost of one model call per candidate (D-037).

The model is Qwen3-Reranker-0.6B, a small language model trained to answer "yes" or "no" to
"does this document meet the query?". Its score is how much likelier "yes" is than "no" as the
next token, read from the log-probabilities that Ollama returns. Ollama has no reranking
endpoint, and runs the model like any other.

Each pair is a new prompt, and the llama-server that Ollama starts keeps a copy of every prompt
in RAM, up to 8 GiB per model: 25 questions took it to 8.8 GB (D-038). Setting
LLAMA_ARG_CACHE_RAM=0 for the Ollama service turns that copy off (README, "Running it on WSL").
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from sentinel1_rag.config import RERANK_MODEL
from sentinel1_rag.ollama import Ollama, OllamaError

if TYPE_CHECKING:
    from sentinel1_rag.retrieval import Hit

# How many of the search's best chunks the reranker reads; it keeps the top k of them.
RERANK_CANDIDATES = 20

# The model card's prompt, with its system turn and an empty thinking block; only the
# instruction is ours. The card's default instruction ("Given a web search query, retrieve
# relevant passages that answer the query") found fewer answers first: hit@1 0.600 against
# 0.840 with this one (D-037).
INSTRUCTION = "Given a question about the Sentinel-1 radar mission, retrieve the documentation passages that answer it"
PROMPT = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct"
    ' provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
# The longest prompt in the eval took 1139 tokens. The same options go with every request, or
# Ollama loads the model again (D-002).
RERANK_OPTIONS = {"num_ctx": 4096, "temperature": 0}


def check_reranker(ollama: Ollama) -> None:
    """Stop with the command to run if the reranker is not pulled."""
    try:
        ollama.digest(RERANK_MODEL)
    except OllamaError as exc:
        raise SystemExit(str(exc)) from None


def reorder(ollama: Ollama, question: str, hits: list[Hit], k: int) -> list[Hit]:
    """The k hits the reranker scores highest, best first, each with its reranker score.
    Equal scores keep the search's order."""
    scored = [(relevance(ollama, question, hit), position, hit) for position, hit in enumerate(hits)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [replace(hit, score=score) for score, _, hit in scored[:k]]


def relevance(ollama: Ollama, question: str, hit: Hit) -> float:
    """log P("yes") - log P("no") for the next token: the log-odds of the model card's score,
    which is the probability of "yes" against "no" alone.

    Ollama returns only the 20 likeliest tokens, and one of the two can be missing: "yes" in 15
    of the eval's 500 pairs, all of them far from the top 5. Its log-probability is then at most
    the 20th's, which stands in for it.
    """
    document = f"{hit.section}\n\n{hit.content}"
    prompt = PROMPT.format(instruction=INSTRUCTION, question=question, document=document)
    top = ollama.next_token_logprobs(RERANK_MODEL, prompt, RERANK_OPTIONS)
    floor = min(top.values())
    return top.get("yes", floor) - top.get("no", floor)
