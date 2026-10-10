"""`make ask Q="..."`: answer a question about Sentinel-1 from the indexed chunks, with citations.

1. Retrieve the chunks closest to the question (retrieval.py).
2. Number them in the prompt and ask Gemma to answer only from them, citing them as [n].
3. Print the answer, then the sources it cites. The code maps each [n] back to its chunk,
   so the section and URL come from the database, never from the model.
"""

import argparse
import re
import textwrap

from sentinel1_rag.config import CHAT_MODEL
from sentinel1_rag.db import connect
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.retrieval import METHODS, SCORE_LABEL, Hit, retrieve

# Sent with every request (D-002): Ollama's default context on this GPU is only 4096 tokens.
# num_predict stops an answer that never ends.
CHAT_OPTIONS = {"num_ctx": 8192, "temperature": 0.2, "num_predict": 1024}

# Wording matters with a small model: "answer in the language of the question, even though
# the passages are in English" got English questions answered in Spanish 8 times out of 10.
SYSTEM_PROMPT = """\
You answer questions about the Sentinel-1 radar satellite mission. Each question comes with \
numbered passages from the SentiWiki documentation.

Rules:
- Use only facts stated in the passages. Do not add facts from your own knowledge, even if you know them.
- After each sentence, cite the passages that support it by number in square brackets, like [2] or [1][3].
- If the passages do not contain the answer, say only that you could not find it in the documentation. \
If they answer part of the question, answer that part and say what is missing.
- Write the answer in the same language as the question. Keep acronyms such as GRD or SLC as they are.
- Be concise: a short paragraph or a few bullet points."""

# [3], or a group such as [1, 3]: the prompt asks for [1][3], but small models vary.
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    """Instructions go in the system turn; the passages, then the question, in the user turn."""
    passages = "\n\n".join(f"[{n}] {hit.section}\n{hit.content}" for n, hit in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question}"},
    ]


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for group in CITATION.findall(answer) for n in group.split(",")})


def print_context(hits: list[Hit], method: str) -> None:
    print(f"Retrieved chunks ({method} retrieval; {SCORE_LABEL[method]})\n")
    for n, hit in enumerate(hits, start=1):
        print(f"[{n}] {hit.score:.3f}  {hit.section}\n    {hit.url}\n")
        print(textwrap.indent(hit.content, "    ") + "\n")


def print_sources(numbers: list[int], hits: list[Hit]) -> None:
    """One entry per cited section: chunks cut from the same section share it, as in [1][2]."""
    sources: dict[tuple[str, str], list[int]] = {}
    for n in numbers:
        if 1 <= n <= len(hits):
            sources.setdefault((hits[n - 1].section, hits[n - 1].url), []).append(n)
    print("Sources:")
    for (section, url), group in sources.items():
        print(f"{''.join(f'[{n}]' for n in group)} {section}\n    {url}")
    if not sources:
        print("    none cited (--show-context prints the chunks that were retrieved)")
    unknown = [n for n in numbers if not 1 <= n <= len(hits)]
    if unknown:
        print(f"    warning: the answer cites {unknown}, but there were only passages 1-{len(hits)}")


def print_stats(response: dict) -> None:
    stats = (
        f"{CHAT_MODEL}: {response['prompt_eval_count']} prompt tokens,"
        f" {response['eval_count']} answer tokens, {response['total_duration'] / 1e9:.1f}s"
    )
    if (load := response["load_duration"] / 1e9) >= 1:
        stats += f" ({load:.0f}s of it loading the model)"
    print(f"\n{stats}")
    if response["done_reason"] == "length":
        print(f"warning: the answer was cut at {CHAT_OPTIONS['num_predict']} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question about Sentinel-1, citing the SentiWiki.")
    parser.add_argument("question")
    parser.add_argument("--show-context", action="store_true", help="also print the retrieved chunks")
    parser.add_argument("--method", choices=METHODS, default="vector", help="retrieval method (default vector)")
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        raise SystemExit('Usage: make ask Q="your question" [ARGS=--show-context]')

    ollama = Ollama()
    with connect() as conn:
        hits = retrieve(conn, ollama, question, method=args.method)
    if not hits:
        raise SystemExit("The index is empty. Run `make ingest` and `make index` first.")
    if args.show_context:
        print_context(hits, args.method)

    # Flushed so the line shows up while waiting, even when the output goes to a pipe.
    print(f"Asking {CHAT_MODEL}...", flush=True)
    response = ollama.chat(CHAT_MODEL, build_messages(question, hits), CHAT_OPTIONS)
    answer = response["message"]["content"].strip()
    print(f"\n{answer}\n")
    print_sources(cited_numbers(answer), hits)
    print_stats(response)


if __name__ == "__main__":
    main()
