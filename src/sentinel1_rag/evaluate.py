"""`make eval`: measure how often retrieval finds the sections that answer known questions.

eval/questions.yaml lists each question with the sections that answer it. Every question goes
through retrieval.retrieve, exactly as in `make ask`, and the eval records the rank of the
first chunk that comes from one of those sections:

- hit@k: the share of questions with such a chunk in the top k.
- MRR@k: the mean of 1/rank, counting 0 when that chunk is below the top k, because
  `make ask` never shows the model anything below the top k (TOP_K, unless --top-k says
  otherwise, to see what another k would give).

Only retrieval is measured, so Gemma is not used. Each run is saved in eval/results/.
"""

import argparse
import json
import time
from datetime import datetime

import yaml

from sentinel1_rag.config import EMBED_MODEL, QUESTIONS_FILE, RESULTS_DIR, ROOT
from sentinel1_rag.db import connect
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.retrieval import TOP_K, Hit, retrieve

# Retrieve deeper than the top k so that a miss still shows how far down the answer was.
# The top k of a deeper search are the same chunks that `make ask` gets.
DEPTH = 20
HIT_AT = (1, 3, 5, 10)


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def check_labels(questions: list[dict], known: dict[str, str]) -> None:
    """Every expected section must exist in the index; otherwise a typo would count as a miss."""
    errors = [
        f"  {q['id']}: {r['section']}\n    {r['url']}"
        for q in questions
        for r in q["relevant"]
        if known.get(r["url"]) != r["section"]
    ]
    if errors:
        raise SystemExit(f"Sections in {QUESTIONS_FILE.name} that are not in the index:\n" + "\n".join(errors))


def first_rank(hits: list[Hit], relevant: set[str]) -> int | None:
    """Rank (from 1) of the first chunk from a relevant section, or None if none was retrieved."""
    return next((rank for rank, hit in enumerate(hits, start=1) if hit.url in relevant), None)


def metrics(ranks: list[int | None], top_k: int) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    result = {f"hit@{k}": sum(r <= k for r in found) / len(ranks) for k in HIT_AT}
    result[f"mrr@{top_k}"] = sum(1 / r for r in found if r <= top_k) / len(ranks)
    return result


def scores_by_group(results: list[dict], top_k: int) -> dict[str, dict[str, float]]:
    """Metrics for all questions, then for each way of writing them (blind vs from-text)."""
    groups = {"all": results}
    for written in sorted({r["written"] for r in results}):
        groups[written] = [r for r in results if r["written"] == written]
    return {
        name: metrics([r["rank"] for r in group], top_k) | {"questions": len(group)}
        for name, group in groups.items()
    }


def print_report(results: list[dict], scores: dict[str, dict[str, float]], chunks: int, top_k: int) -> None:
    print(f"Retrieval eval: {len(results)} questions | {chunks} chunks | {EMBED_MODEL} | top {top_k}")
    for name, group in scores.items():
        values = " | ".join(f"{metric} {value:.3f}" for metric, value in group.items() if metric != "questions")
        print(f"  {name + ' (' + str(group['questions']) + ')':<16} {values}")

    print(f"\n  {'rank':>4}  {'id':<20} question")
    for r in results:
        rank = r["rank"] or "-"
        print(f"  {rank:>4}  {r['id']:<20} {r['question']}")
    print(f"  (rank of the first chunk from an expected section; '-' means not in the top {DEPTH})")

    misses = [r for r in results if r["rank"] is None or r["rank"] > top_k]
    print(f"\nNot in the top {top_k}: {len(misses)}")
    for r in misses:
        where = f"rank {r['rank']}" if r["rank"] else f"not in the top {DEPTH}"
        print(f"\n{r['id']} ({where}): {r['question']}\n  expected:")
        for section in r["expected"]:
            print(f"    {section}")
        print("  retrieved:")
        for rank, hit in enumerate(r["retrieved"][:top_k], start=1):
            print(f"    {rank:>2} {hit['distance']:.3f}  {hit['section']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval on eval/questions.yaml.")
    parser.add_argument("--label", default="", help="short name for this run, added to the results file name")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"chunks passed to the model (default {TOP_K})")
    args = parser.parse_args()
    if not 1 <= args.top_k <= DEPTH:
        raise SystemExit(f"--top-k must be between 1 and {DEPTH}")

    questions = load_questions()
    ollama = Ollama()
    started = time.perf_counter()
    with connect() as conn:
        # Citation URL -> section path, built the same way as in retrieval.py.
        known = dict(
            conn.execute(
                "SELECT DISTINCT d.url || coalesce('#' || c.anchor, ''), c.section"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
            ).fetchall()
        )
        if not known:
            raise SystemExit("The index is empty. Run `make ingest` and `make index` first.")
        check_labels(questions, known)
        configs = [json.loads(c) for (c,) in conn.execute("SELECT DISTINCT index_config FROM documents")]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

        results = []
        for q in questions:
            hits = retrieve(conn, ollama, q["question"], k=DEPTH)
            relevant = {r["url"] for r in q["relevant"]}
            results.append(
                {
                    "id": q["id"],
                    "written": q["written"],
                    "question": q["question"],
                    "expected": [r["section"] for r in q["relevant"]],
                    "rank": first_rank(hits, relevant),
                    "retrieved": [
                        {"section": h.section, "url": h.url, "distance": round(h.distance, 4)} for h in hits
                    ],
                }
            )

    scores = scores_by_group(results, args.top_k)
    print_report(results, scores, chunks, args.top_k)

    now = datetime.now().astimezone()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{now:%Y-%m-%dT%H%M%S}{'-' + args.label if args.label else ''}.json"
    record = {
        "run_at": now.isoformat(timespec="seconds"),
        "label": args.label,
        "top_k": args.top_k,
        "depth": DEPTH,
        "chunks": chunks,
        "index_config": configs,  # one entry per distinct recipe; normally exactly one
        "metrics": scores,
        "questions": results,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)} | {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
