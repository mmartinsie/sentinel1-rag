"""`make eval`: measure how often retrieval finds the sections that answer known questions.

eval/questions.yaml lists each question with the sections that answer it. Every question goes
through retrieval.retrieve, exactly as in `make ask` (--method picks vector, lexical or hybrid
retrieval, and --translate gives the lexical search the question in English), and the eval
records the rank of the first chunk that comes from one of those sections:

- hit@k: the share of questions with such a chunk in the top k.
- MRR@k: the mean of 1/rank, counting 0 when that chunk is below the top k, because
  `make ask` never shows the model anything below the top k (TOP_K, unless --top-k says
  otherwise, to see what another k would give).

Only retrieval is measured, so Gemma is used only to translate the questions for the lexical
search (on by default; --method vector or --no-translate skip it). --backend opensearch runs
the same searches in OpenSearch (D-034), and --rerank reorders the top 20 with the reranker
(D-037). Each run is saved in eval/results/.
"""

import argparse
import json
import time
from datetime import datetime

import yaml

from sentinel1_rag.config import EMBED_MODEL, QUESTIONS_FILE, RERANK_MODEL, RESULTS_DIR, ROOT
from sentinel1_rag.db import connect
from sentinel1_rag.fetch import is_pdf
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.retrieval import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_METHOD,
    DEFAULT_RANKING,
    DEFAULT_RERANK,
    METHODS,
    RANKINGS,
    TOP_K,
    Hit,
    open_backend,
    retrieve,
    score_label,
)
from sentinel1_rag.rerank import INSTRUCTION, RERANK_CANDIDATES, check_reranker
from sentinel1_rag.translate import to_english, use_translation

# Retrieve deeper than the top k so that a miss still shows how far down the answer was.
# The top k of a deeper search are the same chunks that `make ask` gets.
DEPTH = 20
HIT_AT = (1, 3, 5, 10)


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def check_labels(questions: list[dict], known: set[tuple[str, str]]) -> None:
    """Every expected section must exist in the index; otherwise a typo would count as a miss."""
    errors = [
        f"  {q['id']}: {r['section']}\n    {r['url']}"
        for q in questions
        for r in q["relevant"]
        if (r["url"], r["section"]) not in known
    ]
    if errors:
        raise SystemExit(f"Sections in {QUESTIONS_FILE.name} that are not in the index:\n" + "\n".join(errors))


def answer_in(relevant: list[dict]) -> str:
    """Where a question's labelled sections are: only in SentiWiki pages, only in PDFs, or both."""
    kinds = {is_pdf(r["url"]) for r in relevant}
    return "wiki+pdf" if len(kinds) == 2 else "pdf-only" if True in kinds else "wiki-only"


def first_rank(hits: list[Hit], relevant: set[tuple[str, str]]) -> int | None:
    """Rank (from 1) of the first chunk from a relevant section, or None if none was retrieved.
    A section is its citation URL and its heading path: in a PDF, sections that start on the
    same page share the URL (url#page=N)."""
    return next((rank for rank, hit in enumerate(hits, start=1) if (hit.url, hit.section) in relevant), None)


def metrics(ranks: list[int | None], top_k: int) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    result = {f"hit@{k}": sum(r <= k for r in found) / len(ranks) for k in HIT_AT}
    result[f"mrr@{top_k}"] = sum(1 / r for r in found if r <= top_k) / len(ranks)
    return result


def scores_by_group(results: list[dict], top_k: int) -> dict[str, dict[str, float]]:
    """Metrics for all questions, then by how they were written (blind vs from-text), by language
    and by where the answer is (wiki-only, pdf-only, wiki+pdf)."""
    groups = {"all": results}
    for field in ("written", "lang", "answer_in"):
        for value in sorted({r[field] for r in results}):
            groups[value] = [r for r in results if r[field] == value]
    return {
        name: metrics([r["rank"] for r in group], top_k) | {"questions": len(group)}
        for name, group in groups.items()
    }


def print_report(
    results: list[dict],
    scores: dict[str, dict[str, float]],
    chunks: int,
    top_k: int,
    method: str,
    translate: bool,
    ranking: str,
    backend: str,
    rerank: bool,
) -> None:
    model = f" | {EMBED_MODEL}" if method != "lexical" else ""
    lexical = "" if method == "vector" else f" ({ranking}{', question in English' if translate else ''})"
    reranked = f" | reranked by {RERANK_MODEL}" if rerank else ""
    print(f"Retrieval eval: {len(results)} questions | {chunks} chunks | {backend} | {method}{lexical}{model}{reranked} | top {top_k}")
    for name, group in scores.items():
        values = " | ".join(f"{metric} {value:.3f}" for metric, value in group.items() if metric != "questions")
        print(f"  {name + ' (' + str(group['questions']) + ')':<17} {values}")

    print(f"\n  {'rank':>4}  {'id':<20} question")
    for r in results:
        rank = r["rank"] or "-"
        print(f"  {rank:>4}  {r['id']:<20} {r['question']}")
    print(f"  (rank of the first chunk from an expected section; '-' means not in the top {DEPTH})")

    misses = [r for r in results if r["rank"] is None or r["rank"] > top_k]
    print(f"\nNot in the top {top_k}: {len(misses)} (scores are {score_label(method, ranking, rerank)})")
    for r in misses:
        where = f"rank {r['rank']}" if r["rank"] else f"not in the top {DEPTH}"
        print(f"\n{r['id']} ({where}): {r['question']}")
        if r.get("english", r["question"]) != r["question"]:
            print(f"  lexical search: {r['english']}")
        print("  expected:")
        for section in r["expected"]:
            print(f"    {section}")
        print("  retrieved:")
        for rank, hit in enumerate(r["retrieved"][:top_k], start=1):
            print(f"    {rank:>2} {hit['score']:.3f}  {hit['section']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval on eval/questions.yaml.")
    parser.add_argument("--label", default="", help="short name for this run, added to the results file name")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"chunks passed to the model (default {TOP_K})")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD, help=f"retrieval method (default {DEFAULT_METHOD})")
    parser.add_argument(
        "--translate", action=argparse.BooleanOptionalAction,
        help="translate the questions into English for the lexical search (default: on, except with --method vector)",
    )
    parser.add_argument(
        "--lexical-ranking", choices=RANKINGS, default=DEFAULT_RANKING,
        help=f"how the full-text search ranks chunks (default {DEFAULT_RANKING})",
    )
    parser.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND, help=f"where to search (default {DEFAULT_BACKEND})")
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=DEFAULT_RERANK,
        help=f"reorder the top 20 with the reranker (default {'on' if DEFAULT_RERANK else 'off'})",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= DEPTH:
        raise SystemExit(f"--top-k must be between 1 and {DEPTH}")
    args.translate = use_translation(args.method, args.translate)

    questions = load_questions()
    ollama = Ollama()
    started = time.perf_counter()
    with connect() as conn:
        # (citation URL, section path) of every section, built the same way as in retrieval.py.
        known = set(
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
        opensearch = open_backend(args.backend, conn, args.method, args.lexical_ranking)
        if args.rerank:
            check_reranker(ollama)

        results = []
        for q in questions:
            english = to_english(ollama, q["question"]) if args.translate else None
            hits = retrieve(
                conn, ollama, q["question"], k=DEPTH, method=args.method, english=english,
                ranking=args.lexical_ranking, opensearch=opensearch, rerank=args.rerank,
            )
            relevant = {(r["url"], r["section"]) for r in q["relevant"]}
            results.append(
                {
                    "id": q["id"],
                    "written": q["written"],
                    "lang": q["lang"],
                    "answer_in": answer_in(q["relevant"]),
                    "question": q["question"],
                    **({"english": english} if args.translate else {}),
                    "expected": [r["section"] for r in q["relevant"]],
                    "rank": first_rank(hits, relevant),
                    "retrieved": [
                        {"section": h.section, "url": h.url, "score": round(h.score, 4)} for h in hits
                    ],
                }
            )

    scores = scores_by_group(results, args.top_k)
    print_report(results, scores, chunks, args.top_k, args.method, args.translate, args.lexical_ranking, args.backend, args.rerank)

    now = datetime.now().astimezone()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{now:%Y-%m-%dT%H%M%S}{'-' + args.label if args.label else ''}.json"
    record = {
        "run_at": now.isoformat(timespec="seconds"),
        "label": args.label,
        "backend": args.backend,
        "method": args.method,
        "translate": args.translate,
        "lexical_ranking": None if args.method == "vector" else args.lexical_ranking,
        "rerank": {
            "model": RERANK_MODEL, "digest": ollama.digest(RERANK_MODEL), "candidates": RERANK_CANDIDATES,
            "instruction": INSTRUCTION,
        } if args.rerank else None,
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
