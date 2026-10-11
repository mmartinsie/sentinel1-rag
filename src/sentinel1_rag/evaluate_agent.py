"""`make eval-agent`: measure whether the agent calls the right tools with the right arguments (D-029).

Every question in eval/agent_questions.yaml goes through the agent loop of `make agent`, with
the real tools, and the run is checked against what the file expects:

- tools: it called exactly the expected tools;
- arguments: one of its scene searches used the expected product type, dates and filters;
- place: the place it searched for contains the expected point;
- passages: one of the passages it read comes from a section labelled in questions.yaml;
- citations: an answerable question gets a cited answer, an unanswerable one no citations, and an
  answer that read no passages (a question about scenes only) cites none.

A question passes when every check that applies to it passes. Each run is saved in eval/results/.
"""

import argparse
import json
import time
from datetime import date, datetime

import httpx
import yaml

from sentinel1_rag.agent import Run, result_summary, run_agent
from sentinel1_rag.ask import cited_numbers
from sentinel1_rag.config import AGENT_QUESTIONS_FILE, CHAT_MODEL, RESULTS_DIR, ROOT, USER_AGENT
from sentinel1_rag.db import connect
from sentinel1_rag.evaluate import load_questions
from sentinel1_rag.geocode import Geocoder
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.tools import SceneSearch, Toolbox

CHECKS = ("tools", "arguments", "place", "passages", "citations")
SCENE_ARGUMENTS = ("product_type", "start", "end", "mode", "orbit")


def accepted(expected) -> set[str | None]:
    """The values a question accepts for one argument, as strings (YAML reads dates as dates)."""
    values = expected if isinstance(expected, list) else [expected]
    return {None if v is None else str(v) for v in values}


def arguments_match(search: SceneSearch, expected: dict) -> bool:
    return all(
        (None if getattr(search, key) is None else str(getattr(search, key))) in accepted(expected[key])
        for key in SCENE_ARGUMENTS
        if key in expected
    )


def contains(bbox: tuple[float, float, float, float], point: list[float]) -> bool:
    west, south, east, north = bbox
    lat, lon = point
    return south <= lat <= north and west <= lon <= east


def check(q: dict, run: Run, toolbox: Toolbox, labels: dict[str, set[tuple[str, str]]]) -> dict[str, bool]:
    """The checks that apply to a question, each True or False."""
    called = {c.tool for c in run.calls}
    result = {"tools": called == set(q["tools"])}
    if "scenes" in q:
        result["arguments"] = any(arguments_match(s, q["scenes"]) for s in toolbox.searches)
        result["place"] = any(contains(s.place.bbox, q["scenes"]["point"]) for s in toolbox.searches)
    if "labels_from" in q:
        relevant = labels[q["labels_from"]]
        result["passages"] = any((p.url, p.section) in relevant for p in toolbox.passages)
    numbers = cited_numbers(run.answer or "")
    if "search_docs" in q["tools"]:
        if q.get("answerable", True):
            result["citations"] = bool(numbers) and all(1 <= n <= len(toolbox.passages) for n in numbers)
        else:
            result["citations"] = run.answer is not None and not numbers
    elif not toolbox.passages:
        # Nothing was read that could be cited: an [n] here points at a scene search result.
        result["citations"] = not numbers
    return result


def rates(results: list[dict]) -> dict[str, dict]:
    """For each check, and for whole questions, how many passed out of those it applies to."""
    def share(passed: list[bool]) -> dict:
        return {"passed": sum(passed), "of": len(passed), "rate": round(sum(passed) / len(passed), 3) if passed else None}

    groups = {"all": results} | {
        name: [r for r in results if r["set"] == name] for name in sorted({r["set"] for r in results})
    }
    return {
        name: {
            "questions": share([all(r["checks"].values()) for r in group]),
            **{c: share([r["checks"][c] for r in group if c in r["checks"]]) for c in CHECKS},
        }
        for name, group in groups.items()
    }


def print_report(results: list[dict], scores: dict[str, dict], today: date, think: bool) -> None:
    print(f"Agent eval: {len(results)} questions | {CHAT_MODEL}{' (thinking)' if think else ''} | today {today}")
    for name, group in scores.items():
        values = " | ".join(f"{c} {v['passed']}/{v['of']}" for c, v in group.items() if v["of"])
        print(f"  {name:<9} {values}")

    print(f"\n  {'ok':<3} {'id':<24} {'turns':>5} {'time':>6}  tools called")
    for r in results:
        ok = "yes" if all(r["checks"].values()) else "NO"
        tools = ", ".join(c["tool"] + (" (forced)" if c["forced"] else "") for c in r["calls"]) or "none"
        print(f"  {ok:<3} {r['id']:<24} {r['turns']:>5} {r['seconds']:>5.1f}s  {tools}")

    failures = [r for r in results if not all(r["checks"].values())]
    print(f"\nFailed: {len(failures)}")
    for r in failures:
        print(f"\n{r['id']} ({r['set']}): {r['question']}")
        print(f"  failed checks: {', '.join(c for c, ok in r['checks'].items() if not ok)}")
        for c in r["calls"]:
            forced = " (forced by the loop)" if c["forced"] else ""
            print(f"  turn {c['turn']}: {c['tool']}({json.dumps(c['arguments'], ensure_ascii=False)}){forced}")
            print(f"    -> {c['summary']}")
        for p in r["places"]:
            print(f"  place found: {p['name']} {p['bbox']}")
        answer = (r["answer"] or "(no answer)").replace("\n", " ")
        print(f"  answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the agent's tool calls on eval/agent_questions.yaml.")
    parser.add_argument("--label", default="", help="short name for this run, added to the results file name")
    parser.add_argument("--think", action="store_true", help="let Gemma think before each turn")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these questions")
    parser.add_argument("--set", choices=("tuning", "held-out"), help="run only the questions of one set")
    args = parser.parse_args()

    with AGENT_QUESTIONS_FILE.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    today, questions = spec["today"], spec["questions"]
    if args.only:
        unknown = set(args.only) - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in args.only]
    if args.set:
        questions = [q for q in questions if q["set"] == args.set]
    labels = {q["id"]: {(r["url"], r["section"]) for r in q["relevant"]} for q in load_questions()}
    missing = {q["labels_from"] for q in questions if "labels_from" in q} - labels.keys()
    if missing:
        raise SystemExit(f"labels_from names ids that are not in questions.yaml: {', '.join(sorted(missing))}")

    ollama = Ollama()
    started = time.perf_counter()
    results = []
    with connect() as conn, httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        geocoder = Geocoder(http)
        for q in questions:
            print(f"{q['id']}...", flush=True)
            toolbox = Toolbox(conn, ollama, http, geocoder, today)
            question_started = time.perf_counter()
            run = run_agent(q["question"], toolbox, ollama, today, think=args.think)
            results.append(
                {
                    "id": q["id"],
                    "set": q["set"],
                    "lang": q["lang"],
                    "question": q["question"],
                    "checks": check(q, run, toolbox, labels),
                    "calls": [
                        {"turn": c.turn, "tool": c.tool, "forced": c.forced, "arguments": c.arguments, "summary": result_summary(c.tool, c.result)}
                        for c in run.calls
                    ],
                    "places": [{"name": s.place.name, "bbox": list(s.place.bbox)} for s in toolbox.searches],
                    "passages": [p.section for p in toolbox.passages],
                    "answer": run.answer,
                    "turns": run.turns,
                    "prompt_tokens": run.prompt_tokens,
                    "seconds": round(time.perf_counter() - question_started, 1),
                }
            )

    scores = rates(results)
    print()
    print_report(results, scores, today, args.think)

    now = datetime.now().astimezone()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{now:%Y-%m-%dT%H%M%S}-agent{'-' + args.label if args.label else ''}.json"
    record = {
        "run_at": now.isoformat(timespec="seconds"),
        "label": args.label,
        "kind": "agent",
        "model": CHAT_MODEL,
        "think": args.think,
        "today": today.isoformat(),
        "metrics": scores,
        "questions": results,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)} | {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
