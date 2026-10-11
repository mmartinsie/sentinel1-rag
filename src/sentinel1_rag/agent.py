"""`make agent Q="..."`: answer with a hand-written agent loop over two tools (D-027).

The model gets the question and a description of each tool (tools.py). On every turn it either
asks for tool calls or answers. Ollama only returns the calls; this loop runs them, appends each
result to the conversation as a `tool` message and asks the model again, until it answers or
MAX_TURNS is reached.

Then, as in `make ask`, the code prints the sources the answer cites and the scenes found, from
what the tools returned, never from the model's words.
"""

import argparse
import json
import re
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx

from sentinel1_rag.ask import cited_numbers, print_sources
from sentinel1_rag.config import CHAT_MODEL, CHAT_NUM_CTX, USER_AGENT
from sentinel1_rag.db import connect
from sentinel1_rag.geocode import ATTRIBUTION, Geocoder
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.scenes import item_url
from sentinel1_rag.tools import SCHEMAS, Toolbox

# Model turns per question. A question about scenes and documentation takes two: one with both
# calls, one with the answer.
MAX_TURNS = 5
# Temperature 0 makes the tool calls repeatable, so the eval is too.
AGENT_OPTIONS = {"num_ctx": CHAT_NUM_CTX, "temperature": 0, "num_predict": 1024}

SYSTEM_PROMPT = """\
You answer questions about the Sentinel-1 radar satellite mission. Today is {weekday} {today}.

Answer only from what the tools return, never from memory:
- search_docs searches the Sentinel-1 documentation. Call it for every question, or part of a \
question, that is not only about finding scenes, even if you think you know the answer or expect \
the documentation not to have it. Use only facts stated in the passages it returns, and cite them \
by number after each sentence, like [2] or [1][3]. If the passages do not contain the answer, say \
that you could not find it in the documentation.
- search_scenes finds acquisitions over a place in a date range. For "the last N days" or "the \
past N weeks", give days; for months or dates, give start and end ("last month" is the previous \
calendar month). Never ask the user to confirm dates. Say which place and dates were searched and \
summarise what was found; do not list the scenes.

A question can need both tools; then call both.
Write the answer in the same language as the question. Keep acronyms such as GRD or SLC as they are. \
Be concise."""


@dataclass
class Call:
    """One tool call, as the model asked for it and as it was run."""

    turn: int
    tool: str
    arguments: dict[str, Any]
    result: str
    seconds: float
    forced: bool = False  # run by the loop, not asked for by the model (see run_agent)


@dataclass
class Run:
    answer: str | None  # None when the model was still calling tools after MAX_TURNS turns
    calls: list[Call] = field(default_factory=list)
    turns: int = 0
    prompt_tokens: int = 0  # of the last turn, which reads the whole conversation
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    load_seconds: float = 0.0


def system_prompt(today: date) -> str:
    return SYSTEM_PROMPT.format(weekday=today.strftime("%A"), today=today.isoformat())


def run_agent(
    question: str,
    toolbox: Toolbox,
    ollama: Ollama,
    today: date,
    think: bool = False,
    log: Callable[[str], None] = lambda line: None,
) -> Run:
    """The agent loop. `log` gets one line per model turn and per tool call."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(today)},
        {"role": "user", "content": question},
    ]
    run = Run(answer=None)
    for turn in range(1, MAX_TURNS + 1):
        response = ollama.chat(CHAT_MODEL, messages, AGENT_OPTIONS, tools=SCHEMAS, think=think)
        message = response["message"]
        # The model's turn goes back into the conversation as it came, tool calls included, so
        # that the next turn sees what it asked for next to what came back.
        messages.append(message)
        run.turns = turn
        run.prompt_tokens = response["prompt_eval_count"]
        run.model_seconds += response["total_duration"] / 1e9
        run.load_seconds += response["load_duration"] / 1e9
        tool_calls = message.get("tool_calls") or []
        forced = False
        if not tool_calls and not run.calls:
            # An answer before any tool was called comes from the model's memory: in the eval it
            # skipped search_docs for "¿Qué es el modo IW?" and for a question the documentation
            # cannot answer, whatever the prompt said. So the loop discards that answer and searches
            # the documentation with the question itself, as `make ask` does (D-027).
            forced = True
            tool_calls = [{"function": {"name": "search_docs", "arguments": {"query": question}}}]
            messages[-1] = {"role": "assistant", "content": "", "tool_calls": tool_calls}
            log(f"Turn {turn}: answered without calling a tool; the loop searches the documentation instead")
        elif not tool_calls:
            run.answer = message["content"].strip()
            log(f"Turn {turn}: answer ({response['total_duration'] / 1e9:.1f}s)")
            return run
        else:
            log(f"Turn {turn}: {len(tool_calls)} tool call{'s' if len(tool_calls) > 1 else ''} ({response['total_duration'] / 1e9:.1f}s)")
        for tool_call in tool_calls:
            name, arguments = tool_call["function"]["name"], tool_call["function"]["arguments"]
            started = time.perf_counter()
            result = toolbox.call(name, arguments)
            seconds = time.perf_counter() - started
            run.tool_seconds += seconds
            run.calls.append(Call(turn, name, arguments, result, seconds, forced))
            log(f"  {name}({json.dumps(arguments, ensure_ascii=False)})  {seconds:.1f}s")
            log(f"    -> {result_summary(name, result)}")
            messages.append({"role": "tool", "tool_name": name, "content": result})
    log(f"Stopped after {MAX_TURNS} turns without an answer.")
    return run


PASSAGE_NUMBER = re.compile(r"^(\[\d+\]) ", re.MULTILINE)


def result_summary(name: str, result: str) -> str:
    """One line about a tool's result, for the trace."""
    if result.startswith("Error:"):
        return result
    if name == "search_docs":
        return "passages " + " ".join(PASSAGE_NUMBER.findall(result))
    lines = result.splitlines()
    return next((line for line in lines if line.startswith(("Found:", "No scenes"))), lines[0])


def print_scenes(toolbox: Toolbox) -> None:
    for search in toolbox.searches:
        print(f"\nScenes: {search.describe()}")
        if not search.scenes:
            print("    none found")
            continue
        for s in search.scenes:
            print(
                f"  {s.acquired[:10]} {s.acquired[11:16]}  {s.id[:3]} {s.mode:<3} {'+'.join(s.polarisations):<6}"
                f" {s.orbit:<10} track {s.relative_orbit:<4} {s.id}"
            )
        if search.more:
            print(f"  (the {len(search.scenes)} newest; the catalogue has more: narrow the dates to see them)")
        print(f"  Metadata of a scene: {item_url(search.collection, '<name>')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question about Sentinel-1 with an agent that can search the documentation and the scene catalogue.")
    parser.add_argument("question")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help="the date the model takes as today (YYYY-MM-DD)")
    parser.add_argument("--think", action="store_true", help="let Gemma think before each turn")
    parser.add_argument("--show-results", action="store_true", help="also print each tool's result, as the model read it")
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        raise SystemExit('Usage: make agent Q="your question" [ARGS=--show-results]')

    ollama = Ollama()
    with connect() as conn, httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        toolbox = Toolbox(conn, ollama, http, Geocoder(http), args.today)
        print(f"Asking {CHAT_MODEL}, with tools...", flush=True)
        run = run_agent(question, toolbox, ollama, args.today, think=args.think, log=lambda line: print(line, flush=True))

    if args.show_results:
        for call in run.calls:
            print(f"\nResult of {call.tool} (turn {call.turn}):\n" + textwrap.indent(call.result, "    "))
    if run.answer is None:
        print(f"\nNo answer after {MAX_TURNS} turns.")
    else:
        print(f"\n{run.answer}")
    numbers = cited_numbers(run.answer or "")
    if toolbox.passages:
        print()
        print_sources(numbers, toolbox.passages)
    elif numbers:
        print(f"\nwarning: the answer cites {numbers}, but the documentation was never searched")
    print_scenes(toolbox)
    if toolbox.searches:
        print(f"\n{ATTRIBUTION}")

    stats = (
        f"{CHAT_MODEL}: {run.turns} turns, {run.prompt_tokens} prompt tokens in the last one;"
        f" {run.model_seconds:.1f}s in the model, {run.tool_seconds:.1f}s in the tools"
    )
    if run.load_seconds >= 1:
        stats += f" ({run.load_seconds:.0f}s of it loading the model)"
    print(f"\n{stats}")


if __name__ == "__main__":
    main()
