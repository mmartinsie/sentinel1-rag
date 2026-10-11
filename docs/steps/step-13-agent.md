# Step 13: An agent that searches the documentation and the scene catalogue

*Finished on 2026-10-11. The next extension in the README, chosen by the user.*

## What this step was about

Until now every question went through the same fixed pipeline: retrieve five passages, give them to the model, get an answer. An agent works differently. The model gets the question together with a description of some tools, and on each turn it decides whether to call one of them or to answer. The program runs the tools it asks for, gives the results back, and asks again, in a loop, until the model answers.

Tool calling sounds like the model runs code, but it does not. Each tool is described to the model with a JSON Schema: its name, what it does, and its arguments. When the model wants a tool, Ollama returns a structured request ("call `search_scenes` with these arguments") instead of text. The loop in our code runs the function, adds its result to the conversation as a `tool` message, and calls the model again. Everything the agent can do is code we wrote; the model only chooses.

The reason for an agent here is a second source of facts. The documentation explains how Sentinel-1 works, but it cannot say what the satellites have acquired. That is in the Copernicus Data Space Ecosystem's STAC catalogue, a standard API where a search with a place, dates and filters returns the matching scenes. With both as tools, one question can use both: "Are there scenes of Rotterdam from last week, and what does the IW mode acquire?"

## What we built

`make agent Q="..."` runs a loop of about 50 lines in `agent.py`, written by hand without an agent framework. It allows at most 5 turns, uses temperature 0, and prints a trace of every call. `make ask` did not change.

The agent has two tools:

- **`search_docs`** runs the same retrieval as `make ask` on a query the model writes, and returns numbered passages to cite.
- **`search_scenes`** turns a place name into a bounding box and searches the catalogue for Sentinel-1 GRD or SLC scenes over it, optionally only in one acquisition mode or orbit direction. The place lookup uses Nominatim, OpenStreetMap's geocoder, after reading its usage policy. Answers are cached, requests are limited to one per second, and the attribution to OpenStreetMap is printed. The model would not guess coordinates by itself, so a geocoder was needed.

As in `make ask`, what the user sees as evidence is printed by the code, not written by the model. Under the answer come the cited sections and a table of the scenes found, with their names. The model only reads a summary of the scenes: how many, on which days, from which satellites and orbits. Copying ten long scene names had taken it about 20 seconds.

`make eval-agent` checks the agent on 18 questions: 5 about the documentation, 8 about scenes and 5 mixed. It looks at what the agent did rather than at the scenes, which change every day:

- whether it called the right tools;
- whether it gave the right arguments (product type, dates, filters);
- whether the place it found contains a known point;
- whether a passage it read comes from a section known to answer;
- whether its citations make sense.

"Today" is fixed in the question file, so "last month" has one right answer. Twelve questions were used to improve the agent ("tuning"). Six were written after the first run and never used for that ("held-out"), to see whether the improvements carry over to questions nobody tuned for.

## What we found

**A small model follows rules only some of the time.** The first version had a reasonable prompt, and it passed 10 of the 18 questions. It answered "What is the IW mode?" from memory, and wrongly. It said that the mission's cost was not in the documentation without having looked. It turned "the last 10 days" into 20 days. It added an acquisition mode nobody asked for, once one that found nothing. Writing these rules more firmly into the prompt fixed one question and broke another. What worked was moving each rule out of the prompt and into the code or the shape of the tools:

- If the model answers before calling any tool, the loop throws that answer away and searches the documentation itself.
- For "the last N days" the model passes a number, and the code works out the dates.
- The mode and orbit filters offer "any" as an explicit choice, and that is the default.
- Errors say exactly what is wrong, because the model reliably corrects a call when it is told what is missing.

**Place names are ambiguous.** For the geocoder, "Doñana, Spain" is a mountain peak in Almería and "Delta del Ebro, Spain" is a street. The national park and the river delta are not even among its first five matches. The first version searched those tiny places and reported their scenes as if they were right. Now a place less than 1 km across is sent back once, as an error that asks for a more precise name. When the question is only about scenes, the model then searches again with a better name. When the question also asks about the documentation, it answers with what it has and says that the place was not found, which is honest but incomplete.

**Fixes can trade one failure for another.** The stricter prompt stopped the answers from memory, but it made the model put a citation like "[1]" after facts about scenes, as if a scene search were a documentation passage. The eval did not catch it at first, because it only checked citations on documentation questions. It showed up when reading the answers. The eval then gained that check, and every scene result now says that it has no number to cite. That brought the problem down from half of the scene answers to one.

**The final version passes 13 of the 18 questions, and 5 of the 6 held-out ones,** against 10 and 4 for the first version, in a median of about 6 seconds per question. Letting the model think before each turn passes the same 13, but takes about 15 seconds. Thinking first, it also searched the documentation by itself for the two questions it cannot answer, so the code did not have to force that search. Thinking stays off by default.

**Running the eval froze the computer once.** Ollama decided that memory was short and stopped mapping the model file. Without mapping, the whole model is copied into memory, which uses more memory, not less. Gemma's process grew from 3 to 6 GB, the system started swapping to the hard disk, and it stopped responding for almost half an hour. Every request now asks Ollama to map the model files, and WSL was given 12 GB instead of 10. The later runs were watched and would have been stopped if free memory fell below 2 GiB, which it never did.

## What it means for the next steps

The agent works, and its weak points are known and measured: ambiguous places in mixed questions, one remaining stray citation, and the retrieval misses it inherits from `make ask`. Its eval is small, the held-out part has only six questions, and two runs of the same configuration gave the same checks but differently worded answers, so small differences between versions mean little. The tools are already plain functions with JSON Schemas, which is what the next extension, an MCP server, needs to expose them to other programs.

## Check your understanding

**In tool calling, who runs the tool, and what does the model actually produce?**
The model never runs anything. It is told about each tool through a JSON Schema (name, purpose, arguments), and when it wants one it produces a structured request with the tool's name and the arguments. Our code reads that request, runs the matching Python function, appends the result to the conversation as a `tool` message, and calls the model again. The loop stops when the model answers without asking for a tool, or after a fixed number of turns.

**Why put some of an agent's rules in code instead of in the prompt, and what does that cost?**
A small model follows prompt rules unreliably: here it skipped the documentation search, miscounted dates and invented filters even with rules against it, and rewording the prompt broke other questions. Rules that must always hold can be enforced in code or in the tool's design: a forced search when no tool was called, the code counting the days, an explicit "any" value. The cost is that the code overrides the model's choices, for example by searching the documentation even for a greeting, and each guard has to be measured to show it helps.
