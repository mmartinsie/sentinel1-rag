# Step 3: Ingest and chunking

*Finished on 2026-10-08.*

## What this step was about

An embedding squeezes a whole piece of text into one vector, so the size of the pieces decides what can be found later. A chunk that is too big mixes several topics, and its vector ends up as a vague average of all of them. A chunk that is too small loses the context that gives it meaning: "this mode offers 5 m resolution" says little if you don't know which mode.

So we cut along the structure the authors already gave the text. Each heading opens a section, and a section is already a unit of topic. Only long sections are cut further, between sentences. Each of those cuts repeats a little of the previous chunk (an *overlap*), so that an idea sitting on the boundary appears whole somewhere. Every chunk also carries its heading path ("S1 Mission > Acquisition Modes > Interferometric Wide Swath"), which goes in front of the text before it is embedded.

Sizes are measured in *tokens*, the word pieces a model actually reads, because both the embedding model's limits and the prompt budget are counted in tokens.

## What we built

`make ingest` runs four small modules, one per job:

1. **Download** (`fetch.py`). It fetches each page listed in `sources.yaml` politely:
   - it identifies itself;
   - it checks `robots.txt`;
   - it waits a second between requests;
   - it keeps a copy in `data/raw/`, so each page is downloaded only once.
2. **Extraction** (`extract.py`). It finds the article body in the HTML, throws away buttons, icons and images, and walks the page in order, starting a new section at every heading. The text is kept as follows:
   - paragraphs become blocks of text;
   - lists become one line per item;
   - tables become one line per row ("cell | cell | cell");
   - figures keep only their caption.
3. **Chunking** (`chunking.py`). It turns sections into chunks, as described above.
4. **Output** (`ingest.py`). It writes one JSON file per page in `data/chunks/`, with the chunks and a fingerprint of the page text, and prints statistics. Nothing goes into the database yet; that is step 4.

## The corpus

The site's own page index confirmed that the Sentinel-1 part of the SentiWiki is exactly five pages: a landing page and four long chapters (mission, products, processing and applications).

Three other pages were considered and left out for now:

- a page on precise orbits, shared by all missions;
- a glossary covering every mission;
- an almost empty page on the SAFE format.

Adding them can be tested in step 6, when we can measure whether they help or only add noise.

## What we found along the way

**Our token estimate was wrong, and measuring it showed by how much.** Counting tokens exactly needs the model's own tokenizer, so the chunker estimates them from the number of words. The first guess was 1.35 tokens per word, close to a common rule of thumb for English. To check it, we sent every chunk through bge-m3 and read the real count back from Ollama. This corpus actually needs 1.68 tokens per word. Technical text is full of codes, numbers and units ("S1A_IW_GRDH_1SDV", "20.4x22.5"), and those split into many tokens; in tables the ratio is almost 1.9. With the corrected ratio, the estimate is right on average.

**Splitting a table in the middle destroyed its meaning.** One example chunk started halfway through a table: rows of numbers without the title row that says which product and which beams they describe. Tables now stay whole when they fit in a chunk, and 43 of the 44 tables do.

**The wiki's footnote markers had to go.** Some paragraphs contain bibliography markers like "[1]". Our answers will cite their sources with the same notation ([1], [2]…), so these markers could confuse the language model, and they are now removed.

## Results

- **Count:** 195 chunks from 89 sections across the five pages. 32 long sections were split.
- **Real sizes**, measured with bge-m3: 268 tokens on average, median 273, smallest 20, largest 595. Most chunks (133 of 195) have between 200 and 400 tokens. Three table-heavy chunks are slightly above 500, which is harmless.
- **Overlap:** 87 of the 106 cuts carry an overlap. The others come right after a whole table or a very long sentence, which is too big to repeat.
- **Repeatability:** running it a second time downloads nothing and produces byte-identical files.

Three examples:

- **Prose**, from "S1 Mission > Acquisition Modes > Interferometric Wide Swath": "The Interferometric Wide (IW) swath mode is the main acquisition mode over land and satisfies the majority of service requirements. It acquires data with a 250 km swath at 5 m by 20 m spatial resolution (single look)…"
- **A whole table**, from "S1 Products > Level-1 Products": the characteristics of the IW GRD High Resolution product. It runs from "Product ID | IW_GRD_HR" down to the rows for each beam, such as "Beam ID | IW1 | IW2 | IW3" and "Spatial Resolution rg x az m | 20.4x22.5 | 20.3x22.6 | 20.5x22.6".
- **An overlap**, in "SAR Observation Scenario": one chunk ends with "The high level Sentinel-1 observation strategy during full operations capacity is based on:" and the next one starts with that same sentence, followed by the list it introduces.

## What it means for the next steps

Step 4 reads these JSON files, embeds each chunk with its heading path in front, and stores everything in Postgres, one page per transaction. The page fingerprint and the chunking settings tell it which pages it can skip because nothing has changed.

## Check your understanding

**Why cut by sections instead of using fixed windows of, say, 400 tokens?**
Sections are topics chosen by the authors, so a chunk that follows them talks about one thing, and its vector represents that thing well. A fixed window cuts wherever the count runs out, often in the middle of a topic or a table, and mixes the end of one subject with the start of the next. Windows are simpler and give perfectly even sizes. Here, though, the structure is clean, and the headings double as citation anchors.

**How did you choose the tokens-per-word ratio, and why does it matter?**
We measured it. Every chunk went through bge-m3 and Ollama reported the real token count, which gave 1.68 tokens per word on this corpus instead of the 1.35 we had assumed. A ratio that is too low makes the chunker believe chunks are smaller than they are: the first guess let chunks reach almost 600 tokens when we aimed for 500 at most. Chunk size matters for two reasons: it affects how focused each embedding is, and it decides how many chunks fit in the prompt.
