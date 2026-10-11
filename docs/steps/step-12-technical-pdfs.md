# Step 12: Technical PDFs from the Sentinel-1 document library

*Finished on 2026-10-11. The next extension in the README, chosen by the user.*

## What this step was about

Until now the corpus was five SentiWiki pages. The SentiWiki also has a document library with the mission's technical notes and specifications, as PDFs. They explain how to do things the wiki only mentions, such as converting pixel values into calibrated backscatter or removing thermal noise.

A PDF is a hard source for a RAG system. It does not store text with its structure: it stores characters placed at positions on a page. Which characters form a word, which lines form a paragraph, which line is a heading and which numbers belong to which table column all have to be worked out again. The structure is exactly what this system needs. Chunks are cut at section boundaries, every chunk is embedded with its heading path in front, and every citation points to a section. So most of this step was about rebuilding that structure reliably.

Citations needed a new idea too. A web page has an anchor for each heading, but a PDF does not. What a PDF link can do is open the file at a page: `file.pdf#page=20`, a standard fragment that browsers' PDF viewers support. A section is therefore cited by its heading path and the page where it starts.

## What we built

Six documents were chosen, the latest issue of each: notes on radiometric calibration, thermal noise removal, the no-value pixels at the edges of GRD products and TOPS deramping, a guide to geocoding, and the 130-page product definition. They are listed in `sources.yaml` with a short title and a URL that belongs to that issue of the file. They are downloaded and cached like the pages and never committed.

The new extractor, `extract_pdf.py`, reads each PDF with the pdfplumber library. pypdf, the lighter alternative, split words in two ("Faci lity"), which would break the keyword search. The extractor then rebuilds the structure in a series of small steps:

- It drops lines that repeat in the top or bottom margin of most pages, which are running headers, footers and page numbers.
- It treats a line as a heading when it starts with a section number, is set in a larger font than the body text, and continues the numbering (3.1 may follow 3, but not 3.4). Each of the three conditions blocked a real mistake. Only one of the PDFs had bookmarks, so they could not be used.
- It joins lines into paragraphs where the gap between them is small, and turns bulleted lines into list items.
- It turns tables into one line per row. A cell that spans several rows is repeated in each of them, so that every row lines up with its header.
- It drops the debris of charts and equations: rotated axis labels, spaced-out letters, and glyphs that the font does not map to any character.

The result has the same shape as a SentiWiki page, so chunking, indexing and the database did not change at all. The corpus grew from 195 chunks to 568, of which 373 come from the PDFs.

## What we found

**The eval had to be re-labelled before it could be trusted.** Each eval question lists the sections that answer it, and those lists only named wiki sections. With the PDFs in, the correct section in the product definition came first for "what does an SLC product contain", and the old labels counted that as a miss. So the 19 questions were checked against the PDFs with the same strict rule as before, and 11 of them gained PDF sections. Six new questions were drafted from the PDFs, one per document. One more fix came out of this: two sections that start on the same PDF page share the same link, so a label now has to match both the link and the section path.

**The earlier questions are not hurt.** The same 18 of 19 still get a correct section into the 5 passages the model reads, and their answers cite a correct section just as often (36 of 38). The product questions improved the most. The SLC answer now explains that every pixel is a complex number that keeps amplitude and phase, which the wiki passages never told the model.

**Four of the six PDF questions are answered well.** Before this step none of them could cite a correct source. Now the answers explain how thermal noise is subtracted and why the processor does not do it by default, why GRD products have no-value pixels at near range, how errors in the elevation model shift a geocoded image, and the resolution classes of GRD products with their values. That last answer was wrong at first: it gave the pixel spacing as the resolution, because a table's merged cells had shifted its columns. Fixing the table extraction fixed the answer.

**The two misses are the classic failure modes.** "Sigma cero" was translated as "sigma zero", which matches the sections about the *noise-equivalent sigma zero*, while the calibration note speaks of σ0, digital numbers and calibration tables. The question did not use the document's vocabulary. In the other miss, the deramping note's long definition section filled four of the top places, and its short introduction, which explains *why* deramping is needed, ended up 10th.

**New sources bring new problems.** Equations are lost, because their fonts do not say which character each glyph is. Documents written in different years can disagree: the geocoding guide and the wiki give different delivery times for the same orbit files, and one answer mixed the two. Once, the model cited "[9]" after reading "Table 9" in a passage, when there were only 5 passages. The code warns about such citation numbers.

The default retrieval stays as it was: hybrid search with the question translated for the keyword half is still the best of the methods tested at 5 passages. Over the 25 questions it reaches hit@5 0.880 and MRR@5 0.810.

## What it means for the next steps

The system now answers from two kinds of sources and cites both precisely. The eval is fairer for it, but its new questions were all drafted from the text: no question written blind is about a topic that only the PDFs cover, so how a reader who has not seen them fares there is still unmeasured. Two of the problems found here can be measured quickly with the eval: capping how many chunks one section may place in the top 5, and testing whether rephrasing questions toward the documentation's vocabulary helps. The larger extensions that remain are an agent that can search for satellite scenes, an MCP server and OpenSearch.

## Check your understanding

**Why is a PDF harder to ingest than an HTML page, and how does this project recover the sections and cite them?**
HTML marks its headings, paragraphs and tables, but a PDF only places characters on pages, so the structure has to be inferred. Here a heading is a line that starts with a section number, is set in a larger font than the body, and continues the numbering. Repeated margin lines are removed as headers and footers, and lines are grouped into paragraphs by the gaps between them. Since a PDF has no per-heading anchors, a citation gives the section path and a link that opens the file at the page where the section starts (`#page=N`).

**After adding documents, why can't you simply rerun the old eval and compare the numbers?**
The labels only list the sections that existed when they were written. A new document can contain a better answer, and the old labels count retrieving it as a miss. Here that made the old questions appear to drop from hit@5 0.947 to 0.895, even though correct sections were being found. The labels have to be re-checked against the new sources with the same rule first. The answers should also be compared, because the extra sources compete for the same five places in the prompt.
