# sentinel1-rag

A small, local and free RAG (retrieval-augmented generation) system that answers technical questions about Sentinel-1 (acquisition modes, GRD and SLC products, processing levels…) using the [Sentinel-1 SentiWiki](https://sentiwiki.copernicus.eu/web/sentinel-1) as its only source. Every answer cites the sections it comes from.

Everything runs locally: Ollama for embeddings and generation, PostgreSQL + pgvector for retrieval.

> **Status:** work in progress. See [docs/PROGRESS.md](docs/PROGRESS.md) for the current step and [docs/DECISIONS.md](docs/DECISIONS.md) for the design decisions. Each finished step has a plain-English write-up in [docs/steps/](docs/steps/).

## Extensions (not implemented)

Ordered by expected value:

1. **Lexical baseline and hybrid search:** Postgres full-text search vs. vectors on the same eval, then a hybrid of both. SAR acronyms (GRD, SLC, IW, EW, ETAD) are where lexical search usually wins.
2. **Technical PDFs** from the Sentinel-1 document library.
3. **Agent** with a `search_scenes` tool over the Copernicus Data Space Ecosystem STAC API, with a hand-written agent loop.
4. **MCP server** exposing `search_docs` and `search_scenes`.
5. **OpenSearch k-NN** instead of pgvector, comparing results.
