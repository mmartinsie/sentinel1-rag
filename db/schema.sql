-- Database schema for the RAG index.
-- Applied once by the Postgres image when the data volume is empty
-- (docker-entrypoint-initdb.d). After editing it: `make clean && make up`.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source page.
CREATE TABLE documents (
    id           serial PRIMARY KEY,
    url          text UNIQUE NOT NULL,
    title        text NOT NULL,
    -- Hash of the page text. Together with index_config it decides
    -- whether a page has to be chunked and embedded again.
    content_hash text NOT NULL,
    -- Chunking settings and embedding model used to build this page's chunks.
    index_config text NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);

-- One row per chunk, always inserted together with its embedding.
CREATE TABLE chunks (
    id          serial PRIMARY KEY,
    document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,           -- position within the page
    section     text NOT NULL,          -- heading path shown in citations
    anchor      text,                   -- heading id for url#anchor; NULL = top of the page
    content     text NOT NULL,
    embedding   vector(1024) NOT NULL,  -- bge-m3
    -- Words of the same text that is embedded, for full-text search (D-019). Postgres fills it
    -- in. STORED because Postgres 18 makes generated columns virtual by default, and a virtual
    -- column cannot be indexed.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', section || E'\n\n' || content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
-- With a few hundred chunks exact search is already fast: this index is here
-- to learn how HNSW works, not for performance.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted index for full-text search (the @@ operator): lexeme -> chunks that contain it.
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
