# Project entry points. The remaining Python targets (ask, eval) come in later steps.
.PHONY: up down ingest index clean

# Start Postgres + pgvector and wait until it accepts connections.
up:
	docker compose up -d --wait

# Stop the containers. The database volume is kept.
down:
	docker compose down

# Download the pages in sources.yaml (cached in data/raw/) and split them into data/chunks/.
ingest:
	uv run python -m sentinel1_rag.ingest

# Embed the chunks and store them in Postgres. Unchanged pages are skipped.
index:
	uv run python -m sentinel1_rag.index

# Stop the containers, delete the database volume and the downloaded data.
clean:
	docker compose down -v
	rm -rf data/
