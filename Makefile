# Project entry points. The eval target comes in a later step.
.PHONY: up down ingest index ask clean

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

# Answer a question with cited sources: make ask Q="..." [ARGS=--show-context]
# make exports Q to the environment and the shell reads it as "$$Q", so quotes and
# backticks in the question reach Python unchanged (a $ is still expanded by make).
ask:
	uv run python -m sentinel1_rag.ask $(ARGS) "$$Q"

# Stop the containers, delete the database volume and the downloaded data.
clean:
	docker compose down -v
	rm -rf data/
