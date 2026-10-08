# Project entry points. The Python targets (ingest, index, ask, eval) come in later steps.
.PHONY: up down clean

# Start Postgres + pgvector and wait until it accepts connections.
up:
	docker compose up -d --wait

# Stop the containers. The database volume is kept.
down:
	docker compose down

# Stop the containers and delete the database volume (the schema is re-applied on the next `make up`).
clean:
	docker compose down -v
