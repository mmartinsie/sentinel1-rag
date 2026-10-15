# Project entry points.
.PHONY: up down ingest index opensearch-up opensearch-index ask agent mcp mcp-check eval eval-agent clean

# Start Postgres + pgvector and wait until it accepts connections.
up:
	docker compose up -d --wait

# Stop the containers, OpenSearch included. The data volumes are kept.
down:
	docker compose --profile opensearch down

# Download the pages and PDFs in sources.yaml (cached in data/raw/) and split them into data/chunks/.
ingest:
	uv run python -m sentinel1_rag.ingest

# Embed the chunks and store them in Postgres. Unchanged pages are skipped.
index:
	uv run python -m sentinel1_rag.index

# Start OpenSearch, the second search backend (D-034), and wait until it answers. It needs about
# 1.6 GB of RAM and half a minute to start; `make up` does not start it.
opensearch-up:
	docker compose --profile opensearch up -d --wait opensearch

# Copy the chunks and their vectors from Postgres into a new OpenSearch index. Run it again after
# `make index` changes anything; --backend opensearch stops if the copy is out of date.
opensearch-index:
	uv run python -m sentinel1_rag.opensearch

# Answer a question with cited sources: make ask Q="..." [ARGS="--show-context --method vector --top-k 10"]
# The default retrieval is hybrid, with the question translated for the full-text half, and the
# reranker reorders the top 20 to keep 5 (--no-rerank skips it).
# make exports Q to the environment and the shell reads it as "$$Q", so quotes and
# backticks in the question reach Python unchanged (a $ is still expanded by make).
ask:
	uv run python -m sentinel1_rag.ask $(ARGS) "$$Q"

# Answer with the agent, which can search the documentation and the Copernicus catalogue of
# scenes: make agent Q="..." [ARGS="--show-results --today 2026-10-11 --think"]
agent:
	uv run python -m sentinel1_rag.agent $(ARGS) "$$Q"

# Run the MCP server on stdio. An MCP host starts it (see .mcp.json); run by hand, it waits for
# JSON-RPC messages on stdin. The @ keeps make from echoing the command into the protocol channel.
mcp:
	@uv run python -m sentinel1_rag.mcp_server

# Start the MCP server as a host would and check its tools through the protocol, with both the
# initialize handshake and the 2026-07-28 per-request metadata. Needs the network.
mcp-check:
	uv run python -m sentinel1_rag.mcp_check

# Measure retrieval on eval/questions.yaml (hit@k, MRR) and save the run in eval/results/.
# Same defaults and options as ask (--method, --no-translate, --top-k, --lexical-ranking, --backend, --no-rerank);
# name a run with ARGS="--label <name>". The default translates the questions and reranks, so Gemma and the
# reranker are loaded.
eval:
	uv run python -m sentinel1_rag.evaluate $(ARGS)

# Measure the agent's tool calls on eval/agent_questions.yaml and save the run in eval/results/.
# It calls the real tools, so it needs the network: [ARGS="--label <name> --think"]
eval-agent:
	uv run python -m sentinel1_rag.evaluate_agent $(ARGS)

# Stop the containers, delete the data volumes (Postgres and OpenSearch) and the downloaded data.
clean:
	docker compose --profile opensearch down -v
	rm -rf data/
