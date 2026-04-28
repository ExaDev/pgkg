.PHONY: up down logs psql migrate test smoke line-count clean bench-locomo bench-longmemeval bench-smoke bench-mem0-stack bench-zep-stack bench-openrouter-free

# Bring the stack up (build first), detached
up:
	docker compose up --build -d

# Tear the stack down
down:
	docker compose down

# Stream app logs
logs:
	docker compose logs -f app

# Open a psql session in the db container
psql:
	docker compose exec db psql -U pgkg -d pgkg

# Run database migrations inside the running app container
migrate:
	docker compose exec app pgkg migrate

# Run the test suite locally
test:
	uv run pytest -q

# Smoke test: health check + memorize + recall
smoke:
	@echo "--- health ---"
	@curl -sf http://localhost:${PGKG_APP_PORT:-8000}/health | python3 -m json.tool || (echo "FAIL: /health unreachable" && exit 1)
	@echo ""
	@echo "--- memorize ---"
	@MEMO_ID=$$(curl -sf -X POST http://localhost:${PGKG_APP_PORT:-8000}/memorize \
		-H 'Content-Type: application/json' \
		-d '{"content": "pgkg smoke test memory", "tags": ["smoke"]}' | python3 -m json.tool); \
	echo "$$MEMO_ID"; \
	echo ""; \
	echo "--- recall ---"; \
	curl -sf -X POST http://localhost:${PGKG_APP_PORT:-8000}/recall \
		-H 'Content-Type: application/json' \
		-d '{"query": "smoke test memory", "top_k": 3}' | python3 -m json.tool || (echo "FAIL: recall failed" && exit 1)
	@echo ""
	@echo "Smoke test passed."

# Print line counts: Python vs SQL
line-count:
	@bash scripts/line_count.sh

# Destroy everything including volumes, caches, and local artifacts
clean:
	docker compose down -v
	rm -rf .venv .pytest_cache htmlcov .coverage

# Benchmarks
bench-locomo:
	uv run python -m bench.locomo --limit 2

bench-longmemeval:
	uv run python -m bench.longmemeval --variant longmemeval_s --limit 5

bench-smoke:
	PGKG_OFFLINE_EXTRACT=1 uv run python -m bench.locomo --dry-run --limit 1
	PGKG_OFFLINE_EXTRACT=1 uv run python -m bench.longmemeval --variant longmemeval_s --dry-run --limit 1

# Stack preset benchmarks
# Each target warns the user about cost and requires confirmation unless CI=1.

bench-mem0-stack:
	@if [ "$$CI" != "1" ]; then \
		echo "WARNING: This benchmark will call OpenAI APIs and spend real money."; \
		echo "Make sure OPENAI_API_KEY is set and you accept the cost."; \
		read -p "Continue? [y/N] " ans; \
		[ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || (echo "Aborted." && exit 1); \
	fi
	set -a; source .env.bench-mem0-stack; set +a; \
		uv run python -m bench.locomo && \
		uv run python -m bench.longmemeval --variant longmemeval_s

bench-zep-stack:
	@if [ "$$CI" != "1" ]; then \
		echo "WARNING: This benchmark uses gpt-4o and is significantly more expensive."; \
		echo "Make sure OPENAI_API_KEY is set and you accept the cost."; \
		read -p "Continue? [y/N] " ans; \
		[ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || (echo "Aborted." && exit 1); \
	fi
	set -a; source .env.bench-zep-stack; set +a; \
		uv run python -m bench.locomo && \
		uv run python -m bench.longmemeval --variant longmemeval_s

# NOTE: local-claude runs the app on the HOST (not inside Docker) because the
# claude-agent-sdk shells out to the local 'claude' binary on your PATH.
# The db container runs as normal; only the app must stay on the host.
local-claude: ## Run pgkg locally with Claude Code subscription bridge (Pro/Max users)
	@echo "Local experimentation mode — uses your claude CLI subscription."
	@echo "Requires: claude CLI installed and logged in. Run 'claude' once first."
	@command -v claude >/dev/null 2>&1 || { echo "claude CLI not found. Install from https://claude.com/claude-code"; exit 1; }
	set -a; source .env.local-claude; set +a; \
	  ./scripts/dev_db.sh && \
	  uv run pgkg migrate && \
	  uv run pgkg serve

# Ablation baseline: pure chunk RAG (no LLM extraction). Same stack as bench-mem0-stack
# but with --chunks-only on both bench commands. Quantifies the value of proposition extraction.
bench-mem0-stack-chunks:
	@if [ "$$CI" != "1" ]; then \
		echo "WARNING: This benchmark will call OpenAI APIs and spend real money."; \
		echo "Make sure OPENAI_API_KEY is set and you accept the cost."; \
		read -p "Continue? [y/N] " ans; \
		[ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || (echo "Aborted." && exit 1); \
	fi
	set -a; source .env.bench-mem0-stack; set +a; \
		uv run python -m bench.locomo --chunks-only && \
		uv run python -m bench.longmemeval --variant longmemeval_s --chunks-only

# Zero-LLM local mode: chunks-only ingest with hybrid retrieval. No API key, no claude CLI.
local-chunks: ## Run pgkg locally in chunks-only mode (zero LLM at ingest)
	@echo "Local chunks-only mode — pure hybrid RAG. No LLM, no API key, no claude CLI required."
	set -a; source .env.local-chunks; set +a; \
	  ./scripts/dev_db.sh && \
	  uv run pgkg migrate && \
	  uv run pgkg serve

bench-openrouter-free:
	@if [ "$$CI" != "1" ]; then \
		echo "WARNING: Free-tier OpenRouter is rate-limited. Use --limit 5 or expect throttling."; \
		echo "Make sure OPENAI_API_KEY=sk-or-v1-... (OpenRouter key) is set."; \
		read -p "Continue? [y/N] " ans; \
		[ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || (echo "Aborted." && exit 1); \
	fi
	set -a; source .env.bench-openrouter-free; set +a; \
		uv run python -m bench.locomo --limit 5 && \
		uv run python -m bench.longmemeval --variant longmemeval_s --limit 5
