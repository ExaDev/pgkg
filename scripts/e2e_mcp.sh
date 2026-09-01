#!/usr/bin/env bash
# Stand up a throwaway Postgres, migrate, and run the MCP end-to-end check.
#
# pgserver's bundled server has no pg_trgm on every platform, so this uses
# Docker rather than the embedded path the unit suite prefers.
set -euo pipefail

# The claude_code provider is an optional extra, and `uv sync --extra dev
# --extra mcp` REMOVES it from the venv — so a run that looks like a product
# failure ("no provider") is usually a sync that dropped the SDK.  Say so here
# rather than 200 lines into the transcript.
if [ "${PGKG_LLM_PROVIDER:-}" = "claude_code" ]; then
    if ! uv run --python 3.12 python -c "import claude_agent_sdk" 2>/dev/null; then
        echo "PGKG_LLM_PROVIDER=claude_code needs the claude_agent extra:" >&2
        echo "  uv sync --python 3.12 --extra dev --extra mcp --extra claude_agent" >&2
        exit 1
    fi
fi

NAME=pgkg-e2e-$$
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "== postgres on :$PORT =="
docker run -d --name "$NAME" -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pgkg \
    -p "${PORT}:5432" pgvector/pgvector:pg16 >/dev/null
for _ in $(seq 1 40); do
    docker exec "$NAME" pg_isready -U postgres -q 2>/dev/null && break
    sleep 1
done

export PGKG_DATABASE_URL="postgresql://postgres:postgres@localhost:${PORT}/pgkg"
uv run --python 3.12 pgkg migrate

# Not `exec`: exec replaces this shell, and a replaced shell does not run its
# EXIT trap, so the container outlived every run. Five of them accumulated
# before anyone noticed, each holding a port and competing for the same Docker.
uv run --python 3.12 python scripts/e2e_mcp.py
status=$?
exit "$status"
