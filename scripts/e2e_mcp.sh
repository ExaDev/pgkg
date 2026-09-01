#!/usr/bin/env bash
# Stand up a throwaway Postgres, migrate, and run the MCP end-to-end check.
#
# pgserver's bundled server has no pg_trgm on every platform, so this uses
# Docker rather than the embedded path the unit suite prefers.
set -euo pipefail

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
exec uv run --python 3.12 python scripts/e2e_mcp.py
