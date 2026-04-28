#!/usr/bin/env bash
# dev_db.sh
# Spin up only the Postgres db container for local test runs.
# Uses the same pgvector image and credentials as the full stack.
#
# Usage:
#   ./scripts/dev_db.sh            — start db and wait until ready
#   ./scripts/dev_db.sh stop       — stop and remove the db container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-start}"

if [ "${ACTION}" = "stop" ]; then
    echo "Stopping db container..."
    docker compose -f "${ROOT}/docker-compose.yml" stop db
    docker compose -f "${ROOT}/docker-compose.yml" rm -f db
    echo "Done."
    exit 0
fi

echo "Starting db container..."
docker compose -f "${ROOT}/docker-compose.yml" up -d db

echo "Waiting for Postgres to be ready..."
until docker compose -f "${ROOT}/docker-compose.yml" exec -T db \
    pg_isready -U pgkg -d pgkg -q 2>/dev/null; do
    printf "."
    sleep 1
done

echo ""
echo "Postgres is ready."
echo "  Host    : localhost"
echo "  Port    : ${PGKG_DB_PORT:-5432}"
echo "  Database: pgkg"
echo "  User    : pgkg"
echo "  Password: pgkg"
echo ""
echo "  PGKG_DATABASE_URL=postgresql://pgkg:pgkg@localhost:${PGKG_DB_PORT:-5432}/pgkg"
