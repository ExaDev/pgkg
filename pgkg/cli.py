from __future__ import annotations

import argparse
import asyncio
import json
import sys


def cmd_migrate(args: argparse.Namespace) -> None:
    import pathlib

    import asyncpg

    from pgkg.config import get_settings

    async def _run() -> None:
        dsn = get_settings().database_url
        if dsn is None:
            from pgkg.embedded import get_dsn
            dsn = get_dsn()
        migrations_dir = pathlib.Path(__file__).resolve().parent.parent / "migrations"
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS pgkg_schema_migrations ("
                "  filename TEXT PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
            applied = {
                r["filename"]
                for r in await conn.fetch("SELECT filename FROM pgkg_schema_migrations")
            }
            for migration in sorted(migrations_dir.glob("*.sql")):
                if migration.name in applied:
                    print(f"Skipping {migration.name} (already applied).")
                    continue
                print(f"Applying {migration.name}...")
                async with conn.transaction():
                    await conn.execute(migration.read_text())
                    await conn.execute(
                        "INSERT INTO pgkg_schema_migrations (filename) VALUES ($1)",
                        migration.name,
                    )
            print("All migrations applied.")
        finally:
            await conn.close()

    asyncio.run(_run())


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run("pgkg.api:app", host=args.host, port=args.port, reload=False)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Serve the MCP tools over stdio.

    The scope comes from these flags and nothing else.  A tool that took an org
    argument would be taking it from a language model, whose context includes
    whatever the corpus retrieved for it — so the tenant is fixed here, once,
    by the operator launching the server (ADR-0001 D3; pgkg/mcp_server.py).
    """
    import asyncio

    from pgkg.config import get_settings
    from pgkg.db import make_pool, close_pool
    from pgkg.mcp_server import ServerScope, build_server

    scope = _scope(args)

    async def _serve() -> None:
        pool = await make_pool(get_settings().database_url)
        try:
            server = build_server(
                pool,
                ServerScope(
                    org_id=scope.org_id,
                    collection_id=scope.collection_id,
                    user_id=scope.user_id,
                ),
                extract_propositions=get_settings().extract_propositions,
            )
            await server.run_stdio_async()
        finally:
            await close_pool(pool)

    asyncio.run(_serve())


def _scope(args: argparse.Namespace):
    """The tenant this invocation runs as.

    Absent flags resolve to the reserved default org and collection, which is
    the partition every pre-tenancy row was backfilled into — so an existing
    script keeps reading and writing exactly what it did.
    """
    import uuid

    from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID
    from pgkg.memory import Scope

    return Scope(
        org_id=uuid.UUID(args.org) if getattr(args, "org", None) else DEFAULT_ORG_ID,
        collection_id=(
            uuid.UUID(args.collection)
            if getattr(args, "collection", None)
            else DEFAULT_COLLECTION_ID
        ),
        user_id=uuid.UUID(args.user) if getattr(args, "user", None) else None,
    )


def _add_scope_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--org", default=None, help="Org UUID to act as")
    parser.add_argument("--user", default=None, help="User UUID to act as")
    parser.add_argument("--collection", default=None, help="Collection UUID")


async def run_ingest(args: argparse.Namespace) -> None:
    from pgkg.config import get_settings
    from pgkg.db import pool_from_settings
    from pgkg.memory import Memory

    if args.path == "-":
        text = sys.stdin.read()
        source = "stdin"
    else:
        import pathlib
        p = pathlib.Path(args.path)
        text = p.read_text()
        source = str(p)

    settings = get_settings()
    async with pool_from_settings() as pool:
        mem = Memory(
            pool,
            namespace=settings.default_namespace,
            scope=_scope(args),
            extract_propositions=not args.chunks_only,
        )
        try:
            result = await mem.ingest(text, source=source)
        finally:
            await mem.aclose()
        print(json.dumps({
            "documents": result.documents,
            "chunks": result.chunks,
            "propositions": result.propositions,
            "entities": result.entities,
        }))


def cmd_ingest(args: argparse.Namespace) -> None:
    asyncio.run(run_ingest(args))


async def run_recall(args: argparse.Namespace) -> None:
    from pgkg.config import get_settings
    from pgkg.db import pool_from_settings
    from pgkg.memory import Memory

    settings = get_settings()
    async with pool_from_settings() as pool:
        mem = Memory(pool, namespace=settings.default_namespace, scope=_scope(args))
        # A CLI invocation is a whole process lifetime: without the close, the
        # access counts recall() accumulated in memory never reach the database.
        try:
            results = await mem.recall(args.query, k=args.k)
        finally:
            await mem.aclose()
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


def cmd_recall(args: argparse.Namespace) -> None:
    asyncio.run(run_recall(args))


async def run_worker(args: argparse.Namespace) -> None:
    """Drain the corpus ingest queue outside the request process.

    Corpus ingest is batch (ADR 0001, D7): 600k chunks is a GPU-day, and it
    must not compete with online recall for pool slots.  The two knobs are the
    ones an operator actually reaches for — how many documents may be in
    flight, and how long to wait between them.
    """
    from pgkg.corpus import CorpusIngest
    from pgkg.db import pool_from_settings
    from pgkg.ingest_jobs import IngestWorker

    scope = _scope(args)
    async with pool_from_settings() as pool:
        worker = IngestWorker(
            pool,
            ingest=CorpusIngest(
                pool, org_id=scope.org_id, collection_id=scope.collection_id
            ),
            org_id=scope.org_id if getattr(args, "org", None) else None,
            throttle_seconds=args.throttle,
            slots=args.slots,
        )
        drained = await worker.run(concurrency=args.slots, max_jobs=args.max_jobs)
        print(json.dumps({"jobs": drained}))


def cmd_worker(args: argparse.Namespace) -> None:
    asyncio.run(run_worker(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pgkg", description="pgkg knowledge graph CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate", help="Apply database migrations")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    mcp_parser = subparsers.add_parser(
        "mcp", help="Serve the MCP tools over stdio for an agent client"
    )
    _add_scope_flags(mcp_parser)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or stdin")
    ingest_parser.add_argument("path", help="Path to file or '-' for stdin")
    ingest_parser.add_argument(
        "--chunks-only",
        action="store_true",
        default=False,
        help=(
            "Skip LLM proposition extraction; store chunks directly as propositions "
            "(NULL subject/predicate/object). Zero LLM cost at ingest. "
            "Equivalent to PGKG_EXTRACT_PROPOSITIONS=0."
        ),
    )
    _add_scope_flags(ingest_parser)

    recall_parser = subparsers.add_parser("recall", help="Recall memories matching a query")
    recall_parser.add_argument("query", help="Search query")
    recall_parser.add_argument("--k", type=int, default=10, help="Number of results")
    _add_scope_flags(recall_parser)

    worker_parser = subparsers.add_parser(
        "worker", help="Drain the corpus ingest queue"
    )
    worker_parser.add_argument(
        "--slots",
        type=int,
        default=1,
        help="How many documents may be in flight at once",
    )
    worker_parser.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="Seconds to wait between documents",
    )
    worker_parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Stop after this many documents (default: drain the queue)",
    )
    _add_scope_flags(worker_parser)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "recall":
        cmd_recall(args)
    elif args.command == "worker":
        cmd_worker(args)


if __name__ == "__main__":
    main()
