#!/usr/bin/env python3
"""Drive a real `pgkg mcp` server over stdio, the way an agent client does.

The measuring instrument for changes to retrieval, ingest or the MCP surface.
The unit suite covers every layer in isolation; this covers the one thing it
cannot — that the layers still compose when a real model embeds real text into
a real Postgres. Every defect it has found so far was invisible to the suite:
an extraction prompt that read a conversational passage as a request to it, a
provider handed another provider's model id, and a gazetteer nothing calls.

Usage:
    scripts/e2e_mcp.sh                 # starts its own throwaway Postgres
    PGKG_DATABASE_URL=... scripts/e2e_mcp.py   # against an existing one

Exits non-zero on any assertion, so it is usable as a gate.

Sections 6-9 reach past the MCP surface deliberately, and each says why: the
mention sweep is a cron command rather than a tool, a collection is created by
the operator's HTTP surface, `GET /health` is what a monitor watches, and
"both rows carry a vector" is a fact with no product surface at all.  Those are
the seams where the layers meet, which is what this file is for.
"""
import asyncio, contextlib, json, os, socket, subprocess, textwrap
from pathlib import Path
from uuid import UUID

import asyncpg
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID

REPO = Path(__file__).resolve().parent.parent
UV = ["uv", "run", "--python", "3.12"]

SCOPE_ARGS = {"org_id","collection_id","user_id","acl_groups","namespace","include_system_org"}

FACTS = [
    "I'm Will, I run engineering at ExaDev. I prefer terse code review comments over long ones.",
    "We decided on 2026-08-20 that pgkg would keep everything in Postgres rather than adding Neo4j.",
    "The Helios migration is blocked on the Acme renewal being signed.",
    "I always want the cross-encoder reranker enabled for corpus queries, even though it costs latency.",
]

HANDBOOK = textwrap.dedent("""\
    # Engineering handbook: code review

    ## Review turnaround
    A pull request should receive its first review within one working day. If the
    author has not had a review in two days they should escalate in #eng-help.

    ## Approval requirements
    Changes to migrations require two approvals, one of which must be from the
    platform team. Everything else needs one approval.

    ## Style of feedback
    Reviewers should state severity explicitly. Blocking comments must say what
    would need to change for the block to lift.
    """)

VENDOR = textwrap.dedent("""\
    # pgvector 0.8 release notes

    ## Iterative index scans
    HNSW and IVFFlat now support iterative scans, controlled by hnsw.iterative_scan.
    A filtered query that previously under-returned rows will keep scanning until it
    has enough surviving rows.

    ## Halfvec
    halfvec stores 16-bit floats and can be indexed with HNSW up to 4000 dimensions,
    against 2000 for vector.
    """)

# The document D2 is about: it defines the entities a chat fact named, and it
# shares not one content word with the question below.  "Acme renewal" is
# deliberately absent from it — the query names the renewal, so a document
# containing that phrase would be a keyword hit and would arrive on its own
# merits, which is the one thing this section must not allow.
ARCHITECTURE = textwrap.dedent("""\
    # Helios architecture note

    ## Shape
    The Helios migration moves the ledger from one writer to a partitioned one.
    Each partition owns a range of tenants and replays its own journal.

    ## Cutover
    Draining a partition takes thirty seconds, so a two-minute window covers a
    region. The Helios migration keeps its journal for fourteen days.
    """)

# Worded to match the CHAT fact and nothing in the document above: after
# stemming and stopwords it is {block, acme, renew, sign}, none of which the
# note contains.  D2's claim is that the note is retrieved anyway.
D2_QUERY = "what is blocked on the Acme renewal being signed?"

# Why the pool is narrowed to three candidates per arm rather than left at the
# product's default 200.
#
# A graph candidate contributes `w_graph * MIN(seed fused score)` (migration
# 010's pgkg_fuse, 043's "neighbour floor"), so it is by construction the
# lowest-scoring item in the pool, and `pgkg_apply_quotas` keeps the top
# `k_rerank * corpus_fraction` passages — 38 at the product's defaults.  So the
# mention edge only ever shows on a corpus small enough for the passage to
# survive that cut, and a corpus that small puts the passage inside the arms'
# own candidate list, where it is a seed and 043 excludes seeds from expansion.
# Narrowing k_initial reproduces at three candidates the geometry a real corpus
# has at two hundred: the passage is outside the arms and inside the budget.
#
# Measured, on 310 passages in one collection, with a chat fact naming an entity
# the document defines: the graph arm emits the document (raw score 0.003861,
# the seed floor) and `recall()` returns it at neither k=5, 10 nor 20 — with the
# mention rows in place and with them deleted, identically. Widening k_rerank
# from 64 to 1000, which is the same as removing the quota, is what makes it
# appear. So what this section asserts is that the sweep, the mention rows and
# the graph arm are wired to each other; that the edge survives the quota and
# reaches an agent is NOT asserted here, because it does not.
#
# No embedding is passed: with q_embedding NULL the vector arm returns nothing,
# the seeds are whatever the keyword arm found, and the assertion does not
# depend on a model's opinion of two sentences.
_GRAPH_ARM_SQL = """
SELECT r.source_kind
FROM pgkg_retrieve(
        q_text => $1,
        q_embedding => NULL,
        k_retrieve => 100,
        k_initial => 3,
        p_org_ids => $2::uuid[],
        p_collection_ids => $3::uuid[]
     ) r
WHERE r.text LIKE $4
"""

_LEAKPROOF_SQL = "SELECT signature, leakproof FROM pgkg_keyword_match_leakproof()"

_TWO_COLLECTIONS_SQL = """
SELECT collection_id, embedding IS NOT NULL AS has_vector
FROM chunks
WHERE text LIKE $1
ORDER BY collection_id
"""


async def call(s, name, **kw):
    r = await s.call_tool(name, kw)
    text = "".join(b.text for b in r.content if b.type == "text")
    if r.isError:
        raise RuntimeError(f"{name} -> {text}")
    try: return json.loads(text)
    except json.JSONDecodeError: return text

def show(label, payload, limit=5):
    print(f"\n  {label}")
    rows = payload["results"][:limit]
    if not rows:
        print("      (nothing returned)")
    for i, r in enumerate(rows, 1):
        spo = ""
        if r.get("subject"):
            spo = f"   [{r['subject']} | {r['predicate']} | {r['object']}]"
        print(f"      {i}. ({r['kind']}, {r['score']:.4f}) {r['text'][:96]}{spo}")

def server_params(*extra):
    """A `pgkg mcp` server, optionally bound to another collection.

    The collection is a launch flag and not a tool argument on purpose
    (pgkg/mcp_server.py), so a second collection means a second server.
    """
    return StdioServerParameters(
        command="uv",
        args=[*UV[1:], "pgkg", "mcp", *extra],
        env={**os.environ},
    )


def maintain(*args):
    """`pgkg maintain`, the cron entry point, as an operator runs it.

    The mention sweep has no MCP tool and should not have one: a maintenance
    run is an operator action over a whole org, not something a model asks for.
    """
    out = subprocess.run([*UV, "pgkg", "maintain", *args],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        raise RuntimeError(f"pgkg maintain {args} failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.splitlines()[-1])["tasks"][0]


async def graph_arm(conn, like):
    """Which arms carried a passage matching `like`, at a narrowed pool."""
    rows = await conn.fetch(
        _GRAPH_ARM_SQL, D2_QUERY, [DEFAULT_ORG_ID], [DEFAULT_COLLECTION_ID], like
    )
    return sorted(r["source_kind"] for r in rows)


@contextlib.asynccontextmanager
async def api_server():
    """`pgkg serve` on a private port, for the two surfaces MCP does not have.

    Started here rather than assumed running: a monitor's view of a deployment
    is only worth asserting on if the deployment is this fresh database.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    proc = subprocess.Popen(
        [*UV, "pgkg", "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO,
    )
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=30.0
        ) as client:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError(f"pgkg serve exited with {proc.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("pgkg serve never answered /health")
            yield client
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=20)


async def main():
    db_url = os.environ.get("PGKG_DATABASE_URL")
    assert db_url, "PGKG_DATABASE_URL is required: sections 6-9 read the database"
    # Said here rather than discovered in section 6: the offline extractor names
    # every entity "?", so the gazetteer has nothing to match and a sweep that
    # correctly adds no edge reads as a product defect.  This file's premise is
    # a real model.
    assert os.environ.get("PGKG_OFFLINE_EXTRACT", "0") in ("0", "", "false"), (
        "PGKG_OFFLINE_EXTRACT is set: dummy extraction produces no entity names, "
        "so sections 6 and 7 cannot mean anything. Run with "
        "PGKG_OFFLINE_EXTRACT=0 and a real provider."
    )
    conn = await asyncpg.connect(db_url)
    params = server_params()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            tools = (await s.list_tools()).tools
            print(f"\n== 1. handshake: {len(tools)} tools ==")
            for t in tools:
                props = sorted(t.inputSchema.get("properties", {}))
                leak = SCOPE_ARGS & set(props)
                print(f"   {t.name:16} args={props} {'LEAKS ' + str(leak) if leak else ''}")
            assert not any(SCOPE_ARGS & set(t.inputSchema.get("properties", {})) for t in tools)

            print("\n== 2. remember 4 facts ==")
            for f in FACTS:
                out = await call(s, "remember", text=f)
                print(f"   props={out['propositions']} chunks={out['chunks']} ents={out['entities']}  {f[:56]}...")

            print("\n== 3. add 2 documents ==")
            for eid, body, pub in [("handbook://code-review", HANDBOOK, None),
                                   ("https://pgvector.org/0.8", VENDOR, "2024-10-30T00:00:00+00:00")]:
                out = await call(s, "add_document", external_id=eid, text=body, published_at=pub)
                print(f"   {eid}: changed={out['changed']} chunks={out['chunks']} new={out['chunks_new']} embedded={out['embedded']}")
            out = await call(s, "add_document", external_id="handbook://code-review", text=HANDBOOK)
            print(f"   re-offer unchanged: changed={out['changed']} embedded={out['embedded']}  <- should be False/0")

            print("\n== 4. retrieval ==")
            show("recall 'how fast should a PR be reviewed?'", await call(s, "recall", query="how fast should a PR be reviewed?"))
            show("recall 'what did we decide about the database?'", await call(s, "recall", query="what did we decide about the database?"))
            show("recall 'do I want reranking on?'", await call(s, "recall", query="do I want reranking on?"))
            show("recall 'what is blocking Helios?'", await call(s, "recall", query="what is blocking Helios?"))
            show("search_corpus 'migration approvals'", await call(s, "search_corpus", query="how many approvals for a migration?"))
            show("recall_memory 'migration approvals'", await call(s, "recall_memory", query="how many approvals for a migration?"))
            show("search_corpus 'halfvec dimensions'", await call(s, "search_corpus", query="how many dimensions can halfvec index?"))

            print("\n== 5. forget ==")
            found = await call(s, "recall_memory", query="do I want reranking on?")
            target = next((r for r in found["results"] if r["forgettable"]), None)
            assert target is not None, "no forgettable result: forget is unreachable"
            print(f"   forgetting: {target['text'][:70]}")
            await call(s, "forget", proposition_id=target["id"], reason="user_deleted")
            after = await call(s, "recall_memory", query="do I want reranking on?")
            gone = target["id"] not in {r["id"] for r in after["results"]}
            assert gone, "a forgotten fact is still retrievable"
            print("   forgotten fact is no longer retrievable")
            show("after forget", after)

            # The edge D2 calls the reason the corpus and the graph are worth
            # joining: a chat fact names an entity, a document defines it, and
            # the gazetteer is the only thing that connects them.  Nothing in
            # the product called the gazetteer until `pgkg maintain` existed
            # (issue #19), so this is the first section that can exist at all.
            print("\n== 6. the mention edge: chat fact -> entity -> document ==")
            out = await call(s, "add_document",
                             external_id="wiki://helios-architecture", text=ARCHITECTURE)
            print(f"   wiki://helios-architecture: chunks={out['chunks']} "
                  f"embedded={out['embedded']}")

            before = await graph_arm(conn, "# Helios architecture note%")
            print(f"   arms carrying the note before the sweep: {before or 'none'}")
            assert not before, (
                "the note reached retrieval before any mention edge existed, so "
                f"this section cannot attribute anything to the sweep: {before}"
            )

            swept = maintain("--task", "mentions")
            print(f"   pgkg maintain --task mentions -> {json.dumps(swept)}")
            assert swept["ran"], "the mention sweep declined to run"
            assert swept["changed"] > 0, (
                "the sweep added no mention edges, so entity_mentions is still "
                "empty and the graph has no corpus arm"
            )

            # A sweep is a cron job, so "did it stop when it was done" is as
            # much of its contract as "did it do anything": both watermarks
            # advance or the next tick redoes the same batch for ever.
            again = maintain("--task", "mentions")
            print(f"   second run -> {json.dumps(again)}")
            assert again["ran"] and again["changed"] == 0, (
                f"a second sweep was not a no-op, so a watermark is not advancing: {again}"
            )

            after_sweep = await graph_arm(conn, "# Helios architecture note%")
            print(f"   arms carrying the note after the sweep:  {after_sweep or 'none'}")
            assert "graph" in after_sweep, (
                "the mention edge exists but retrieval does not carry it: a "
                "question sharing no word with the note failed to reach it"
            )
            print("   the note arrived on the graph arm alone: D2's edge is wired")

            # Said out loud on every run, because the section above stops one
            # step short of the promise and a green run should not read as if
            # it did not.  What an agent asking `recall` sees is measured in the
            # comment on _GRAPH_ARM_SQL: at the product's own pool and quota,
            # the passage the mention edge added is cut before the reranker.
            print("   KNOWN GAP: at the default k_initial/k_rerank a graph-only")
            print("   passage scores below every keyword and vector candidate and")
            print("   the corpus quota drops it, so this edge does not yet change")
            print("   what recall() returns. Mechanism asserted, payoff is not.")

            # #17 moved this predicate from similarity() to `%` with the
            # threshold pinned in the function.  The claim is that the candidate
            # set is unchanged, and the observable is here: 'ExaDev' comes from
            # fact 1, 'ExaDev Ltd' from this one, they are 0.64 apart on
            # trigrams, and one entity is what 0.6 has always meant.
            print("\n== 7. entity dedup on a near-identical name ==")
            out = await call(s, "remember", text="ExaDev Ltd was incorporated in 2019.")
            print(f"   remember: props={out['propositions']} ents={out['entities']}")
            named = [r["name"] for r in await conn.fetch(
                "SELECT name FROM entities WHERE name ILIKE 'exadev%' ORDER BY name"
            )]
            print(f"   entities named ExaDev-something: {named}")
            assert len(named) == 1, (
                "'ExaDev' and 'ExaDev Ltd' resolved to separate entities, so the "
                f"fuzzy stage is unreachable or its threshold moved: {named}"
            )

    # Two surfaces MCP does not have, on the database the run just built.
    print("\n== 8. the operator's surfaces on a fresh database ==")
    async with api_server() as client:
        health = (await client.get("/health")).json()
        print(f"   GET /health -> {json.dumps(health)}")
        assert health["db"] is True, "health reports no database"

        dim = await conn.fetchval("SELECT pgkg_embedding_dim('propositions', 'embedding')")
        assert health["embedding"]["dim"] == dim, (
            f"health reports width {health['embedding']['dim']}, the vectors are {dim}"
        )
        assert health["embedding"]["generations"], "health reports no live generation"

        # Leakproof cannot be enforced — 043 and 046 degrade to a NOTICE where a
        # managed Postgres refuses the ALTER — so the only thing worth asserting
        # is that the report is TRUE of this database rather than a constant.
        operators = {r["signature"]: r["leakproof"]
                     for r in await conn.fetch(_LEAKPROOF_SQL)}
        assert health["keyword_index"]["operators"] == operators, (
            f"health says {health['keyword_index']['operators']}, the catalog says {operators}"
        )
        assert health["keyword_index"]["leakproof"] == (
            bool(operators) and all(operators.values())
        ), "health's summary disagrees with the operators it just listed"
        print(f"   keyword arms leakproof: {health['keyword_index']['leakproof']} "
              f"(catalog agrees)")

        applied = {r["filename"] for r in
                   await conn.fetch("SELECT filename FROM pgkg_schema_migrations")}
        on_disk = {p.name for p in (REPO / "migrations").glob("*.sql")}
        print(f"   migrations applied: {len(applied)} of {len(on_disk)} on disk")
        assert applied == on_disk, (
            f"the ledger and the directory disagree: {on_disk ^ applied}"
        )

        second = (await client.post("/collections", json={
            "name": f"e2e-second-{os.getpid()}",
            "kind": "corpus",
            "decay_profile": "timeless",
        })).json()["collection_id"]
        print(f"   second collection: {second}")

    # #16 under #18's address: one passage offered to two collections is two
    # rows, because the content address carries the collection — and the defect
    # was that the second row was created and then never vectored, leaving it
    # retrievable by the keyword arm alone, for good.
    print("\n== 9. one passage, two collections ==")
    async with stdio_client(server_params("--collection", second)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            out = await call(s, "add_document",
                             external_id="handbook://code-review", text=HANDBOOK)
            print(f"   the same handbook into the second collection: {json.dumps(out)}")
            assert out["chunks_new"] == 1, (
                f"expected one new row at this collection's address: {out}"
            )
            assert out["embedded"] == 1, (
                f"the new row was created and not vectored: {out}"
            )
            show("search_corpus in the second collection",
                 await call(s, "search_corpus", query="how many approvals for a migration?"))

    rows = await conn.fetch(_TWO_COLLECTIONS_SQL, "# Engineering handbook%")
    for row in rows:
        print(f"   {row['collection_id']} has_vector={row['has_vector']}")
    vectored = {row["collection_id"] for row in rows if row["has_vector"]}
    # Both named addresses, rather than a count: run against a database that
    # already has collections and the count says nothing, while "these two hold
    # it and both hold a vector" says the same thing either way.
    assert {DEFAULT_COLLECTION_ID, UUID(second)} <= vectored, (
        "a collection holds the passage without a vector, so only the keyword "
        f"arm can ever find it: {[(str(r['collection_id']), r['has_vector']) for r in rows]}"
    )

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
    print("\n== all end-to-end checks passed ==")
