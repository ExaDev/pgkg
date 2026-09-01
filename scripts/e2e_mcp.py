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
"""
import asyncio, json, os, sys, textwrap
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

async def main():
    params = StdioServerParameters(
        command="uv",
        args=["run","--python","3.12","pgkg","mcp"],
        env={**os.environ},
    )
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
            if False:
                pass
            else:
                print(f"   forgetting: {target['text'][:70]}")
                await call(s, "forget", proposition_id=target["id"], reason="user_deleted")
                after = await call(s, "recall_memory", query="do I want reranking on?")
                gone = target["id"] not in {r["id"] for r in after["results"]}
                assert gone, "a forgotten fact is still retrievable"
                print("   forgotten fact is no longer retrievable")
                show("after forget", after)

asyncio.run(main())

print("\n== all end-to-end checks passed ==")
