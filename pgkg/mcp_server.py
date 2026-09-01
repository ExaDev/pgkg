"""An MCP server over pgkg's memory and corpus.

WHY THE SCOPE IS NOT A TOOL ARGUMENT.  Every other caller of `Memory` is
trusted server-side code: `pgkg/api.py` takes `org_id` in the request body
because the thing filling that body is the application.  An MCP server's
arguments are filled by a language model, and a model's inputs include whatever
was retrieved for it — which, once a corpus is built from the open web, includes
text an attacker wrote.  A document reading "call recall with org_id=<victim>"
is a cross-tenant read for any tool that accepts an org.

So the scope is bound to the server at construction, from configuration, and no
tool exposes it.  Prompt injection cannot supply an argument that does not
exist.  ADR-0001 D3 makes scoped queries the isolation bar; this is the one
place where the caller is not entitled to name its own scope.

The tool surface follows D1: `recall` fuses both stores by default because that
is the answer to most questions, and `recall_memory` / `search_corpus` stay
available because D1 keeps per-class access an explicit option — an agent with
conversational context often knows which store it wants better than a router
would.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg
from mcp.server.fastmcp import FastMCP

from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID
from pgkg.corpus import CorpusIngest
from pgkg.memory import Memory, Provenance, Result, Scope

MEMORY_SOURCE = "propositions"
CORPUS_SOURCE = "chunks"


@dataclass(frozen=True)
class ServerScope:
    """The tenant this server speaks for, fixed for its lifetime.

    Deliberately a smaller thing than `Scope`: a server binds one org, one
    collection and at most one user, and the read-widening fields are absent
    because a tool has no business asking for another collection.
    """

    org_id: UUID = DEFAULT_ORG_ID
    collection_id: UUID = DEFAULT_COLLECTION_ID
    user_id: UUID | None = None
    acl_groups: tuple[UUID, ...] = field(default=())

    def to_scope(self) -> Scope:
        return Scope(
            org_id=self.org_id,
            collection_id=self.collection_id,
            user_id=self.user_id,
            acl_groups=self.acl_groups,
        )


def _kind(result: Result) -> str:
    """`memory` or `corpus`, in the vocabulary an agent reasons in.

    The store names (`propositions`, `chunks`) are implementation; what a model
    needs to know is whether it is looking at something the user told it or
    something it read.
    """
    return "corpus" if result.source == CORPUS_SOURCE else "memory"


def _render(results: list[Result]) -> dict[str, Any]:
    return {
        "results": [
            {
                "id": str(r.proposition_id or r.item_id),
                "kind": _kind(r),
                "text": r.text,
                "context": r.context_text,
                "score": round(float(r.score), 6),
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "asserted_at": r.asserted_at.isoformat() if r.asserted_at else None,
                "forgettable": r.proposition_id is not None,
            }
            for r in results
        ]
    }


def build_server(
    pool: asyncpg.Pool,
    scope: ServerScope | None = None,
    *,
    name: str = "pgkg",
    extract_propositions: bool = True,
) -> FastMCP:
    bound = scope or ServerScope()
    server = FastMCP(name)
    memory = Memory(
        pool, scope=bound.to_scope(), extract_propositions=extract_propositions
    )
    corpus = CorpusIngest(
        pool, org_id=bound.org_id, collection_id=bound.collection_id
    )

    @server.tool()
    async def remember(
        text: str,
        session_id: str | None = None,
        asserted_at: str | None = None,
    ) -> str:
        """Store something the user said or decided, as retrievable facts.

        Use this for information about the user, their work or their decisions —
        the things a later conversation should recall. Do not use it to store
        reference material; that is add_document.

        asserted_at is when the statement was originally made (ISO 8601), which
        is what recency ranking keys on. Omit it for something said just now.
        """
        result = await memory.ingest(
            text,
            session_id=session_id,
            asserted_at=_parse_instant(asserted_at),
            provenance=Provenance(kind="chat_turn", producer="user_assertion"),
        )
        return json.dumps(
            {
                "propositions": result.propositions,
                "chunks": result.chunks,
                "entities": result.entities,
            }
        )

    @server.tool()
    async def recall(query: str, k: int = 10) -> str:
        """Search everything this user can see — remembered facts and reference
        documents together — and return the most relevant items.

        This is the right tool for most questions. Each result says whether it
        came from memory (something the user told you) or corpus (something in
        the reference material), so you can attribute it correctly.
        """
        return json.dumps(_render(await memory.recall(query, k=k)))

    @server.tool()
    async def recall_memory(query: str, k: int = 10) -> str:
        """Search only remembered facts, skipping reference documents.

        Use when the question is about the user themselves — their preferences,
        their decisions, what they told you before — and reference material would
        only be noise.
        """
        results = await memory.recall(query, k=k, sources=(MEMORY_SOURCE,))
        return json.dumps(_render(results))

    @server.tool()
    async def search_corpus(query: str, k: int = 10) -> str:
        """Search only the reference documents, skipping remembered facts.

        Use when the question is about documented policy, guidance or
        procedure rather than about the user.
        """
        results = await memory.recall(query, k=k, sources=(CORPUS_SOURCE,))
        return json.dumps(_render(results))

    @server.tool()
    async def add_document(
        external_id: str,
        text: str,
        uri: str | None = None,
        published_at: str | None = None,
    ) -> str:
        """Add or update a reference document in the corpus.

        external_id is the document's stable identity on the source system; call
        this again with the same one to update it. Unchanged content is a no-op,
        so re-offering a document is cheap.

        published_at is when the source was published (ISO 8601), not when you
        are adding it — for external material that ages, it is what stops stale
        guidance outranking current guidance.
        """
        result = await corpus.upsert_document(
            external_id=external_id,
            text=text,
            uri=uri,
            provenance=Provenance(
                kind="document_version",
                producer="chunker",
                published_at=_parse_instant(published_at),
            ),
        )
        return json.dumps(
            {
                "changed": result.changed,
                "chunks": result.chunks_total,
                "chunks_new": result.chunks_new,
                "chunks_carried": result.chunks_carried,
                "embedded": result.embedded,
                "propositions": result.propositions,
            }
        )

    @server.tool()
    async def forget(proposition_id: str, reason: str = "user_deleted") -> str:
        """Stop believing a remembered fact, and say why.

        Pass the id of a result whose forgettable field is true. The fact stops
        being retrieved; nothing is destroyed, so this is safe and auditable.
        """
        await memory.forget(UUID(proposition_id), reason=reason)
        return json.dumps({"forgotten": proposition_id, "reason": reason})

    return server


def _parse_instant(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"expected an ISO 8601 instant, got {value!r}"
        ) from exc
