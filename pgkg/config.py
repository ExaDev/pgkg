from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict


# The rows migrations 020-022 reserve, as constants rather than lookups: a
# column default, an RLS policy and an application default all have to name the
# same partition, and a round trip to learn a value that cannot change would be
# one per request.  test_api_scoping pins them against the SQL functions.
SYSTEM_ORG_ID = UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_COLLECTION_ID = UUID("00000000-0000-0000-0000-000000000002")
GENERATION_1_ID = UUID("00000000-0000-0000-0000-000000000010")

# The GUC the RLS policies read.  Every connection the application takes sets
# it, which is also what gives the entities.org_id default a value to resolve
# to — pgkg_link_entity() takes no org argument.
ORG_GUC = "pgkg.org_id"


# The model a provider is asked for when the caller named a provider and no
# model.  `llm_model` has to default to something, and defaulting it to an
# OpenAI id is right for the common case — but it meant selecting claude_code
# alone sent that id to the `claude` CLI, which failed with a message about
# logging in.  A provider's default belongs with the provider.
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-haiku-4-5-20251001",
    "claude_code": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PGKG_",
        extra="ignore",
    )

    # When None, pgkg auto-starts an embedded Postgres via pgserver (no Docker).
    # Set explicitly to connect to an external Postgres instance.
    database_url: str | None = None
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # The embedding width is a property of the schema, not of configuration:
    # read it with pgkg_embedding_dim('propositions', 'embedding').  A settings
    # field here would only be able to disagree with the column.
    #
    # Pinned model IDs — dated suffixes ensure reproducible benchmark comparisons.
    llm_model: str = "gpt-4o-mini-2024-07-18"
    llm_provider: Literal["openai", "anthropic", "ollama", "claude_code"] = "openai"
    # When set, overrides llm_model for extraction only.
    # Useful for "extract with one model, answer with another" Mem0-style setups.
    extractor_model: str | None = None
    # Pinned judge model — matches LongMemEval/LoCoMo published evaluation setups.
    judge_model: str = "gpt-4o-2024-08-06"
    judge_provider: str = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Point at OpenRouter (https://openrouter.ai/api/v1) or Groq, etc.
    openai_base_url: str | None = None
    default_namespace: str = "default"
    offline_extract: str = "0"
    # When False, skip LLM proposition extraction entirely; store chunks directly
    # as propositions (NULL subject/predicate/object). Zero LLM cost at ingest.
    extract_propositions: bool = True
    # Informational: the prompt version used for extraction (source of truth is
    # the PROMPT_VERSION constant in ml.py; this field is logged into BenchReport).
    prompt_version: str = "v2"


    @property
    def resolved_extractor_model(self) -> str:
        """The model to extract with, honouring the provider when unasked.

        Precedence: an explicit `extractor_model`, then an explicitly-set
        `llm_model`, then the provider's own default.  The middle step is why
        this reads `model_fields_set` rather than comparing against the default
        value — a caller who deliberately sets the OpenAI id while pointing at
        another provider is doing something unusual, and is entitled to.
        """
        if self.extractor_model:
            return self.extractor_model
        if "llm_model" in self.model_fields_set:
            return self.llm_model
        return PROVIDER_DEFAULT_MODELS.get(self.llm_provider, self.llm_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Alias for external use
MemoryConfig = Settings


class _Queryable(Protocol):
    """The part of an asyncpg connection the registry readers need."""

    async def fetch(self, query: str, *args: object) -> list: ...

    async def fetchval(self, query: str, *args: object) -> object: ...


@dataclass(frozen=True)
class Generation:
    """One embedding model space, as the registry describes it.

    `query_prefix` travels with the generation because a cutover window runs two
    generations with different prefixes at once, so it cannot live in settings.
    """

    generation_id: UUID
    name: str
    dim: int
    storage_type: str
    normalize: bool
    query_prefix: str | None
    role: str


# `normalize` is a reserved word, so the output column of pgkg_live_generations
# has to be quoted wherever it is named.
_LIVE_GENERATIONS_SQL = """
SELECT generation_id, name, dim, storage_type, "normalize", query_prefix, role
FROM pgkg_live_generations($1)
"""


async def live_generations(
    conn: _Queryable, org_id: UUID = DEFAULT_ORG_ID
) -> tuple[Generation, ...]:
    """Every generation this org must embed a query with, primary first."""
    rows = await conn.fetch(_LIVE_GENERATIONS_SQL, org_id)
    return tuple(
        Generation(
            generation_id=row["generation_id"],
            name=row["name"],
            dim=row["dim"],
            storage_type=row["storage_type"],
            normalize=row["normalize"],
            query_prefix=row["query_prefix"],
            role=row["role"],
        )
        for row in rows
    )


# The signatures are the database's list, not this module's: 046 marks them and
# names them in one place, and a copy here could only ever disagree with what
# was marked.
_KEYWORD_LEAKPROOF_SQL = """
SELECT signature, leakproof FROM pgkg_keyword_match_leakproof()
"""


async def keyword_match_leakproof(conn: _Queryable) -> dict[str, bool | None]:
    """Whether each function behind `@@` may be used as an index condition.

    `ALTER FUNCTION ... LEAKPROOF` needs ownership of a built-in, which a
    managed Postgres will not grant, so 043 and 046 degrade to a NOTICE and the
    keyword arms stay correct and lose the GIN index under a role with row
    security.  Nothing can assert the mark; something has to be able to see it.
    """
    rows = await conn.fetch(_KEYWORD_LEAKPROOF_SQL)
    return {row["signature"]: row["leakproof"] for row in rows}


async def embed_dim(conn: _Queryable, org_id: UUID = DEFAULT_ORG_ID) -> int:
    """The width of the org's primary embedding space.

    Read from the registry rather than declared here.  A settings field would
    only be able to disagree with the column it describes, which is what
    `config.embed_dim` did before D8 gave the width an owner.
    """
    dim = await conn.fetchval(
        """
        SELECT g.dim
        FROM org_embedders oe
        JOIN embedder_generations g ON g.id = oe.generation_id
        WHERE oe.org_id = $1 AND oe.role = 'primary'
        """,
        org_id,
    )
    if dim is None:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )
    return dim
