from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol
from uuid import UUID

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pgkg.migrate import plain_identifier


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


# The two values of the `pgkg.keyword_arm` GUC that 059's dispatcher tells
# apart.  Anything else reaches the database as the policy path.
KeywordArm = Literal["policy", "owner"]
KEYWORD_ARM_GUC = "pgkg.keyword_arm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PGKG_",
        extra="ignore",
    )

    # When None, pgkg auto-starts an embedded Postgres via pgserver (no Docker).
    # Set explicitly to connect to an external Postgres instance.
    database_url: str | None = None
    # The schema pgkg is installed into and queried in (issue #30).  A host
    # application can vendor pgkg into a schema of its own; the migrations
    # qualify every reference in a function, trigger or policy body with it, so
    # those bodies work whatever the caller's search_path holds.
    db_schema: str = "public"
    # Where `pgkg migrate` creates an extension the database does not have yet.
    # One it already has is used wherever it is.
    extension_schema: str = "public"
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
    # Which keyword arm pgkg_bm25_candidates() runs (migration 059).  "policy"
    # reads under row security, and reaches the GIN index only where the `@@`
    # functions are marked LEAKPROOF — which a managed Postgres will not let
    # pgkg do.  "owner" runs a SECURITY DEFINER arm that restates the read
    # policies and reaches the index without the mark.  Opt-in: it is a second
    # statement of the policies, and a deployment that can set the mark should.
    keyword_arm: KeywordArm = "policy"

    @field_validator("db_schema", "extension_schema")
    @classmethod
    def _plain_schema_name(cls, value: str, info: ValidationInfo) -> str:
        return plain_identifier(value, info.field_name or "schema")

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


_LEAKPROOF_STATE_SQL = """
SELECT signature, serves, leakproof, fix FROM pgkg_leakproof_state()
"""

_OWNER_ARM_BYPASSES_SQL = "SELECT pgkg_owner_arm_bypasses_policy()"

# What 059's dispatcher will do on this connection, which is the GUC and not the
# setting: a pool built by something other than make_pool() carries no option.
_ARM_IN_FORCE_SQL = f"""
SELECT CASE WHEN current_setting('{KEYWORD_ARM_GUC}', TRUE) = 'owner'
            THEN 'owner' ELSE 'policy' END
"""


async def gazetteer_match_leakproof(conn: _Queryable) -> dict[str, bool | None]:
    """The gazetteer half of pgkg_leakproof_state(): 047's `%` and `@>`."""
    rows = await conn.fetch(_LEAKPROOF_STATE_SQL)
    return {
        row["signature"]: row["leakproof"]
        for row in rows
        if row["serves"] == "gazetteer"
    }


async def keyword_arm_in_force(conn: _Queryable) -> KeywordArm:
    return await conn.fetchval(_ARM_IN_FORCE_SQL)


async def owner_arm_bypasses_policy(conn: _Queryable) -> bool:
    return bool(await conn.fetchval(_OWNER_ARM_BYPASSES_SQL))


async def row_security_warnings(
    conn: _Queryable, *, keyword_arm: KeywordArm
) -> tuple[str, ...]:
    """What row security is costing this deployment's indexes, in words.

    One warning per operator 043, 046 or 047 could not mark, carrying the
    statement a superuser runs to mark it — except the keyword operators when
    the owner arm is selected and actually escapes the policy, because that is
    the remedy already taken.  And one when the owner arm is selected but its
    owner is under the policy after all, where it buys nothing.  A schema that
    predates 059 has none of the functions this reads, and is told so.
    """
    if not await conn.fetchval(_HAS_059_SQL):
        return (_NOT_MIGRATED,)
    owner_effective = await owner_arm_bypasses_policy(conn)
    owner_remedies_keyword = keyword_arm == "owner" and owner_effective
    unmarked = tuple(
        _unmarked_warning(row)
        for row in await conn.fetch(_LEAKPROOF_STATE_SQL)
        if row["leakproof"] is not True
        and not (row["serves"] == "keyword" and owner_remedies_keyword)
    )
    ineffective = (
        (_OWNER_ARM_UNDER_POLICY,)
        if keyword_arm == "owner" and not owner_effective
        else ()
    )
    return unmarked + ineffective


_HAS_059_SQL = """
SELECT to_regprocedure('pgkg_leakproof_state()') IS NOT NULL
   AND to_regprocedure('pgkg_owner_arm_bypasses_policy()') IS NOT NULL
"""

_NOT_MIGRATED = (
    "this schema predates migration 059, so the LEAKPROOF state of the keyword "
    "and gazetteer operators cannot be read; run `pgkg migrate` first."
)

_OWNER_ARM_UNDER_POLICY = (
    "PGKG_KEYWORD_ARM=owner is selected, but pgkg_bm25_candidates_as_owner() "
    "runs under row security: its owner does not own every table it reads, or "
    "one of them is FORCE ROW LEVEL SECURITY. It stays correct and cannot "
    "reach the GIN index; make the table owner its owner, or drop FORCE."
)


def _unmarked_warning(row: Mapping[str, object]) -> str:
    fix = row["fix"] or "install the function first"
    remedy = (
        " Without one, PGKG_KEYWORD_ARM=owner restores the index (README, "
        "Known limitations)."
        if row["serves"] == "keyword"
        else ""
    )
    return (
        f"{row['signature']} is not LEAKPROOF, so the {row['serves']} arm "
        "cannot use its index under row security and scans the whole tenant. "
        f"As a superuser: {fix}.{remedy}"
    )


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
