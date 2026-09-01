-- Explicit tenancy columns, one shared retrieval predicate, and RLS behind it.
--
-- Scope was a string.  `namespace` carried the whole hierarchy a product needs
-- — customer, user, thread — with no referential integrity, no way to express
-- "this user's private facts plus the org's shared corpus", and nothing for the
-- planner to prune on (ADR 0001, D3).  It is replaced by columns:
--
--   org_id        hard isolation boundary, and the future partition key
--   collection_id decay profile, claim scope and ACL mode live on the collection
--   claim_scope   what the claim is ABOUT: world | org | user (D4)
--   visibility    who holds the copy: private | shared
--   owner_user_id set iff visibility = 'private', enforced by CHECK
--   acl_group_id  low-cardinality group, filtered INSIDE the candidate source
--
-- `namespace` is deliberately left in place and still filtered.  Retiring it is
-- an application change — every caller passes a namespace today — and doing it
-- in the same migration that introduces the columns would mean no state in
-- which both are known good.
--
-- WHY THE ARRAY FORM, TODAY.  The predicate is `org_id = ANY($1)` and
-- `collection_id = ANY($2)`, not equality, even though nothing is shared yet.
-- D4's ownership seam makes a subscribed shared collection a second partition
-- rather than a special case; equality now means a second pass over every
-- query later, and this is the hot path.  A single-element array prunes exactly
-- as well as equality.
--
-- ONE PREDICATE, NOT THREE COPIES.  pgkg_visible() holds it once and each
-- candidate source calls it.  It is IMMUTABLE and a single SELECT, so the
-- planner inlines it back into column comparisons: the HNSW index stays usable,
-- which is the point — pgvector's graph walk does not know about the WHERE
-- clause, so a predicate it cannot see degrades top-k into a sequential scan
-- over every tenant.
--
-- THE GRAPH ARM IS RE-FILTERED, NOT TRUSTED.  Entity resolution is org-wide by
-- design (D3, accepted risks), so a shared entity is a bridge between users.
-- Every neighbour proposition reached over that bridge passes the same
-- predicate as the seed.  Without that the graph arm is a permission-laundering
-- machine, and the seed filter is decoration.
--
-- RLS IS DEFENCE IN DEPTH, keyed on `pgkg.org_id` and read through a STABLE
-- wrapper so it can still be an index qualifier.  An unset GUC resolves to the
-- backfill org rather than to "everything": there is no state in which the
-- policy is a no-op.  Postgres exempts table owners and BYPASSRLS roles, so the
-- policies only bite for a role that has neither — hence the `pgkg_app` role
-- below, which is what a deployment should connect as.  Migrations keep running
-- as the owner.
--
-- BACKFILL.  Every existing row maps to one org and one collection, both
-- reserved ids, applied as the column default.  Un-scoped callers keep working
-- unchanged: they write into the default partition and read with a NULL scope
-- array, which the predicate treats as "unrestricted".
--
-- NOT HERE, DELIBERATELY.  `collections` and `collection_subscriptions` (D4)
-- carry decay profile, claim scope and rrf_weight and land with the corpus, so
-- collection_id has no foreign key yet.  `invalidated_at`, the last conjunct of
-- D3's predicate, arrives with the bitemporal columns that replace the
-- `superseded_by` filter.  entities' unique key is still (namespace, name,
-- type) rather than (org_id, ...) because widening it means re-declaring
-- pgkg_link_entity's ON CONFLICT target.


-- 1. Reserved partitions.  Ids are constants rather than lookups so that a
-- column default, an RLS policy and a backfill can all name the same row
-- without a join.  A reserved system org, never org_id IS NULL: NULL breaks the
-- NOT NULL invariant, complicates every policy, and is a trap in a partition
-- key (D4).
CREATE FUNCTION pgkg_system_org() RETURNS UUID
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT '00000000-0000-0000-0000-000000000000'::UUID $$;

CREATE FUNCTION pgkg_default_org() RETURNS UUID
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT '00000000-0000-0000-0000-000000000001'::UUID $$;

CREATE FUNCTION pgkg_default_collection() RETURNS UUID
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT '00000000-0000-0000-0000-000000000002'::UUID $$;


CREATE TABLE orgs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    is_system  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE orgs IS
    'Hard isolation boundary. Nothing a tenant ingests is ever promoted into a '
    'system-org collection: not automatically, not by heuristic, not as a '
    '"contribute back" default. Shared collections are populated by the '
    'operator from operator-licensed sources (ADR 0001, D4).';

INSERT INTO orgs (id, name, is_system) VALUES
    (pgkg_system_org(),  'system',  TRUE),
    (pgkg_default_org(), 'default', FALSE);


CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    external_id TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, external_id)
);


-- Partition placement, as a row rather than a hash of the org id.  Postgres
-- moves rows across partitions on a partition-key UPDATE, so a whale tenant is
-- promoted out of the shared pool by updating this table, not by migrating
-- (D3).  The pool default keeps per-tenant physical isolation available as a
-- sales option at no present cost.
CREATE TABLE tenant_shards (
    org_id     UUID PRIMARY KEY REFERENCES orgs(id) ON DELETE CASCADE,
    shard_key  TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- 2. The per-request scope.  STABLE, not VOLATILE: the planner may then treat
-- the call as a constant for the duration of the statement and use it as an
-- index qualifier or a pruning constraint.  An unset or blank GUC resolves to
-- the backfill org.
CREATE FUNCTION pgkg_current_org() RETURNS UUID
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT COALESCE(
        NULLIF(current_setting('pgkg.org_id', TRUE), '')::UUID,
        pgkg_default_org()
    )
$$;


CREATE FUNCTION pgkg_tenant_shard(p_org_id UUID) RETURNS TEXT
LANGUAGE SQL STABLE
AS $$
    SELECT COALESCE(
        (SELECT ts.shard_key FROM tenant_shards ts WHERE ts.org_id = p_org_id),
        'pool_' || (abs(hashtext(p_org_id::TEXT)) % 64)::TEXT
    )
$$;


-- 3. Scoping columns.  The org default is pgkg_current_org(), not the constant:
-- a writer that set the GUC but does not yet pass org_id explicitly still lands
-- in its own partition, and would otherwise be rejected by its own RLS policy.
ALTER TABLE propositions
    ADD COLUMN org_id        UUID NOT NULL DEFAULT pgkg_current_org()
                                  REFERENCES orgs(id),
    ADD COLUMN collection_id UUID NOT NULL DEFAULT pgkg_default_collection(),
    ADD COLUMN claim_scope   TEXT NOT NULL DEFAULT 'org',
    ADD COLUMN visibility    TEXT NOT NULL DEFAULT 'shared',
    ADD COLUMN owner_user_id UUID REFERENCES users(id),
    ADD COLUMN acl_group_id  UUID,
    ADD CONSTRAINT propositions_claim_scope_check
        CHECK (claim_scope IN ('world', 'org', 'user')),
    ADD CONSTRAINT propositions_visibility_check
        CHECK (visibility IN ('private', 'shared')),
    ADD CONSTRAINT propositions_private_has_owner
        CHECK ((visibility = 'private') = (owner_user_id IS NOT NULL));

ALTER TABLE chunks
    ADD COLUMN org_id        UUID NOT NULL DEFAULT pgkg_current_org()
                                  REFERENCES orgs(id),
    ADD COLUMN collection_id UUID NOT NULL DEFAULT pgkg_default_collection(),
    ADD COLUMN visibility    TEXT NOT NULL DEFAULT 'shared',
    ADD COLUMN owner_user_id UUID REFERENCES users(id),
    ADD COLUMN acl_group_id  UUID,
    ADD CONSTRAINT chunks_visibility_check
        CHECK (visibility IN ('private', 'shared')),
    ADD CONSTRAINT chunks_private_has_owner
        CHECK ((visibility = 'private') = (owner_user_id IS NOT NULL));

ALTER TABLE documents
    ADD COLUMN org_id        UUID NOT NULL DEFAULT pgkg_current_org()
                                  REFERENCES orgs(id),
    ADD COLUMN collection_id UUID NOT NULL DEFAULT pgkg_default_collection();

-- Entities carry the isolation boundary but no visibility: resolution is
-- org-wide, which is where most of the graph's value lives, and the residual
-- exposure of an entity NAME across users inside one org is the accepted risk
-- that the graph re-filter below pays for (D3).
ALTER TABLE entities
    ADD COLUMN org_id UUID NOT NULL DEFAULT pgkg_current_org()
                           REFERENCES orgs(id);


CREATE INDEX prop_tenancy_idx ON propositions (org_id, collection_id)
    WHERE superseded_by IS NULL;
CREATE INDEX prop_owner_idx ON propositions (owner_user_id)
    WHERE owner_user_id IS NOT NULL;
CREATE INDEX chunk_tenancy_idx ON chunks (org_id, collection_id);
CREATE INDEX doc_tenancy_idx ON documents (org_id, collection_id);
CREATE INDEX entities_org_idx ON entities (org_id);


-- 4. The shared retrieval predicate, in one place.  IMMUTABLE and a single
-- SELECT so the planner inlines it: the caller's arrays become ordinary column
-- comparisons and the index paths survive.
--
-- NULL scope arrays mean unrestricted, which is what keeps a pre-tenancy caller
-- working.  A NULL user or group list is NOT unrestricted: `= ANY(NULL)` is
-- NULL, so a private row with no matching caller and an ACL-gated row with no
-- groups named both drop out.  Restriction fails closed; only the two partition
-- keys have an opt-out, and those are chosen by the application, never by a
-- missing argument.
CREATE FUNCTION pgkg_visible(
    p_row_org        UUID,
    p_row_collection UUID,
    p_row_visibility TEXT,
    p_row_owner      UUID,
    p_row_acl_group  UUID,
    p_org_ids        UUID[],
    p_collection_ids UUID[],
    p_user_id        UUID,
    p_acl_groups     UUID[]
) RETURNS BOOLEAN
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$
    SELECT (p_org_ids IS NULL OR p_row_org = ANY(p_org_ids))
       AND (p_collection_ids IS NULL OR p_row_collection = ANY(p_collection_ids))
       AND (p_row_visibility = 'shared' OR p_row_owner = p_user_id)
       AND (p_row_acl_group IS NULL OR p_row_acl_group = ANY(p_acl_groups))
$$;


-- 5. The candidate sources, scoped.  A parameter list is part of a function's
-- identity, so adding one means DROP then CREATE: CREATE OR REPLACE would leave
-- the unscoped overload in place beside the scoped one and make every existing
-- four-argument call ambiguous.  Each body is carried over verbatim from the
-- migration that last defined it, with the predicate added and nothing else
-- touched — no retrieval stage is duplicated, because each still lives in its
-- own function.

DROP FUNCTION pgkg_bm25_candidates(TEXT, TEXT, TEXT, INT);

CREATE FUNCTION pgkg_bm25_candidates(
    q_text           TEXT,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
WITH

-- Stemmed lexemes of the query.  Empty for a NULL, blank or stop-word-only
-- query, which is what short-circuits the whole arm.
query_lexemes AS (
    SELECT trim(BOTH '''' FROM t.lexeme) AS lexeme
    FROM unnest(
        string_to_array(plainto_tsquery('english', q_text)::text, ' & ')
    ) AS t(lexeme)
    WHERE q_text IS NOT NULL
      AND q_text <> ''
      AND trim(BOTH '''' FROM t.lexeme) <> ''
),

-- OR-joined match query: string_agg over zero lexemes yields NULL, and
-- to_tsquery(NULL) is NULL, so an empty query matches nothing rather than
-- raising.
query_or AS (
    SELECT to_tsquery('simple', string_agg(lexeme, ' | ')) AS q
    FROM query_lexemes
),

-- One row, always.  A domain with no statistics row falls back to n_total = 1
-- and avgdl = 1.0, which flattens IDF to a constant and leaves ranking on term
-- frequency alone — degraded but sane, and never an error.
stats AS (
    SELECT
        GREATEST(COALESCE(cs.n_total, 1), 1)::FLOAT8 AS n_total,
        COALESCE(cs.avgdl, 1.0)                      AS avgdl
    FROM (VALUES (1)) AS present(x)
    LEFT JOIN corpus_stats cs
        ON cs.namespace = p_namespace
       AND cs.kind = 'proposition'
),

-- Robertson-Sparck-Jones IDF.  A lexeme with no row has df = 0, the same
-- value the old correlated COUNT(*) returned for an unseen term.
idf AS (
    SELECT
        ql.lexeme,
        LN(
            (s.n_total - COALESCE(ld.df, 0)::FLOAT8 + 0.5)
            / (COALESCE(ld.df, 0)::FLOAT8 + 0.5)
            + 1.0
        ) AS idf_val
    FROM query_lexemes ql
    CROSS JOIN stats s
    LEFT JOIN lexeme_df ld
        ON ld.namespace = p_namespace
       AND ld.kind = 'proposition'
       AND ld.lexeme = ql.lexeme
)

SELECT
    sub.prop_id,
    'kw'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sub.bm25_score DESC))::INT,
    sub.bm25_score::REAL
FROM (
    SELECT
        p.id AS prop_id,
        (
            SELECT COALESCE(SUM(
                i.idf_val
                * (COALESCE(array_length(u.positions, 1), 0)::FLOAT8 * 2.2)
                / (COALESCE(array_length(u.positions, 1), 0)::FLOAT8
                   + 1.2 * (1.0 - 0.75 + 0.75 * p.doc_len::FLOAT8 / s.avgdl))
            ), 0.0)
            FROM unnest(p.tsv) AS u(lexeme, positions, weights)
            JOIN idf i ON i.lexeme = u.lexeme
            CROSS JOIN stats s
        ) AS bm25_score
    FROM propositions p
    CROSS JOIN query_or
    WHERE q_text IS NOT NULL
      AND q_text <> ''
      AND p.namespace = p_namespace
      AND p.superseded_by IS NULL
      AND query_or.q IS NOT NULL
      AND p.tsv @@ query_or.q
      AND (
            p_session_id IS NULL
            OR p.session_id = p_session_id
            OR p.session_id IS NULL
          )
      AND pgkg_visible(
            p.org_id, p.collection_id, p.visibility,
            p.owner_user_id, p.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
) sub
WHERE sub.bm25_score > 0.0
ORDER BY sub.bm25_score DESC
LIMIT k_initial;
$$;


DROP FUNCTION pgkg_vector_candidates(halfvec, TEXT, TEXT, INT);

CREATE FUNCTION pgkg_vector_candidates(
    q_embedding      halfvec,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
SELECT
    p.id,
    'vec'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY p.embedding <=> q_embedding))::INT,
    (1.0 - (p.embedding <=> q_embedding))::REAL
FROM propositions p
WHERE q_embedding IS NOT NULL
  AND p.embedding IS NOT NULL
  AND p.namespace = p_namespace
  AND p.superseded_by IS NULL
  AND (
        p_session_id IS NULL
        OR p.session_id = p_session_id
        OR p.session_id IS NULL
      )
  AND pgkg_visible(
        p.org_id, p.collection_id, p.visibility,
        p.owner_user_id, p.acl_group_id,
        p_org_ids, p_collection_ids, p_user_id, p_acl_groups
      )
ORDER BY p.embedding <=> q_embedding
LIMIT k_initial;
$$;


-- The re-filter D3 makes a hard requirement.  Seed entities are org-wide, so
-- the neighbour propositions are filtered exactly as the seeds were: a walk
-- from a shared entity into another user's private fact ends here.
DROP FUNCTION pgkg_graph_candidates(pgkg_candidate[], TEXT, INT, INT, INT);

CREATE FUNCTION pgkg_graph_candidates(
    p_seeds          pgkg_candidate[],
    p_namespace      TEXT   DEFAULT 'default',
    k_seed_entities  INT    DEFAULT 20,
    k_per_seed       INT    DEFAULT 10,
    k_total          INT    DEFAULT 100,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
WITH

seeds AS (
    SELECT s.item_id AS prop_id, s.raw_score AS score
    FROM unnest(p_seeds) AS s(item_id, kind, cand_rank, raw_score)
),

-- Entities named by the seed propositions, best-scoring first.
seed_entities AS (
    SELECT entity_id
    FROM (
        SELECT entity_id, MAX(score) AS best_score
        FROM (
            SELECT p.subject_id AS entity_id, s.score
            FROM seeds s
            JOIN propositions p ON p.id = s.prop_id
            WHERE p.subject_id IS NOT NULL

            UNION ALL

            SELECT p.object_id AS entity_id, s.score
            FROM seeds s
            JOIN propositions p ON p.id = s.prop_id
            WHERE p.object_id IS NOT NULL
        ) combined
        GROUP BY entity_id
    ) deduped
    ORDER BY best_score DESC
    LIMIT k_seed_entities
),

-- Neighbours of each seed entity, numbered within that entity so the cap
-- below is per seed.
per_seed AS (
    SELECT
        np.id AS prop_id,
        ROW_NUMBER() OVER (
            PARTITION BY se.entity_id
            ORDER BY COALESCE(e.weight, 0.0) DESC, np.id
        ) AS seed_rank
    FROM seed_entities se
    JOIN edges e
      ON e.src_entity = se.entity_id
      OR e.dst_entity = se.entity_id
    JOIN propositions np ON np.id = e.proposition_id
    WHERE np.superseded_by IS NULL
      AND np.namespace = p_namespace
      AND NOT EXISTS (SELECT 1 FROM seeds s WHERE s.prop_id = np.id)
      AND pgkg_visible(
            np.org_id, np.collection_id, np.visibility,
            np.owner_user_id, np.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
),

capped AS (
    SELECT prop_id, MIN(seed_rank) AS best_seed_rank
    FROM per_seed
    WHERE seed_rank <= k_per_seed
    GROUP BY prop_id
)

SELECT
    capped.prop_id,
    'graph'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY capped.best_seed_rank, capped.prop_id))::INT,
    COALESCE((SELECT MIN(score) FROM seeds), 0.0)::REAL
FROM capped
ORDER BY capped.best_seed_rank, capped.prop_id
LIMIT k_total;
$$;


-- 6. pgkg_search(), threading the scope through every arm.  The four new
-- parameters go at the END with defaults, because pgkg/memory.py calls this
-- positionally: a parameter inserted mid-signature can still type-check and
-- silently mis-bind.
DROP FUNCTION pgkg_search(TEXT, halfvec, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT);

CREATE FUNCTION pgkg_search(
    q_text                 TEXT,
    q_embedding            halfvec,
    k_retrieve             INT     DEFAULT 100,
    k_initial              INT     DEFAULT 200,
    p_namespace            TEXT    DEFAULT 'default',
    p_session_id           TEXT    DEFAULT NULL,
    recency_half_life_days REAL    DEFAULT 30.0,
    expand_graph           BOOLEAN DEFAULT TRUE,
    rrf_k                  INT     DEFAULT 60,
    p_org_ids              UUID[]  DEFAULT NULL,
    p_collection_ids       UUID[]  DEFAULT NULL,
    p_user_id              UUID    DEFAULT NULL,
    p_acl_groups           UUID[]  DEFAULT NULL
) RETURNS TABLE (
    proposition_id UUID,
    text           TEXT,
    embedding      vector,
    rrf_score      REAL,
    adjusted_score REAL,
    source_kind    TEXT,
    chunk_id       UUID,
    subject_id     UUID,
    predicate      TEXT,
    object_id      UUID,
    asserted_at    TIMESTAMPTZ
)
LANGUAGE SQL STABLE
AS $$
WITH

retrieved AS (
    SELECT
        ARRAY(
            SELECT (b.item_id, b.kind, b.rank, b.raw_score)::pgkg_candidate
            FROM pgkg_bm25_candidates(
                q_text, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups
            ) b
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups
            ) v
        ) AS candidates
),

-- Seeds for graph expansion are the fused lexical+vector candidates; their
-- fused scores set both the seed-entity ordering and the neighbour floor.
seeds AS (
    SELECT ARRAY(
        SELECT (f.item_id, 'fused'::TEXT, 0, f.fused_score)::pgkg_candidate
        FROM retrieved r, pgkg_fuse(r.candidates, rrf_k) f
    ) AS candidates
),

expanded AS (
    SELECT ARRAY(
        SELECT (g.item_id, g.kind, g.rank, g.raw_score)::pgkg_candidate
        FROM pgkg_graph_candidates(
            CASE WHEN expand_graph THEN s.candidates ELSE '{}'::pgkg_candidate[] END,
            p_namespace, 20, 10, 100,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
        ) g
    ) AS candidates
    FROM seeds s
),

fused AS (
    SELECT f.*
    FROM retrieved r, expanded x, pgkg_fuse(r.candidates || x.candidates, rrf_k) f
),

profiled AS (
    SELECT ap.*
    FROM pgkg_apply_profile(
        ARRAY(
            SELECT (f.item_id, 'fused'::TEXT, 0, f.fused_score)::pgkg_candidate
            FROM fused f
        ),
        recency_half_life_days
    ) ap
)

SELECT
    fused.item_id,
    p.text,
    p.embedding::vector,
    fused.fused_score,
    profiled.adjusted_score,
    CASE
        WHEN fused.in_kw AND fused.in_vec THEN 'both'
        WHEN fused.in_kw                  THEN 'kw'
        WHEN fused.in_vec                 THEN 'vec'
        ELSE                                   'graph'
    END,
    p.chunk_id,
    p.subject_id,
    p.predicate,
    p.object_id,
    p.asserted_at
FROM fused
JOIN propositions p ON p.id = fused.item_id
JOIN profiled ON profiled.item_id = fused.item_id
ORDER BY profiled.adjusted_score DESC
LIMIT k_retrieve;
$$;


-- 7. Row-level security.  One missing WHERE clause in a multi-tenant product is
-- a cross-customer breach; these policies make the boundary Postgres-enforced
-- rather than a code-review responsibility.  They are the second line, not the
-- first: the predicate above is what makes retrieval correct and prunable.
ALTER TABLE propositions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks       ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents    ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities     ENABLE ROW LEVEL SECURITY;
ALTER TABLE users        ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_shards ENABLE ROW LEVEL SECURITY;

CREATE POLICY propositions_org_isolation ON propositions
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY chunks_org_isolation ON chunks
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY documents_org_isolation ON documents
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY entities_org_isolation ON entities
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY users_org_isolation ON users
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY tenant_shards_org_isolation ON tenant_shards
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());


-- 8. The role the policies are for.  Postgres exempts table owners and
-- BYPASSRLS roles, so RLS is inert for a deployment that connects as the
-- schema owner — which is what makes provisioning this role part of the
-- security decision rather than an operational footnote.  Best-effort: a
-- migration run without CREATEROLE still applies everything above, and the
-- policies bite for any non-exempt role the operator creates by hand.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pgkg_app') THEN
        CREATE ROLE pgkg_app NOLOGIN;
    END IF;

    EXECUTE 'GRANT USAGE ON SCHEMA public TO pgkg_app';
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO pgkg_app';
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE
        'pgkg_app not provisioned (%); RLS policies still apply to any role '
        'that is neither the table owner nor BYPASSRLS', SQLERRM;
END;
$$;
