-- The embedder registry, and retrieval across two generations at once.
--
-- The embedder will change.  When it does, 4M vectors have to be re-embedded
-- while the system keeps serving, so for a window two model spaces coexist and
-- retrieval has to read both.  Retrofitting that later is a project; the
-- columns and the extra candidate source are cheap now (ADR 0001, D8).
--
-- WHY THE REGISTRY IS PER ORG, NOT PER COLLECTION.  Every live generation costs
-- one query embedding per recall.  Per-collection embedders would mean
-- embedding the query once per collection on every request, which is the wrong
-- shape at any scale.  So `org_embedders` binds an org to its generations and
-- at most one of them is `primary`.
--
-- WHY FUSION NEEDS NO CALIBRATION.  A cosine from bge-m3 and a cosine from its
-- 768-wide successor are not comparable; their RANKS are.  RRF already consumes
-- ranks only, so a second generation is literally just another candidate source
-- into pgkg_fuse() with kind 'vec' — no weight, no normalisation, no threshold.
-- That is also what makes a partial backfill safe: an item the backfill has not
-- reached is absent from the new source and still arrives at its old-generation
-- rank, because rank in one source does not depend on the other.
--
-- WHY STORAGE IS ASYMMETRIC.  The primary generation stays inline on the
-- content row, because the scoping columns the vector search must filter on
-- live there — move it to a side table and the filter is on one relation while
-- the index is on another, which is the filtered-HNSW problem the whole design
-- avoids.  Transitional generations go in side tables, where the extra join is
-- acceptable because the window is temporary.  A side table also solves what
-- the inline column cannot express: one column has one width, so a 768-wide
-- second generation has nowhere else to go.  This is the same reason at most
-- one generation per org may be `primary`.
--
-- WHY THE SIDE TABLE IS NAMED FROM THE GENERATION ID.  D8 sketches
-- `emb_prop_g2`, but generations are identified by UUID and there is no
-- sequence to number them by.  pgkg_generation_table() derives the name from
-- the id instead, so the physical location is a pure function of the
-- generation and needs no column in the registry to record it.  The presence
-- of that table is what tells retrieval whether a generation is inline or
-- transitional, which means creating and dropping the table IS the storage
-- decision — there is no second place for it to disagree with.
--
-- WHY THE STATUS IS READ ON THE READ PATH.  Every step of the D8 cutover
-- protocol rolls back by flipping a status or a role.  That is only true if
-- retrieval reads them, so pgkg_generation_candidates() returns nothing for a
-- generation that is `building` or `retired`, or that the querying org is not
-- bound to.  Rollback is then an UPDATE, never a deploy.
--
-- The dimension of generation 1 is READ from the propositions column rather
-- than written down, preserving the single-source property migration 012
-- established.  This migration contains no embedding-width literal except the
-- 4000-dimension halfvec index ceiling, which is a property of pgvector.


-- 1. The registry.  Reserved id for generation 1 so a column default, a
-- backfill and a test can all name the row without a lookup, exactly as 020
-- does for the reserved orgs.
CREATE FUNCTION pgkg_generation_1() RETURNS UUID
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT '00000000-0000-0000-0000-000000000010'::UUID $$;


CREATE TABLE embedder_generations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT NOT NULL,
    dim          INT  NOT NULL CHECK (dim > 0 AND dim <= 4000),
    storage_type TEXT NOT NULL CHECK (storage_type IN ('halfvec', 'bit+rescore')),
    normalize    BOOLEAN NOT NULL,
    query_prefix TEXT,
    status       TEXT NOT NULL
                 CHECK (status IN ('building', 'live', 'primary',
                                   'retiring', 'retired')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE embedder_generations IS
    'One row per embedding model space. MRL truncation is a generation, not a '
    'special case: bge-m3 and bge-m3@256 produce incomparable vectors and are '
    'registered separately (ADR 0001, D8).';

COMMENT ON COLUMN embedder_generations.dim IS
    'Bounded by the halfvec HNSW ceiling of 4000 dimensions: a wider '
    'generation could be registered but never indexed, and so never retrieved.';

COMMENT ON COLUMN embedder_generations.query_prefix IS
    'Asymmetric models prefix the query but not the document. It travels with '
    'the generation rather than the application config because a cutover '
    'window runs two generations with different prefixes at once.';


CREATE TABLE org_embedders (
    org_id        UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    generation_id UUID NOT NULL REFERENCES embedder_generations(id),
    role          TEXT NOT NULL CHECK (role IN ('primary', 'secondary')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, generation_id)
);

-- One primary per org, because the primary is the generation that owns the
-- single inline embedding column and a column has one width.
CREATE UNIQUE INDEX org_embedders_one_primary_idx
    ON org_embedders (org_id) WHERE role = 'primary';

CREATE INDEX org_embedders_generation_idx ON org_embedders (generation_id);


-- 2. Generation 1: the model in use today, at the width the column declares.
INSERT INTO embedder_generations
    (id, name, dim, storage_type, normalize, query_prefix, status)
SELECT
    pgkg_generation_1(),
    'bge-m3',
    pgkg_embedding_dim('propositions', 'embedding'),
    'halfvec',
    TRUE,
    NULL,
    'primary';

INSERT INTO org_embedders (org_id, generation_id, role)
SELECT o.id, pgkg_generation_1(), 'primary' FROM orgs o;


-- 3. Which generation produced the vector on a content row.  The default is
-- non-volatile, so the backfill of existing rows is a catalog change rather
-- than a table rewrite.
ALTER TABLE propositions
    ADD COLUMN embedder_generation_id UUID NOT NULL
        DEFAULT pgkg_generation_1()
        REFERENCES embedder_generations(id);

ALTER TABLE chunks
    ADD COLUMN embedder_generation_id UUID NOT NULL
        DEFAULT pgkg_generation_1()
        REFERENCES embedder_generations(id);

CREATE INDEX prop_generation_idx ON propositions (embedder_generation_id);
CREATE INDEX chunk_generation_idx ON chunks (embedder_generation_id);


-- 4. What a caller must embed with, for one org.  A dual window is the only
-- reason this is a query and not a constant: the caller embeds the query once
-- per row returned here, concurrently, and passes the results to pgkg_search().
--
-- NORMALIZE is a reserved word, so the output column has to be quoted here and
-- by anything selecting it by name.  D8 names the registry column `normalize`
-- and renaming it in one of the two places it appears would be worse.
CREATE FUNCTION pgkg_live_generations(p_org_id UUID)
RETURNS TABLE (
    generation_id UUID,
    name          TEXT,
    dim           INT,
    storage_type  TEXT,
    "normalize"   BOOLEAN,
    query_prefix  TEXT,
    role          TEXT
)
LANGUAGE SQL STABLE
AS $$
    SELECT g.id, g.name, g.dim, g.storage_type, g.normalize, g.query_prefix,
           oe.role
    FROM org_embedders oe
    JOIN embedder_generations g ON g.id = oe.generation_id
    WHERE oe.org_id = p_org_id
      AND g.status IN ('live', 'primary', 'retiring')
    ORDER BY (oe.role = 'primary') DESC, g.name;
$$;


-- 5. Where a transitional generation's vectors live.  A pure function of the
-- source and the generation, so nothing has to record it.
CREATE FUNCTION pgkg_generation_table(p_source TEXT, p_generation_id UUID)
RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$
    SELECT format('emb_%s_g%s', p_source,
                  left(replace(p_generation_id::TEXT, '-', ''), 8));
$$;


-- 6. Create that table and its index.  Same shape whatever the source: the
-- item's id, the vector at the generation's width, and an HNSW cosine index of
-- its own.  Cascading from the source row means a deleted item cannot leave a
-- vector behind to be retrieved.
CREATE FUNCTION pgkg_create_generation_storage(
    p_generation_id UUID,
    p_source        TEXT DEFAULT 'prop'
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_dim          INT;
    v_source_table TEXT := CASE p_source
                               WHEN 'prop'  THEN 'propositions'
                               WHEN 'chunk' THEN 'chunks'
                           END;
    v_table        TEXT := pgkg_generation_table(p_source, p_generation_id);
BEGIN
    IF v_source_table IS NULL THEN
        RAISE EXCEPTION 'unknown embedding source %', p_source;
    END IF;

    SELECT g.dim INTO v_dim
    FROM embedder_generations g WHERE g.id = p_generation_id;

    IF v_dim IS NULL THEN
        RAISE EXCEPTION 'no such embedder generation %', p_generation_id;
    END IF;

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I ('
        '  item_id UUID PRIMARY KEY REFERENCES %I(id) ON DELETE CASCADE,'
        '  vec halfvec(%s) NOT NULL)',
        v_table, v_source_table, v_dim
    );

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (vec halfvec_cosine_ops)',
        v_table || '_vec_idx', v_table
    );

    RETURN v_table;
END;
$$;


-- 7. A generation, as a candidate source.  Same output signature as every other
-- source, so pgkg_fuse() needs no change and neither does anything downstream;
-- kind is 'vec' because that is what it is — a vector arm in another model
-- space, and two vector arms are what rank fusion is for.
--
-- plpgsql rather than SQL because the relation is chosen at run time.  That
-- costs the planner's inlining, which is the acceptable price D8 names for a
-- transitional generation and does not apply to the primary one.
--
-- Four ways this returns nothing, all of them deliberate: no query vector, an
-- unregistered generation, a status outside the queryable set, or an org not
-- bound to the generation.  A wrong-width query vector is the one case that
-- raises instead, because silently dropping the arm would look exactly like a
-- finished backfill with collapsed recall.
CREATE TYPE pgkg_gen_query AS (
    generation_id UUID,
    q_embedding   halfvec
);

CREATE FUNCTION pgkg_generation_candidates(
    p_generation_id  UUID,
    q_embedding      halfvec,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_dim    INT;
    v_status TEXT;
    v_table  TEXT;
BEGIN
    IF p_generation_id IS NULL OR q_embedding IS NULL THEN
        RETURN;
    END IF;

    SELECT g.dim, g.status INTO v_dim, v_status
    FROM embedder_generations g WHERE g.id = p_generation_id;

    IF v_dim IS NULL OR v_status NOT IN ('live', 'primary', 'retiring') THEN
        RETURN;
    END IF;

    IF p_org_ids IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM org_embedders oe
        WHERE oe.generation_id = p_generation_id
          AND oe.org_id = ANY(p_org_ids)
    ) THEN
        RETURN;
    END IF;

    IF vector_dims(q_embedding) <> v_dim THEN
        RAISE EXCEPTION
            'query vector has % dimensions but generation % declares %',
            vector_dims(q_embedding), p_generation_id, v_dim;
    END IF;

    v_table := pgkg_generation_table('prop', p_generation_id);

    -- No side table means this generation is the inline one, and its vectors
    -- are on the rows that name it.
    IF to_regclass(v_table) IS NULL THEN
        RETURN QUERY
        SELECT
            p.id,
            'vec'::TEXT,
            (ROW_NUMBER() OVER (ORDER BY p.embedding <=> q_embedding))::INT,
            (1.0 - (p.embedding <=> q_embedding))::REAL
        FROM propositions p
        WHERE p.embedding IS NOT NULL
          AND p.embedder_generation_id = p_generation_id
          AND p.namespace = p_namespace
          AND pgkg_temporal_visible(
                p.invalidated_at, p.valid_from, p.valid_to,
                COALESCE(p_valid_at, now())
              )
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
        RETURN;
    END IF;

    RETURN QUERY EXECUTE format($q$
        SELECT
            s.item_id,
            'vec'::TEXT,
            (ROW_NUMBER() OVER (ORDER BY s.vec <=> $1))::INT,
            (1.0 - (s.vec <=> $1))::REAL
        FROM %I s
        JOIN propositions p ON p.id = s.item_id
        WHERE p.namespace = $2
          AND pgkg_temporal_visible(
                p.invalidated_at, p.valid_from, p.valid_to,
                COALESCE($9, now())
              )
          AND ($3 IS NULL OR p.session_id = $3 OR p.session_id IS NULL)
          AND pgkg_visible(
                p.org_id, p.collection_id, p.visibility,
                p.owner_user_id, p.acl_group_id,
                $5, $6, $7, $8
              )
        ORDER BY s.vec <=> $1
        LIMIT $4
    $q$, v_table)
    USING q_embedding, p_namespace, p_session_id, k_initial,
          p_org_ids, p_collection_ids, p_user_id, p_acl_groups, p_valid_at;
END;
$$;


-- 8. pgkg_search(), with the additional generations appended last.  q_embedding
-- stays the primary generation's query vector, so every existing positional
-- caller is untouched; p_gen_queries carries the generations BEYOND that one,
-- and naming the primary in both would double its vote.
--
-- Only the `retrieved` CTE changes.  Fusion, graph expansion, the memory
-- profile, the column list and the ordering are exactly as 021 left them.
DROP FUNCTION pgkg_search(
    TEXT, halfvec, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT,
    UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ
);

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
    p_acl_groups           UUID[]  DEFAULT NULL,
    p_valid_at             TIMESTAMPTZ DEFAULT NULL,
    p_gen_queries          pgkg_gen_query[] DEFAULT NULL
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
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at
            ) b
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at
            ) v
        )
        ||
        ARRAY(
            SELECT (g.item_id, g.kind, g.rank, g.raw_score)::pgkg_candidate
            FROM unnest(COALESCE(p_gen_queries, '{}'::pgkg_gen_query[]))
                 AS gq(generation_id, q_embedding),
                 LATERAL pgkg_generation_candidates(
                     gq.generation_id, gq.q_embedding,
                     p_namespace, p_session_id, k_initial,
                     p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                     p_valid_at
                 ) g
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
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
            p_valid_at
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


-- 9. The registry is org-scoped data, so it carries the same isolation as every
-- other org-scoped table (D3).  embedder_generations is a global catalog of
-- model spaces and deliberately is not.
ALTER TABLE org_embedders ENABLE ROW LEVEL SECURITY;

CREATE POLICY org_embedders_org_isolation ON org_embedders
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());
