-- The corpus as the application surface reaches it (ADR 0001, D1, D3, D6).
--
-- 031 built the two-store retriever and nothing in Python called it: recall()
-- still went to pgkg_search(), which is proposition-shaped and can only ever
-- answer half of "what did we agree about the refund policy".  Wiring the
-- application to pgkg_retrieve() needs three things the database did not yet
-- say.
--
-- WHAT A CHUNK HAS TO BE TO COUNT AS A PASSAGE.  Chat ingest writes a chunk per
-- turn so a fact can cite the text it was extracted from.  Those chunks are
-- provenance, not corpus: retrieving them returns the same content twice, once
-- as the fact and once as the passage it came from, and — because a chunk has
-- no belief clock — keeps returning it after the fact has been forgotten.  They
-- also carry no namespace, so a chat chunk would leak across every namespace in
-- its collection.  The rule that separates them from the corpus is already in
-- the schema: a corpus chunk belongs to a document VERSION and to no single
-- document, and a chat chunk belongs to a document and to no version.  A chunk
-- that belongs to neither is a standalone passage and stands on its own, which
-- is what 031 said and stays true.
--
-- WHY THE COMPOSITION IS RESTATED.  pgkg_retrieve() is the composition, and
-- both of the things added here — restricting the candidate arms to one source
-- class, and the second embedder generation's arm — are decisions taken at the
-- point where the arms are chosen.  There is no smaller unit: every stage below
-- `retrieved` already takes what it is given.  The arms themselves are
-- untouched, and the two new parameters are appended with defaults that leave
-- every existing caller with the behaviour it has.
--
-- WHY THE GENERATION ARM COMES ACROSS FROM pgkg_search.  A cutover window
-- serves queries from two embedding spaces at once (D8); pgkg_search() has
-- carried that arm since 022 and recall() has passed it since.  Moving recall()
-- onto pgkg_retrieve() without it would silently retire dual-generation
-- retrieval.  It stays proposition-only, like the graph arm: chunk vectors have
-- no generation side table.
--
-- THE THREE TABLES 030 AND 032 LEFT WITHOUT RLS.  document_versions,
-- document_version_chunks and ingest_jobs hold, respectively, what a tenant has
-- ingested, in what order, and the full text of it while it waits to be
-- indexed.  All three were shipped without a policy because the coverage test
-- that pins the RLS-enabled table set lives in a file their authors did not
-- own.  A policy and its test land together here.


-- 1. A chat chunk is provenance, not a passage.
DROP FUNCTION pgkg_chunk_live(UUID, INT);

CREATE FUNCTION pgkg_chunk_live(p_chunk_id UUID, p_refcount INT)
RETURNS BOOLEAN
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT (
            p_refcount = 0
            AND NOT EXISTS (
                SELECT 1 FROM chunks c
                WHERE c.id = p_chunk_id AND c.document_id IS NOT NULL
            )
        )
        OR EXISTS (
            SELECT 1
            FROM document_version_chunks dvc
            JOIN document_versions dv ON dv.id = dvc.document_version_id
            JOIN documents d ON d.id = dv.document_id
            WHERE dvc.chunk_id = p_chunk_id
              AND dv.status = 'current'
              AND d.deleted_at IS NULL
        )
$$;

COMMENT ON FUNCTION pgkg_chunk_live(UUID, INT) IS
    'Whether retrieval may see a chunk. A chunk carried by the current version '
    'of a live document is a passage; so is a chunk that belongs to no '
    'document at all. A chunk that belongs to a document but to no version is '
    'that document''s provenance — which is what chat ingest writes — and is '
    'not retrievable (ADR 0001, D1).';


-- 2. Row-level security for the three lifecycle tables.
--
-- document_version_chunks carries no org column and gains none: the link is
-- owned by the version, and denormalising the org onto it would create a second
-- statement of the same fact that could disagree with the first.  Its policy
-- reads through document_versions, whose own policy applies inside the
-- subquery, so a stranger's session finds no version to link through.
ALTER TABLE document_versions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_version_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingest_jobs             ENABLE ROW LEVEL SECURITY;

CREATE POLICY document_versions_org_isolation ON document_versions
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY document_version_chunks_org_isolation ON document_version_chunks
    USING (
        EXISTS (
            SELECT 1 FROM document_versions dv
            WHERE dv.id = document_version_id
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM document_versions dv
            WHERE dv.id = document_version_id
        )
    );

CREATE POLICY ingest_jobs_org_isolation ON ingest_jobs
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

COMMENT ON POLICY ingest_jobs_org_isolation ON ingest_jobs IS
    'A queued job holds the document text, so the queue is as sensitive as the '
    'corpus it feeds. A worker draining several orgs therefore has to run as '
    'an owner or set the org per claim (ADR 0001, D3).';


-- 3. A repeated passage keeps every position it appears in.
--
-- Boilerplate repeats inside one document — a disclaimer under every section —
-- and content addressing makes that one chunk row, which is the point.  It is
-- still at several ordinals, and the link table is where an ordinal lives: with
-- the chunk in the primary key the second link collided with the first and was
-- dropped, so the window expansion read an ordering the document does not have
-- and the refcount undercounted a live chunk.  The position is what identifies
-- a link, so the position is what the key is on; the UNIQUE it replaces said
-- the same thing about the same two columns.
ALTER TABLE document_version_chunks
    DROP CONSTRAINT document_version_chunks_pkey,
    DROP CONSTRAINT document_version_chunks_document_version_id_ord_key,
    ADD CONSTRAINT document_version_chunks_pkey
        PRIMARY KEY (document_version_id, ord);

COMMENT ON TABLE document_version_chunks IS
    'Which passages a version carries, and in what order. Keyed on the '
    'position rather than on the chunk: one passage may appear at several '
    'positions in the same document (ADR 0001, D6).';


-- 4. The composition, with the two parameters the application surface needs.
DROP FUNCTION pgkg_retrieve(
    TEXT, halfvec, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT, UUID[], UUID[],
    UUID, UUID[], TIMESTAMPTZ, INT, REAL, INT, REAL, REAL, REAL, REAL, INT, INT
);

CREATE FUNCTION pgkg_retrieve(
    q_text                    TEXT,
    q_embedding               halfvec DEFAULT NULL,
    k_retrieve                INT     DEFAULT 100,
    k_initial                 INT     DEFAULT 200,
    p_namespace               TEXT    DEFAULT 'default',
    p_session_id              TEXT    DEFAULT NULL,
    recency_half_life_days    REAL    DEFAULT 30.0,
    expand_graph              BOOLEAN DEFAULT TRUE,
    rrf_k                     INT     DEFAULT 60,
    p_org_ids                 UUID[]  DEFAULT NULL,
    p_collection_ids          UUID[]  DEFAULT NULL,
    p_user_id                 UUID    DEFAULT NULL,
    p_acl_groups              UUID[]  DEFAULT NULL,
    p_valid_at                TIMESTAMPTZ DEFAULT NULL,
    k_rerank                  INT     DEFAULT 64,
    corpus_fraction           REAL    DEFAULT 0.6,
    memory_floor              INT     DEFAULT 8,
    w_scope_world             REAL    DEFAULT 1.0,
    w_scope_org               REAL    DEFAULT 1.0,
    w_scope_user              REAL    DEFAULT 1.0,
    perishable_half_life_days REAL    DEFAULT 730.0,
    window_before             INT     DEFAULT 1,
    window_after              INT     DEFAULT 1,
    p_sources                 TEXT[]  DEFAULT NULL,
    p_gen_queries             pgkg_gen_query[] DEFAULT NULL
) RETURNS TABLE (
    item_id        UUID,
    source         TEXT,
    text           TEXT,
    context_text   TEXT,
    rrf_score      REAL,
    adjusted_score REAL,
    source_kind    TEXT,
    bucket         TEXT,
    claim_scope    TEXT,
    collection_id  UUID,
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
                p_valid_at, 'propositions'
            ) b
            WHERE p_sources IS NULL OR 'propositions' = ANY(p_sources)
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'propositions'
            ) v
            WHERE p_sources IS NULL OR 'propositions' = ANY(p_sources)
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
            WHERE p_sources IS NULL OR 'propositions' = ANY(p_sources)
        )
        ||
        ARRAY(
            SELECT (b.item_id, b.kind, b.rank, b.raw_score)::pgkg_candidate
            FROM pgkg_bm25_candidates(
                q_text, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'chunks'
            ) b
            WHERE p_sources IS NULL OR 'chunks' = ANY(p_sources)
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'chunks'
            ) v
            WHERE p_sources IS NULL OR 'chunks' = ANY(p_sources)
        ) AS candidates
),
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
            CASE
                WHEN expand_graph
                 AND (p_sources IS NULL OR 'propositions' = ANY(p_sources))
                THEN s.candidates
                ELSE '{}'::pgkg_candidate[]
            END,
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

weighted AS (
    SELECT
        f.item_id,
        f.fused_score,
        f.in_kw,
        f.in_vec,
        s.source,
        s.collection_id,
        (f.fused_score
         * CASE s.claim_scope
               WHEN 'world' THEN w_scope_world
               WHEN 'user'  THEN w_scope_user
               ELSE              w_scope_org
           END)::REAL AS weighted_score
    FROM fused f
    JOIN pgkg_item_scope(ARRAY(SELECT inner_f.item_id FROM fused inner_f)) s
      ON s.item_id = f.item_id
),

quota AS (
    SELECT q.*
    FROM pgkg_apply_quotas(
        ARRAY(
            SELECT (w.item_id, 'fused'::TEXT, 0, w.weighted_score)::pgkg_candidate
            FROM weighted w
        ),
        k_rerank, corpus_fraction, memory_floor
    ) q
),

profiled AS (
    SELECT ap.*
    FROM pgkg_apply_profile(
        ARRAY(
            SELECT (q.item_id, 'fused'::TEXT, 0, q.quota_score)::pgkg_candidate
            FROM quota q
        ),
        recency_half_life_days, perishable_half_life_days
    ) ap
),

payload AS (
    SELECT p.id AS item_id, p.text, p.asserted_at
    FROM propositions p
    WHERE p.id IN (SELECT q.item_id FROM quota q)

    UNION ALL

    SELECT c.id, c.text, c.asserted_at
    FROM chunks c
    WHERE c.id IN (SELECT q.item_id FROM quota q)
),

-- One window per chunk.  Boilerplate deduplication means a chunk can belong to
-- the current version of more than one document, and a result row has one
-- context: the lowest version id is an arbitrary choice, but a stable one.
windows AS (
    SELECT DISTINCT ON (cw.chunk_id) cw.chunk_id, cw.context_text
    FROM pgkg_chunk_window(
        ARRAY(SELECT q.item_id FROM quota q), window_before, window_after
    ) cw
    ORDER BY cw.chunk_id, cw.document_version_id
)

SELECT
    quota.item_id,
    weighted.source,
    payload.text,
    COALESCE(windows.context_text, payload.text),
    weighted.fused_score,
    profiled.adjusted_score,
    CASE
        WHEN weighted.in_kw AND weighted.in_vec THEN 'both'
        WHEN weighted.in_kw                     THEN 'kw'
        WHEN weighted.in_vec                    THEN 'vec'
        ELSE                                         'graph'
    END,
    quota.bucket,
    quota.claim_scope,
    weighted.collection_id,
    payload.asserted_at
FROM quota
JOIN weighted ON weighted.item_id = quota.item_id
JOIN payload  ON payload.item_id = quota.item_id
JOIN profiled ON profiled.item_id = quota.item_id
LEFT JOIN windows ON windows.chunk_id = quota.item_id
ORDER BY profiled.adjusted_score DESC
LIMIT k_retrieve;
$$;

COMMENT ON FUNCTION pgkg_retrieve IS
    'The two-store retrieval surface. p_sources restricts the candidate arms '
    'to one class before the quota is computed, so a caller that asks for '
    'passages alone gets the whole budget spent on passages rather than the '
    'share the fused split left them (ADR 0001, D1). p_gen_queries carries the '
    'second embedding space during a cutover window (D8).';
