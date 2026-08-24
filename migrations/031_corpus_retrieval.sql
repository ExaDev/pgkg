-- The corpus as a candidate source of its own: separate statistics, per-scope
-- quotas, three decay profiles, and small-to-big context (ADR 0001, D1 and D6).
--
-- WHY THE CHUNK STORE NEEDS ITS OWN CORPUS STATISTICS.  BM25's length
-- normalisation divides every document length by the corpus mean, so a mean
-- taken over a mixture of 12-token facts and 600-token passages describes
-- neither.  Every passage then looks pathologically long, its term frequencies
-- are discounted to nothing, and the corpus loses every query it should win —
-- while a mean taken the other way round makes short facts look empty.  This
-- is the concrete reason D1 keeps two physical stores rather than one
-- polymorphic table, so a chunk arm that reused the proposition statistics
-- would be the polymorphic table with extra steps.
--
-- 011 anticipated this: corpus_stats and lexeme_df are keyed on
-- (namespace, kind) precisely so a second store could register its own row,
-- and `kind` has meant 'proposition' alone until now.  Chunks have no
-- namespace column — their scope is org and collection (020) — so the chunk
-- domain key is the collection, rendered by pgkg_stats_domain() into the
-- existing TEXT domain column with a 'collection:' prefix.  The prefix is not
-- decoration: it makes a chunk domain unable to collide with any namespace a
-- caller could pass, and it says at a glance that the value is a domain key
-- rather than a namespace.  Widening the key to a typed column would mean
-- re-declaring 011's and 021's three statistics functions in a migration whose
-- subject is retrieval, and the proposition domain is the thing phase 1 moves
-- to collection_id anyway.
--
-- BM25 STATISTICS ARE ADDITIVE, which is what makes a per-collection key work
-- for a query spanning several collections: n_total and total_len sum, df sums,
-- and avgdl falls out of the two sums.  That is also why 011 stores the sum
-- rather than the mean.
--
-- QUOTAS ARE NOT A REFINEMENT.  600k org chunks against 4k personal facts win
-- nearly every query on candidate volume, and the product symptom D1 names is
-- exact: the assistant stops remembering you and starts quoting the handbook.
-- The reranker cannot fix it, because it only reorders what reached it, and MMR
-- cannot either, because it only diversifies the survivors.  So the cap is
-- applied in SQL, before the row budget the reranker sees, and it is applied by
-- claim scope rather than by store: general knowledge is topically unbounded
-- and competes on every query, where org policy only matches queries about the
-- org.  Every parameter is a function argument, because a quota is a tuning
-- decision and tuning must not need a migration.
--
-- THREE PROFILES, ONE FUNCTION.  023 added collections.decay_profile and left
-- reading it to this phase.  Conversational keeps exactly the behaviour
-- pgkg_apply_profile has had since 010 — exponential decay on asserted_at plus
-- the logarithmic access-count boost — so every existing caller in the default
-- collection is unaffected.  Timeless is a flat factor of 1.0, because a 2019
-- expenses policy is the expenses policy.  Perishable decays on the same clock
-- with a multi-year constant, because a 2019 post about a framework's API is
-- not stale, it is wrong.  The frequency boost is off for both corpus profiles:
-- on reference material it is a popularity feedback loop, and on shared
-- material it carries one tenant's usage into another's ranking.
--
-- RETRIEVE ON THE CHUNK, RETURN THE WINDOW.  A 300-token chunk is the right
-- unit to match and the wrong unit to read, so pgkg_chunk_window() widens a hit
-- to its neighbours by document_version_chunks.ord.  It is a separate function
-- because the window is a presentation decision and has no business inside a
-- scoring stage.
--
-- pgkg_search() IS NOT TOUCHED.  Its column list is propositions-shaped and its
-- contract is what the existing suite pins; the unified surface is
-- pgkg_retrieve() below, which returns rows from both stores and names its
-- source.  pgkg_apply_profile() is redefined underneath both, and the default
-- collection's profile is conversational, so pgkg_search() behaves as before.


-- 1. The chunk statistics domain.
CREATE FUNCTION pgkg_stats_domain(p_collection_id UUID) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT 'collection:' || p_collection_id::TEXT $$;

COMMENT ON FUNCTION pgkg_stats_domain(UUID) IS
    'The retrieval-statistics domain key for a chunk collection. Prefixed so a '
    'domain key can never collide with a caller-supplied namespace.';


-- Maintenance, in the shape 011 established: statement-level triggers with
-- transition tables, one set-based upsert per statement rather than one per row.
--
-- INSERT and DELETE only.  A chunk's text is immutable by trigger (030), so no
-- UPDATE can change its lexemes or its length, and the highest-volume UPDATE on
-- the table — refcount maintenance — must not rewrite the statistics tables at
-- all.  Moving a chunk between collections is not a supported operation and is
-- the one thing that would drift; pgkg_refresh_chunk_stats() repairs it.
CREATE FUNCTION pgkg_chunks_stats_delta() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    delta_sign INT := TG_ARGV[0]::INT;
BEGIN
    INSERT INTO corpus_stats AS cs (namespace, kind, n_total, total_len)
    SELECT pgkg_stats_domain(d.collection_id),
           'chunk',
           delta_sign * COUNT(*),
           delta_sign * COALESCE(SUM(d.doc_len), 0)
    FROM delta_rows d
    GROUP BY d.collection_id
    ON CONFLICT (namespace, kind) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now();

    INSERT INTO lexeme_df AS ld (namespace, kind, lexeme, df)
    SELECT pgkg_stats_domain(d.collection_id), 'chunk', u.lexeme,
           delta_sign * COUNT(*)
    FROM delta_rows d, unnest(d.tsv) AS u(lexeme, positions, weights)
    GROUP BY d.collection_id, u.lexeme
    ON CONFLICT (namespace, kind, lexeme) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;

CREATE TRIGGER pgkg_chunk_stats_insert
    AFTER INSERT ON chunks
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_chunks_stats_delta('1');

CREATE TRIGGER pgkg_chunk_stats_delete
    AFTER DELETE ON chunks
    REFERENCING OLD TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_chunks_stats_delta('-1');


-- The backfill and drift repair, alongside 011's proposition equivalent rather
-- than inside it: the two stores are keyed differently, so one function cannot
-- take one scope argument.  NULL means every collection.
CREATE FUNCTION pgkg_refresh_chunk_stats(p_collection_id UUID DEFAULT NULL)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM corpus_stats
    WHERE kind = 'chunk'
      AND (p_collection_id IS NULL
           OR namespace = pgkg_stats_domain(p_collection_id));

    DELETE FROM lexeme_df
    WHERE kind = 'chunk'
      AND (p_collection_id IS NULL
           OR namespace = pgkg_stats_domain(p_collection_id));

    INSERT INTO corpus_stats (namespace, kind, n_total, total_len)
    SELECT pgkg_stats_domain(c.collection_id), 'chunk',
           COUNT(*), COALESCE(SUM(c.doc_len), 0)
    FROM chunks c
    WHERE p_collection_id IS NULL OR c.collection_id = p_collection_id
    GROUP BY c.collection_id;

    INSERT INTO lexeme_df (namespace, kind, lexeme, df)
    SELECT pgkg_stats_domain(c.collection_id), 'chunk', u.lexeme, COUNT(*)
    FROM chunks c, unnest(c.tsv) AS u(lexeme, positions, weights)
    WHERE p_collection_id IS NULL OR c.collection_id = p_collection_id
    GROUP BY c.collection_id, u.lexeme;
END;
$$;

SELECT pgkg_refresh_chunk_stats();


-- 2. Which chunks retrieval may see.  A chunk with no version links is a
-- pre-lifecycle chunk and stands on its own; a linked chunk is visible only
-- through the current version of a live document, so promoting a new version
-- withdraws the passages it dropped without touching a single chunk row.
-- refcount is the cheap half of the test and is maintained by trigger (030).
CREATE FUNCTION pgkg_chunk_live(p_chunk_id UUID, p_refcount INT)
RETURNS BOOLEAN
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT p_refcount = 0
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


-- 3. The keyword arm over either store.
--
-- p_source is appended last with the existing behaviour as its default, so
-- every positional caller keeps the proposition arm it has.  An unrecognised
-- source retrieves nothing rather than raising: both the statistics row and the
-- candidate scan are gated on the same equality, so neither branch produces a
-- row.
--
-- The two branches are UNION ALL rather than two functions because D1 fixes one
-- signature for every candidate source, and because the ranking machinery
-- either side of the branch — lexemes, IDF, the score, the cap — is the same
-- machinery.  Only the table, the statistics domain and the visibility
-- predicate differ.  p_source is a constant at every call site, so the planner
-- folds the unused branch away and the proposition path keeps the plan shape
-- 011 left it with.
DROP FUNCTION pgkg_bm25_candidates(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ
);

CREATE FUNCTION pgkg_bm25_candidates(
    q_text           TEXT,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL,
    p_source         TEXT   DEFAULT 'propositions'
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

-- The collection domains a chunk query reads its statistics from.  NULL means
-- unrestricted, matching pgkg_visible().
chunk_domains AS (
    SELECT pgkg_stats_domain(c) AS domain
    FROM unnest(p_collection_ids) AS c
),

-- One row for a recognised source, none otherwise.  A domain with no
-- statistics row falls back to n_total = 1 and avgdl = 1.0, which flattens IDF
-- to a constant and leaves ranking on term frequency alone — degraded but sane,
-- and never an error.
stats AS (
    SELECT
        GREATEST(COALESCE(cs.n_total, 1), 1)::FLOAT8 AS n_total,
        COALESCE(cs.avgdl, 1.0)                      AS avgdl
    FROM (VALUES (1)) AS present(x)
    LEFT JOIN corpus_stats cs
        ON cs.namespace = p_namespace
       AND cs.kind = 'proposition'
    WHERE p_source = 'propositions'

    UNION ALL

    -- Summed, not averaged: BM25 statistics are additive across sub-corpora,
    -- so the statistics of a multi-collection query are the sums of the
    -- collections' own.
    SELECT
        GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8,
        GREATEST(
            COALESCE(SUM(cs.total_len), 0)::FLOAT8
            / GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8,
            1.0
        )
    FROM corpus_stats cs
    WHERE p_source = 'chunks'
      AND cs.kind = 'chunk'
      AND (
            p_collection_ids IS NULL
            OR cs.namespace IN (SELECT domain FROM chunk_domains)
          )
    -- An un-grouped aggregate returns one row over zero input rows, so the
    -- source gate has to be applied to the group, not to the scan: without
    -- this, a proposition query would see two statistics rows and score
    -- against both.
    HAVING p_source = 'chunks'
),

-- Document frequency for the source's domain, summed for the same reason.
term_df AS (
    SELECT ld.lexeme, SUM(ld.df)::FLOAT8 AS df
    FROM lexeme_df ld
    WHERE (
            p_source = 'propositions'
            AND ld.kind = 'proposition'
            AND ld.namespace = p_namespace
          )
       OR (
            p_source = 'chunks'
            AND ld.kind = 'chunk'
            AND (
                  p_collection_ids IS NULL
                  OR ld.namespace IN (SELECT domain FROM chunk_domains)
                )
          )
    GROUP BY ld.lexeme
),

-- Robertson-Sparck-Jones IDF.  A lexeme with no row has df = 0, the same
-- value the old correlated COUNT(*) returned for an unseen term.
idf AS (
    SELECT
        ql.lexeme,
        LN(
            (s.n_total - COALESCE(d.df, 0.0) + 0.5)
            / (COALESCE(d.df, 0.0) + 0.5)
            + 1.0
        ) AS idf_val
    FROM query_lexemes ql
    CROSS JOIN stats s
    LEFT JOIN term_df d ON d.lexeme = ql.lexeme
)

SELECT
    sub.cand_id,
    'kw'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sub.bm25_score DESC))::INT,
    sub.bm25_score::REAL
FROM (
    SELECT
        p.id AS cand_id,
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
    WHERE p_source = 'propositions'
      AND q_text IS NOT NULL
      AND q_text <> ''
      AND p.namespace = p_namespace
      AND pgkg_temporal_visible(
            p.invalidated_at, p.valid_from, p.valid_to,
            COALESCE(p_valid_at, now())
          )
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

    UNION ALL

    -- The chunk branch carries no session and no validity interval: a passage
    -- is not asserted in a conversation and has no validity clock of its own.
    -- Its temporal filter is the document lifecycle, which pgkg_chunk_live()
    -- resolves.
    SELECT
        c.id,
        (
            SELECT COALESCE(SUM(
                i.idf_val
                * (COALESCE(array_length(u.positions, 1), 0)::FLOAT8 * 2.2)
                / (COALESCE(array_length(u.positions, 1), 0)::FLOAT8
                   + 1.2 * (1.0 - 0.75 + 0.75 * c.doc_len::FLOAT8 / s.avgdl))
            ), 0.0)
            FROM unnest(c.tsv) AS u(lexeme, positions, weights)
            JOIN idf i ON i.lexeme = u.lexeme
            CROSS JOIN stats s
        )
    FROM chunks c
    CROSS JOIN query_or
    WHERE p_source = 'chunks'
      AND q_text IS NOT NULL
      AND q_text <> ''
      AND query_or.q IS NOT NULL
      AND c.tsv @@ query_or.q
      AND pgkg_chunk_live(c.id, c.refcount)
      AND pgkg_visible(
            c.org_id, c.collection_id, c.visibility,
            c.owner_user_id, c.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
) sub
WHERE sub.bm25_score > 0.0
ORDER BY sub.bm25_score DESC
LIMIT k_initial;
$$;


-- 4. The vector arm over either store, on the same appended parameter.
DROP FUNCTION pgkg_vector_candidates(
    halfvec, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ
);

CREATE FUNCTION pgkg_vector_candidates(
    q_embedding      halfvec,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL,
    p_source         TEXT   DEFAULT 'propositions'
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
SELECT
    sub.cand_id,
    'vec'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sub.distance))::INT,
    (1.0 - sub.distance)::REAL
FROM (
    (
    SELECT p.id AS cand_id, (p.embedding <=> q_embedding) AS distance
    FROM propositions p
    WHERE p_source = 'propositions'
      AND q_embedding IS NOT NULL
      AND p.embedding IS NOT NULL
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
    LIMIT k_initial
    )

    UNION ALL

    (
    SELECT c.id, (c.embedding <=> q_embedding)
    FROM chunks c
    WHERE p_source = 'chunks'
      AND q_embedding IS NOT NULL
      AND c.embedding IS NOT NULL
      AND pgkg_chunk_live(c.id, c.refcount)
      AND pgkg_visible(
            c.org_id, c.collection_id, c.visibility,
            c.owner_user_id, c.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
    ORDER BY c.embedding <=> q_embedding
    LIMIT k_initial
    )
) sub
ORDER BY sub.distance
LIMIT k_initial;
$$;


-- 5. The memory profile, resolved from the collection (D6).
--
-- One constant cannot serve chat, policy and vendor documentation, and the
-- choice is a property of the material rather than of the row: every passage of
-- the handbook ages the same way, so it is the collection that carries the
-- profile and 023 put the column there.
--
--   conversational  exponential decay on asserted_at, plus the access-count
--                   boost.  Exactly what 010 did, which is why the default
--                   collection — and therefore every existing caller — is
--                   unaffected.
--   timeless        factor 1.0.  Not a very long half-life: a flat factor, so
--                   ordering inside a policy corpus is decided by relevance
--                   alone and never by which page was written last.
--   perishable      the same decay on the same clock, on a multi-year
--                   constant.  asserted_at IS the publication date for a
--                   corpus chunk (D6), so no fourth clock is needed.
--
-- Both corpus profiles drop the frequency boost.  On reference material it is a
-- popularity feedback loop — retrieved because retrieved — and on shared
-- material it carries one tenant's usage into another tenant's ranking.  It
-- also deletes the read-path write amplification on the largest table.
--
-- Both stores are scored here rather than in two functions: an item's profile
-- comes from its collection whichever table holds it, and the caller has one
-- fused candidate set, not one per store.  A chunk has no access count and no
-- confidence of its own, and needs neither — the profiles that apply to chunks
-- are the two that ignore both.
DROP FUNCTION pgkg_apply_profile(pgkg_candidate[], REAL);

CREATE FUNCTION pgkg_apply_profile(
    p_scored                  pgkg_candidate[],
    recency_half_life_days    REAL DEFAULT 30.0,
    perishable_half_life_days REAL DEFAULT 730.0
) RETURNS TABLE (
    item_id        UUID,
    adjusted_score REAL
)
LANGUAGE SQL STABLE
AS $$
WITH

cand AS (
    SELECT c.item_id, c.raw_score
    FROM unnest(p_scored) AS c(item_id, kind, cand_rank, raw_score)
),

profiled AS (
    SELECT
        p.id AS item_id,
        col.decay_profile,
        COALESCE(p.asserted_at, p.last_accessed_at) AS decay_clock,
        p.access_count,
        p.confidence
    FROM cand
    JOIN propositions p ON p.id = cand.item_id
    JOIN collections col ON col.id = p.collection_id

    UNION ALL

    SELECT
        c.id,
        col.decay_profile,
        COALESCE(c.asserted_at, c.created_at),
        0,
        1.0::REAL
    FROM cand
    JOIN chunks c ON c.id = cand.item_id
    JOIN collections col ON col.id = c.collection_id
)

SELECT
    cand.item_id,
    CAST(GREATEST(
        cand.raw_score::FLOAT8
        * CASE pr.decay_profile
              WHEN 'timeless' THEN 1.0
              -- The -87.0 floor keeps EXP() above REAL underflow.
              ELSE EXP(GREATEST(
                  -EXTRACT(EPOCH FROM (now() - pr.decay_clock))
                  / (86400.0 * CASE pr.decay_profile
                                   WHEN 'perishable'
                                       THEN perishable_half_life_days::FLOAT8
                                   ELSE recency_half_life_days::FLOAT8
                               END),
                  -87.0
              ))
          END
        * CASE pr.decay_profile
              WHEN 'conversational'
                  THEN 1.0 + LN(1.0 + pr.access_count::FLOAT8)
              ELSE 1.0
          END
        * pr.confidence::FLOAT8,
        0.0
    ) AS REAL)
FROM cand
JOIN profiled pr ON pr.item_id = cand.item_id;
$$;


-- 6. What a candidate is, for a stage that has to treat both stores alike.
--
-- The item ids of the two stores are disjoint, so a fused candidate set does
-- not need to carry its provenance through fusion — it can be resolved.  That
-- is what keeps pgkg_candidate at four columns and leaves every function
-- written against it, pgkg_fuse() included, untouched.
--
-- `bucket` is the quota axis and comes from the collection's kind, not from the
-- store: a proposition extracted from the handbook (D2's opt-in) is corpus
-- material however it is stored, and a chunk of a chat transcript is not.
-- claim_scope is the row's own where a row has one and the collection's
-- otherwise, because a proposition can be about the world while living in a
-- mixed collection.
CREATE FUNCTION pgkg_item_scope(p_item_ids UUID[])
RETURNS TABLE (
    item_id       UUID,
    source        TEXT,
    collection_id UUID,
    bucket        TEXT,
    claim_scope   TEXT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        p.id,
        'propositions'::TEXT,
        p.collection_id,
        CASE WHEN col.kind = 'corpus' THEN 'corpus' ELSE 'memory' END,
        p.claim_scope
    FROM propositions p
    JOIN collections col ON col.id = p.collection_id
    WHERE p.id = ANY(p_item_ids)

    UNION ALL

    SELECT
        c.id,
        'chunks'::TEXT,
        c.collection_id,
        CASE WHEN col.kind = 'corpus' THEN 'corpus' ELSE 'memory' END,
        col.claim_scope
    FROM chunks c
    JOIN collections col ON col.id = c.collection_id
    WHERE c.id = ANY(p_item_ids)
$$;


-- 7. Source quotas, applied to the fused set before anything downstream sees
-- it (D1).
--
-- Three numbers, all arguments, because a quota is a tuning decision and tuning
-- must not need a migration:
--
--   k_rerank         the row budget being divided — the reranker's input, not
--                    the caller's k.
--   corpus_fraction  the ceiling on corpus material as a share of that budget.
--   memory_floor     slots held for the caller's own memory even when the
--                    corpus outscores it everywhere.  A floor and a ceiling are
--                    not the same constraint: the ceiling stops the corpus
--                    filling the budget, the floor stops it filling everything
--                    the ceiling left.
--
-- The floor never invents rows: it is clamped to the memory candidates that
-- actually exist, so a query with no personal material gives the corpus the
-- whole budget rather than returning a short result.
--
-- Within the corpus allowance, each claim scope is capped at an equal share.
-- This is the part D1 insists on: unbounded general knowledge matches every
-- query, org policy only matches queries about the org, and a single corpus cap
-- would let the former consume the latter's share on every request.
CREATE FUNCTION pgkg_apply_quotas(
    p_scored        pgkg_candidate[],
    k_rerank        INT  DEFAULT 64,
    corpus_fraction REAL DEFAULT 0.6,
    memory_floor    INT  DEFAULT 8
) RETURNS TABLE (
    item_id     UUID,
    quota_score REAL,
    bucket      TEXT,
    claim_scope TEXT
)
LANGUAGE SQL STABLE
AS $$
WITH

cand AS (
    SELECT c.item_id, c.raw_score
    FROM unnest(p_scored) AS c(item_id, kind, cand_rank, raw_score)
),

labelled AS (
    SELECT cand.item_id, cand.raw_score, s.bucket, s.claim_scope
    FROM cand
    JOIN pgkg_item_scope(
        (SELECT array_agg(inner_cand.item_id) FROM cand inner_cand)
    ) s ON s.item_id = cand.item_id
),

allowance AS (
    SELECT
        corpus_allow,
        LEAST(n_memory, GREATEST(k_rerank, 0) - corpus_allow) AS memory_allow,
        GREATEST(CEIL(corpus_allow::FLOAT8 / n_scopes)::BIGINT, 1)
            AS per_scope_cap
    FROM (
        SELECT
            n_memory,
            n_scopes,
            GREATEST(LEAST(
                n_corpus,
                FLOOR(GREATEST(k_rerank, 0)::FLOAT8
                      * GREATEST(corpus_fraction, 0.0)::FLOAT8)::BIGINT,
                GREATEST(k_rerank, 0)
                    - LEAST(GREATEST(memory_floor, 0), n_memory)
            ), 0) AS corpus_allow
        FROM (
            SELECT
                COUNT(*) FILTER (WHERE bucket = 'memory') AS n_memory,
                COUNT(*) FILTER (WHERE bucket = 'corpus') AS n_corpus,
                GREATEST(
                    COUNT(DISTINCT claim_scope) FILTER (WHERE bucket = 'corpus'),
                    1
                ) AS n_scopes
            FROM labelled
        ) counted
    ) budget
),

-- Rank inside the claim scope first, so the per-scope cap is applied before
-- the scopes compete with one another for the corpus allowance.
scoped AS (
    SELECT
        l.*,
        ROW_NUMBER() OVER (
            PARTITION BY l.bucket,
                CASE WHEN l.bucket = 'corpus' THEN l.claim_scope ELSE '' END
            ORDER BY l.raw_score DESC, l.item_id
        ) AS scope_rank
    FROM labelled l
),

admitted AS (
    SELECT
        s.*,
        ROW_NUMBER() OVER (
            PARTITION BY s.bucket ORDER BY s.raw_score DESC, s.item_id
        ) AS bucket_rank
    FROM scoped s, allowance a
    WHERE s.bucket = 'memory' OR s.scope_rank <= a.per_scope_cap
)

SELECT a.item_id, a.raw_score, a.bucket, a.claim_scope
FROM admitted a, allowance al
WHERE (a.bucket = 'corpus' AND a.bucket_rank <= al.corpus_allow)
   OR (a.bucket = 'memory' AND a.bucket_rank <= al.memory_allow);
$$;


-- 8. Small-to-big.  A passage is matched at chunk granularity and read at
-- section granularity, so a hit is widened to its neighbours by ordinal within
-- the version that carries it.  Separate from every scoring stage on purpose:
-- the window changes what is returned, never what is ranked.
--
-- Only the current version of a live document supplies a window, for the same
-- reason retrieval only sees those chunks: a retired ordering is not the
-- document's ordering any more.  A chunk with no version links — the
-- pre-lifecycle shape — has no window and no row here, which the caller reads
-- as "the passage is its own context".
CREATE FUNCTION pgkg_chunk_window(
    p_chunk_ids UUID[],
    p_before    INT DEFAULT 1,
    p_after     INT DEFAULT 1
) RETURNS TABLE (
    chunk_id            UUID,
    document_version_id UUID,
    ord_from            INT,
    ord_to              INT,
    context_text        TEXT
)
LANGUAGE SQL STABLE
AS $$
SELECT
    anchor.chunk_id,
    anchor.document_version_id,
    MIN(neighbour.ord)::INT,
    MAX(neighbour.ord)::INT,
    string_agg(n.text, E'\n\n' ORDER BY neighbour.ord)
FROM document_version_chunks anchor
JOIN document_versions dv
  ON dv.id = anchor.document_version_id
 AND dv.status = 'current'
JOIN documents d
  ON d.id = dv.document_id
 AND d.deleted_at IS NULL
JOIN document_version_chunks neighbour
  ON neighbour.document_version_id = anchor.document_version_id
 AND neighbour.ord BETWEEN anchor.ord - GREATEST(p_before, 0)
                       AND anchor.ord + GREATEST(p_after, 0)
JOIN chunks n ON n.id = neighbour.chunk_id
WHERE anchor.chunk_id = ANY(p_chunk_ids)
GROUP BY anchor.chunk_id, anchor.document_version_id;
$$;


-- 9. The unified retrieval surface (D1).
--
-- pgkg_search() is left exactly as it is: its column list is
-- propositions-shaped and its contract is what the existing suite pins.  This
-- is the function that returns both stores in one ranked list and names which
-- store each row came from, in the order D1 lays out —
--
--   bm25(propositions) ┐
--   vec (propositions) ├→ fuse → scope weights → quotas → profile → order
--   bm25(chunks)       │
--   vec (chunks)       │
--   graph(seeds)       ┘
--
-- Each store contributes its own rank sequence, so reciprocal-rank fusion
-- already puts the best chunk and the best fact on equal footing; what it
-- cannot do is stop the corpus owning every slot below the first, which is the
-- quota's job.
--
-- Per-scope RRF weights ship at parity and multiply the fused score.  They are
-- the knob for a tenant that wants general knowledge turned down, or off,
-- without a rebuild — the retrieval-side counterpart of
-- collection_subscriptions.rrf_weight.
--
-- The graph arm stays proposition-only.  Chunks reach the graph through
-- entity_mentions (D2), which is not built yet; adding it later is one more
-- array in `retrieved` and nothing else.
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
    window_after              INT     DEFAULT 1
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
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'propositions'
            ) v
        )
        ||
        ARRAY(
            SELECT (b.item_id, b.kind, b.rank, b.raw_score)::pgkg_candidate
            FROM pgkg_bm25_candidates(
                q_text, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'chunks'
            ) b
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at, 'chunks'
            ) v
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
