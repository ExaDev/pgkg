-- Decompose pgkg_search() into composable set-returning functions.
--
-- Migrations 003/007/008/009 each carry a full copy of a ~270-line pgkg_search:
-- every change to one retrieval stage meant re-declaring the whole pipeline, so
-- the stages could never be tested — or replaced — independently.  Adding the
-- corpus as a second retrievable store (ADR 0001, D1) makes that untenable:
-- a new candidate source must be a new function, not another copy.
--
-- Every candidate source now shares one output signature
-- (item_id UUID, kind TEXT, rank INT, raw_score REAL), carried between stages
-- as an array of the pgkg_candidate composite type.  pgkg_search() becomes a
-- thin composition:
--
--   bm25 ─┐
--   vector┼→ fuse → graph(seeds) → fuse → profile → order/limit
--   graph ┘
--
-- The graph arm is fused twice because its score is propagated from the fused
-- lexical+vector floor, exactly as the pre-refactor neighbour CTE did.
--
-- Two deliberate behaviour changes, both defects the old shape hid:
--   1. Graph fan-out is capped PER SEED ENTITY (ADR 0001, D7 latency table).
--      The old global LIMIT 100 let one hub entity consume the whole neighbour
--      budget and crowd out every other seed.
--   2. Fusion groups by item, so a proposition reachable over several edges is
--      one result row.  The old UNION ALL emitted one row per traversed edge.
-- Everything else — signature, column list, scores, ordering — is unchanged.

CREATE TYPE pgkg_candidate AS (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
);


-- 1. Keyword retrieval: BM25 over OR-joined query lexemes.
CREATE FUNCTION pgkg_bm25_candidates(
    q_text       TEXT,
    p_namespace  TEXT DEFAULT 'default',
    p_session_id TEXT DEFAULT NULL,
    k_initial    INT  DEFAULT 200
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

corpus_stats AS (
    SELECT
        GREATEST(COUNT(*), 1)::FLOAT8              AS n_total,
        GREATEST(AVG(length(p.tsv)), 1.0)::FLOAT8  AS avgdl
    FROM propositions p
    WHERE p.namespace = p_namespace
      AND p.superseded_by IS NULL
),

doc_freq AS (
    SELECT
        ql.lexeme,
        (
            SELECT COUNT(*)::FLOAT8
            FROM propositions p
            WHERE p.tsv @@ to_tsquery('simple', ql.lexeme)
              AND p.namespace = p_namespace
              AND p.superseded_by IS NULL
        ) AS df
    FROM query_lexemes ql
),

idf AS (
    SELECT
        df.lexeme,
        LN((cs.n_total - df.df + 0.5) / (df.df + 0.5) + 1.0) AS idf_val
    FROM doc_freq df
    CROSS JOIN corpus_stats cs
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
                   + 1.2 * (1.0 - 0.75 + 0.75 * length(p.tsv)::FLOAT8 / cs.avgdl))
            ), 0.0)
            FROM unnest(p.tsv) AS u(lexeme, positions, weights)
            JOIN idf i ON i.lexeme = u.lexeme
            CROSS JOIN corpus_stats cs
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
) sub
WHERE sub.bm25_score > 0.0
ORDER BY sub.bm25_score DESC
LIMIT k_initial;
$$;


-- 2. Vector retrieval over the HNSW index.  raw_score is cosine similarity;
-- only the rank feeds fusion.
CREATE FUNCTION pgkg_vector_candidates(
    q_embedding  vector(1024),
    p_namespace  TEXT DEFAULT 'default',
    p_session_id TEXT DEFAULT NULL,
    k_initial    INT  DEFAULT 200
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
ORDER BY p.embedding <=> q_embedding
LIMIT k_initial;
$$;


-- 3. One-hop graph expansion.  p_seeds is a scored candidate set (raw_score
-- carries each seed's fused score): its propositions name the seed entities,
-- its minimum score is the floor propagated to every neighbour, and its items
-- are excluded from the result.  An empty seed set expands nothing, which is
-- how callers express "no graph expansion".
--
-- Fan-out is capped per seed entity, not globally, so a hub entity with
-- thousands of edges cannot exhaust the neighbour budget.
CREATE FUNCTION pgkg_graph_candidates(
    p_seeds         pgkg_candidate[],
    p_namespace     TEXT DEFAULT 'default',
    k_seed_entities INT  DEFAULT 20,
    k_per_seed      INT  DEFAULT 10,
    k_total         INT  DEFAULT 100
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


-- 4. Weighted reciprocal-rank fusion.  Rank-based sources contribute
-- w / (rrf_k + rank); graph candidates carry a propagated score instead of a
-- comparable rank, so they contribute w_graph * raw_score.  Grouping by item
-- means the same item found by several sources is one row.
CREATE FUNCTION pgkg_fuse(
    p_candidates pgkg_candidate[],
    rrf_k        INT  DEFAULT 60,
    w_kw         REAL DEFAULT 1.0,
    w_vec        REAL DEFAULT 1.0,
    w_graph      REAL DEFAULT 0.5
) RETURNS TABLE (
    item_id     UUID,
    fused_score REAL,
    in_kw       BOOLEAN,
    in_vec      BOOLEAN,
    in_graph    BOOLEAN
)
LANGUAGE SQL IMMUTABLE
AS $$
SELECT
    c.item_id,
    CAST(SUM(
        CASE c.kind
            WHEN 'kw'    THEN w_kw::FLOAT8    * (1.0::FLOAT8 / (rrf_k + c.cand_rank)::FLOAT8)
            WHEN 'vec'   THEN w_vec::FLOAT8   * (1.0::FLOAT8 / (rrf_k + c.cand_rank)::FLOAT8)
            WHEN 'graph' THEN w_graph::FLOAT8 * c.raw_score::FLOAT8
            ELSE 0.0::FLOAT8
        END
    ) AS REAL),
    bool_or(c.kind = 'kw'),
    bool_or(c.kind = 'vec'),
    bool_or(c.kind = 'graph')
FROM unnest(p_candidates) AS c(item_id, kind, cand_rank, raw_score)
GROUP BY c.item_id;
$$;


-- 5. Memory profile: exponential recency decay, logarithmic access-frequency
-- boost and the stored confidence, applied to each candidate's fused score.
-- The -87.0 floor keeps EXP() above REAL underflow.
CREATE FUNCTION pgkg_apply_profile(
    p_scored               pgkg_candidate[],
    recency_half_life_days REAL DEFAULT 30.0
) RETURNS TABLE (
    item_id        UUID,
    adjusted_score REAL
)
LANGUAGE SQL STABLE
AS $$
SELECT
    c.item_id,
    CAST(GREATEST(
        c.raw_score::FLOAT8
        * EXP(
            GREATEST(
                -EXTRACT(EPOCH FROM (now() - COALESCE(p.asserted_at, p.last_accessed_at)))
                / (86400.0 * recency_half_life_days::FLOAT8),
                -87.0
            )
          )
        * (1.0 + LN(1.0 + p.access_count::FLOAT8))
        * p.confidence::FLOAT8,
        0.0
    ) AS REAL)
FROM unnest(p_scored) AS c(item_id, kind, cand_rank, raw_score)
JOIN propositions p ON p.id = c.item_id;
$$;


-- 6. pgkg_search() as a composition of the stages above.  CREATE OR REPLACE
-- cannot change a function's return type, so the old definition goes first.
DROP FUNCTION IF EXISTS pgkg_search(TEXT, vector, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT);

CREATE FUNCTION pgkg_search(
    q_text               TEXT,
    q_embedding          vector(1024),
    k_retrieve           INT DEFAULT 100,
    k_initial            INT DEFAULT 200,
    p_namespace          TEXT DEFAULT 'default',
    p_session_id         TEXT DEFAULT NULL,
    recency_half_life_days REAL DEFAULT 30.0,
    expand_graph         BOOLEAN DEFAULT TRUE,
    rrf_k                INT DEFAULT 60
) RETURNS TABLE (
    proposition_id UUID,
    text           TEXT,
    embedding      vector(1024),
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
            FROM pgkg_bm25_candidates(q_text, p_namespace, p_session_id, k_initial) b
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(q_embedding, p_namespace, p_session_id, k_initial) v
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
            p_namespace
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
    p.embedding,
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
