-- Query decomposition: switch keyword matching from AND to OR semantics.
--
-- plainto_tsquery('english', 'fix permission handler update pipeline')
-- produces 'fix' & 'permiss' & 'handl' & 'updat' & 'pipelin' — a document
-- must contain ALL terms to match.  For agent memory queries (often
-- multi-topic), this misses propositions that match a subset of terms.
--
-- The fix: build an OR-joined tsquery from the individual stemmed lexemes.
-- BM25 scoring already sums per-term IDF*TF contributions, so propositions
-- matching more terms and rarer terms naturally rank higher.  A proposition
-- matching 1 of 5 query terms still appears in results (with a low BM25
-- score) instead of being filtered out entirely.
--
-- The query_lexemes CTE (from 008_bm25_search.sql) already extracts
-- individual stemmed lexemes.  We add a new CTE that OR-joins them into
-- a single tsquery for the @@ filter, while BM25 scoring remains unchanged.

CREATE OR REPLACE FUNCTION pgkg_search(
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

-- Extract stemmed lexemes from the query.
query_lexemes AS (
    SELECT trim(BOTH '''' FROM t.lexeme) AS lexeme
    FROM unnest(
        string_to_array(plainto_tsquery('english', q_text)::text, ' & ')
    ) AS t(lexeme)
    WHERE q_text IS NOT NULL
      AND q_text <> ''
      AND trim(BOTH '''' FROM t.lexeme) <> ''
),

-- Build an OR-joined tsquery from the individual lexemes.
-- e.g. 'fix' | 'permiss' | 'handl' — matches propositions containing ANY term.
query_or AS (
    SELECT to_tsquery('simple',
        string_agg(lexeme, ' | ')
    ) AS q
    FROM query_lexemes
),

-- Corpus-level statistics for BM25.
corpus_stats AS (
    SELECT
        GREATEST(COUNT(*), 1)::FLOAT8              AS n_total,
        GREATEST(AVG(length(p.tsv)), 1.0)::FLOAT8  AS avgdl
    FROM propositions p
    WHERE p.namespace = p_namespace
      AND p.superseded_by IS NULL
),

-- Document frequency per query lexeme (corpus-wide).
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

-- IDF per query lexeme.
idf AS (
    SELECT
        df.lexeme,
        LN((cs.n_total - df.df + 0.5) / (df.df + 0.5) + 1.0) AS idf_val
    FROM doc_freq df
    CROSS JOIN corpus_stats cs
),

-- 1. Keyword retrieval with BM25 scoring and OR matching.
-- The @@ filter uses the OR-joined query (any term matches).
-- BM25 scoring sums contributions from whichever query terms appear
-- in each document — more matching terms = higher score.
kw AS (
    SELECT
        sub.prop_id,
        ROW_NUMBER() OVER (ORDER BY sub.bm25_score DESC) AS rank
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
    LIMIT k_initial
),

-- 2. Vector retrieval (unchanged)
vec AS (
    SELECT
        p.id AS prop_id,
        ROW_NUMBER() OVER (
            ORDER BY p.embedding <=> q_embedding
        )    AS rank
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
    LIMIT k_initial
),

-- 3. RRF fusion (unchanged)
fused AS (
    SELECT
        COALESCE(kw.prop_id, vec.prop_id)               AS prop_id,
        CAST(
            COALESCE(1.0 / (rrf_k + kw.rank), 0.0) +
            COALESCE(1.0 / (rrf_k + vec.rank), 0.0)
        AS REAL)                                         AS rrf_score,
        (kw.prop_id IS NOT NULL)                         AS in_kw,
        (vec.prop_id IS NOT NULL)                        AS in_vec
    FROM kw
    FULL OUTER JOIN vec ON kw.prop_id = vec.prop_id
),

-- 4. Seed entities (unchanged)
seed_entities AS (
    SELECT entity_id
    FROM (
        SELECT entity_id, MAX(rrf_score) AS best_rrf
        FROM (
            SELECT p.subject_id AS entity_id, f.rrf_score
            FROM fused f
            JOIN propositions p ON p.id = f.prop_id
            WHERE p.subject_id IS NOT NULL

            UNION ALL

            SELECT p.object_id AS entity_id, f.rrf_score
            FROM fused f
            JOIN propositions p ON p.id = f.prop_id
            WHERE p.object_id IS NOT NULL
        ) combined
        GROUP BY entity_id
    ) deduped
    ORDER BY best_rrf DESC
    LIMIT 20
),

-- 5. Graph-expanded neighbours (unchanged)
neighbor_props AS (
    SELECT
        np.id                                                          AS prop_id,
        CAST(
            0.5 * COALESCE((SELECT MIN(rrf_score) FROM fused), 0.0)
        AS REAL)                                                       AS rrf_score,
        FALSE                                                          AS in_kw,
        FALSE                                                          AS in_vec
    FROM edges e
    JOIN propositions np ON np.id = e.proposition_id
    WHERE expand_graph = TRUE
      AND (
            e.src_entity IN (SELECT entity_id FROM seed_entities)
            OR e.dst_entity IN (SELECT entity_id FROM seed_entities)
          )
      AND e.proposition_id NOT IN (SELECT prop_id FROM fused)
      AND np.superseded_by IS NULL
      AND np.namespace = p_namespace
    LIMIT 100
),

-- 6. All candidates unified (unchanged)
all_candidates AS (
    SELECT prop_id, rrf_score, in_kw, in_vec FROM fused
    UNION ALL
    SELECT prop_id, rrf_score, in_kw, in_vec FROM neighbor_props
),

-- 7. Adjusted score with recency/frequency decay (unchanged)
scored AS (
    SELECT
        ac.prop_id,
        p.text,
        p.embedding,
        ac.rrf_score,
        CAST(GREATEST(
            ac.rrf_score::FLOAT8
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
        ) AS REAL)              AS adjusted_score,
        CASE
            WHEN ac.in_kw AND ac.in_vec THEN 'both'
            WHEN ac.in_kw             THEN 'kw'
            WHEN ac.in_vec            THEN 'vec'
            ELSE                           'graph'
        END                     AS source_kind,
        p.chunk_id,
        p.subject_id,
        p.predicate,
        p.object_id,
        p.asserted_at
    FROM all_candidates ac
    JOIN propositions p ON p.id = ac.prop_id
)

SELECT
    prop_id      AS proposition_id,
    text,
    embedding,
    rrf_score,
    adjusted_score,
    source_kind,
    chunk_id,
    subject_id,
    predicate,
    object_id,
    asserted_at
FROM scored
ORDER BY adjusted_score DESC
LIMIT k_retrieve;
$$;
