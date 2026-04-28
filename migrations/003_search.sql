-- Hero search function: hybrid keyword + vector + graph expansion with RRF
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
    object_id      UUID
)
LANGUAGE SQL STABLE
AS $$
WITH

-- 1. Keyword retrieval (skipped when q_text is NULL or empty)
kw AS (
    SELECT
        p.id                                                          AS prop_id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(p.tsv, plainto_tsquery('english', q_text)) DESC
        )                                                             AS rank
    FROM propositions p
    WHERE q_text IS NOT NULL
      AND q_text <> ''
      AND p.namespace = p_namespace
      AND p.superseded_by IS NULL
      AND p.tsv @@ plainto_tsquery('english', q_text)
      AND (
            p_session_id IS NULL
            OR p.session_id = p_session_id
            OR p.session_id IS NULL
          )
    ORDER BY ts_rank_cd(p.tsv, plainto_tsquery('english', q_text)) DESC
    LIMIT k_initial
),

-- 2. Vector retrieval (skipped when q_embedding is NULL)
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

-- 3. RRF fusion of keyword + vector results
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

-- 4. Seed entities: top 20 entity ids from fused propositions
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

-- 5. Graph-expanded neighbor propositions (only when expand_graph = TRUE)
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

-- 6. All candidates unified
all_candidates AS (
    SELECT prop_id, rrf_score, in_kw, in_vec FROM fused
    UNION ALL
    SELECT prop_id, rrf_score, in_kw, in_vec FROM neighbor_props
),

-- 7. Join with proposition data and compute adjusted score
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
                    -EXTRACT(EPOCH FROM (now() - p.last_accessed_at))
                    / (86400.0 * recency_half_life_days::FLOAT8),
                    -87.0  -- clamp so EXP result >= ~1.6e-38, safe for REAL cast
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
        p.object_id
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
    object_id
FROM scored
ORDER BY adjusted_score DESC
LIMIT k_retrieve;
$$;


-- Bump access counters for a set of propositions
CREATE OR REPLACE FUNCTION pgkg_bump_access(prop_ids UUID[])
RETURNS VOID
LANGUAGE SQL
AS $$
    UPDATE propositions
    SET last_accessed_at = now(),
        access_count     = access_count + 1
    WHERE id = ANY(prop_ids);
$$;


-- Find or create an entity by name/type/embedding similarity
CREATE OR REPLACE FUNCTION pgkg_link_entity(
    p_namespace  TEXT,
    p_name       TEXT,
    p_type       TEXT,
    p_embedding  vector(1024),
    p_threshold  REAL DEFAULT 0.85
) RETURNS UUID
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    v_id UUID;
BEGIN
    -- 1. Exact name + type match within namespace
    SELECT id INTO v_id
    FROM entities
    WHERE namespace = p_namespace
      AND name = p_name
      AND (type = p_type OR (type IS NULL AND p_type IS NULL))
    LIMIT 1;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    -- 2. Trigram + embedding similarity match
    IF p_embedding IS NOT NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE namespace = p_namespace
          AND similarity(name, p_name) > 0.6
          AND (1 - (embedding <=> p_embedding)) > p_threshold
        ORDER BY (embedding <=> p_embedding)
        LIMIT 1;
    END IF;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    -- 3. Create new entity
    INSERT INTO entities (name, type, embedding, namespace)
    VALUES (p_name, p_type, p_embedding, p_namespace)
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$;
