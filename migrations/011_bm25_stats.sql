-- Materialise the BM25 corpus statistics that were recomputed per query.
--
-- pgkg_bm25_candidates() derived every ranking statistic at query time:
-- `avgdl` from AVG(length(tsv)) over every active proposition in the
-- namespace, and document frequency from one correlated COUNT(*) per query
-- lexeme.  That is O(corpus) per query, once for the average plus once per
-- term — seconds at 10M rows, and the largest latency blocker in the system
-- (ADR 0001, D7 latency table).  It also recomputed length(tsv) for every
-- candidate row on every query.
--
-- Three pieces of state replace those scans:
--   * propositions.doc_len — the tsvector's distinct-lexeme count, stored once
--     at write time.  A generated column cannot reference another generated
--     column, so it re-derives to_tsvector('english', text) rather than
--     reading tsv; the expression is identical, so doc_len = length(tsv).
--   * corpus_stats — (n_total, total_len) per retrieval domain.  The SUM, not
--     the average, is what is stored: a mean cannot be maintained
--     incrementally, a sum can, and avgdl falls out as a generated column.
--   * lexeme_df — document frequency per lexeme per retrieval domain.
--
-- The domain key is (namespace, kind).  `kind` is 'proposition' throughout
-- today and exists so that the corpus retriever of ADR 0001 D1 registers its
-- own statistics — chunks average 300-800 tokens against a proposition's 5-20,
-- and length normalisation over the mixture is meaningless.  Phase 1 replaces
-- namespace with collection_id, which is then an ALTER plus a widened primary
-- key rather than a redesign.
--
-- MAINTENANCE: statement-level triggers with transition tables, not a
-- scheduled refresh.
--
-- BM25 is a ranking heuristic, so stale statistics would have been tolerable
-- and a scheduled refresh was the obvious alternative.  Triggers win on two
-- grounds.  First, exactness is free here: the read path never pays for it,
-- and nothing outside the database has to be scheduled for correct ranking —
-- a refresh-only design makes every fresh namespace rank on absent statistics
-- until an operator remembers to run something.  Second, the overhead the ADR
-- warns about is per-row overhead, and transition tables remove it: one
-- statement inserting N propositions fires each trigger once and performs one
-- set-based upsert of the distinct lexemes in the whole batch, not N upserts.
-- That is the shape phase 0's set-based ingest wants.
--
-- The escape hatch is preserved rather than argued away.  The read path cannot
-- tell which mechanism populated the tables, so if ingest throughput ever makes
-- the triggers unacceptable they can be dropped and pgkg_refresh_corpus_stats()
-- scheduled instead; the read path degrades to uniform IDF when a domain has no
-- statistics row at all, so it never errors on a missing or lagging refresh.
--
-- Two functions rather than one, because Postgres forbids sharing a transition
-- table across trigger events: INSERT and DELETE differ only in sign and share
-- one function, while UPDATE needs both transition tables at once so it can
-- take the net delta and, crucially, skip statements that changed nothing
-- ranking-relevant.  The access-count flush rewrites access_count and
-- last_accessed_at in bulk; without that guard it would rewrite the statistics
-- tables too, on every read.


-- 1. Stored document length.  Adding a generated column rewrites the table;
-- that is acceptable while the schema is pre-production and forward-only.
ALTER TABLE propositions
    ADD COLUMN doc_len INT
    GENERATED ALWAYS AS (length(to_tsvector('english', text))) STORED;


-- 2. Corpus-level statistics per retrieval domain.
CREATE TABLE corpus_stats (
    namespace  TEXT   NOT NULL,
    kind       TEXT   NOT NULL DEFAULT 'proposition',
    n_total    BIGINT NOT NULL DEFAULT 0,
    total_len  BIGINT NOT NULL DEFAULT 0,
    avgdl      DOUBLE PRECISION GENERATED ALWAYS AS (
                   GREATEST(total_len::FLOAT8 / GREATEST(n_total, 1)::FLOAT8, 1.0)
               ) STORED,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, kind)
);


-- 3. Document frequency per lexeme.  Rows are allowed to fall to df = 0
-- rather than being deleted: a zero row reads the same as an absent one, and
-- pruning them would mean a scan on the ingest path to find them.
CREATE TABLE lexeme_df (
    namespace TEXT   NOT NULL,
    kind      TEXT   NOT NULL DEFAULT 'proposition',
    lexeme    TEXT   NOT NULL,
    df        BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (namespace, kind, lexeme)
);


-- 4. Full recomputation.  NULL namespace means every namespace.  This is the
-- backfill path, the drift repair, and the fallback if the triggers are ever
-- dropped in favour of scheduled refreshes.
CREATE FUNCTION pgkg_refresh_corpus_stats(p_namespace TEXT DEFAULT NULL)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM corpus_stats
    WHERE kind = 'proposition'
      AND (p_namespace IS NULL OR namespace = p_namespace);

    DELETE FROM lexeme_df
    WHERE kind = 'proposition'
      AND (p_namespace IS NULL OR namespace = p_namespace);

    INSERT INTO corpus_stats (namespace, kind, n_total, total_len)
    SELECT p.namespace, 'proposition', COUNT(*), COALESCE(SUM(p.doc_len), 0)
    FROM propositions p
    WHERE p.superseded_by IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace;

    INSERT INTO lexeme_df (namespace, kind, lexeme, df)
    SELECT p.namespace, 'proposition', u.lexeme, COUNT(*)
    FROM propositions p, unnest(p.tsv) AS u(lexeme, positions, weights)
    WHERE p.superseded_by IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace, u.lexeme;
END;
$$;


-- 5. Incremental maintenance for INSERT and DELETE.  The sign comes from the
-- trigger argument, so both events share this body; the transition table is
-- named delta_rows by both triggers.  Counters are clamped at zero so that a
-- delete of a row that predates the triggers cannot drive them negative.
CREATE FUNCTION pgkg_propositions_stats_delta() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    delta_sign INT := TG_ARGV[0]::INT;
BEGIN
    INSERT INTO corpus_stats AS cs (namespace, kind, n_total, total_len)
    SELECT d.namespace,
           'proposition',
           delta_sign * COUNT(*),
           delta_sign * COALESCE(SUM(d.doc_len), 0)
    FROM delta_rows d
    WHERE d.superseded_by IS NULL
    GROUP BY d.namespace
    ON CONFLICT (namespace, kind) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now();

    INSERT INTO lexeme_df AS ld (namespace, kind, lexeme, df)
    SELECT d.namespace, 'proposition', u.lexeme, delta_sign * COUNT(*)
    FROM delta_rows d, unnest(d.tsv) AS u(lexeme, positions, weights)
    WHERE d.superseded_by IS NULL
    GROUP BY d.namespace, u.lexeme
    ON CONFLICT (namespace, kind, lexeme) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;


-- 6. Incremental maintenance for UPDATE.  Needs both transition tables: a row
-- contributes -1 in its old shape and +1 in its new one, and a row whose
-- namespace, tsvector and active/superseded state are all unchanged
-- contributes nothing at all.  That last clause is what keeps the recall-path
-- access-count bump from touching these tables.
CREATE FUNCTION pgkg_propositions_stats_update() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    WITH delta AS (
        SELECT o.namespace, o.tsv, o.doc_len, -1 AS delta_sign
        FROM old_rows o
        WHERE o.superseded_by IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM new_rows n
              WHERE n.id = o.id
                AND n.namespace = o.namespace
                AND n.tsv = o.tsv
                AND (n.superseded_by IS NULL) = (o.superseded_by IS NULL)
          )

        UNION ALL

        SELECT n.namespace, n.tsv, n.doc_len, 1
        FROM new_rows n
        WHERE n.superseded_by IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM old_rows o
              WHERE o.id = n.id
                AND o.namespace = n.namespace
                AND o.tsv = n.tsv
                AND (o.superseded_by IS NULL) = (n.superseded_by IS NULL)
          )
    ),
    corpus AS (
        INSERT INTO corpus_stats AS cs (namespace, kind, n_total, total_len)
        SELECT d.namespace,
               'proposition',
               SUM(d.delta_sign),
               SUM(d.delta_sign * d.doc_len)
        FROM delta d
        GROUP BY d.namespace
        ON CONFLICT (namespace, kind) DO UPDATE
            SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
                total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
                updated_at = now()
        RETURNING 1
    )
    INSERT INTO lexeme_df AS ld (namespace, kind, lexeme, df)
    SELECT d.namespace, 'proposition', u.lexeme, SUM(d.delta_sign)
    FROM delta d, unnest(d.tsv) AS u(lexeme, positions, weights)
    GROUP BY d.namespace, u.lexeme
    ON CONFLICT (namespace, kind, lexeme) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;


CREATE TRIGGER pgkg_prop_stats_insert
    AFTER INSERT ON propositions
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_propositions_stats_delta('1');

CREATE TRIGGER pgkg_prop_stats_delete
    AFTER DELETE ON propositions
    REFERENCING OLD TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_propositions_stats_delta('-1');

CREATE TRIGGER pgkg_prop_stats_update
    AFTER UPDATE ON propositions
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_propositions_stats_update();


-- 7. Backfill whatever already exists.
SELECT pgkg_refresh_corpus_stats();


-- 8. Rewire the keyword arm onto the materialised statistics.  The BM25
-- formula, its constants and the output contract are untouched: only where
-- n_total, avgdl, df and the per-document length come from changes, so a
-- domain with maintained statistics ranks exactly as before.
CREATE OR REPLACE FUNCTION pgkg_bm25_candidates(
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
) sub
WHERE sub.bm25_score > 0.0
ORDER BY sub.bm25_score DESC
LIMIT k_initial;
$$;
