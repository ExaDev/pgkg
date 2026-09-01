-- Retrieval statistics get a tenant, a population and one pass over the score
-- (ADR 0001, D1, D4, D6, D8).
--
-- WHY THE STATISTICS DOMAIN GAINS AN ORG.  corpus_stats and lexeme_df were
-- keyed (namespace, kind) and read with a namespace match alone, so N, avgdl
-- and every term's document frequency were computed over every tenant's rows at
-- once: every tenant's Memory uses one namespace.  D4's second hard rule is
-- explicit that "ranking signals are never computed globally over shared
-- content ... a real cross-tenant inference channel", and this is that channel
-- in its plainest form — a tenant measures how much other tenants have written
-- about a term by watching its own scores move, with no access to a single row.
--
-- The key becomes typed: (kind, namespace, org_id, collection_id).  031 keyed
-- the chunk domain as the TEXT 'collection:<uuid>' inside the namespace column
-- to avoid re-declaring 011's and 021's three maintenance functions in a
-- migration about retrieval; that deferral is what left the proposition half
-- without an org, so this migration takes the rework instead of layering a
-- second string convention on top.  `namespace` keeps its meaning — the
-- proposition namespace — and is empty for the chunk domain, whose retrieval
-- scope has no namespace axis to split on.
--
-- THE RULE THE READ PATH NOW FOLLOWS: the statistics filter mirrors the
-- candidate scan's scope filter, term for term.  Both arms restrict on
-- org_id and collection_id exactly as pgkg_visible() does, and NULL means
-- unrestricted in both places.  A statistics domain wider than the scan is the
-- leak above; a domain narrower than the scan is the population defect below.
-- Because BM25 statistics are additive, "mirror the scan" is a SUM over the
-- domains in scope rather than a different key per query shape.
--
-- WHY CHUNK RETRIEVABILITY BECOMES A STORED COLUMN.  The chunk statistics were
-- maintained over every row of `chunks`, but retrieval only returns rows
-- passing pgkg_chunk_live(): chat-provenance chunks (which memory.py writes on
-- the default path, into the same collection corpus documents land in), chunks
-- linked only to retired versions, and chunks of soft-deleted documents all
-- moved the ranking of passages they can never appear beside.  21 chat turns
-- collapsed the only retrievable passage's score by 10.5x.  021 already stated
-- the principle for the proposition half — "the maintenance predicate has to be
-- the retrieval predicate" — and the reason the chunk half could not follow it
-- is timing: a passage is not yet linked to its version when its INSERT
-- statement ends, so no predicate evaluated at insert time can classify it.
--
-- So retrievability is derived state, stored on the row and reconciled at every
-- transition that can change it, in the same spirit as doc_len (011) and
-- refcount (030): the statistics population and the scan predicate then read
-- one column that cannot disagree with itself.  Reconciliation, not a signed
-- delta, is what makes the trigger set idempotent — a chunk inserted and linked
-- inside one statement is visited twice and counted once.
--
-- It also takes the liveness test off the hot path.  pgkg_chunk_live() contains
-- two sublinks, and a SQL function whose body has a sublink can never be
-- inlined, so the predicate ran as a nested-loop subquery once per candidate
-- chunk — 28,937 shared buffers against 927 for the same test written inline,
-- and ~950 ms added to a single keyword arm at 40,000 candidates.  A column
-- comparison costs nothing and is what the arms below use.  pgkg_chunk_live()
-- is kept, reading the same column, because 040's graph arm calls it.
--
-- WHY THE QUOTA BUCKET STOPS KEYING ON collections.kind.  031 labelled a
-- candidate 'corpus' only when collections.kind = 'corpus'.  The vocabulary is
-- ('chat','corpus','mixed') and the reserved DEFAULT collection — where
-- POST /documents, Memory and CorpusIngest all land when no collection is named
-- — is 'mixed'.  So on the default path every passage was bucketed 'memory',
-- D1's 60% corpus ceiling and its per-claim-scope split did not apply to it,
-- and 200 passages took all 64 reranker slots from 20 personal facts.  That is
-- D1's failure-mode blockquote verbatim ("the corpus drowns the memory ... the
-- assistant stops remembering you and starts quoting the handbook"), live in
-- the default configuration, with the 8-row memory floor never engaging.
--
-- The class of blind spot matters more than the missing value.  Keying a
-- safety limit on a vocabulary column means every value added to that
-- vocabulary later silently reclassifies rows, and silently in the unprotected
-- direction; the existing quota tests all missed it because each one builds an
-- explicit kind='corpus' collection, so the default path was never exercised.
-- The bucket therefore keys on structure that has no vocabulary to extend:
--
--   * a chunk is corpus material.  D1 keeps two physical stores precisely
--     because passages and facts are different kinds of thing, and the volume
--     asymmetry the ceiling exists to bound ("600k org chunks against a user's
--     4k personal facts") is the chunk store.  A retrievable chunk is a
--     document passage by construction: chat provenance is not retrievable at
--     all.  No collection kind can turn a passage into a personal memory.
--   * a proposition is memory unless it is corpus-derived — extracted under
--     D2's per-collection opt-in, or citing (D5's derivation edge) a chunk that
--     a document version carries.  A fact extracted from the handbook is
--     corpus material however it is stored, which is 031's point and is kept;
--     a fact extracted from a chat turn cites a chunk no version carries.
--
-- claim_scope is untouched: it remains the row's own where a row has one and
-- the collection's otherwise, and stays the axis the corpus allowance is split
-- along (D4: claim_scope "governs decay profile, contradiction partitioning and
-- retrieval quotas").  Bucketing on claim_scope instead was rejected: a
-- user-scoped passage would then be exempt from the ceiling and a chat fact
-- about the org would compete against the handbook for corpus slots, which
-- inverts both halves of D1's protection.
--
-- WHY THE PRIMARY VECTOR ARM FILTERS ON GENERATION (D8).  Cosines from two
-- model spaces are incomparable, and a stale vector is still a vector: it
-- returns a number rather than an error and sorts among the rest.  During a
-- cutover the retiring generation's rows are still inline, so the arm ranked
-- them against the new space's query vector — and the same row got a second,
-- correctly-scoped vote through pgkg_generation_candidates, double-counting it
-- in RRF.  memory.py's MMR query already nulls an embedding whose generation is
-- not the org's primary; this is the same guard, one stage earlier.
-- The restriction applies only when a primary generation is actually visible,
-- so a role or scope that cannot read org_embedders degrades to the behaviour
-- it has today rather than losing the arm.
--
-- WHY THE SCORE IS AGGREGATED RATHER THAN CORRELATED.  `... ) sub WHERE
-- sub.bm25_score > 0 ORDER BY sub.bm25_score` over a correlated sub-select made
-- Postgres plan the per-row aggregate twice, once for the filter and once for
-- the sort: two SubPlans, each with loops = the candidate count.  The scoring
-- is a sum over each candidate's matching lexemes, which is a join and a GROUP
-- BY — one hash aggregate for the whole candidate set instead of one aggregate
-- per row, evaluated once.  The inner join to the IDF terms also subsumes the
-- `> 0` filter: RSJ IDF with the +1 inside LN() is strictly positive, so a
-- candidate scores zero exactly when no query lexeme matched it, and such a
-- candidate now produces no group at all.
--
-- WHY IDF READS ITS OWN TERMS.  011 removed the per-row correlated COUNT(*) but
-- left the lexeme restriction as a downstream join, so the term_df aggregate
-- read the entire vocabulary of the corpus — 4,009 rows for a three-term query,
-- growing with the corpus, 17.3 ms per query at 40k propositions.  Aggregating
-- the query's lexemes into one array puts them in the scan's own qualifier,
-- where the primary key can serve them.


-- 1. The statistics domain becomes typed.
--
-- The rows are discarded rather than backfilled: the split changes every count,
-- so there is nothing in the old rows to carry over, and both refresh functions
-- rebuild from the content tables at the end of this migration.  That is the
-- same escape hatch 011 shipped for drift repair.
ALTER TABLE corpus_stats
    ADD COLUMN org_id        UUID,
    ADD COLUMN collection_id UUID;

ALTER TABLE lexeme_df
    ADD COLUMN org_id        UUID,
    ADD COLUMN collection_id UUID;

DELETE FROM corpus_stats;
DELETE FROM lexeme_df;

ALTER TABLE corpus_stats
    ALTER COLUMN org_id SET NOT NULL,
    ALTER COLUMN collection_id SET NOT NULL,
    ALTER COLUMN namespace SET DEFAULT '',
    DROP CONSTRAINT corpus_stats_pkey,
    ADD PRIMARY KEY (kind, namespace, org_id, collection_id);

-- The lexeme is third in the key, not last, because the read path knows the
-- kind, the namespace and its own terms on every query but may leave either
-- partition key unrestricted.
ALTER TABLE lexeme_df
    ALTER COLUMN org_id SET NOT NULL,
    ALTER COLUMN collection_id SET NOT NULL,
    ALTER COLUMN namespace SET DEFAULT '',
    DROP CONSTRAINT lexeme_df_pkey,
    ADD PRIMARY KEY (kind, namespace, lexeme, org_id, collection_id);

COMMENT ON COLUMN corpus_stats.namespace IS
    'The proposition namespace, and empty for the chunk domain: a chunk''s '
    'retrieval scope is org and collection and has no namespace axis '
    '(ADR 0001, D4).';

COMMENT ON COLUMN lexeme_df.namespace IS
    'The proposition namespace, and empty for the chunk domain: a chunk''s '
    'retrieval scope is org and collection and has no namespace axis '
    '(ADR 0001, D4).';


-- 2. Chunk retrievability, stored.
ALTER TABLE chunks
    ADD COLUMN retrievable BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN chunks.retrievable IS
    'Whether retrieval may see this chunk: derived state, maintained by '
    'trigger from the document lifecycle. It is both the scan predicate and '
    'the chunk statistics population, so the two cannot disagree '
    '(ADR 0001, D1 and D6).';

-- The one statement of what liveness means.  Not on the read path — the
-- read path reads the column — so the two sublinks that stop it inlining
-- cost nothing here.
CREATE FUNCTION pgkg_chunk_retrievable(
    p_chunk_id    UUID,
    p_document_id UUID,
    p_refcount    INT
) RETURNS BOOLEAN
LANGUAGE SQL STABLE
AS $$
    SELECT (p_refcount = 0 AND p_document_id IS NULL)
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

UPDATE chunks c
SET retrievable = pgkg_chunk_retrievable(c.id, c.document_id, c.refcount);


-- 3. Reconciliation: bring a set of chunks' flag up to date and move the
-- statistics by exactly the rows whose retrievability changed.
--
-- One function for both halves, in one statement, because a flag updated
-- without its delta — or a delta applied twice — is drift in the tables the
-- ranking reads.  Only the rows that flipped are touched, so the function is
-- idempotent and the trigger set below can visit a chunk any number of times.
CREATE FUNCTION pgkg_chunk_retrievability_sync(p_chunk_ids UUID[])
RETURNS VOID
LANGUAGE SQL
AS $$
WITH target AS (
    SELECT c.id, c.org_id, c.collection_id, c.doc_len, c.tsv,
           c.retrievable AS was_retrievable,
           pgkg_chunk_retrievable(c.id, c.document_id, c.refcount)
               AS is_retrievable
    FROM chunks c
    WHERE c.id = ANY(p_chunk_ids)
),
flipped AS (
    SELECT t.*, CASE WHEN t.is_retrievable THEN 1 ELSE -1 END AS delta_sign
    FROM target t
    WHERE t.is_retrievable IS DISTINCT FROM t.was_retrievable
),
marked AS (
    UPDATE chunks c
    SET retrievable = f.is_retrievable
    FROM flipped f
    WHERE c.id = f.id
    RETURNING c.id
),
totals AS (
    INSERT INTO corpus_stats AS cs
        (kind, namespace, org_id, collection_id, n_total, total_len)
    SELECT 'chunk', '', f.org_id, f.collection_id,
           SUM(f.delta_sign), SUM(f.delta_sign * f.doc_len)
    FROM flipped f
    GROUP BY f.org_id, f.collection_id
    ON CONFLICT (kind, namespace, org_id, collection_id) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now()
    RETURNING 1
)
INSERT INTO lexeme_df AS ld
    (kind, namespace, lexeme, org_id, collection_id, df)
SELECT 'chunk', '', u.lexeme, f.org_id, f.collection_id, SUM(f.delta_sign)
FROM flipped f, unnest(f.tsv) AS u(lexeme, positions, weights)
GROUP BY u.lexeme, f.org_id, f.collection_id
ON CONFLICT (kind, namespace, lexeme, org_id, collection_id) DO UPDATE
    SET df = GREATEST(ld.df + EXCLUDED.df, 0);
$$;


-- 4. The transitions.  Statement-level with transition tables, in the shape
-- 011 established: one set-based reconciliation per statement, never one per
-- row.  A chunk becomes retrievable, or stops being retrievable, only here:
--
--   chunks INSERT                   a chunk that names no document is its own
--                                   passage and is retrievable at once; one
--                                   that names a document is not, until linked.
--   document_version_chunks         the link that carries a passage, and the
--                                   purge that drops it (the version-side
--                                   cascade fires this DELETE).
--   document_versions               promotion and retirement: the same chunks,
--                                   a different answer.
--   documents                       soft delete and its reversal.
--   chunks UPDATE                   a chunk that acquires a document becomes
--                                   that document's provenance.
--
-- The two lifecycle triggers carry no column list because Postgres forbids a
-- transition table on a trigger that has one; reconciliation makes the extra
-- statements they see a no-op rather than a wrong delta.
--
-- The chunks UPDATE trigger is narrowed to document_id for two reasons.  It is
-- the only column on the row that changes the answer and is not already covered
-- by the link trigger above — refcount moves only when a link does — and
-- refcount maintenance is the highest-volume write on this table, which 031
-- required not to reach the statistics at all.  The empty-array guard is what
-- terminates the recursion the reconciliation would otherwise start against
-- itself: its own UPDATE touches `retrievable` alone, so the second invocation
-- finds nothing to reconcile and stops.  A chunk moved between collections is
-- still outside this: 031 named that unsupported, and pgkg_refresh_chunk_stats()
-- repairs it.
CREATE FUNCTION pgkg_chunks_retrievability_update() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_ids UUID[];
BEGIN
    SELECT array_agg(DISTINCT n.id) INTO v_ids
    FROM new_rows n
    JOIN old_rows o ON o.id = n.id
    WHERE n.document_id IS DISTINCT FROM o.document_id;

    IF v_ids IS NOT NULL THEN
        PERFORM pgkg_chunk_retrievability_sync(v_ids);
    END IF;

    RETURN NULL;
END;
$$;

CREATE FUNCTION pgkg_chunks_retrievability_insert() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pgkg_chunk_retrievability_sync(
        (SELECT array_agg(DISTINCT d.id) FROM delta_rows d)
    );
    RETURN NULL;
END;
$$;

CREATE FUNCTION pgkg_version_chunks_retrievability() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pgkg_chunk_retrievability_sync(
        (SELECT array_agg(DISTINCT d.chunk_id) FROM delta_rows d)
    );
    RETURN NULL;
END;
$$;

-- The two lifecycle tables share a body: both transition tables expose the
-- parent id as `id`, and only the path down to the chunk differs.
CREATE FUNCTION pgkg_lifecycle_retrievability() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_TABLE_NAME = 'document_versions' THEN
        PERFORM pgkg_chunk_retrievability_sync((
            SELECT array_agg(DISTINCT dvc.chunk_id)
            FROM delta_rows d
            JOIN document_version_chunks dvc
                 ON dvc.document_version_id = d.id
        ));
    ELSE
        PERFORM pgkg_chunk_retrievability_sync((
            SELECT array_agg(DISTINCT dvc.chunk_id)
            FROM delta_rows d
            JOIN document_versions dv ON dv.document_id = d.id
            JOIN document_version_chunks dvc
                 ON dvc.document_version_id = dv.id
        ));
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER pgkg_chunk_retrievability_insert
    AFTER INSERT ON chunks
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_chunks_retrievability_insert();

CREATE TRIGGER pgkg_chunk_retrievability_update
    AFTER UPDATE ON chunks
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_chunks_retrievability_update();

CREATE TRIGGER pgkg_version_chunks_retrievability_insert
    AFTER INSERT ON document_version_chunks
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_retrievability();

CREATE TRIGGER pgkg_version_chunks_retrievability_delete
    AFTER DELETE ON document_version_chunks
    REFERENCING OLD TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_retrievability();

CREATE TRIGGER pgkg_versions_retrievability_update
    AFTER UPDATE ON document_versions
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_lifecycle_retrievability();

CREATE TRIGGER pgkg_documents_retrievability_update
    AFTER UPDATE ON documents
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_lifecycle_retrievability();


-- 5. The chunk statistics triggers on the chunks table itself.  INSERT is now
-- the reconciliation above; DELETE is the only remaining signed delta, and it
-- subtracts exactly the rows that were counted.
DROP TRIGGER pgkg_chunk_stats_insert ON chunks;
DROP TRIGGER pgkg_chunk_stats_delete ON chunks;

CREATE OR REPLACE FUNCTION pgkg_chunks_stats_delta() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO corpus_stats AS cs
        (kind, namespace, org_id, collection_id, n_total, total_len)
    SELECT 'chunk', '', d.org_id, d.collection_id,
           -COUNT(*), -COALESCE(SUM(d.doc_len), 0)
    FROM delta_rows d
    WHERE d.retrievable
    GROUP BY d.org_id, d.collection_id
    ON CONFLICT (kind, namespace, org_id, collection_id) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now();

    INSERT INTO lexeme_df AS ld
        (kind, namespace, lexeme, org_id, collection_id, df)
    SELECT 'chunk', '', u.lexeme, d.org_id, d.collection_id, -COUNT(*)
    FROM delta_rows d, unnest(d.tsv) AS u(lexeme, positions, weights)
    WHERE d.retrievable
    GROUP BY u.lexeme, d.org_id, d.collection_id
    ON CONFLICT (kind, namespace, lexeme, org_id, collection_id) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;

CREATE TRIGGER pgkg_chunk_stats_delete
    AFTER DELETE ON chunks
    REFERENCING OLD TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_chunks_stats_delta();


-- 6. Backfill and drift repair for the chunk half.  The flag is repaired first,
-- from the same predicate the triggers reconcile against, so one call fixes
-- both a drifted flag and drifted statistics.
CREATE OR REPLACE FUNCTION pgkg_refresh_chunk_stats(p_collection_id UUID DEFAULT NULL)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE chunks c
    SET retrievable = pgkg_chunk_retrievable(c.id, c.document_id, c.refcount)
    WHERE (p_collection_id IS NULL OR c.collection_id = p_collection_id)
      AND c.retrievable IS DISTINCT FROM
          pgkg_chunk_retrievable(c.id, c.document_id, c.refcount);

    DELETE FROM corpus_stats
    WHERE kind = 'chunk'
      AND (p_collection_id IS NULL OR collection_id = p_collection_id);

    DELETE FROM lexeme_df
    WHERE kind = 'chunk'
      AND (p_collection_id IS NULL OR collection_id = p_collection_id);

    INSERT INTO corpus_stats
        (kind, namespace, org_id, collection_id, n_total, total_len)
    SELECT 'chunk', '', c.org_id, c.collection_id,
           COUNT(*), COALESCE(SUM(c.doc_len), 0)
    FROM chunks c
    WHERE c.retrievable
      AND (p_collection_id IS NULL OR c.collection_id = p_collection_id)
    GROUP BY c.org_id, c.collection_id;

    INSERT INTO lexeme_df
        (kind, namespace, lexeme, org_id, collection_id, df)
    SELECT 'chunk', '', u.lexeme, c.org_id, c.collection_id, COUNT(*)
    FROM chunks c, unnest(c.tsv) AS u(lexeme, positions, weights)
    WHERE c.retrievable
      AND (p_collection_id IS NULL OR c.collection_id = p_collection_id)
    GROUP BY u.lexeme, c.org_id, c.collection_id;
END;
$$;


-- 7. The proposition half, on the typed key.  The three bodies are 021's with
-- the domain widened and nothing else touched: the population predicate stays
-- `invalidated_at IS NULL`, which is still the retrieval predicate.
CREATE OR REPLACE FUNCTION pgkg_refresh_corpus_stats(p_namespace TEXT DEFAULT NULL)
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

    INSERT INTO corpus_stats
        (kind, namespace, org_id, collection_id, n_total, total_len)
    SELECT 'proposition', p.namespace, p.org_id, p.collection_id,
           COUNT(*), COALESCE(SUM(p.doc_len), 0)
    FROM propositions p
    WHERE p.invalidated_at IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace, p.org_id, p.collection_id;

    INSERT INTO lexeme_df
        (kind, namespace, lexeme, org_id, collection_id, df)
    SELECT 'proposition', p.namespace, u.lexeme, p.org_id, p.collection_id,
           COUNT(*)
    FROM propositions p, unnest(p.tsv) AS u(lexeme, positions, weights)
    WHERE p.invalidated_at IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace, u.lexeme, p.org_id, p.collection_id;
END;
$$;


CREATE OR REPLACE FUNCTION pgkg_propositions_stats_delta() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    delta_sign INT := TG_ARGV[0]::INT;
BEGIN
    INSERT INTO corpus_stats AS cs
        (kind, namespace, org_id, collection_id, n_total, total_len)
    SELECT 'proposition', d.namespace, d.org_id, d.collection_id,
           delta_sign * COUNT(*),
           delta_sign * COALESCE(SUM(d.doc_len), 0)
    FROM delta_rows d
    WHERE d.invalidated_at IS NULL
    GROUP BY d.namespace, d.org_id, d.collection_id
    ON CONFLICT (kind, namespace, org_id, collection_id) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now();

    INSERT INTO lexeme_df AS ld
        (kind, namespace, lexeme, org_id, collection_id, df)
    SELECT 'proposition', d.namespace, u.lexeme, d.org_id, d.collection_id,
           delta_sign * COUNT(*)
    FROM delta_rows d, unnest(d.tsv) AS u(lexeme, positions, weights)
    WHERE d.invalidated_at IS NULL
    GROUP BY d.namespace, u.lexeme, d.org_id, d.collection_id
    ON CONFLICT (kind, namespace, lexeme, org_id, collection_id) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;


CREATE OR REPLACE FUNCTION pgkg_propositions_stats_update() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    WITH delta AS (
        SELECT o.namespace, o.org_id, o.collection_id, o.tsv, o.doc_len,
               -1 AS delta_sign
        FROM old_rows o
        WHERE o.invalidated_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM new_rows n
              WHERE n.id = o.id
                AND n.namespace = o.namespace
                AND n.org_id = o.org_id
                AND n.collection_id = o.collection_id
                AND n.tsv = o.tsv
                AND (n.invalidated_at IS NULL) = (o.invalidated_at IS NULL)
          )

        UNION ALL

        SELECT n.namespace, n.org_id, n.collection_id, n.tsv, n.doc_len, 1
        FROM new_rows n
        WHERE n.invalidated_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM old_rows o
              WHERE o.id = n.id
                AND o.namespace = n.namespace
                AND o.org_id = n.org_id
                AND o.collection_id = n.collection_id
                AND o.tsv = n.tsv
                AND (o.invalidated_at IS NULL) = (n.invalidated_at IS NULL)
          )
    ),
    corpus AS (
        INSERT INTO corpus_stats AS cs
            (kind, namespace, org_id, collection_id, n_total, total_len)
        SELECT 'proposition', d.namespace, d.org_id, d.collection_id,
               SUM(d.delta_sign), SUM(d.delta_sign * d.doc_len)
        FROM delta d
        GROUP BY d.namespace, d.org_id, d.collection_id
        ON CONFLICT (kind, namespace, org_id, collection_id) DO UPDATE
            SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
                total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
                updated_at = now()
        RETURNING 1
    )
    INSERT INTO lexeme_df AS ld
        (kind, namespace, lexeme, org_id, collection_id, df)
    SELECT 'proposition', d.namespace, u.lexeme, d.org_id, d.collection_id,
           SUM(d.delta_sign)
    FROM delta d, unnest(d.tsv) AS u(lexeme, positions, weights)
    GROUP BY d.namespace, u.lexeme, d.org_id, d.collection_id
    ON CONFLICT (kind, namespace, lexeme, org_id, collection_id) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;


-- 8. Rebuild both domains on the new key.
SELECT pgkg_refresh_corpus_stats();
SELECT pgkg_refresh_chunk_stats();


-- 9. Liveness now reads the flag, so 040's graph arm pays one lookup where it
-- paid four, and there is one statement of what liveness means rather than two
-- that can drift apart.  p_refcount is retained in the signature: it is part of
-- the identity every caller compiled against, and the column it summarises is
-- an input to the predicate the flag was computed from.
CREATE OR REPLACE FUNCTION pgkg_chunk_live(p_chunk_id UUID, p_refcount INT)
RETURNS BOOLEAN
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM chunks c WHERE c.id = p_chunk_id AND c.retrievable
    )
$$;

COMMENT ON FUNCTION pgkg_chunk_live(UUID, INT) IS
    'Whether retrieval may see a chunk, read from the maintained '
    'chunks.retrievable flag. A chunk carried by the current version of a live '
    'document is a passage; so is a chunk that belongs to no document at all. '
    'A chunk that belongs to a document but to no version is that document''s '
    'provenance — which is what chat ingest writes — and is not retrievable '
    '(ADR 0001, D1).';


-- 10. The keyword arm.
DROP FUNCTION pgkg_bm25_candidates(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
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

-- The same lexemes as one array, so the document-frequency lookup can name
-- them in its own qualifier instead of joining to them afterwards.
query_terms AS (
    SELECT COALESCE(array_agg(lexeme), ARRAY[]::TEXT[]) AS terms
    FROM query_lexemes
),

-- OR-joined match query: string_agg over zero lexemes yields NULL, and
-- to_tsquery(NULL) is NULL, so an empty query matches nothing rather than
-- raising.
query_or AS (
    SELECT to_tsquery('simple', string_agg(lexeme, ' | ')) AS q
    FROM query_lexemes
),

-- One row, always: an un-grouped aggregate over no input rows returns the
-- fallback n_total = 1, avgdl = 1.0, which flattens IDF to a constant and
-- leaves ranking on term frequency alone — degraded but sane, and never an
-- error.  Summed rather than averaged because BM25 statistics are additive
-- across sub-corpora, which is what lets one key serve a query spanning
-- several collections.  The scope filter is pgkg_visible()'s two partition
-- terms, so the population these numbers describe is the population the scan
-- below reads.
stats AS (
    SELECT
        GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8 AS n_total,
        GREATEST(
            COALESCE(SUM(cs.total_len), 0)::FLOAT8
            / GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8,
            1.0
        ) AS avgdl
    FROM corpus_stats cs
    WHERE cs.kind = CASE p_source
                        WHEN 'propositions' THEN 'proposition'
                        WHEN 'chunks'       THEN 'chunk'
                    END
      AND cs.namespace = CASE p_source WHEN 'chunks' THEN '' ELSE p_namespace END
      AND (p_org_ids IS NULL OR cs.org_id = ANY(p_org_ids))
      AND (p_collection_ids IS NULL OR cs.collection_id = ANY(p_collection_ids))
),

-- Document frequency for the query's own terms, summed over the same domains.
term_df AS (
    SELECT ld.lexeme, SUM(ld.df)::FLOAT8 AS df
    FROM lexeme_df ld
    CROSS JOIN query_terms qt
    WHERE ld.kind = CASE p_source
                        WHEN 'propositions' THEN 'proposition'
                        WHEN 'chunks'       THEN 'chunk'
                    END
      AND ld.namespace = CASE p_source WHEN 'chunks' THEN '' ELSE p_namespace END
      AND ld.lexeme = ANY(qt.terms)
      AND (p_org_ids IS NULL OR ld.org_id = ANY(p_org_ids))
      AND (p_collection_ids IS NULL OR ld.collection_id = ANY(p_collection_ids))
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
),

-- The two stores, one signature.  p_source is a constant at every call site, so
-- the planner folds the unused branch away and the proposition path keeps the
-- plan shape it has.  The chunk branch carries no session and no validity
-- interval: a passage is not asserted in a conversation and has no validity
-- clock of its own.  Its temporal filter is the document lifecycle, which
-- chunks.retrievable carries.
candidates AS (
    SELECT p.id AS cand_id, p.tsv AS tsv, p.doc_len AS doc_len
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

    SELECT c.id, c.tsv, c.doc_len
    FROM chunks c
    CROSS JOIN query_or
    WHERE p_source = 'chunks'
      AND q_text IS NOT NULL
      AND q_text <> ''
      AND query_or.q IS NOT NULL
      AND c.tsv @@ query_or.q
      AND c.retrievable
      AND pgkg_visible(
            c.org_id, c.collection_id, c.visibility,
            c.owner_user_id, c.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
),

-- One aggregate for the whole candidate set.  The join to idf is what selects
-- the scoring terms, and it is an inner join: a candidate no query lexeme
-- reached produces no group, which is the `score > 0` filter the old shape
-- applied afterwards.
scored AS (
    SELECT
        cd.cand_id,
        SUM(
            i.idf_val
            * (COALESCE(array_length(u.positions, 1), 0)::FLOAT8 * 2.2)
            / (COALESCE(array_length(u.positions, 1), 0)::FLOAT8
               + 1.2 * (1.0 - 0.75 + 0.75 * cd.doc_len::FLOAT8 / s.avgdl))
        ) AS bm25_score
    FROM candidates cd
    CROSS JOIN stats s
    CROSS JOIN LATERAL unnest(cd.tsv) AS u(lexeme, positions, weights)
    JOIN idf i ON i.lexeme = u.lexeme
    GROUP BY cd.cand_id
)

SELECT
    sc.cand_id,
    'kw'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sc.bm25_score DESC))::INT,
    sc.bm25_score::REAL
FROM scored sc
WHERE sc.bm25_score > 0.0
ORDER BY sc.bm25_score DESC
LIMIT k_initial;
$$;


-- 11. The vector arm, inside one model space (D8).
DROP FUNCTION pgkg_vector_candidates(
    halfvec, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
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
WITH

-- The model space this arm is entitled to compare in: the primary generation
-- of every org in scope.  Aggregated into one array so the restriction reaches
-- the index scan's own filter rather than becoming a join the ordered scan
-- cannot push its LIMIT through.  NULL — no primary generation visible at all,
-- which is what a role that cannot read org_embedders sees — leaves the arm
-- unrestricted rather than empty.
primary_space AS (
    SELECT array_agg(oe.generation_id) AS generations
    FROM org_embedders oe
    WHERE oe.role = 'primary'
      AND (p_org_ids IS NULL OR oe.org_id = ANY(p_org_ids))
)

SELECT
    sub.cand_id,
    'vec'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sub.distance))::INT,
    (1.0 - sub.distance)::REAL
FROM (
    (
    SELECT p.id AS cand_id, (p.embedding <=> q_embedding) AS distance
    FROM propositions p
    CROSS JOIN primary_space ps
    WHERE p_source = 'propositions'
      AND q_embedding IS NOT NULL
      AND p.embedding IS NOT NULL
      AND p.namespace = p_namespace
      AND (
            ps.generations IS NULL
            OR p.embedder_generation_id = ANY(ps.generations)
          )
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
    CROSS JOIN primary_space ps
    WHERE p_source = 'chunks'
      AND q_embedding IS NOT NULL
      AND c.embedding IS NOT NULL
      AND c.retrievable
      AND (
            ps.generations IS NULL
            OR c.embedder_generation_id = ANY(ps.generations)
          )
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


-- 12. What a candidate is, for the stage that has to treat both stores alike.
--
-- The item ids of the two stores are disjoint, so a fused candidate set does
-- not need to carry its provenance through fusion — it can be resolved.  That
-- is what keeps pgkg_candidate at four columns and leaves every function
-- written against it, pgkg_fuse() included, untouched.
CREATE OR REPLACE FUNCTION pgkg_item_scope(p_item_ids UUID[])
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
        CASE
            WHEN col.kind = 'corpus' THEN 'corpus'
            WHEN EXISTS (
                SELECT 1 FROM document_version_chunks dvc
                WHERE dvc.chunk_id = p.chunk_id
            ) THEN 'corpus'
            ELSE 'memory'
        END,
        p.claim_scope
    FROM propositions p
    JOIN collections col ON col.id = p.collection_id
    WHERE p.id = ANY(p_item_ids)

    UNION ALL

    SELECT
        c.id,
        'chunks'::TEXT,
        c.collection_id,
        'corpus'::TEXT,
        col.claim_scope
    FROM chunks c
    JOIN collections col ON col.id = c.collection_id
    WHERE c.id = ANY(p_item_ids)
$$;

COMMENT ON FUNCTION pgkg_item_scope(UUID[]) IS
    'Resolves a fused candidate to its store, its collection, its quota bucket '
    'and its claim scope. The bucket keys on structure, never on '
    'collections.kind: a retrievable chunk is a document passage, and a '
    'proposition is corpus material when it was extracted under D2''s opt-in '
    'or cites a passage a document version carries (ADR 0001, D1 and D4).';


-- 13. The chunk statistics domain is a pair of typed columns now, so the
-- rendered domain key has nothing left to render.
DROP FUNCTION pgkg_stats_domain(UUID);
