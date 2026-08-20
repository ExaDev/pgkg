-- Provenance as a shared, append-only derivation record, and the four clocks
-- that replace `superseded_by IS NULL` (ADR 0001, D5 and D6).
--
-- WHY PROVENANCE IS A TABLE AND A COLUMN, NOT A JSONB BLOB.  Every operation
-- that matters is a set-based query over it: a source changes and the facts
-- derived from it must be withdrawn; an extractor deploy goes wrong and one
-- batch must be undone; a prompt improves and everything extracted by the old
-- one must be re-run.  With provenance_id on propositions each of those is one
-- indexed UPDATE.  Without it, each is a traversal, which is why the column is
-- NOT NULL: a nullable hot-path key means a fallback scan for every cascade.
--
-- ONE ROW PER CHUNK-AND-MODEL-RUN, so provenance cardinality tracks chunk
-- count and not fact count.  proposition_provenance carries the many-to-one
-- that arrives with deduplication — the same claim extracted from forty chats
-- is one proposition with forty sources, and forty sources is the signal that
-- should raise confidence.  It also gives erasure for free: remove a user's
-- derivation records and a proposition left with none is the caller's to
-- delete, while one still corroborated by another source survives.
--
-- APPEND-ONLY IS ENFORCED, NOT DOCUMENTED.  How a fact was derived is the
-- basis of every citation already emitted; rewriting it makes those citations
-- retrospectively false.  UPDATE is refused outright.  DELETE is refused too,
-- except through pgkg_erase_provenance(), which sets a transaction-local GUC
-- the trigger looks for — subject erasure has to be possible, but it has to be
-- deliberate and it has to be one identifiable call site.
--
-- WHY THE FILTER MOVES OFF superseded_by.  Retiring a document version has to
-- withdraw every proposition derived from the chunks it dropped, in bulk.  A
-- self-referencing UUID cannot be set in bulk to anything meaningful — hence
-- today's `superseded_by = id` idiom, which says "replaced by itself".  One
-- nullable timestamp can, it supports a cheap partial index, and it carries an
-- enumerated reason so the audit trail distinguishes a supersession from a TTL
-- expiry from a retracted run.  superseded_by stays exactly what its name says:
-- the pointer to the replacement.  It is the reason, not the filter.
--
-- The two are kept in step by a BEFORE ROW trigger rather than by convention,
-- because a caller that sets one and not the other produces a row that is
-- either invisible with no recorded reason or visible after being replaced.
-- The trigger's WHEN clause fires only when superseded_by actually moves, so
-- the access-count flush — the highest-volume UPDATE on this table — never
-- pays for it.
--
-- THREE MODES, DELIBERATELY SEPARATE.  Current state is the default and is
-- served by the partial index on invalidated_at IS NULL.  As-of validity is the
-- same belief at a different world instant, so it threads a timestamp through
-- the existing arms.  As-of belief cannot use the partial index and defeats
-- HNSW pre-filtering, so it is a different function — pgkg_believed_at() — with
-- no vector arm and no ranking at all.  Making it a parameter of pgkg_search()
-- would make it one mistaken argument away from being the hot path.
--
-- IDF IS NOT BITEMPORAL.  corpus_stats and lexeme_df are maintained against
-- invalidated_at, so they follow belief but not validity: an as-of-validity
-- query ranks against today's document frequencies.  Making the statistics
-- time-travel would mean storing a df series per lexeme, and the error it
-- avoids is a second-order change to a logarithm.
--
-- WHAT IS RE-DECLARED, AND WHY THERE IS NO SMALLER UNIT.  A parameter list is
-- part of a function's identity, so threading p_valid_at through the three
-- candidate sources and pgkg_search() means DROP then CREATE; CREATE OR REPLACE
-- would leave the old overload beside the new one and make every existing call
-- ambiguous.  The bodies are carried over verbatim from the migration that last
-- defined them with the active predicate swapped and nothing else touched.  The
-- three statistics functions from 011 are replaced for the same reason in
-- reverse: the predicate is embedded in a plpgsql body, so the body is the
-- smallest unit that can be altered.  The predicate itself lives in exactly one
-- place — pgkg_temporal_visible() — so there is one definition to change next
-- time, not four.
--
-- NOT HERE, DELIBERATELY.  chunks take provenance_id but not the temporal
-- columns: retrieval filters propositions, and chunk lifecycle belongs with
-- document_versions and reference counting (D6), which land with the corpus.
-- The three decay profiles are a ranking change, not a lifecycle one, and stay
-- with pgkg_apply_profile().  Physical GC of expired rows is a separate sweeper:
-- pgkg_expire_due() withdraws them from belief, which is what retrieval reads,
-- and leaves reclaiming the space to a job that can respect the HNSW rebuild
-- cost.


-- 1. The reserved derivation record.  Every pre-provenance row needs a source
-- and does not have one; naming that honestly beats back-dating a plausible
-- lie, and it makes "which facts predate provenance" an indexed query.  It
-- belongs to the operator's system org because it is schema metadata, not any
-- tenant's content.
CREATE FUNCTION pgkg_unattributed_provenance() RETURNS UUID
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$ SELECT '00000000-0000-0000-0000-000000000003'::UUID $$;


CREATE TABLE provenance (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id         UUID NOT NULL REFERENCES orgs(id),
    kind           TEXT NOT NULL,
    source_id      UUID,
    source_locator JSONB,
    producer       TEXT NOT NULL,
    producer_model TEXT,
    prompt_version TEXT,
    ingest_run_id  UUID,
    -- No ON DELETE action: a referential action is an UPDATE or a DELETE on
    -- this table, and both are refused.  Deleting an actor therefore fails
    -- until their derivation records are erased, which is D5's erasure order
    -- rather than an obstacle to it.
    actor_user_id  UUID REFERENCES users(id),
    source_url     TEXT,
    publisher      TEXT,
    published_at   TIMESTAMPTZ,
    retrieved_at   TIMESTAMPTZ,
    licence        TEXT,
    source_authority SMALLINT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT provenance_kind_check CHECK (kind IN (
        'chat_turn', 'document_version', 'api_assertion', 'inference',
        'consolidation', 'backfill'
    )),
    CONSTRAINT provenance_producer_check CHECK (producer IN (
        'llm_extract', 'chunker', 'user_assertion', 'consolidation', 'backfill'
    ))
);

COMMENT ON TABLE provenance IS
    'Append-only derivation record, one row per chunk and model run. UPDATE is '
    'refused; DELETE only through pgkg_erase_provenance(). source_id is '
    'polymorphic by kind, which is why it carries no foreign key (ADR 0001, D5).';

COMMENT ON COLUMN provenance.source_authority IS
    'Tie-breaker WITHIN world scope only. Ranking a tenant claim against a '
    'shared one by source authority would let operator content outrank what a '
    'customer said about themselves (ADR 0001, D4).';

COMMENT ON COLUMN provenance.retrieved_at IS
    'When we fetched it. Distinct from published_at, which feeds the perishable '
    'decay profile, and from created_at, which is when we ingested it.';

INSERT INTO provenance (id, org_id, kind, producer)
VALUES (
    pgkg_unattributed_provenance(),
    pgkg_system_org(),
    'backfill',
    'backfill'
);


CREATE INDEX provenance_source_idx ON provenance (source_id)
    WHERE source_id IS NOT NULL;
CREATE INDEX provenance_ingest_run_idx ON provenance (ingest_run_id)
    WHERE ingest_run_id IS NOT NULL;
CREATE INDEX provenance_reextract_idx ON provenance (producer_model, prompt_version);
CREATE INDEX provenance_org_idx ON provenance (org_id);


-- 2. Append-only, as a trigger rather than a rule: a rule that swallowed the
-- write would leave the caller believing it succeeded.
CREATE FUNCTION pgkg_provenance_append_only() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION
            'provenance is append-only: row % cannot be updated', OLD.id;
    END IF;

    IF COALESCE(
        current_setting('pgkg.allow_provenance_erasure', TRUE), 'off'
    ) <> 'on' THEN
        RAISE EXCEPTION
            'provenance row % may only be deleted through '
            'pgkg_erase_provenance()', OLD.id;
    END IF;

    RETURN OLD;
END;
$$;

CREATE TRIGGER pgkg_provenance_append_only
    BEFORE UPDATE OR DELETE ON provenance
    FOR EACH ROW
    EXECUTE FUNCTION pgkg_provenance_append_only();


-- The one sanctioned way past the trigger.  The GUC is transaction-local, so
-- the exemption cannot outlive the call that granted it.
CREATE FUNCTION pgkg_erase_provenance(p_ids UUID[]) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    erased BIGINT;
BEGIN
    PERFORM set_config('pgkg.allow_provenance_erasure', 'on', TRUE);

    WITH gone AS (
        DELETE FROM provenance
        WHERE id = ANY(p_ids)
          AND id <> pgkg_unattributed_provenance()
        RETURNING 1
    )
    SELECT COUNT(*) INTO erased FROM gone;

    PERFORM set_config('pgkg.allow_provenance_erasure', 'off', TRUE);
    RETURN erased;
END;
$$;


-- 3. The many-to-one that deduplication brings.  Cascading on both sides is
-- what makes the erasure story fall out of the schema: the link goes when
-- either end goes, and a proposition's remaining source count is a COUNT(*).
CREATE TABLE proposition_provenance (
    proposition_id UUID NOT NULL REFERENCES propositions(id) ON DELETE CASCADE,
    provenance_id  UUID NOT NULL REFERENCES provenance(id) ON DELETE CASCADE,
    PRIMARY KEY (proposition_id, provenance_id)
);

CREATE INDEX prop_prov_provenance_idx ON proposition_provenance (provenance_id);


-- 4. The hot-path column.  RESTRICT on delete rather than CASCADE: losing the
-- derivation record must not silently take the facts with it, because "delete
-- this source" and "delete everything we learned from it" are different
-- decisions and only one of them is reversible.
ALTER TABLE propositions
    ADD COLUMN provenance_id UUID NOT NULL
        DEFAULT pgkg_unattributed_provenance()
        REFERENCES provenance(id);

ALTER TABLE chunks
    ADD COLUMN provenance_id UUID NOT NULL
        DEFAULT pgkg_unattributed_provenance()
        REFERENCES provenance(id);

CREATE INDEX prop_provenance_idx ON propositions (provenance_id);
CREATE INDEX chunk_provenance_idx ON chunks (provenance_id);


-- 5. The four clocks.  Assertion already exists as asserted_at and still
-- drives decay; these are the other three.
ALTER TABLE propositions
    ADD COLUMN valid_from          TIMESTAMPTZ,
    ADD COLUMN valid_to            TIMESTAMPTZ,
    ADD COLUMN recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN invalidated_at      TIMESTAMPTZ,
    ADD COLUMN invalidation_reason TEXT,
    ADD COLUMN expires_at          TIMESTAMPTZ,
    ADD COLUMN legal_hold          BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN propositions.valid_from IS
    'Validity clock: when the claim is true in the world. NULL means unbounded.';
COMMENT ON COLUMN propositions.recorded_at IS
    'Belief clock: when we came to hold the claim. Read only by the audit path.';
COMMENT ON COLUMN propositions.legal_hold IS
    'Retention clock: policy, not truth. A row under hold outlives its TTL.';

-- Rows already withdrawn under the old idiom keep their meaning: the pointer is
-- what said so, and the timestamp is now what retrieval reads.
UPDATE propositions
SET invalidated_at = COALESCE(last_accessed_at, now()),
    invalidation_reason = 'superseded'
WHERE superseded_by IS NOT NULL
  AND invalidated_at IS NULL;

ALTER TABLE propositions
    ADD CONSTRAINT propositions_invalidation_reason_check
        CHECK (invalidation_reason IN (
            'superseded', 'source_updated', 'source_deleted', 'ttl',
            'user_deleted', 'contradicted', 'retracted_run'
        )),
    ADD CONSTRAINT propositions_invalidation_has_reason
        CHECK ((invalidated_at IS NULL) = (invalidation_reason IS NULL)),
    ADD CONSTRAINT propositions_validity_runs_forwards
        CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_to > valid_from);


-- The pointer and the filter, kept in step.  The WHEN clauses mean this fires
-- only when superseded_by moves: the recall-path access flush never touches it,
-- and a caller invalidating for any other reason states its own reason.
CREATE FUNCTION pgkg_sync_invalidation() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.superseded_by IS NOT NULL AND NEW.invalidated_at IS NULL THEN
        NEW.invalidated_at := now();
        NEW.invalidation_reason := 'superseded';
    ELSIF NEW.superseded_by IS NULL
          AND NEW.invalidation_reason = 'superseded' THEN
        NEW.invalidated_at := NULL;
        NEW.invalidation_reason := NULL;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER pgkg_prop_sync_invalidation_insert
    BEFORE INSERT ON propositions
    FOR EACH ROW
    WHEN (NEW.superseded_by IS NOT NULL)
    EXECUTE FUNCTION pgkg_sync_invalidation();

CREATE TRIGGER pgkg_prop_sync_invalidation_update
    BEFORE UPDATE ON propositions
    FOR EACH ROW
    WHEN (NEW.superseded_by IS DISTINCT FROM OLD.superseded_by)
    EXECUTE FUNCTION pgkg_sync_invalidation();


-- 6. Indexes for the modes.  The partial predicate is what serves current
-- state; the tenancy indexes from 020 carried the old filter and are replaced
-- rather than left to cost write bandwidth for a predicate nothing reads.
DROP INDEX prop_tenancy_idx;
DROP INDEX prop_active_idx;

CREATE INDEX prop_live_tenancy_idx ON propositions (org_id, collection_id)
    WHERE invalidated_at IS NULL;
CREATE INDEX prop_live_ns_idx ON propositions (namespace)
    WHERE invalidated_at IS NULL;

-- No index on the validity interval and none on recorded_at.  Both modes that
-- read them are already scoped to one namespace, and namespace is the more
-- selective column, so the planner reaches those rows through the indexes above
-- and applies the timestamps as a filter.  An index nothing chooses is write
-- bandwidth on the largest table, and the audit path is explicitly budgeted as
-- a sequential pass over one tenant's rows (ADR 0001, D6).

CREATE INDEX prop_expiry_idx ON propositions (expires_at)
    WHERE expires_at IS NOT NULL AND legal_hold = FALSE AND invalidated_at IS NULL;


-- 7. The temporal predicate, in one place, IMMUTABLE so the planner inlines it
-- back into column comparisons and the partial index stays usable.  Callers
-- resolve the instant themselves — COALESCE(p_valid_at, now()) — which keeps
-- this function free of now() and therefore free of a STABLE marking that
-- would block inlining into an index-qualifying expression.
CREATE FUNCTION pgkg_temporal_visible(
    p_invalidated_at TIMESTAMPTZ,
    p_valid_from     TIMESTAMPTZ,
    p_valid_to       TIMESTAMPTZ,
    p_valid_at       TIMESTAMPTZ
) RETURNS BOOLEAN
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
AS $$
    SELECT p_invalidated_at IS NULL
       AND (p_valid_from IS NULL OR p_valid_from <= p_valid_at)
       AND (p_valid_to   IS NULL OR p_valid_to   >  p_valid_at)
$$;


-- 8. Lifecycle operations.  Each is one set-based UPDATE keyed on an index, and
-- each refuses to overwrite a reason already recorded: the first withdrawal is
-- the true one, and a later cascade that relabelled it would destroy the only
-- trace of why belief changed.
CREATE FUNCTION pgkg_invalidate_source(
    p_source_id UUID,
    p_reason    TEXT DEFAULT 'source_updated'
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    affected BIGINT;
BEGIN
    WITH withdrawn AS (
        UPDATE propositions p
        SET invalidated_at = now(),
            invalidation_reason = p_reason
        WHERE p.invalidated_at IS NULL
          AND p.provenance_id IN (
              SELECT pr.id FROM provenance pr WHERE pr.source_id = p_source_id
          )
        RETURNING 1
    )
    SELECT COUNT(*) INTO affected FROM withdrawn;

    RETURN affected;
END;
$$;


CREATE FUNCTION pgkg_retract_ingest_run(p_ingest_run_id UUID) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    affected BIGINT;
BEGIN
    WITH withdrawn AS (
        UPDATE propositions p
        SET invalidated_at = now(),
            invalidation_reason = 'retracted_run'
        WHERE p.invalidated_at IS NULL
          AND p.provenance_id IN (
              SELECT pr.id FROM provenance pr
              WHERE pr.ingest_run_id = p_ingest_run_id
          )
        RETURNING 1
    )
    SELECT COUNT(*) INTO affected FROM withdrawn;

    RETURN affected;
END;
$$;


-- A contradiction is the world changing, not us having been wrong.  Belief is
-- untouched: the row stays live, so the as-of-validity mode can still answer
-- with it, and the audit path still sees it as something we held.  Deleting it
-- would throw away the only evidence of what the answer used to be.
CREATE FUNCTION pgkg_contradict(
    p_proposition_id UUID,
    p_effective_at   TIMESTAMPTZ DEFAULT now()
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    affected BIGINT;
BEGIN
    WITH closed AS (
        UPDATE propositions p
        SET valid_to = p_effective_at
        WHERE p.id = p_proposition_id
          AND (p.valid_to IS NULL OR p.valid_to > p_effective_at)
        RETURNING 1
    )
    SELECT COUNT(*) INTO affected FROM closed;

    RETURN affected;
END;
$$;


-- Retention answers to policy, so legal_hold wins over the TTL.  This withdraws
-- from belief only; reclaiming the space is a separate sweep, because deleting
-- rows out of an HNSW index leaves tombstones that need a rebuild to clear.
CREATE FUNCTION pgkg_expire_due(p_namespace TEXT DEFAULT NULL) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    affected BIGINT;
BEGIN
    WITH withdrawn AS (
        UPDATE propositions p
        SET invalidated_at = now(),
            invalidation_reason = 'ttl'
        WHERE p.invalidated_at IS NULL
          AND p.legal_hold = FALSE
          AND p.expires_at IS NOT NULL
          AND p.expires_at <= now()
          AND (p_namespace IS NULL OR p.namespace = p_namespace)
        RETURNING 1
    )
    SELECT COUNT(*) INTO affected FROM withdrawn;

    RETURN affected;
END;
$$;


-- 9. The audit mode.  No vector arm, no BM25, no scoring: the question is what
-- the system held at an instant, and the answer is a filter, not a ranking.
-- Kept a separate function precisely so it cannot become the hot path by
-- someone passing one extra argument to pgkg_search().
CREATE FUNCTION pgkg_believed_at(
    p_belief_at      TIMESTAMPTZ,
    p_namespace      TEXT   DEFAULT 'default',
    k_retrieve       INT    DEFAULT 100,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL
) RETURNS TABLE (
    proposition_id      UUID,
    text                TEXT,
    recorded_at         TIMESTAMPTZ,
    invalidated_at      TIMESTAMPTZ,
    invalidation_reason TEXT,
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,
    provenance_id       UUID
)
LANGUAGE SQL STABLE
AS $$
SELECT
    p.id,
    p.text,
    p.recorded_at,
    p.invalidated_at,
    p.invalidation_reason,
    p.valid_from,
    p.valid_to,
    p.provenance_id
FROM propositions p
WHERE p.namespace = p_namespace
  AND p.recorded_at <= p_belief_at
  AND (p.invalidated_at IS NULL OR p.invalidated_at > p_belief_at)
  AND pgkg_visible(
        p.org_id, p.collection_id, p.visibility,
        p.owner_user_id, p.acl_group_id,
        p_org_ids, p_collection_ids, p_user_id, p_acl_groups
      )
ORDER BY p.recorded_at DESC, p.id
LIMIT k_retrieve;
$$;


-- 10. The candidate sources, on the new filter.  Each body is the one from 020
-- with `superseded_by IS NULL` replaced by the shared temporal predicate and
-- nothing else changed.
DROP FUNCTION pgkg_bm25_candidates(TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[]);

CREATE FUNCTION pgkg_bm25_candidates(
    q_text           TEXT,
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
) sub
WHERE sub.bm25_score > 0.0
ORDER BY sub.bm25_score DESC
LIMIT k_initial;
$$;


DROP FUNCTION pgkg_vector_candidates(halfvec, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[]);

CREATE FUNCTION pgkg_vector_candidates(
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
$$;


-- The graph arm is re-filtered on time for the same reason 020 re-filtered it
-- on scope: a seed entity is a bridge, and a walk across it must not resurrect
-- a proposition the seed's own filter would have withdrawn.
DROP FUNCTION pgkg_graph_candidates(
    pgkg_candidate[], TEXT, INT, INT, INT, UUID[], UUID[], UUID, UUID[]
);

CREATE FUNCTION pgkg_graph_candidates(
    p_seeds          pgkg_candidate[],
    p_namespace      TEXT   DEFAULT 'default',
    k_seed_entities  INT    DEFAULT 20,
    k_per_seed       INT    DEFAULT 10,
    k_total          INT    DEFAULT 100,
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
    WHERE np.namespace = p_namespace
      AND pgkg_temporal_visible(
            np.invalidated_at, np.valid_from, np.valid_to,
            COALESCE(p_valid_at, now())
          )
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


-- 11. pgkg_search(), with the validity instant appended last.  NULL means now,
-- so every existing positional caller keeps the current-state mode it has, and
-- as-of validity is one named argument away.
DROP FUNCTION pgkg_search(
    TEXT, halfvec, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT,
    UUID[], UUID[], UUID, UUID[]
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
    p_valid_at             TIMESTAMPTZ DEFAULT NULL
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


-- 12. The ranking statistics follow belief.  IDF computed over rows retrieval
-- cannot return would understate the document frequency of every withdrawn
-- term, so the maintenance predicate has to be the retrieval predicate.  These
-- three bodies are 011's with `superseded_by IS NULL` swapped for
-- `invalidated_at IS NULL`; the predicate is embedded in plpgsql, so the body
-- is the smallest unit that can be altered.
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

    INSERT INTO corpus_stats (namespace, kind, n_total, total_len)
    SELECT p.namespace, 'proposition', COUNT(*), COALESCE(SUM(p.doc_len), 0)
    FROM propositions p
    WHERE p.invalidated_at IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace;

    INSERT INTO lexeme_df (namespace, kind, lexeme, df)
    SELECT p.namespace, 'proposition', u.lexeme, COUNT(*)
    FROM propositions p, unnest(p.tsv) AS u(lexeme, positions, weights)
    WHERE p.invalidated_at IS NULL
      AND (p_namespace IS NULL OR p.namespace = p_namespace)
    GROUP BY p.namespace, u.lexeme;
END;
$$;


CREATE OR REPLACE FUNCTION pgkg_propositions_stats_delta() RETURNS TRIGGER
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
    WHERE d.invalidated_at IS NULL
    GROUP BY d.namespace
    ON CONFLICT (namespace, kind) DO UPDATE
        SET n_total    = GREATEST(cs.n_total + EXCLUDED.n_total, 0),
            total_len  = GREATEST(cs.total_len + EXCLUDED.total_len, 0),
            updated_at = now();

    INSERT INTO lexeme_df AS ld (namespace, kind, lexeme, df)
    SELECT d.namespace, 'proposition', u.lexeme, delta_sign * COUNT(*)
    FROM delta_rows d, unnest(d.tsv) AS u(lexeme, positions, weights)
    WHERE d.invalidated_at IS NULL
    GROUP BY d.namespace, u.lexeme
    ON CONFLICT (namespace, kind, lexeme) DO UPDATE
        SET df = GREATEST(ld.df + EXCLUDED.df, 0);

    RETURN NULL;
END;
$$;


CREATE OR REPLACE FUNCTION pgkg_propositions_stats_update() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    WITH delta AS (
        SELECT o.namespace, o.tsv, o.doc_len, -1 AS delta_sign
        FROM old_rows o
        WHERE o.invalidated_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM new_rows n
              WHERE n.id = o.id
                AND n.namespace = o.namespace
                AND n.tsv = o.tsv
                AND (n.invalidated_at IS NULL) = (o.invalidated_at IS NULL)
          )

        UNION ALL

        SELECT n.namespace, n.tsv, n.doc_len, 1
        FROM new_rows n
        WHERE n.invalidated_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM old_rows o
              WHERE o.id = n.id
                AND o.namespace = n.namespace
                AND o.tsv = n.tsv
                AND (o.invalidated_at IS NULL) = (n.invalidated_at IS NULL)
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


-- 13. Row-level security on the new tables, matching 020.  The reserved
-- backfill record is exempt by id: it belongs to the operator's system org and
-- every tenant's pre-provenance rows point at it, so a policy that hid it would
-- make those propositions unjoinable to their own derivation record.
ALTER TABLE provenance ENABLE ROW LEVEL SECURITY;

CREATE POLICY provenance_org_isolation ON provenance
    USING (org_id = pgkg_current_org() OR id = pgkg_unattributed_provenance())
    WITH CHECK (org_id = pgkg_current_org());

COMMENT ON TABLE proposition_provenance IS
    'No RLS policy: the table holds only id pairs, and both tables it keys into '
    'carry their own org isolation.';

DO $$
BEGIN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON provenance, '
            'proposition_provenance TO pgkg_app';
EXCEPTION WHEN undefined_object OR insufficient_privilege THEN
    RAISE NOTICE 'pgkg_app not granted on the provenance tables (%)', SQLERRM;
END;
$$;
