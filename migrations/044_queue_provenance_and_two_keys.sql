-- The three keys the fix pass left stated in Python rather than in the schema
-- (ADR 0001, D4, D5, D6, D7).
--
-- WHY THE EXTERNAL-ID INDEX BECOMES PARTIAL.  documents_external_id_key was
-- UNIQUE (org_id, collection_id, external_id) over every row, live or
-- withdrawn, while the ingest pipeline's own lookup says "a soft-deleted
-- document is not a collision: re-ingesting a withdrawn external id is a new
-- document".  Those two cannot both be true, and the pipeline was made right by
-- taking the id off the withdrawn row before inserting the new one — which
-- erases the only thing a deletion audit asks of that row, the id it was
-- withdrawn under.  The index is where the rule belongs: uniqueness is a
-- property of the live corpus, because a withdrawn document is not something a
-- crawl can collide with.
--
-- WHY THE EXTRACTION CACHE IS KEYED BY ORG.  043 gave proposition_cache an
-- org column and a policy, and left the primary key on cache_key alone because
-- the statements that write it name that key in ON CONFLICT.  A single-column
-- key is not a smaller version of the right one here: it says one row per text
-- per model for the whole installation, so the second org to hold a passage
-- either reads the first org's claims or overwrites them, and under the owning
-- role — which is how the reference deployment connects — the policy is inert
-- and cannot stop either.  The key is what the payload is: the facts extracted
-- from a passage by an org that holds it (D4).  Two orgs holding one text now
-- hold one cached extraction each, which is the cost D4 already accepts for the
-- embedding cache on everything that is not operator-licensed.
--
-- WHY THE QUEUE CARRIES PROVENANCE.  D7 makes batch the default posture for a
-- corpus, so the queue is the path most documents take, and a job carried the
-- org, the collection, the external id, the content hash, the text and the uri
-- and nothing else.  Everything the connector knew — publisher, licence,
-- publication date, the run that produced it — was dropped at the door, and the
-- worker then re-derived asserted_at from its own clock.  D5 says published_at
-- feeds the perishable profile and D6 keys that profile on asserted_at, so the
-- inline path deriving one from the other buys nothing while the queued path
-- still decays an eleven-year-old article from the moment a worker reached it.
-- The whole provenance record travels as JSONB rather than as fourteen columns:
-- ingest_jobs is a queue, and what it holds for the pipeline it feeds is the
-- pipeline's shape to state, not the queue's.


-- 1. Uniqueness is a property of the live corpus.
DROP INDEX documents_external_id_key;

CREATE UNIQUE INDEX documents_external_id_key
    ON documents (org_id, collection_id, external_id)
    WHERE external_id IS NOT NULL AND deleted_at IS NULL;

COMMENT ON INDEX documents_external_id_key IS
    'One live document per external id per collection. A withdrawn document '
    'keeps the id it was ingested under: it is the record of what was '
    'withdrawn, and a re-crawl of that id is a new document (ADR 0001, D6).';


-- 2. A cached extraction belongs to the org that paid for it.
ALTER TABLE proposition_cache DROP CONSTRAINT proposition_cache_pkey;

ALTER TABLE proposition_cache
    ADD CONSTRAINT proposition_cache_pkey PRIMARY KEY (cache_key, org_id);

COMMENT ON TABLE proposition_cache IS
    'Extracted propositions, keyed by the text and model that produced them '
    'and by the org that holds that text. A hit is the extracted claims '
    'themselves, so the org is part of the key and not only of the row policy: '
    'the deployments that connect as the owning role are the ones RLS cannot '
    'help (ADR 0001, D4).';


-- 3. The queue carries what the connector knew.
ALTER TABLE ingest_jobs
    ADD COLUMN source      TEXT,
    ADD COLUMN asserted_at TIMESTAMPTZ,
    ADD COLUMN provenance  JSONB;

COMMENT ON COLUMN ingest_jobs.asserted_at IS
    'The world-time of this document''s content, as the connector stated it. '
    'NULL means the pipeline derives it from the provenance record''s '
    'published_at, which is what D5 and D6 ask for.';

COMMENT ON COLUMN ingest_jobs.provenance IS
    'The derivation record this document arrives with (ADR 0001, D5), as the '
    'ingest pipeline shapes it. Held as JSONB because what a queue carries for '
    'the pipeline it feeds is that pipeline''s shape to state.';

DROP FUNCTION pgkg_enqueue_ingest_job(UUID, UUID, TEXT, BYTEA, TEXT, TEXT);

CREATE FUNCTION pgkg_enqueue_ingest_job(
    p_org_id        UUID,
    p_collection_id UUID,
    p_external_id   TEXT,
    p_content_hash  BYTEA,
    p_payload       TEXT,
    p_uri           TEXT        DEFAULT NULL,
    p_source        TEXT        DEFAULT NULL,
    p_asserted_at   TIMESTAMPTZ DEFAULT NULL,
    p_provenance    JSONB       DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_job UUID;
BEGIN
    INSERT INTO ingest_jobs
        (org_id, collection_id, external_id, content_hash, payload, uri,
         source, asserted_at, provenance)
    VALUES (p_org_id, p_collection_id, p_external_id, p_content_hash,
            p_payload, p_uri, p_source, p_asserted_at, p_provenance)
    ON CONFLICT (org_id, collection_id, external_id, content_hash)
        WHERE status <> 'done'
        DO NOTHING
    RETURNING id INTO v_job;

    -- The conflicting row is the work already queued for this exact content,
    -- and its provenance is the account the connector gave when it first
    -- offered it. A second offer of the same bytes gets a handle to that work
    -- rather than a second copy of it, so it does not restate its origin
    -- either.
    IF v_job IS NULL THEN
        SELECT j.id INTO v_job
        FROM ingest_jobs j
        WHERE j.org_id = p_org_id
          AND j.collection_id = p_collection_id
          AND j.external_id = p_external_id
          AND j.content_hash = p_content_hash
          AND j.status <> 'done';
    END IF;

    RETURN v_job;
END;
$$;

DROP FUNCTION pgkg_claim_ingest_job(UUID, INTERVAL);

CREATE FUNCTION pgkg_claim_ingest_job(
    p_org_id UUID     DEFAULT NULL,
    p_lease  INTERVAL DEFAULT INTERVAL '5 minutes'
) RETURNS TABLE (
    job_id        UUID,
    org_id        UUID,
    collection_id UUID,
    external_id   TEXT,
    uri           TEXT,
    payload       TEXT,
    content_hash  BYTEA,
    attempts      INT,
    source        TEXT,
    asserted_at   TIMESTAMPTZ,
    provenance    JSONB
)
LANGUAGE SQL
AS $$
    WITH claimable AS (
        SELECT j.id
        FROM ingest_jobs j
        WHERE (p_org_id IS NULL OR j.org_id = p_org_id)
          AND (j.status = 'pending'
               OR (j.status = 'running'
                   AND j.heartbeat_at < now() - p_lease))
        ORDER BY j.enqueued_at, j.id
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    UPDATE ingest_jobs j
    SET status = 'running',
        attempts = j.attempts + 1,
        started_at = COALESCE(j.started_at, now()),
        heartbeat_at = now(),
        error = NULL
    FROM claimable
    WHERE j.id = claimable.id
    RETURNING j.id, j.org_id, j.collection_id, j.external_id, j.uri,
              j.payload, j.content_hash, j.attempts, j.source, j.asserted_at,
              j.provenance;
$$;
