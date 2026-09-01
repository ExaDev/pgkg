-- The batch corpus ingest pipeline: the shared embedding cache, and the queue
-- the workers drain (ADR 0001, D4 and D7).
--
-- WHY A CACHE AND NOT SHARED ROWS.  Chunks are content-addressed, so physical
-- row dedup across tenants looks free.  D4 rejects it: a row shared between
-- tenants cannot live in a tenant partition, so it lands in a
-- content-hash-partitioned pool, and vector search over that pool cannot prune
-- by org.  The recall lost exceeds the disk saved, and it couples tenants for
-- index rebuilds.  So the expensive half is shared instead — the GPU cost of
-- turning a passage into a vector, paid once per unique passage per generation
-- — while every tenant keeps its own chunk row in its own partition.
-- Partitioning intact, retrieval untouched.
--
-- THE DELIBERATE BOUNDARY.  A content-hash cache is PROBE-ABLE: observing
-- cache-hit latency reveals whether a passage is already cached.  For public
-- web content that is uninteresting; for a confidential document it would
-- confirm another tenant holds it.  Hence the restriction, enforced in the
-- ingest path and stated on the table: only collections flagged public_source
-- — crawled or operator-licensed material, which 023's CHECK already confines
-- to operator-owned collections — take part.  User uploads and chat text never
-- enter this cache, and the pipeline does not read it for them either.  Reading
-- is gated on the same flag as writing on purpose: a cache HIT is itself the
-- observation, so a private ingest that consulted the cache would be the probe
-- this rule exists to prevent, whichever way the row got in.
--
-- WHY THE VECTOR COLUMN HAS NO WIDTH.  Every other embedding column in this
-- schema is one generation wide because it is indexed and an index needs a
-- width.  This one is looked up by primary key and never searched, so it can
-- hold two generations of different widths at once — which is exactly what a
-- D8 cutover across a fleet with overlapping public content needs, and the
-- reason the cache makes that cutover cheap rather than merely correct.
--

-- WHY THERE IS A QUEUE AT ALL.  Corpus ingest is batch: 600k chunks is one to
-- two GPU-hours, or a day-plus on CPU, and it must not compete with online
-- recall for pool slots (D7).  A queue is what lets it be throttled, resumed
-- and observed.  Observed matters most: "is my corpus indexed yet" is the first
-- question a customer asks, and per-document progress is the only honest answer.
--
-- IDEMPOTENCY IS BY CONTENT HASH, NOT BY DOCUMENT.  A connector re-enqueues
-- everything it can see every night, so enqueue has to be a no-op for work
-- already queued.  The partial unique index makes a second enqueue of the same
-- (document, content) collide while the job is still open, and stops colliding
-- once it is done — because a document that reverts to an earlier revision is
-- new work again, and forbidding that would turn an ordinary A-B-A edit into a
-- permanent refusal.
--
-- RESUMABILITY IS A LEASE, NOT A FLAG.  A worker that dies leaves its job in
-- `running`, and no observer can tell that from a worker still working.  The
-- heartbeat is what distinguishes them, so claiming reclaims a job whose lease
-- has lapsed rather than requiring a separate sweeper to notice.
--
-- NO ROW-LEVEL SECURITY ON ingest_jobs, DELIBERATELY.  It carries org_id and
-- belongs behind a policy, but every policy in this schema is pinned by a
-- cross-org read in the isolation test module, and that module enumerates the
-- protected tables exhaustively — a policy added here without a case there
-- would either ship an untested isolation boundary or break that guard.  The
-- queue is reached only through the functions below, which take the org as an
-- argument.


-- 1. The cache.  No org column: this table is the one place in the schema where
-- content is deliberately not partitioned by tenant, which is only sound
-- because of what may enter it.
CREATE TABLE embedding_cache (
    content_hash  BYTEA NOT NULL,
    generation_id UUID  NOT NULL REFERENCES embedder_generations(id)
                        ON DELETE CASCADE,
    vec           halfvec NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (content_hash, generation_id)
);

COMMENT ON TABLE embedding_cache IS
    'Consulted before calling the embedder, and populated ONLY for collections '
    'flagged public_source: crawled or licensed public content. A content-hash '
    'cache is probe-able, so for a confidential document a hit would confirm '
    'another tenant holds it. User uploads and chat text never enter this '
    'cache and the ingest path never reads it for them (ADR 0001, D4).';

COMMENT ON COLUMN embedding_cache.vec IS
    'Deliberately unconstrained in width: keyed by primary key and never '
    'indexed, so two generations of different widths coexist here, which is '
    'what makes an embedder cutover cheap across a fleet (ADR 0001, D8).';


-- 2. The queue.  The payload travels with the job rather than being re-fetched
-- on claim: a connector that has already read the document should not have to
-- still be alive when a worker gets to it, and the content hash the job carries
-- is only meaningful next to the text it was taken from.
CREATE TABLE ingest_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    collection_id   UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    external_id     TEXT NOT NULL,
    uri             TEXT,
    content_hash    BYTEA NOT NULL,
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INT  NOT NULL DEFAULT 0,
    chunks_total    INT,
    chunks_embedded INT,
    -- What the job produced, recorded rather than referenced.  A job row is an
    -- operational record of work done, and a document deleted or a version
    -- purged months later must not rewrite the history of the ingest that
    -- created it — the same reason provenance.source_id carries no key (D5).
    document_id     UUID,
    version_id      UUID,
    error           TEXT,
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    heartbeat_at    TIMESTAMPTZ,
    CONSTRAINT ingest_jobs_status_check
        CHECK (status IN ('pending', 'running', 'done', 'failed'))
);

COMMENT ON TABLE ingest_jobs IS
    'One row per document a connector offered. Corpus ingest is batch and must '
    'not compete with online recall for pool slots, so it runs from here under '
    'a worker''s slot budget rather than on the request path (ADR 0001, D7).';

CREATE UNIQUE INDEX ingest_jobs_open_work_key
    ON ingest_jobs (org_id, collection_id, external_id, content_hash)
    WHERE status <> 'done';

CREATE INDEX ingest_jobs_claimable_idx
    ON ingest_jobs (org_id, enqueued_at)
    WHERE status IN ('pending', 'running');


-- 3. Enqueue.  Returns the job that now holds this work, whether or not this
-- call created it, so a re-crawl gets a handle rather than an error.
CREATE FUNCTION pgkg_enqueue_ingest_job(
    p_org_id        UUID,
    p_collection_id UUID,
    p_external_id   TEXT,
    p_content_hash  BYTEA,
    p_payload       TEXT,
    p_uri           TEXT DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_job UUID;
BEGIN
    INSERT INTO ingest_jobs
        (org_id, collection_id, external_id, content_hash, payload, uri)
    VALUES (p_org_id, p_collection_id, p_external_id, p_content_hash,
            p_payload, p_uri)
    ON CONFLICT (org_id, collection_id, external_id, content_hash)
        WHERE status <> 'done'
        DO NOTHING
    RETURNING id INTO v_job;

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


-- 4. Claim one job.  SKIP LOCKED is what lets several workers drain the queue
-- without coordinating, and the lease folds recovery into the same statement:
-- a job whose worker stopped sending heartbeats is claimable again, and one
-- whose worker is still alive is not.  Oldest first, so a slow queue stays fair.
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
    attempts      INT
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
              j.payload, j.content_hash, j.attempts;
$$;


-- 5. Progress, and the heartbeat that comes with it.  The two are one call
-- because a worker that reports progress is by definition alive, and a
-- liveness signal nobody has a reason to send is a liveness signal that stops
-- being sent.
--
-- A caller must report this from OUTSIDE the ingest transaction: progress
-- written inside it is invisible until commit, which is precisely when it stops
-- being progress.
CREATE FUNCTION pgkg_report_ingest_progress(
    p_job_id          UUID,
    p_chunks_total    INT DEFAULT NULL,
    p_chunks_embedded INT DEFAULT NULL
) RETURNS VOID
LANGUAGE SQL
AS $$
    UPDATE ingest_jobs
    SET chunks_total = COALESCE(p_chunks_total, chunks_total),
        chunks_embedded = COALESCE(p_chunks_embedded, chunks_embedded),
        heartbeat_at = now()
    WHERE id = p_job_id;
$$;


-- 6. Finish, and fail.  A failure returns the job to the queue until the
-- attempt budget runs out: the usual reason a corpus job dies is a transient
-- one — an embedder restart, a lost connection — and a queue that gave up on
-- the first of those would need a human for something a retry fixes.  The error
-- is kept either way, because a job that eventually succeeded after failing is
-- still something an operator wants to see.
CREATE FUNCTION pgkg_finish_ingest_job(
    p_job_id          UUID,
    p_document_id     UUID DEFAULT NULL,
    p_version_id      UUID DEFAULT NULL,
    p_chunks_total    INT  DEFAULT NULL,
    p_chunks_embedded INT  DEFAULT NULL
) RETURNS VOID
LANGUAGE SQL
AS $$
    UPDATE ingest_jobs
    SET status = 'done',
        document_id = COALESCE(p_document_id, document_id),
        version_id = COALESCE(p_version_id, version_id),
        chunks_total = COALESCE(p_chunks_total, chunks_total),
        chunks_embedded = COALESCE(p_chunks_embedded, chunks_embedded),
        finished_at = now(),
        heartbeat_at = now()
    WHERE id = p_job_id;
$$;

CREATE FUNCTION pgkg_fail_ingest_job(
    p_job_id       UUID,
    p_error        TEXT,
    p_max_attempts INT DEFAULT 3
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_status TEXT;
BEGIN
    UPDATE ingest_jobs j
    SET status = CASE WHEN j.attempts >= p_max_attempts
                      THEN 'failed' ELSE 'pending' END,
        error = p_error,
        finished_at = CASE WHEN j.attempts >= p_max_attempts
                           THEN now() ELSE NULL END,
        heartbeat_at = now()
    WHERE j.id = p_job_id
    RETURNING j.status INTO v_status;

    RETURN v_status;
END;
$$;


-- 7. The application role, again: 020's GRANT was a snapshot of the tables that
-- existed then, so a table created here is unreachable by the role a deployment
-- connects as until it is granted by name.
DO $$
BEGIN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON '
            'embedding_cache, ingest_jobs TO pgkg_app';
EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
    RAISE NOTICE 'pgkg_app not granted on the corpus ingest tables (%)', SQLERRM;
END;
$$;
