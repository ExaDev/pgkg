-- Document ACLs: the write half of the seam, failing closed (ADR 0001, D3).
--
-- The read half has been finished since 020.  pgkg_visible() drops an ACL-gated
-- row unless the caller presents its group, 042 made the ACL group part of a
-- chunk's content address, and 041's arms carry p_acl_groups into every CTE.
-- Nothing ever wrote the column.  acl_group_id was NULL on every row every
-- pipeline produced, and the predicate reads
--
--     (p_row_acl_group IS NULL OR p_row_acl_group = ANY(p_acl_groups))
--
-- so a NULL group passes for every caller.  A collection could therefore
-- declare acl_mode = 'group' and get no enforcement whatsoever, while the
-- schema, the retrieval predicate and the HTTP scope fields all read as though
-- it had some.  D3 is explicit about the shape of that mistake: ingesting a
-- corporate corpus without modelling its permissions "builds a
-- permission-laundering machine".
--
-- WHAT THIS MIGRATION IS NOT.  It is not ACL support.  That needs a permissions
-- source to synchronise groups from — SharePoint, Drive — and none exists, so
-- nothing populates a group automatically.  What it does is stop the absence
-- from being silent: a collection that declares an ACL mode now refuses content
-- that names no group, so an operator who sets acl_mode gets an error rather
-- than an unenforced setting that looks finished.
--
-- WHY THE REFUSAL LIVES ON THE TABLES.  The ingest pipeline checks the
-- collection too, before it spends the embedder budget, because a ValueError
-- with a sentence in it is worth more to a connector author than a trigger's
-- exception.  But a rule enforced only there is one pipeline's rule: chat
-- ingest writes chunks and propositions through its own statements, the
-- maintenance functions write both, and phase 4's group sync will write more.
-- Every one of them passes through these two tables, which is the only place
-- the rule can be stated once.  Both halves exist for the same reason 042's
-- same-org check is a trigger and not a widened policy: a row that is ACL-gated
-- for nobody is meaningless in any session, RLS on or off, owner or not.
--
-- WHY THE GUARD IS "NOT EXEMPT" RATHER THAN "IS BOUNDED".  The obvious
-- predicate joins to collections and refuses a row whose collection has
-- acl_mode <> 'none'.  A trigger function runs as the invoker, so under RLS a
-- collection row the session cannot see makes that join find nothing and the
-- guard pass — a guard that fails open exactly when visibility is in question.
-- The predicate is inverted instead: a row with no group is refused unless a
-- collection saying acl_mode = 'none' is visible from this session.  023's
-- policy shows every collection of the caller's own org and of the operator's,
-- which is every collection a write may legally name, so this refuses nothing
-- legitimate today and turns any future narrowing of that policy into a loud
-- failure rather than a quiet one.
--
-- THE INSERT GUARD IS STATEMENT-LEVEL, like every other derived-state guard
-- here: the corpus path links a 300 page handbook's chunks in one statement,
-- and a per-row trigger would make this one extra query per chunk instead of
-- one per statement.  The UPDATE guard is the opposite shape, because Postgres
-- refuses a transition table on a trigger with a column list and an
-- unrestricted AFTER UPDATE would put this in the way of every refcount,
-- retrievability and invalidation statement on the two largest tables.  It is
-- 030's immutability trigger's shape instead — BEFORE UPDATE FOR EACH ROW with
-- the whole condition in the WHEN clause — so the only UPDATEs that reach the
-- function are the ones that could launder a row.  Watching UPDATE at all is
-- 045's lesson: the statement nobody guarded was the one that moved a row
-- across a boundary without inserting or deleting anything, and dropping the
-- group from a tagged row is that same statement here.


-- 1. The rule, in one place: a row with no group belongs only to a collection
-- that says so.  Phrased as "not exempt" rather than "is bounded" for the
-- fail-closed reason above, and STABLE so it inlines into the trigger's scan.
CREATE FUNCTION pgkg_acl_bounded(p_collection_id UUID) RETURNS BOOLEAN
LANGUAGE SQL STABLE
AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM collections c
        WHERE c.id = p_collection_id
          AND c.acl_mode = 'none'
    )
$$;

COMMENT ON FUNCTION pgkg_acl_bounded(UUID) IS
    'Whether a row of this collection must name an ACL group. True unless a '
    'collection with acl_mode = none is visible from this session, so a '
    'collection nobody can see is treated as bounded (ADR 0001, D3).';


-- 2. The guards.  The read predicate lets a NULL group past for every caller,
-- so an untagged row in an ACL-bounded collection is not a narrower grant but
-- no grant at all.
CREATE FUNCTION pgkg_acl_group_required() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_id         UUID;
    v_collection UUID;
BEGIN
    SELECT n.id, n.collection_id
    INTO v_id, v_collection
    FROM new_rows n
    WHERE n.acl_group_id IS NULL
      AND pgkg_acl_bounded(n.collection_id)
    LIMIT 1;

    IF v_id IS NOT NULL THEN
        RAISE EXCEPTION
            'collection % is ACL-bounded, so %.% must name an acl_group_id: '
            'a row with none is visible to every caller of the tenant',
            v_collection, TG_TABLE_NAME, v_id;
    END IF;

    RETURN NULL;
END;
$$;

CREATE FUNCTION pgkg_acl_group_kept() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF pgkg_acl_bounded(NEW.collection_id) THEN
        RAISE EXCEPTION
            'collection % is ACL-bounded, so %.% must name an acl_group_id: '
            'a row with none is visible to every caller of the tenant',
            NEW.collection_id, TG_TABLE_NAME, NEW.id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER pgkg_chunks_acl_group_required
    AFTER INSERT ON chunks
    REFERENCING NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_acl_group_required();

CREATE TRIGGER pgkg_chunks_acl_group_kept
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    WHEN (NEW.acl_group_id IS NULL
          AND (NEW.acl_group_id IS DISTINCT FROM OLD.acl_group_id
               OR NEW.collection_id IS DISTINCT FROM OLD.collection_id))
    EXECUTE FUNCTION pgkg_acl_group_kept();

CREATE TRIGGER pgkg_propositions_acl_group_required
    AFTER INSERT ON propositions
    REFERENCING NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_acl_group_required();

CREATE TRIGGER pgkg_propositions_acl_group_kept
    BEFORE UPDATE ON propositions
    FOR EACH ROW
    WHEN (NEW.acl_group_id IS NULL
          AND (NEW.acl_group_id IS DISTINCT FROM OLD.acl_group_id
               OR NEW.collection_id IS DISTINCT FROM OLD.collection_id))
    EXECUTE FUNCTION pgkg_acl_group_kept();


-- 3. The queue carries the group, for the same reason 044 made it carry the
-- provenance: a connector that has read a document and knows which group it
-- belongs to should not have to still be alive when a worker reaches it, and
-- the group is not recoverable from the payload.  Without this the HTTP field
-- would work inline and be dropped on the queued path — the silent half-wiring
-- this migration exists to remove.
--
-- The open-work key is unchanged, so a second offer of the same bytes still
-- gets a handle to the work already queued rather than a second copy of it,
-- and does not restate the group any more than it restates its provenance.
ALTER TABLE ingest_jobs ADD COLUMN acl_group_id UUID;

COMMENT ON COLUMN ingest_jobs.acl_group_id IS
    'The ACL group the connector says this document belongs to, carried to the '
    'worker. NULL is refused by the ingest path when the collection is '
    'ACL-bounded (ADR 0001, D3).';

DROP FUNCTION pgkg_enqueue_ingest_job(
    UUID, UUID, TEXT, BYTEA, TEXT, TEXT, TEXT, TIMESTAMPTZ, JSONB
);

CREATE FUNCTION pgkg_enqueue_ingest_job(
    p_org_id        UUID,
    p_collection_id UUID,
    p_external_id   TEXT,
    p_content_hash  BYTEA,
    p_payload       TEXT,
    p_uri           TEXT        DEFAULT NULL,
    p_source        TEXT        DEFAULT NULL,
    p_asserted_at   TIMESTAMPTZ DEFAULT NULL,
    p_provenance    JSONB       DEFAULT NULL,
    p_acl_group_id  UUID        DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_job UUID;
BEGIN
    INSERT INTO ingest_jobs
        (org_id, collection_id, external_id, content_hash, payload, uri,
         source, asserted_at, provenance, acl_group_id)
    VALUES (p_org_id, p_collection_id, p_external_id, p_content_hash,
            p_payload, p_uri, p_source, p_asserted_at, p_provenance,
            p_acl_group_id)
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
    provenance    JSONB,
    acl_group_id  UUID
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
              j.provenance, j.acl_group_id;
$$;
