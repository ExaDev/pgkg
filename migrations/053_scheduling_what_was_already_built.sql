-- The scheduled path: what a timer needs that the jobs themselves do not have
-- (ADR 0001, D2, D4; docs/adrs/0001-implementation-notes.md §4; issue #19).
--
-- WHY THIS MIGRATION EXISTS.  Four jobs were built, tested and never run.
-- `pgkg_match_entity_mentions()` is the one that mattered most: D2 makes the
-- mention edge the answer to "is a corpus-graph relationship worth having",
-- and nothing in the product called it, so `entity_mentions` was empty in every
-- deployment while the tests that drove the matcher directly stayed green.
-- `pgkg_recompute_pagerank()`, `pgkg_contradict()` and `pgkg_expire_due()` are
-- recorded in the implementation notes' phase-4 table as existing and
-- unscheduled for the same reason.  Nothing below re-implements any of them.
-- What is added is only what a scheduler needs and a hand-run does not: a
-- watermark for the direction the sweep could not reach, a candidate rule that
-- turns a per-row function into a job, an org argument on the one job that had
-- none, and a lock so a cron entry that overlaps itself declines instead of
-- repeating work.
--
-- WHY THE NAME SIDE NEEDS A WATERMARK OF ITS OWN.  040 put the watermark on
-- the chunk, and `pgkg_unmatched_chunks()` reads it: a sweep asks which
-- passages have never been matched and matches those against every name the org
-- knows.  That is complete only for names that already existed when the passage
-- arrived.  In steady state the order is the other way round — a corpus is bulk
-- loaded and swept, and the chat turns naming what it talks about arrive for
-- months afterwards.  Those passages are stamped, so no chunk-side sweep will
-- ever look at them again, and the entity a chat fact creates has no route
-- back: the D2 win ("a retrieved chat fact seeds an entity which pulls in the
-- Helios architecture doc") is exactly the case the chunk watermark cannot
-- serve.  A watermark on the entity closes it, and the two together are
-- complete over pairs: for any (entity, passage) pair one of the two arrived
-- last, and the sweep for that side sees the other already stored.
--
-- The stamp is cleared when either key the matcher probes changes, because
-- entity resolution keeps adding aliases to rows that already exist and a new
-- alias is a new phrase to match.  A watermark that outlives the name it was
-- stamped for is a stale answer, not a saved scan.
--
-- WHY A SUPERSESSION IS THE ONLY CONTRADICTION CANDIDATE THIS SCHEMA CAN NAME.
-- The ADR's contradiction stage is "same (subject, predicate), different
-- object, where the predicate is single-valued", and it says plainly that
-- free-text predicates make that undecidable: without a `predicates` table
-- carrying `is_functional`, "works at" and "is employed by" are unrelated
-- relations and auto-invalidating a multi-valued one is a bug.  That table is
-- deferred, and this migration does not pre-empt it.  What it schedules instead
-- is the contradiction someone already asserted: `superseded_by` records that
-- one claim replaced another, 021's trigger withdraws the replaced row from
-- belief, and nothing has ever closed its `valid_to`.  So the as-of-validity
-- mode still answers with the replaced claim for every instant after its
-- replacement — a real defect in the audit path, with a candidate rule that
-- guesses nothing.  D4's first rule bounds it: resolution runs only within a
-- claim scope, so a user-scoped note superseding an org-scoped policy is a
-- personal exception and the policy's validity stays open.
--
-- WHY pgkg_expire_due() GAINS AN ORG AND KEEPS ITS OLD MEANING.  The TTL sweep
-- has to run per tenant, because that is how an operator schedules anything in
-- a multi-tenant product, and as written it had no way to say which tenant: a
-- run for one customer withdrew every customer's due rows wherever the
-- connection is not subject to row security (a migration role, a maintenance
-- role, the owner).  Unlike 025, NULL here means every org rather than the
-- request scope, so the operator-facing one-argument call keeps doing exactly
-- what it did; the scheduled path always states its org.
--
-- WHY THE LOCK IS ADVISORY AND SESSION SCOPED.  A cron entry that overlaps
-- itself is the normal failure mode of anything on a timer, and the failure it
-- causes here is wasted work rather than corruption — the mention insert is
-- ON CONFLICT DO NOTHING and both watermarks are set under an IS NULL
-- predicate.  Wasted work is still the wrong answer for a sweep whose yield an
-- operator reads, so a second run declines the task.  Advisory rather than a
-- row in a table because a lock in a table needs a lease, a heartbeat and a
-- sweeper for the runs that died holding it, all of which the ingest queue
-- already pays for and none of which this needs: a session lock dies with the
-- backend, and the pool releases it when the connection goes back (asyncpg's
-- reset runs pg_advisory_unlock_all()).  Keyed per (task, org) so one tenant's
-- long sweep does not stop every other tenant's.


-- 1. The name-side watermark.
ALTER TABLE entities ADD COLUMN mentions_matched_at TIMESTAMPTZ;

COMMENT ON COLUMN entities.mentions_matched_at IS
    'When this name was last matched against the passages already stored. The '
    'mirror of chunks.mentions_matched_at: that one makes a new passage find '
    'the standing names, this one makes a new name find the standing corpus, '
    'and neither direction is reachable from the other (ADR 0001, D2; #19).';

CREATE INDEX entities_mentions_unmatched_idx
    ON entities (org_id, created_at)
    WHERE mentions_matched_at IS NULL;


-- The stamp is only valid for the keys it was taken over.
CREATE FUNCTION pgkg_reset_entity_mention_watermark() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.mentions_matched_at := NULL;
    RETURN NEW;
END;
$$;

CREATE TRIGGER pgkg_entity_mention_watermark_reset
    BEFORE UPDATE ON entities
    FOR EACH ROW
    WHEN (NEW.name IS DISTINCT FROM OLD.name
          OR NEW.aliases IS DISTINCT FROM OLD.aliases)
    EXECUTE FUNCTION pgkg_reset_entity_mention_watermark();


-- 2. The name-side sweep: one batch of the names this org has never matched.
--
-- One function rather than a queue view plus a matcher call, because the stamp
-- has to be taken over exactly the rows that were matched — a caller that read
-- a candidate list, matched it and stamped it in three round trips would stamp
-- a set the matcher never saw if it died in the middle.  FOR UPDATE SKIP LOCKED
-- so two overlapping runs take different names rather than the same ones; the
-- advisory lock above normally means there is only one, and this is what makes
-- the function safe for the operator who runs it by hand anyway.
CREATE FUNCTION pgkg_sweep_entity_mentions(
    p_org_id     UUID,
    p_limit      INT  DEFAULT 1000,
    p_max_chunks INT  DEFAULT 1000,
    p_max_words  INT  DEFAULT 5,
    p_threshold  REAL DEFAULT 0.9
) RETURNS TABLE (names_scanned BIGINT, mentions_added BIGINT)
LANGUAGE plpgsql
AS $$
DECLARE
    v_ids   UUID[];
    v_added BIGINT;
BEGIN
    SELECT array_agg(pending.id) INTO v_ids
    FROM (
        SELECT e.id
        FROM entities e
        WHERE e.org_id = p_org_id
          AND e.mentions_matched_at IS NULL
        ORDER BY e.created_at, e.id
        LIMIT GREATEST(p_limit, 0)
        FOR UPDATE SKIP LOCKED
    ) AS pending;

    IF v_ids IS NULL THEN
        RETURN QUERY SELECT 0::BIGINT, 0::BIGINT;
        RETURN;
    END IF;

    v_added := pgkg_match_chunk_mentions(
        v_ids, p_max_chunks, p_max_words, p_threshold
    );

    UPDATE entities SET mentions_matched_at = now()
    WHERE id = ANY(v_ids)
      AND mentions_matched_at IS NULL;

    RETURN QUERY SELECT array_length(v_ids, 1)::BIGINT, v_added;
END;
$$;

COMMENT ON FUNCTION pgkg_sweep_entity_mentions(UUID, INT, INT, INT, REAL) IS
    'One batch of the names this org has never matched against its passages, '
    'matched and stamped together. The reverse of the chunk-side sweep and not '
    'reachable from it (ADR 0001, D2; #19). Restricted to the org''s own names: '
    'a shared entity in the system org is readable by every subscriber and '
    'writable by none of them, so the operator''s own run sweeps those.';


-- 3. The contradiction candidate: a supersession whose validity is still open.
--
-- The work is pgkg_contradict()'s, unchanged and called per row, so the
-- semantics of closing a validity interval live in exactly one place. The
-- effective instant is the replacement's own clock, not now(): the world
-- changed when the newer claim became true, and a sweep that ran late would
-- otherwise record the change as having happened when the sweep noticed it.
CREATE FUNCTION pgkg_contradict_superseded(
    p_org_id UUID DEFAULT NULL,
    p_limit  INT  DEFAULT 1000
) RETURNS TABLE (considered BIGINT, closed BIGINT)
LANGUAGE plpgsql
AS $$
DECLARE
    v_org        UUID := COALESCE(p_org_id, pgkg_current_org());
    v_candidate  RECORD;
    v_considered BIGINT := 0;
    v_closed     BIGINT := 0;
BEGIN
    FOR v_candidate IN
        SELECT stale.id AS id,
               COALESCE(fresh.valid_from, fresh.recorded_at) AS effective
        FROM propositions stale
        JOIN propositions fresh ON fresh.id = stale.superseded_by
        WHERE stale.org_id = v_org
          AND fresh.org_id = stale.org_id
          -- 021: `superseded_by = id` was the old retirement idiom, "replaced
          -- by itself", and says nothing about validity.
          AND stale.superseded_by <> stale.id
          AND stale.valid_to IS NULL
          -- D4: never across a claim scope. Cross-scope disagreement is
          -- tension, and tension is surfaced, not resolved.
          AND fresh.claim_scope = stale.claim_scope
          AND COALESCE(fresh.valid_from, fresh.recorded_at) IS NOT NULL
          -- propositions_validity_runs_forwards would reject the close, and an
          -- error would take the whole run down rather than this one row: a
          -- replacement that predates what it replaces is a data problem for a
          -- human, not a job to retry every tick.
          AND (stale.valid_from IS NULL
               OR COALESCE(fresh.valid_from, fresh.recorded_at)
                  > stale.valid_from)
        ORDER BY stale.recorded_at, stale.id
        LIMIT GREATEST(p_limit, 0)
        FOR UPDATE OF stale SKIP LOCKED
    LOOP
        v_considered := v_considered + 1;
        v_closed := v_closed
                  + pgkg_contradict(v_candidate.id, v_candidate.effective);
    END LOOP;

    RETURN QUERY SELECT v_considered, v_closed;
END;
$$;

COMMENT ON FUNCTION pgkg_contradict_superseded(UUID, INT) IS
    'Closes the validity interval of claims a supersession withdrew from '
    'belief and left valid forever. The one contradiction candidate this '
    'schema can name without the deferred predicate vocabulary, and it never '
    'crosses a claim scope (ADR 0001, D4).';


-- 4. The TTL sweep, per tenant.
--
-- Dropped and recreated rather than edited: the body is 021's, with one
-- predicate added. A one-argument call resolves to this definition and behaves
-- as it did, so the operator-facing signature is unchanged.
DROP FUNCTION pgkg_expire_due(TEXT);

CREATE FUNCTION pgkg_expire_due(
    p_namespace TEXT DEFAULT NULL,
    p_org_id    UUID DEFAULT NULL
) RETURNS BIGINT
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
          AND (p_org_id IS NULL OR p.org_id = p_org_id)
        RETURNING 1
    )
    SELECT COUNT(*) INTO affected FROM withdrawn;

    RETURN affected;
END;
$$;

COMMENT ON FUNCTION pgkg_expire_due(TEXT, UUID) IS
    'Withdraws due claims from belief; legal_hold wins over the TTL. NULL org '
    'means every org, which is what the pre-053 signature did — the scheduled '
    'path always states one, because a maintenance run belongs to a tenant.';


-- 5. The overlap guard.
--
-- Two functions rather than an inline pg_try_advisory_lock at each call site,
-- so the key derivation exists once: two callers that hash the task name
-- differently do not exclude each other, and that failure is invisible until a
-- night when both ran.
CREATE FUNCTION pgkg_try_maintenance_lock(
    p_task TEXT, p_org_id UUID DEFAULT NULL
) RETURNS BOOLEAN
LANGUAGE SQL
AS $$
    SELECT pg_try_advisory_lock(
        hashtext('pgkg_maintenance'),
        hashtext(p_task || ':' || COALESCE(p_org_id, pgkg_current_org())::TEXT)
    )
$$;

CREATE FUNCTION pgkg_release_maintenance_lock(
    p_task TEXT, p_org_id UUID DEFAULT NULL
) RETURNS BOOLEAN
LANGUAGE SQL
AS $$
    SELECT pg_advisory_unlock(
        hashtext('pgkg_maintenance'),
        hashtext(p_task || ':' || COALESCE(p_org_id, pgkg_current_org())::TEXT)
    )
$$;

COMMENT ON FUNCTION pgkg_try_maintenance_lock(TEXT, UUID) IS
    'FALSE when another run of this task for this org is in flight. Session '
    'scoped: it dies with the backend, so a run that crashed holding it blocks '
    'nothing (#19).';
