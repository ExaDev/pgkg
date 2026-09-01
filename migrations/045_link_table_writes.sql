-- The link table as a write surface, and liveness that survives reclamation
-- (ADR 0001, D3, D4, D6).
--
-- Re-verification of the fix pass found one residual and one regression on
-- document_version_chunks, and pulling on the second found a third defect on
-- the shipped lifecycle path.  All three are the same table reached by
-- statements the phase-2 and phase-3 work did not consider: 042's trigger
-- refuses a link whose two sides belong to different orgs and 041's triggers
-- move refcount and retrievability when a link appears or is purged, both on
-- INSERT and DELETE only.  A link is (document_version_id, ord) -> chunk_id, so
-- an UPDATE is a repoint: it changes which passage a position of a document
-- carries without inserting or deleting anything, which is exactly the event
-- neither guard watches.
--
-- WHY VISIBILITY IS NOT THE WRITE RULE.  033 gave the table one FOR ALL policy
-- whose USING and WITH CHECK both read `EXISTS (SELECT 1 FROM document_versions
-- dv WHERE dv.id = document_version_id)` — the version's own policy applied
-- inside the subquery, so a stranger's version read as absent and could not be
-- linked through.  043 then widened document_versions' USING to the operator's
-- system org, because D4's sharing seam is exactly that a subscriber can read
-- the operator's corpus, and the link table inherited the widening on both
-- sides of its policy.  So a tenant connected as pgkg_app, holding no
-- subscription and naming only its own org, could UPDATE the operator's links:
-- with both sides of the link already in the system org 042's same-org trigger
-- is satisfied, and 041's triggers do not fire on UPDATE at all.  Measured
-- after such a write, the operator's document came back with another document's
-- opening passage spliced into its context_text for every subscribing tenant —
-- C2's laundering shape reached by a write instead of a read — and the passage
-- the repoint displaced kept a refcount with no link behind it.
--
-- The rule the table needs is D4's first hard rule, the one 043 states
-- everywhere else: reads widen to the operator's shelf and writes never do.  A
-- write to the link table is own-org on BOTH sides, which is also what C4 asked
-- for and 033 never said — the chunk side was never checked at all.  It is two
-- policies rather than one because the two rules are genuinely different: FOR
-- SELECT keeps 033's inherited visibility, so the seam still carries a shared
-- document's ordering to a subscriber, and FOR ALL carries the own-org rule
-- that governs INSERT, UPDATE and DELETE.  Permissive policies are OR-ed, so
-- the pair reads as "visible to read, mine to write".  042's trigger stays and
-- is not weakened by this: a policy is inert for an owner or a BYPASSRLS
-- connection, where a link between two orgs is still meaningless.
--
-- Of the two halves USING is the load-bearing one and WITH CHECK states the
-- rule, which probing showed rather than assumed.  USING is what makes a
-- repoint, a renumber and an unlink of another org's link find no row, so the
-- statement completes having touched nothing.  On the INSERT path a tenant
-- appending to the operator's document is refused before WITH CHECK is
-- consulted, because the link fires 030's refcount trigger, whose UPDATE of
-- the operator's chunk violates that table's own policy — and a link between
-- two orgs never reaches either, because 042's trigger raises first.  So the
-- WITH CHECK term buys no reachable protection today; it is here because the
-- rule belongs on the table that holds the row rather than being an emergent
-- property of two other objects' triggers.
--
-- WHY LIVENESS STOPS BEING A FUNCTION OF refcount.  Reconciling the refcount on
-- a repoint is not enough, and chasing why exposed the third defect.  Liveness
-- read `(refcount = 0 AND document_id IS NULL) OR EXISTS (current live version
-- link)` — 031's predicate, which 041 stored on the row without changing it.
-- The first arm is meant for a standalone passage: a chunk that belongs to no
-- document and no version stands on its own, which is what 033 says.  But an
-- orphan matches it too.  A passage dropped by a new version is correctly not
-- retrievable while a retired version still carries it, and then
-- pgkg_purge_retired_versions() reclaims that version, the last link goes, the
-- refcount falls to zero — and the first arm readmits the passage as a
-- standalone one.  Measured on the shipped path with no manual write anywhere:
-- a passage dropped at v2, retrievable FALSE after promotion, comes back
-- retrievable TRUE the moment v1 is purged and is returned by the keyword arm
-- again.  Withdrawn content is resurrected by garbage collection, and
-- permanently for any passage a proposition was extracted from, since
-- pgkg_gc_chunks() deliberately refuses to collect those.
--
-- This is not a regression of the fix pass — 031 and 033's pgkg_chunk_live()
-- computed the same boolean — but it is the defect the re-verifier's finding is
-- about, reached through the front door, and reconciling the refcount without
-- it would move the repoint route from accidentally right to definitionally
-- wrong.  The two cases are indistinguishable from (document_id, refcount)
-- because the purge destroys the only record that the chunk was ever carried,
-- so that record becomes a column: version_scoped, set when a link to the chunk
-- first exists and never cleared.  Liveness then reads "carried by a live
-- current version, or never carried by anything", and refcount leaves the
-- predicate entirely — which is the shape the H5 analysis argued for, since
-- liveness never was a function of a count of links.
--
-- WHY THE REPOINT IS RECONCILED RATHER THAN REFUSED.  Derived state has to
-- survive every write that can change it, not only the two the ingest path
-- uses.  Repointing a link is a legitimate manual repair for an operator, and
-- the honest response to a repair is to make the derived state agree with it
-- rather than to forbid the repair.  What a repoint means for the window is
-- already 042's answer: the substituted passage now has two carriers, so it is
-- its own context.  Refcount is reconciled from the link table rather than
-- moved by a signed delta, because a repoint is a decrement and an increment in
-- one statement and an ord-only UPDATE is neither; reconciliation is also what
-- makes the trigger idempotent, the property 041 relies on to let one statement
-- visit a chunk twice and count it once.


-- 1. Reads widen, writes do not.
DROP POLICY document_version_chunks_org_isolation ON document_version_chunks;

CREATE POLICY document_version_chunks_read ON document_version_chunks
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM document_versions dv
            WHERE dv.id = document_version_id
        )
    );

CREATE POLICY document_version_chunks_write ON document_version_chunks
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM document_versions dv
            WHERE dv.id = document_version_id
              AND dv.org_id = pgkg_current_org()
        )
        AND EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.id = chunk_id
              AND c.org_id = pgkg_current_org()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM document_versions dv
            WHERE dv.id = document_version_id
              AND dv.org_id = pgkg_current_org()
        )
        AND EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.id = chunk_id
              AND c.org_id = pgkg_current_org()
        )
    );

COMMENT ON POLICY document_version_chunks_read ON document_version_chunks IS
    'What a version carries is as readable as the version, which 043 widened to '
    'the operator''s shared org so a subscriber can read a shared document''s '
    'ordering (ADR 0001, D4).';

COMMENT ON POLICY document_version_chunks_write ON document_version_chunks IS
    'A link is written only by the org that owns both sides of it. The table '
    'carries no org column, so the rule is read through the version and the '
    'chunk — both, because a policy that checked only the version let a tenant '
    'graft a stranger''s passage into its own document and let a subscriber '
    'repoint the operator''s (ADR 0001, D3, D4).';


-- 2. The record the purge used to destroy.
ALTER TABLE chunks
    ADD COLUMN version_scoped BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN chunks.version_scoped IS
    'Whether a document version has ever carried this passage. Set when the '
    'first link to it exists and never cleared, because it is what separates a '
    'standalone passage from one whose last carrier was reclaimed: without it '
    'the purge readmits withdrawn content as a passage that belongs to nothing '
    '(ADR 0001, D6).';

UPDATE chunks c
SET version_scoped = TRUE
WHERE c.refcount > 0
   OR EXISTS (
       SELECT 1 FROM document_version_chunks dvc WHERE dvc.chunk_id = c.id
   );


-- 3. What liveness means, without refcount in it.
DROP FUNCTION pgkg_chunk_retrievable(UUID, UUID, INT);

CREATE FUNCTION pgkg_chunk_retrievable(
    p_chunk_id       UUID,
    p_document_id    UUID,
    p_version_scoped BOOLEAN
) RETURNS BOOLEAN
LANGUAGE SQL STABLE
AS $$
    SELECT (p_document_id IS NULL AND NOT p_version_scoped)
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

COMMENT ON FUNCTION pgkg_chunk_retrievable(UUID, UUID, BOOLEAN) IS
    'Whether retrieval may see a chunk: carried by the current version of a '
    'live document, or belonging to no document and never carried by any '
    'version. Not on the read path — the read path reads chunks.retrievable — '
    'so the sublink that stops it inlining costs nothing here (ADR 0001, D1, '
    'D6).';


-- 4. Reconciliation, now maintaining both columns.
--
-- One statement for both, because a flag written without its statistics delta —
-- or a delta applied twice — is drift in the tables the ranking reads, and
-- because version_scoped is an input to the flag computed in the same pass.
-- Only rows whose answer moved are written, which is what keeps the function
-- idempotent.
CREATE OR REPLACE FUNCTION pgkg_chunk_retrievability_sync(p_chunk_ids UUID[])
RETURNS VOID
LANGUAGE SQL
AS $$
WITH target AS (
    SELECT c.id, c.org_id, c.collection_id, c.doc_len, c.tsv, c.document_id,
           c.retrievable   AS was_retrievable,
           c.version_scoped AS was_scoped,
           c.version_scoped OR EXISTS (
               SELECT 1 FROM document_version_chunks dvc
               WHERE dvc.chunk_id = c.id
           ) AS scoped
    FROM chunks c
    WHERE c.id = ANY(p_chunk_ids)
),
decided AS (
    SELECT t.*,
           pgkg_chunk_retrievable(t.id, t.document_id, t.scoped)
               AS is_retrievable
    FROM target t
),
marked AS (
    UPDATE chunks c
    SET retrievable    = d.is_retrievable,
        version_scoped = d.scoped
    FROM decided d
    WHERE c.id = d.id
      AND (c.retrievable    IS DISTINCT FROM d.is_retrievable
        OR c.version_scoped IS DISTINCT FROM d.scoped)
    RETURNING c.id
),
flipped AS (
    SELECT d.*, CASE WHEN d.is_retrievable THEN 1 ELSE -1 END AS delta_sign
    FROM decided d
    WHERE d.is_retrievable IS DISTINCT FROM d.was_retrievable
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


-- 5. The event the trigger set was missing.
--
-- Refcount first, then the flag, in one trigger rather than two: they are read
-- by the collector and by the scan respectively, and a statement that corrected
-- one without the other would leave the pair disagreeing for the length of the
-- transaction.
CREATE FUNCTION pgkg_version_chunks_relink() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_ids UUID[];
BEGIN
    SELECT array_agg(DISTINCT touched.chunk_id) INTO v_ids
    FROM (
        SELECT chunk_id FROM old_links
        UNION
        SELECT chunk_id FROM new_links
    ) touched;

    IF v_ids IS NULL THEN
        RETURN NULL;
    END IF;

    UPDATE chunks c
    SET refcount = counted.links
    FROM (
        SELECT c2.id,
               (SELECT count(*)::INT
                FROM document_version_chunks dvc
                WHERE dvc.chunk_id = c2.id) AS links
        FROM chunks c2
        WHERE c2.id = ANY(v_ids)
    ) counted
    WHERE c.id = counted.id
      AND c.refcount IS DISTINCT FROM counted.links;

    PERFORM pgkg_chunk_retrievability_sync(v_ids);

    RETURN NULL;
END;
$$;

COMMENT ON FUNCTION pgkg_version_chunks_relink() IS
    'Brings a chunk''s refcount and retrievability back into agreement with the '
    'link table after a link is repointed or renumbered. Reconciles rather than '
    'applying a delta, so an ord-only update costs one comparison and a repoint '
    'is one decrement and one increment in the same statement (ADR 0001, D6).';

CREATE TRIGGER pgkg_version_chunks_relink_update
    AFTER UPDATE ON document_version_chunks
    REFERENCING OLD TABLE AS old_links NEW TABLE AS new_links
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_relink();


-- 6. The full refresh, on the new predicate.
CREATE OR REPLACE FUNCTION pgkg_refresh_chunk_stats(p_collection_id UUID DEFAULT NULL)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE chunks c
    SET version_scoped = TRUE
    WHERE NOT c.version_scoped
      AND (p_collection_id IS NULL OR c.collection_id = p_collection_id)
      AND EXISTS (
          SELECT 1 FROM document_version_chunks dvc WHERE dvc.chunk_id = c.id
      );

    UPDATE chunks c
    SET retrievable =
            pgkg_chunk_retrievable(c.id, c.document_id, c.version_scoped)
    WHERE (p_collection_id IS NULL OR c.collection_id = p_collection_id)
      AND c.retrievable IS DISTINCT FROM
          pgkg_chunk_retrievable(c.id, c.document_id, c.version_scoped);

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


-- 7. The drift an installation may already carry.
--
-- A refcount that does not count links, from an UPDATE taken between 041 and
-- here, and a flag derived from the old predicate.  Reconciled from the link
-- table by hash join rather than by a correlated count per chunk, and the
-- statistics rebuilt from the repaired flag.  What cannot be recovered is a
-- passage already orphaned by a purge: the record that a version once carried
-- it is exactly what was destroyed, so it stays a standalone passage and the
-- collector remains its only exit.
WITH links AS (
    SELECT chunk_id, count(*)::INT AS n
    FROM document_version_chunks
    GROUP BY chunk_id
)
UPDATE chunks c
SET refcount = COALESCE(l.n, 0)
FROM chunks c2
LEFT JOIN links l ON l.chunk_id = c2.id
WHERE c.id = c2.id
  AND c.refcount IS DISTINCT FROM COALESCE(l.n, 0);

SELECT pgkg_refresh_chunk_stats();
