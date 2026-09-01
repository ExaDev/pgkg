-- 052  Retrievability stops being inferred from parentage, and the content
--      address stops reading chunks.document_id (ADR 0001, D1, D6; issue #18).
--
-- WHY THIS MIGRATION EXISTS.  #18 asks for chunks.document_id to be removed:
-- it is the single-parent pointer that makes a total content address
-- unrepresentable, because two pre-lifecycle documents in one collection
-- sharing a paragraph would have to be one row and that row can name only one
-- parent.  The column is not removed here.  What is removed is every semantic
-- use of it, which is the part that needed a decision rather than a rename, and
-- the measurement of what removing the column itself still costs is recorded
-- below so the next pass does not have to take it again.
--
-- WHAT THE POINTER WAS ACTUALLY DECIDING.  Three things read it, and only one
-- of them was about parentage:
--
--   * liveness (045: `document_id IS NULL AND NOT version_scoped`) read it to
--     mean "this passage is provenance for the facts extracted from it, not
--     retrievable content";
--   * the content address (030, narrowed by 042 and 049) read it in its partial
--     predicate to mean "this row is not one of the rows content addressing
--     governs";
--   * the join from a chat-provenance chunk back to the document it came out
--     of, which is parentage, and is the only one of the three the column is
--     the right shape for.
--
-- The first two are the same claim, and it is not a claim about parentage at
-- all.  041's quota bucketing turns on it ("a retrievable chunk is a document
-- passage by construction: chat provenance is not retrievable at all") and so
-- does the chunks-versus-propositions ablation; 049 recorded that the only
-- shape holding an extraction chunk under a document version without making it
-- retrievable is a version that is never promoted, which is not what 030 means
-- by `pending`.  That is the shape #18 asks for a deliberate answer to, and the
-- deliberate answer is that a writer states it.  A passage is provenance
-- because the pipeline that stored it says so, not because of what else it
-- happens to point at.
--
-- SO THE ANSWER IS A COLUMN, NOT A SHAPE.  045 already moved one inference off
-- parentage for exactly this reason: liveness could not tell a standalone
-- passage from an orphan whose last carrier had been purged, because the record
-- had been destroyed, so the record became `version_scoped`.  This is the same
-- move on the other axis.  `provenance_only` is the writer's statement that a
-- row was not stored as retrievable content, and it is an INPUT to
-- retrievability rather than derived from anything: `retrievable` stays the
-- derived column the read path reads, and the two must not be confused.
--
-- WHY IT IS THE STANDALONE ARM'S GUARD AND NOT AN ABSOLUTE VETO.  045's
-- predicate is two arms: carried by the current version of a live document, or
-- standalone.  `document_id IS NULL` guarded the second arm only, so a row that
-- named a document and was also carried by a version was retrievable — and the
-- shipped plan-shape corpus is exactly that shape, four thousand rows that
-- write the pointer and the links both.  `NOT provenance_only` therefore
-- replaces it in the same position, which makes this a substitution and not a
-- change: every row in every installation keeps the answer it has today.  An
-- absolute veto — withheld even when a version carries it — is the stronger
-- property, and it is the one the extraction path will need if it ever moves
-- onto document versions, because that is the shape 049 could not find.  It is
-- not taken here because the bridge below cannot tell a row that states nothing
-- and will be linked from one that states nothing and will not, so an absolute
-- veto would withdraw content that is retrievable today.  It needs the direct
-- writers of the pointer retired first, which is the same prerequisite as
-- dropping the column.
--
-- WHY THE ADDRESS STAYS PARTIAL, AND WHY THAT IS NOT A HALF MEASURE.  A
-- provenance row must never be reused by another writer, so it belongs OUTSIDE
-- the content-address index, not at a different value inside it: making
-- `provenance_only` a key column would content-address the extraction path,
-- where a turn that repeats a paragraph is two rows on purpose because each
-- carries its own span and its own per-chunk derivation record.  Partial on
-- `NOT provenance_only` says which rows content addressing governs, in the
-- terms that decide it, and it is total over those rows — which is what 030
-- was reaching for when it wrote `WHERE document_id IS NULL` and called the
-- widening "one DROP and CREATE".  After this migration the address does not
-- read the pointer at all, so #16's generated lookup has one fewer axis to
-- drift against and gains no new one: the predicate travels with the address,
-- read from pg_get_expr(indpred) as it already was.
--
-- WHY A BRIDGE TRIGGER RATHER THAN A REWRITE OF EVERY WRITER.  Migrations are
-- forward-only and cannot reach the callers.  Rows already written, and writers
-- that still state parentage instead of provenance, would otherwise become
-- retrievable content-addressed passages the moment the predicates moved.
-- Measured by deleting the trigger: five tests across three modules assert
-- exactly that inference by writing `document_id` directly, and two more
-- collide on the address.  So the old statement is translated into the new one
-- in one named place.  The trigger
-- only ever sets TRUE and never FALSE, so it cannot overrule a writer that
-- states its own answer, and its WHEN clause is the whole condition: a path
-- that has moved off the pointer — corpus ingest, chunks-only chat ingest,
-- every write through pgkg_add_version_chunk() — never enters the function, the
-- same property 030 relies on for the immutability trigger.  This is the only
-- remaining reader of chunks.document_id in the schema, which is what makes
-- dropping the column a mechanical change: retire the writers, drop the
-- trigger, drop the column.
--
-- WHAT REMOVING THE COLUMN STILL COSTS, MEASURED HERE SO IT IS NOT GUESSED
-- AGAIN.  chunks.document_id is still the only record of which document a
-- chat-provenance chunk came out of, and eleven test modules read it as that
-- join.  Giving those chunks their parentage through document_version_chunks
-- instead — which is what #18 scopes — is not one column drop: it needs a
-- document_versions row and a link per chat turn, it moves the extraction
-- path's provenance from per-chunk to per-version (049's rule: a shared row
-- cannot record which ingest produced it) and therefore gives up the span a
-- citation names, and it makes every chat chunk carried by a version, which
-- flips 041's proposition quota bucket — `EXISTS (document_version_chunks ...)`
-- — from 'memory' to 'corpus' for every chat fact, which is D1's drowning
-- failure mode restored.  That re-keying is a change to the retrieval statistics
-- and belongs with them.  Nothing below depends on which way it goes.


-- 1. The statement itself.
ALTER TABLE chunks
    ADD COLUMN provenance_only BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN chunks.provenance_only IS
    'Whether this row was stored as provenance for the facts extracted from it '
    'rather than as retrievable content (041). Stated by the writer, not '
    'inferred from what the row points at, because "not retrievable content" '
    'was never a fact about parentage. It guards the standalone arm of '
    'liveness, in the position chunks.document_id held: a document version that '
    'carries the passage still makes it retrievable. An input to '
    'chunks.retrievable, which stays derived (ADR 0001, D1, D6).';

-- The rows the old inference held.  `NOT version_scoped` is part of it: a row
-- that names a document AND is carried by a version was retrievable under 045's
-- second arm, and marking it provenance would withdraw content the read path
-- returns today.
UPDATE chunks
SET provenance_only = TRUE
WHERE document_id IS NOT NULL
  AND NOT version_scoped;


-- 2. The bridge, for the writers a migration cannot reach.
CREATE FUNCTION pgkg_chunks_provenance_bridge() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.provenance_only := TRUE;
    RETURN NEW;
END;
$$;

COMMENT ON FUNCTION pgkg_chunks_provenance_bridge() IS
    'Translates the pre-lifecycle parent pointer into the statement that '
    'replaced it: a row that names one document is that document''s '
    'provenance, which is what chunks.document_id has meant since 030. Only '
    'ever sets TRUE, so a writer that states its own answer is untouched, and '
    'the WHEN clause keeps every writer that has moved off the pointer out of '
    'the function. Delete this with the column (ADR 0001, D1; issue #18).';

-- UPDATE OF document_id, not UPDATE: the bridge's job is to translate a write
-- of the pointer, and refcount maintenance is the highest-volume write on this
-- table (031, 041).  Without the column list a legacy row that names a document
-- and states nothing would be translated by whichever refcount update reached
-- it first — harmless, since a carried row stays retrievable either way, but it
-- would put a per-row trigger and one statistics reconciliation on a path 041
-- required not to reach the statistics at all.
CREATE TRIGGER pgkg_chunks_provenance_bridge
    BEFORE INSERT OR UPDATE OF document_id ON chunks
    FOR EACH ROW
    WHEN (NEW.document_id IS NOT NULL AND NOT NEW.provenance_only)
    EXECUTE FUNCTION pgkg_chunks_provenance_bridge();


-- 3. Liveness, on the statement instead of the pointer.
--
-- A redeclaration because the signature moves — the pointer leaves the
-- parameter list and the statement takes its place — which CREATE OR REPLACE
-- cannot do.  The two arms and their order are 045's, unchanged; only the
-- standalone arm's guard is now the writer's statement rather than the pointer.
DROP FUNCTION pgkg_chunk_retrievable(UUID, UUID, BOOLEAN);

CREATE FUNCTION pgkg_chunk_retrievable(
    p_chunk_id        UUID,
    p_provenance_only BOOLEAN,
    p_version_scoped  BOOLEAN
) RETURNS BOOLEAN
LANGUAGE SQL STABLE
AS $$
    SELECT (NOT p_provenance_only AND NOT p_version_scoped)
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

COMMENT ON FUNCTION pgkg_chunk_retrievable(UUID, BOOLEAN, BOOLEAN) IS
    'Whether retrieval may see a chunk: carried by the current version of a '
    'live document, or stored as retrievable content and never carried by any '
    'version. Not on the read path — the read path reads chunks.retrievable — '
    'so the sublink that stops it inlining costs nothing here (ADR 0001, D1, '
    'D6).';


-- 4. The reconciliation, reading the new input.
--
-- Unchanged except for the column it carries into the decision, so an edit
-- rather than a redeclaration.
CREATE OR REPLACE FUNCTION pgkg_chunk_retrievability_sync(p_chunk_ids UUID[])
RETURNS VOID
LANGUAGE SQL
AS $$
WITH target AS (
    SELECT c.id, c.org_id, c.collection_id, c.doc_len, c.tsv,
           c.provenance_only,
           c.retrievable    AS was_retrievable,
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
           pgkg_chunk_retrievable(t.id, t.provenance_only, t.scoped)
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


-- 5. The chunks UPDATE trigger watches the column that decides the answer.
--
-- 041 narrowed this to document_id because it was "the only column on the row
-- that changes the answer and is not already covered by the link trigger".
-- That column is now provenance_only, and the narrowing is the same narrowing:
-- refcount maintenance is the highest-volume write on this table and 031
-- required it not to reach the statistics at all.  A row that acquires a parent
-- pointer still reaches here, because the bridge above turns that write into a
-- change of provenance_only.
CREATE OR REPLACE FUNCTION pgkg_chunks_retrievability_update() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_ids UUID[];
BEGIN
    SELECT array_agg(DISTINCT n.id) INTO v_ids
    FROM new_rows n
    JOIN old_rows o ON o.id = n.id
    WHERE n.provenance_only IS DISTINCT FROM o.provenance_only;

    IF v_ids IS NOT NULL THEN
        PERFORM pgkg_chunk_retrievability_sync(v_ids);
    END IF;

    RETURN NULL;
END;
$$;


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
            pgkg_chunk_retrievable(c.id, c.provenance_only, c.version_scoped)
    WHERE (p_collection_id IS NULL OR c.collection_id = p_collection_id)
      AND c.retrievable IS DISTINCT FROM
          pgkg_chunk_retrievable(c.id, c.provenance_only, c.version_scoped);

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


-- 7. The content address, over the rows content addressing governs, said in the
-- terms that decide it.
--
-- Same key columns as 049 — nothing about which columns key the address has
-- changed, and widening it is a different question from which rows it covers.
DROP INDEX chunks_content_addressed_key;

CREATE UNIQUE INDEX chunks_content_addressed_key
    ON chunks (
        org_id,
        collection_id,
        (COALESCE(acl_group_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        visibility,
        (COALESCE(owner_user_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        content_hash
    )
    WHERE NOT provenance_only;

COMMENT ON INDEX chunks_content_addressed_key IS
    'The content address. Keyed on every column pgkg_visible() reads as well '
    'as the hash (049), over every row that is retrievable content: a '
    'provenance row is outside the index rather than at a different value '
    'inside it, because it must never be reused by another writer and because '
    'the extraction path repeats a paragraph as two rows on purpose — each '
    'carries its own span and its own derivation record (ADR 0001, D1, D3, D6).';


-- 8. The reuse path, on the same predicate.
--
-- An edit rather than a redeclaration: the signature is 049's, and only the two
-- statements of which rows are content-addressed move.  A plpgsql body has no
-- smaller unit than itself, so the body is restated with those two clauses
-- changed and nothing else.
--
-- provenance_only is not passed and not written: a passage added to a document
-- version is retrievable content by construction, which is the whole reason
-- 049's chunks-only path writes through here.
CREATE OR REPLACE FUNCTION pgkg_add_version_chunk(
    p_version_id    UUID,
    p_ord           INT,
    p_text          TEXT,
    p_acl_group_id  UUID        DEFAULT NULL,
    p_asserted_at   TIMESTAMPTZ DEFAULT NULL,
    p_visibility    TEXT        DEFAULT 'shared',
    p_owner_user_id UUID        DEFAULT NULL
) RETURNS TABLE (chunk_id UUID, is_new BOOLEAN)
LANGUAGE plpgsql
AS $$
DECLARE
    v_org        UUID;
    v_collection UUID;
    v_provenance UUID;
    v_hash       BYTEA := digest(p_text, 'sha256');
    v_chunk      UUID;
    v_is_new     BOOLEAN := FALSE;
BEGIN
    SELECT d.org_id, d.collection_id, dv.provenance_id
    INTO v_org, v_collection, v_provenance
    FROM document_versions dv
    JOIN documents d ON d.id = dv.document_id
    WHERE dv.id = p_version_id;

    IF v_org IS NULL THEN
        RAISE EXCEPTION 'no such document version %', p_version_id;
    END IF;

    INSERT INTO chunks (text, org_id, collection_id, acl_group_id,
                        provenance_id, asserted_at, visibility, owner_user_id)
    VALUES (p_text, v_org, v_collection, p_acl_group_id,
            v_provenance, p_asserted_at, p_visibility, p_owner_user_id)
    ON CONFLICT (
        org_id,
        collection_id,
        (COALESCE(acl_group_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        visibility,
        (COALESCE(owner_user_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        content_hash
    ) WHERE NOT provenance_only DO NOTHING
    RETURNING chunks.id INTO v_chunk;

    IF v_chunk IS NULL THEN
        SELECT c.id INTO v_chunk
        FROM chunks c
        WHERE c.org_id = v_org
          AND c.collection_id = v_collection
          AND c.acl_group_id IS NOT DISTINCT FROM p_acl_group_id
          AND c.visibility = p_visibility
          AND c.owner_user_id IS NOT DISTINCT FROM p_owner_user_id
          AND c.content_hash = v_hash
          AND NOT c.provenance_only;
    ELSE
        v_is_new := TRUE;
    END IF;

    INSERT INTO document_version_chunks (document_version_id, chunk_id, ord)
    VALUES (p_version_id, v_chunk, p_ord)
    ON CONFLICT ON CONSTRAINT document_version_chunks_pkey DO NOTHING;

    RETURN QUERY SELECT v_chunk, v_is_new;
END;
$$;


-- 9. What the pointer is for now.
COMMENT ON COLUMN chunks.document_id IS
    'The pre-lifecycle single-parent pointer, and now only the record of which '
    'document a chat-provenance chunk came out of. Retrievability and the '
    'content address both read provenance_only instead, so the pointer decides '
    'nothing; pgkg_chunks_provenance_bridge() is the one object left that reads '
    'it, and it exists to translate a write of this column into that one. It '
    'cannot be dropped while it is still that record — document_version_chunks '
    'is the shape that can hold it, and moving the extraction path onto '
    'versions re-keys 041''s quota bucket (issue #18).';


-- 10. Reconcile every row against the definition that now applies.
SELECT pgkg_refresh_chunk_stats();
