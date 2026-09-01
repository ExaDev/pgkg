-- 049  Chunks-only chat ingest joins the chunk store, so the content address
--      has to carry who may read a passage.
--
-- WHY THIS MIGRATION EXISTS.  ADR 0001, D1 lists "chunk embeddings in
-- `propositions`" under what not to build — it corrupts BM25 length
-- normalisation, conflates lifecycles and blocks per-store index tuning — and
-- says of the existing instance: undo it.  Phase 2 names the replacement:
-- chunks-only mode becomes "retrieve from the chunk source" and the
-- fake-proposition rows disappear.  Memory.ingest(extract_propositions=False)
-- now writes real chunks through pgkg_add_version_chunk(), which makes it the
-- first writer of a *retrievable private* passage — and that is what this
-- migration is for.
--
-- WHY VISIBILITY AND OWNER JOIN THE CONTENT ADDRESS.  042 narrowed the address
-- to (org_id, collection_id, acl_group, content_hash) on one rule: dedup stops
-- at every boundary a read predicate consults, because a row shared across a
-- scoping boundary cannot carry the boundary.  pgkg_visible() reads five
-- columns, and 042 put three of them in the address.  The corpus never
-- exercised the other two — every corpus chunk is 'shared' with no owner — so
-- the gap cost nothing until chat ingest arrived, where D3's private lane is
-- the point: a user's own note and the org's copy of the same sentence are one
-- row under the old address, and one row cannot hold two owners.  Whichever
-- writer landed first would decide, and the answer is wrong in both directions
-- — a private note published to the org, or an org passage that only one user
-- can retrieve.
--
-- So the address gains `visibility` and `owner_user_id`, on the same rule and
-- for the same reason as `collection_id` and `acl_group_id`.  What it costs is
-- disk on the private lane only: `shared`/NULL is one address, so every corpus
-- row and every default-scoped chat row dedups exactly as before.
--
-- WHAT IS NOT HERE.  The address stays partial on `document_id IS NULL`.  030
-- said widening it to every row is one DROP and CREATE "once ingest moves onto
-- the function below", and only half of ingest has moved: extraction mode still
-- writes the pre-lifecycle parent pointer, deliberately.  A chat turn whose
-- passages exist only as provenance for the facts extracted from them is not
-- retrievable content (041), and the only way to hold a passage under a
-- document version without making it retrievable is a version that is never
-- promoted — which is not what 030 means by 'pending'.  Paying a
-- document_versions row and a link row per chat turn, on the hottest ingest
-- path, to content-address rows no read predicate will ever reach is the
-- pipeline conflation D7 separates the two ingests to avoid.  While
-- chunks.document_id remains a single-parent pointer a total address is
-- unrepresentable anyway: two pre-lifecycle documents in one collection sharing
-- a paragraph would have to be one row, and that row can name only one parent.


-- 1. The content address, over every column pgkg_visible() reads.
--
-- COALESCE on owner_user_id for the reason 042 gives for acl_group_id: "no
-- owner" is one value of the axis, not the absence of one, and an expression
-- index says so on every supported server version.
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
    WHERE document_id IS NULL;

COMMENT ON INDEX chunks_content_addressed_key IS
    'The content address. Keyed on every column pgkg_visible() reads as well '
    'as the hash: a chunk derives its scope from the writer that stored it, '
    'and two writers that disagree about who may read a passage cannot share '
    'one row without one of them being scoped as the other (ADR 0001, D3, D6).';


-- 2. The reuse path learns the two columns.
--
-- A redeclaration rather than an edited clause because the signature moves:
-- CREATE OR REPLACE cannot add a parameter, and two overloads with defaults are
-- ambiguous at every call site.  The new parameters go last, so the corpus
-- pipeline's five positional arguments still resolve — its passages are shared
-- and ownerless, which is what the defaults say.
DROP FUNCTION pgkg_add_version_chunk(UUID, INT, TEXT, UUID, TIMESTAMPTZ);

CREATE FUNCTION pgkg_add_version_chunk(
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
    ) WHERE document_id IS NULL DO NOTHING
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
          AND c.document_id IS NULL;
    ELSE
        v_is_new := TRUE;
    END IF;

    INSERT INTO document_version_chunks (document_version_id, chunk_id, ord)
    VALUES (p_version_id, v_chunk, p_ord)
    ON CONFLICT ON CONSTRAINT document_version_chunks_pkey DO NOTHING;

    RETURN QUERY SELECT v_chunk, v_is_new;
END;
$$;

COMMENT ON FUNCTION
    pgkg_add_version_chunk(UUID, INT, TEXT, UUID, TIMESTAMPTZ, TEXT, UUID) IS
    'Adds one passage to an open version. is_new is the signal to embed. A '
    'passage is reused only by writers that agree with it about org, '
    'collection, ACL group, visibility and owner, because those are the '
    'columns it derives from the writer that stored it and the ones retrieval '
    'reads (ADR 0001, D3, D6).';
