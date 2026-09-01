-- Context windowing and document lifecycle, corrected (ADR 0001, D3, D5, D6).
--
-- The context window and the content address share one root cause: content
-- addressing was taken to be org-wide while every attribute a content-addressed
-- chunk carries is derived per document.  Opening a version and linking a chunk
-- to one are separate races on the same pair of tables, and land here with them.
--
-- WHY THE CONTENT ADDRESS NARROWS TO THE SCOPING COLUMNS.  030 keyed the
-- address on (org_id, content_hash) and derived org, collection, ACL group and
-- provenance from the version's document, so the second document to contain a
-- passage got the first document's scope: ON CONFLICT DO NOTHING keeps the row
-- it found.  D3 puts collection_id and acl_group_id on every retrievable row
-- because they are read by the retrieval predicate — collection also carries
-- the claim scope, the decay profile, the BM25 statistics domain and the quota
-- bucket — and one row cannot hold two values of a column two documents
-- disagree about.  Copying the second document's values over the first only
-- moves the wrong answer to the other document.  So the address gains every
-- column the visibility predicate reads and the chunk derives from its
-- document: dedup now stops at the collection and ACL boundary for exactly the
-- reason D4 already stops it at the org boundary, that a row shared across a
-- scoping boundary cannot carry the boundary.  What the narrower address costs
-- is disk, not model calls: the expensive half is shared through
-- embedding_cache, which is keyed on content_hash alone (D4), and the case the
-- nightly full crawl depends on — the same document re-crawled, and boilerplate
-- repeated inside one collection — dedups exactly as before.
--
-- Provenance and asserted_at are deliberately NOT in the address.  They are
-- derivation and belief clocks (D5, D6), not scoping columns: no read predicate
-- consults them, and D5 makes provenance a shared immutable record, so the
-- first derivation of a passage stays the true one — the same rule 021 applies
-- to an invalidation reason.
--
-- WHY THE WINDOW CARRIES ITS OWN PREDICATE.  Every scoring stage was scoped and
-- then pgkg_chunk_window() reached outside the scope entirely, so
-- pgkg_retrieve()'s context_text laundered whole documents from collections and
-- ACL groups the caller was never granted — the permission-laundering machine
-- D3 names, arriving through the one path with no filter.  The window takes no
-- caller scope even now, and must not: a hit is widened after it is ranked, so
-- the honest predicate is not "what may the caller read" but "is this neighbour
-- indistinguishable from the anchor to every read predicate".  A neighbour that
-- agrees with the anchor chunk on org, collection, ACL group, visibility and
-- owner passes pgkg_visible() for exactly the callers the anchor passes it for,
-- whatever arguments they passed — so the context can never widen the grant
-- that admitted the anchor, and no caller has to remember to scope it.
--
-- WHY AN AMBIGUOUS WINDOW IS NO WINDOW.  A passage carried by the current
-- version of several live documents has no single context.  031 picked the
-- lowest document_version_id and called it arbitrary but stable; it was neither
-- right nor stable in the sense that mattered, since which of two random UUIDs
-- sorts lower decided whose prose was returned.  There is no correct answer to
-- pick: the neighbours of a boilerplate footer in one handbook are not the
-- context of the same footer in another, and attributing them to a citation
-- that does not carry them is a fabrication whichever document wins.  So the
-- passage is its own context, which is what 031 already returns for a chunk
-- with no version links at all, and pgkg_retrieve()'s COALESCE already reads.
--
-- WHY OPENING A VERSION TAKES THE DOCUMENT LOCK.  pgkg_promote_document_version
-- takes it and says why; opening a version computed MAX(version_no) + 1 without
-- it, against a UNIQUE (document_id, version_no).  Two connectors crawling one
-- document in overlapping transactions — a nightly full crawl racing a webhook
-- re-crawl, or two IngestWorker loops — both computed the same number and the
-- second failed with a unique violation out of upsert_document.  The lock makes
-- them serialise into version n and n+1, which is what a second crawl means.
--
-- WHY THE LINK TABLE VALIDATES THE CHUNK SIDE IN A TRIGGER.  033's policy
-- validates only the version side, so a tenant could graft another tenant's
-- chunk id into its own document version: a cross-org write into a table the
-- window then reads, and a way to hold a stranger's chunk live against GC
-- forever.  A trigger rather than a widened policy because this is not a
-- visibility rule a table owner may bypass — a link between two orgs is
-- meaningless in any session, RLS on or off, owner or not — and because 030
-- already enforces the other two invariants of this pair of tables, immutable
-- chunks and exact refcounts, the same way.


-- 1. The content address, over the columns a chunk derives from its document.
--
-- COALESCE rather than NULLS NOT DISTINCT: "no ACL group" is one value of the
-- axis, not the absence of one, and an expression index says so on every
-- supported server version.
DROP INDEX chunks_content_addressed_key;

CREATE UNIQUE INDEX chunks_content_addressed_key
    ON chunks (
        org_id,
        collection_id,
        (COALESCE(acl_group_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        content_hash
    )
    WHERE document_id IS NULL;

COMMENT ON INDEX chunks_content_addressed_key IS
    'The content address. Keyed on the scoping columns as well as the hash: a '
    'chunk derives collection and ACL group from its document, and two '
    'documents that disagree about either cannot share one row without one of '
    'them being scoped as the other (ADR 0001, D3, D6).';


CREATE OR REPLACE FUNCTION pgkg_add_version_chunk(
    p_version_id   UUID,
    p_ord          INT,
    p_text         TEXT,
    p_acl_group_id UUID DEFAULT NULL,
    p_asserted_at  TIMESTAMPTZ DEFAULT NULL
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
                        provenance_id, asserted_at)
    VALUES (p_text, v_org, v_collection, p_acl_group_id,
            v_provenance, p_asserted_at)
    ON CONFLICT (
        org_id,
        collection_id,
        (COALESCE(acl_group_id, '00000000-0000-0000-0000-000000000000'::UUID)),
        content_hash
    ) WHERE document_id IS NULL DO NOTHING
    RETURNING chunks.id INTO v_chunk;

    IF v_chunk IS NULL THEN
        SELECT c.id INTO v_chunk
        FROM chunks c
        WHERE c.org_id = v_org
          AND c.collection_id = v_collection
          AND c.acl_group_id IS NOT DISTINCT FROM p_acl_group_id
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

COMMENT ON FUNCTION pgkg_add_version_chunk(UUID, INT, TEXT, UUID, TIMESTAMPTZ) IS
    'Adds one passage to an open version. is_new is the signal to embed. A '
    'passage is reused only by documents that agree with it about org, '
    'collection and ACL group, because those are the columns it derives from '
    'its document and the ones retrieval reads (ADR 0001, D3, D6).';


-- 2. Small-to-big, inside the grant that admitted the anchor.
CREATE OR REPLACE FUNCTION pgkg_chunk_window(
    p_chunk_ids UUID[],
    p_before    INT DEFAULT 1,
    p_after     INT DEFAULT 1
) RETURNS TABLE (
    chunk_id            UUID,
    document_version_id UUID,
    ord_from            INT,
    ord_to              INT,
    context_text        TEXT
)
LANGUAGE SQL STABLE
AS $$
WITH anchor AS (
    SELECT c.id, c.org_id, c.collection_id, c.acl_group_id,
           c.visibility, c.owner_user_id
    FROM chunks c
    WHERE c.id = ANY(p_chunk_ids)
),

-- A version may supply a window only if the document carrying it is the
-- anchor's own tenant and collection: 033's link policy validates the version
-- side alone, so a link is not on its own evidence that the two belong together.
carrier AS (
    SELECT a.id AS chunk_id, dvc.document_version_id
    FROM anchor a
    JOIN document_version_chunks dvc ON dvc.chunk_id = a.id
    JOIN document_versions dv
      ON dv.id = dvc.document_version_id
     AND dv.status = 'current'
    JOIN documents d
      ON d.id = dv.document_id
     AND d.deleted_at IS NULL
     AND d.org_id = a.org_id
     AND d.collection_id = a.collection_id
    GROUP BY a.id, dvc.document_version_id
),

-- Exactly one carrier, or no window at all.
sole AS (
    SELECT c.chunk_id, c.document_version_id
    FROM carrier c
    WHERE NOT EXISTS (
        SELECT 1 FROM carrier other
        WHERE other.chunk_id = c.chunk_id
          AND other.document_version_id <> c.document_version_id
    )
),

-- DISTINCT because a repeated passage holds several positions in one version
-- (033) and their windows overlap: without it the shared neighbour would be
-- aggregated once per anchor position.
span AS (
    SELECT DISTINCT
        s.chunk_id,
        s.document_version_id,
        neighbour.ord,
        neighbour.chunk_id AS neighbour_id
    FROM sole s
    JOIN document_version_chunks anchor_link
      ON anchor_link.document_version_id = s.document_version_id
     AND anchor_link.chunk_id = s.chunk_id
    JOIN document_version_chunks neighbour
      ON neighbour.document_version_id = s.document_version_id
     AND neighbour.ord BETWEEN anchor_link.ord - GREATEST(p_before, 0)
                           AND anchor_link.ord + GREATEST(p_after, 0)
)

SELECT
    span.chunk_id,
    span.document_version_id,
    MIN(span.ord)::INT,
    MAX(span.ord)::INT,
    string_agg(n.text, E'\n\n' ORDER BY span.ord)
FROM span
JOIN anchor a ON a.id = span.chunk_id
JOIN chunks n
  ON n.id = span.neighbour_id
 AND n.org_id = a.org_id
 AND n.collection_id = a.collection_id
 AND n.acl_group_id IS NOT DISTINCT FROM a.acl_group_id
 AND n.visibility = a.visibility
 AND n.owner_user_id IS NOT DISTINCT FROM a.owner_user_id
GROUP BY span.chunk_id, span.document_version_id;
$$;

COMMENT ON FUNCTION pgkg_chunk_window(UUID[], INT, INT) IS
    'Widens a passage to its neighbours by ordinal within the one live document '
    'version that carries it. Every neighbour agrees with the anchor on every '
    'column the retrieval predicate reads, so context_text can never widen the '
    'grant that admitted the anchor; a passage several live documents carry has '
    'no single context and gets no row (ADR 0001, D3, D6).';


-- 3. Opening a version serialises on its document.
CREATE OR REPLACE FUNCTION pgkg_open_document_version(
    p_document_id   UUID,
    p_content_hash  BYTEA,
    p_provenance_id UUID DEFAULT NULL
) RETURNS TABLE (version_id UUID, is_new BOOLEAN)
LANGUAGE plpgsql
AS $$
DECLARE
    v_document  UUID;
    v_unchanged UUID;
    v_new       UUID;
BEGIN
    -- Before the version number is read, not after: the lock is what makes
    -- MAX(version_no) + 1 still true by the time the INSERT uses it.  A
    -- statement of its own, so the read below takes a fresh snapshot and sees
    -- the version the crawl we waited for committed.
    SELECT d.id INTO v_document
    FROM documents d WHERE d.id = p_document_id FOR UPDATE;

    IF v_document IS NULL THEN
        RAISE EXCEPTION 'no such document %', p_document_id;
    END IF;

    SELECT dv.id INTO v_unchanged
    FROM documents d
    JOIN document_versions dv ON dv.id = d.current_version_id
    WHERE d.id = p_document_id
      AND dv.content_hash = p_content_hash;

    IF v_unchanged IS NOT NULL THEN
        RETURN QUERY SELECT v_unchanged, FALSE;
        RETURN;
    END IF;

    INSERT INTO document_versions
        (document_id, org_id, version_no, content_hash, status, provenance_id)
    SELECT
        d.id,
        d.org_id,
        COALESCE(
            (SELECT MAX(dv.version_no) FROM document_versions dv
             WHERE dv.document_id = d.id),
            0
        ) + 1,
        p_content_hash,
        'pending',
        COALESCE(p_provenance_id, pgkg_unattributed_provenance())
    FROM documents d
    WHERE d.id = p_document_id
    RETURNING document_versions.id INTO v_new;

    RETURN QUERY SELECT v_new, TRUE;
END;
$$;

COMMENT ON FUNCTION pgkg_open_document_version(UUID, BYTEA, UUID) IS
    'Opens the next version of a document, or returns the current one when the '
    'content has not moved. Serialises on the document row: nightly full crawls '
    'are what connectors do, so two of them meeting on one document is the '
    'common case and has to queue rather than collide (ADR 0001, D6).';


-- 4. A link belongs to one org on both sides.
--
-- Statement level with a transition table, the shape 030 uses for the
-- refcounts: the bulk path links every chunk of a document in one statement,
-- and a per-row check would turn that into two queries per passage.
CREATE FUNCTION pgkg_version_chunks_same_org() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_chunk   UUID;
    v_version UUID;
BEGIN
    -- LEFT JOIN on both sides, because under RLS a stranger's row reads as
    -- absent and "absent" is the answer that must not link.
    SELECT l.chunk_id, l.document_version_id INTO v_chunk, v_version
    FROM new_links l
    LEFT JOIN document_versions dv ON dv.id = l.document_version_id
    LEFT JOIN chunks c ON c.id = l.chunk_id
    WHERE dv.org_id IS NULL
       OR c.org_id IS NULL
       OR dv.org_id <> c.org_id
    LIMIT 1;

    IF v_chunk IS NOT NULL THEN
        RAISE EXCEPTION
            'chunk % does not belong to the org of document version %',
            v_chunk, v_version;
    END IF;

    RETURN NULL;
END;
$$;

-- One trigger per event, because a transition table may name only one.
CREATE TRIGGER pgkg_version_chunks_same_org_insert
    AFTER INSERT ON document_version_chunks
    REFERENCING NEW TABLE AS new_links
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_same_org();

CREATE TRIGGER pgkg_version_chunks_same_org_update
    AFTER UPDATE ON document_version_chunks
    REFERENCING NEW TABLE AS new_links
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_same_org();
