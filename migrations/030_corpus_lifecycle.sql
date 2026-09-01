-- The corpus lifecycle: versioned documents, immutable content-addressed
-- chunks, and the many-to-many between them (ADR 0001, D6).
--
-- WHY A DOCUMENT IS NOT A ROW OF TEXT ANY MORE.  Connectors do nightly full
-- crawls; that is what a connector is.  A 100k-document corpus re-crawled
-- nightly has to cost nothing when nothing changed, which means the ingest path
-- has to be able to answer "is this the same content?" before it does any work.
-- Two hashes answer it at two granularities: the document hash short-circuits
-- the whole document, and the per-chunk hash short-circuits the expensive half
-- — a typo fixed in a 300-page handbook is one embedding call, not 300.
--
-- WHY THE CHUNK IS THE UNIT OF IDENTITY.  A chunk row keyed on its content
-- rather than on its position is shared by every version and every document
-- that contains that content, which is where boilerplate dedup falls out for
-- free.  It also means a chunk has no single parent document, so document_id
-- stops being the truth and document_version_chunks becomes it.  The hash is a
-- GENERATED column: a caller that could write it could write the wrong one, and
-- a content address nobody can forge is the entire basis of the reuse above.
--
-- REFCOUNTS ARE MAINTAINED BY THE DATABASE.  An application-maintained refcount
-- drifts, and a drifted refcount is either a leak or a GC that deletes live
-- content.  Statement-level triggers with transition tables keep it exact at
-- one set-based UPDATE per statement, the same shape 011 uses for the BM25
-- statistics.
--
-- WHY DEDUP STOPS AT THE ORG BOUNDARY.  UNIQUE (org_id, content_hash), not
-- UNIQUE (content_hash).  D4 rejects physical row sharing across tenants: a row
-- shared between tenants cannot live in a tenant partition, so it lands in a
-- content-hash-partitioned pool and vector search over that pool cannot prune
-- by org.  The recall lost exceeds the disk saved.  The expensive half is
-- shared instead, through an embedding cache keyed on the same hash, and only
-- for collections flagged public_source.
--
-- THE FLIP IS ONE STATEMENT BECAUSE IT HAS TO BE ONE TRANSACTION.  Retiring the
-- old version and promoting the new one in two round trips leaves a window in
-- which retrieval sees both versions or neither.  pgkg_promote_document_version
-- does the whole ordered sequence D6 specifies — retire, promote, flip the
-- pointer, withdraw the facts whose chunks were dropped — so a caller cannot
-- get the order wrong or stop half way.  A partial unique index makes "two
-- current versions" unrepresentable rather than merely unlikely, which is why
-- the retire has to precede the promote.
--
-- NO VERSION IS DELETED ON PROMOTION.  A retired version keeps its chunk links,
-- so refcounts stay up and nothing is collected while an in-flight reader may
-- still resolve it.  Physical reclamation is a separate, grace-period pass:
-- pgkg_purge_retired_versions() drops the links, refcounts fall, and
-- pgkg_gc_chunks() collects what nothing points at.  pgvector's HNSW does not
-- reclaim deleted-element space eagerly, so churn is something to schedule
-- rather than something to do on the ingest path.
--
-- THE OWNERSHIP SEAM IS RESOLVED, NOT POPULATED.  023 shipped the subscription
-- table and left resolution to phase 2; this migration resolves it, and creates
-- no shared collection.  Org-owned is and stays the default: the capability
-- exists unused, which is the only way a hot-path predicate can be built on it
-- before there is anything to share.
--
-- NOT HERE, DELIBERATELY.  Reading collections.decay_profile in
-- pgkg_apply_profile(), and the chunk arms of retrieval, belong with the
-- retrieval functions rather than with the schema that makes them possible.
-- embedding_cache (D4) is not here either: it is keyed on the content hash this
-- migration introduces, so it can land once something sets public_source.


-- 1. Documents grow the four columns a lifecycle needs.  external_id is the
-- customer's own identifier, which is what a re-crawl collides on: matching on
-- `source` would make a moved file a new document.  deleted_at is a soft
-- delete, because the read path has to stop returning a withdrawn document long
-- before its rows can be physically reclaimed.
ALTER TABLE documents
    ADD COLUMN external_id        TEXT,
    ADD COLUMN uri                TEXT,
    ADD COLUMN current_version_id UUID,
    ADD COLUMN deleted_at         TIMESTAMPTZ;

COMMENT ON COLUMN documents.external_id IS
    'The identifier on the customer''s side. A connector re-crawls by this, so '
    'it — not source or uri — is what a second crawl collides on.';

-- Partial, because every pre-lifecycle document has no external id and two of
-- those are not a collision.
CREATE UNIQUE INDEX documents_external_id_key
    ON documents (org_id, collection_id, external_id)
    WHERE external_id IS NOT NULL;

CREATE INDEX documents_live_idx ON documents (org_id, collection_id)
    WHERE deleted_at IS NULL;


-- 2. The versions themselves.  No UNIQUE (document_id, content_hash): a
-- document that reverts to a previous revision is a new version of the same
-- content, and forbidding it would turn an ordinary A-B-A edit into an error.
-- The no-op check compares against the CURRENT version only, which is what
-- "unchanged since we last looked" actually means.
CREATE TABLE document_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    org_id        UUID NOT NULL DEFAULT pgkg_current_org() REFERENCES orgs(id),
    version_no    INT  NOT NULL,
    content_hash  BYTEA NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    provenance_id UUID NOT NULL DEFAULT pgkg_unattributed_provenance()
                       REFERENCES provenance(id),
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    retired_at    TIMESTAMPTZ,
    UNIQUE (document_id, version_no),
    CONSTRAINT document_versions_status_check
        CHECK (status IN ('pending', 'current', 'retired')),
    CONSTRAINT document_versions_retired_has_timestamp
        CHECK ((status = 'retired') = (retired_at IS NOT NULL))
);

COMMENT ON TABLE document_versions IS
    'One row per ingested revision of a document. A version is retired, never '
    'overwritten: retrieval must be able to resolve exactly one current '
    'version at every instant (ADR 0001, D6).';

-- "Two current versions" is unrepresentable, not merely unlikely.  This is also
-- why pgkg_promote_document_version() retires before it promotes.
CREATE UNIQUE INDEX document_versions_one_current_idx
    ON document_versions (document_id)
    WHERE status = 'current';

CREATE INDEX document_versions_document_idx
    ON document_versions (document_id, version_no DESC);

CREATE INDEX document_versions_retired_idx ON document_versions (retired_at)
    WHERE status = 'retired';

ALTER TABLE documents
    ADD CONSTRAINT documents_current_version_fkey
        FOREIGN KEY (current_version_id) REFERENCES document_versions(id)
        ON DELETE SET NULL;


-- 3. Chunks become content-addressed, immutable and retrievable.
--
-- The hash is derived from the text by the database, so the address cannot
-- disagree with the content.  tsv and doc_len mirror what 002 and 011 gave
-- propositions, because D2 retrieves the corpus as chunks by default and a
-- chunk arm needs the same two statistics a proposition arm does.  The
-- embedding column takes its width from the propositions column rather than
-- restating a dimension: 012 made the catalog the single statement of it.
ALTER TABLE chunks
    ADD COLUMN content_hash BYTEA
        GENERATED ALWAYS AS (digest(text, 'sha256')) STORED,
    ADD COLUMN refcount INT NOT NULL DEFAULT 0,
    ADD COLUMN tsv TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    ADD COLUMN doc_len INT
        GENERATED ALWAYS AS (length(to_tsvector('english', text))) STORED;

DO $$
BEGIN
    EXECUTE format(
        'ALTER TABLE chunks ADD COLUMN embedding halfvec(%s)',
        pgkg_embedding_dim('propositions', 'embedding')
    );
END;
$$;

CREATE INDEX chunk_emb_idx ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX chunk_tsv_idx ON chunks USING gin (tsv);

-- The content address, over the rows content addressing governs.
--
-- document_id IS NULL is the mark of a lifecycle chunk, and not a coincidence:
-- a content-addressed chunk is shared by every version and document containing
-- that content, so it cannot name one parent, and pgkg_add_version_chunk()
-- below never writes one.  The pre-lifecycle ingest path always does, and it
-- inserts a document's chunks in one statement with no conflict handling — so a
-- total constraint would turn a document with a repeated paragraph, or a second
-- ingest of the same text into one org, into an error on a path that is
-- correct today.  Widening this index to every row is one DROP and CREATE once
-- ingest moves onto the function below, and nothing else has to change.
CREATE UNIQUE INDEX chunks_content_addressed_key
    ON chunks (org_id, content_hash)
    WHERE document_id IS NULL;

CREATE INDEX chunk_unreferenced_idx ON chunks (org_id) WHERE refcount = 0;

COMMENT ON COLUMN chunks.content_hash IS
    'Derived, never supplied: a caller that could write this could write the '
    'wrong one, and the reuse path trusts it (ADR 0001, D6).';

COMMENT ON COLUMN chunks.document_id IS
    'The pre-lifecycle single-parent pointer. A content-addressed chunk is '
    'shared by every version and document containing that content, so it has '
    'no single parent: document_version_chunks is the truth.';

COMMENT ON COLUMN chunks.refcount IS
    'Number of document_version_chunks links. Maintained by trigger, because a '
    'drifted refcount is either a leak or a GC that deletes live content.';


-- Immutability, enforced.  Content addressing is a lie the moment the text can
-- change under a hash, and every citation already emitted names this row.  The
-- WHEN clause is the whole condition, so the highest-volume UPDATE on this
-- table — the refcount maintenance below — never enters the function.
CREATE FUNCTION pgkg_chunks_immutable() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'chunks are immutable: chunk % is addressed by its content, so a new '
        'text is a new chunk', OLD.id;
END;
$$;

CREATE TRIGGER pgkg_chunks_immutable
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    WHEN (NEW.text IS DISTINCT FROM OLD.text
          OR NEW.org_id IS DISTINCT FROM OLD.org_id)
    EXECUTE FUNCTION pgkg_chunks_immutable();


-- 4. The many-to-many, carrying order within the version.  RESTRICT on the
-- chunk side: a referenced chunk cannot be deleted at all, which leaves GC with
-- exactly one way in — driving the refcount to zero first.  CASCADE on the
-- version side is what makes the grace-period purge one DELETE.
CREATE TABLE document_version_chunks (
    document_version_id UUID NOT NULL
        REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_id            UUID NOT NULL
        REFERENCES chunks(id) ON DELETE RESTRICT,
    ord                 INT  NOT NULL,
    PRIMARY KEY (document_version_id, chunk_id),
    UNIQUE (document_version_id, ord)
);

CREATE INDEX document_version_chunks_chunk_idx
    ON document_version_chunks (chunk_id);


CREATE FUNCTION pgkg_version_chunks_refcount() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    delta_sign INT := TG_ARGV[0]::INT;
BEGIN
    UPDATE chunks c
    SET refcount = GREATEST(c.refcount + delta_sign * d.links, 0)
    FROM (
        SELECT chunk_id, COUNT(*)::INT AS links
        FROM delta_rows
        GROUP BY chunk_id
    ) d
    WHERE c.id = d.chunk_id;

    RETURN NULL;
END;
$$;

CREATE TRIGGER pgkg_version_chunks_refcount_insert
    AFTER INSERT ON document_version_chunks
    REFERENCING NEW TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_refcount('1');

CREATE TRIGGER pgkg_version_chunks_refcount_delete
    AFTER DELETE ON document_version_chunks
    REFERENCING OLD TABLE AS delta_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION pgkg_version_chunks_refcount('-1');


-- 5. Open a version.  Returns the current version and is_new = FALSE when the
-- content has not moved, which is the whole no-op path: the caller stops there
-- and never chunks, never embeds, never extracts.
CREATE FUNCTION pgkg_open_document_version(
    p_document_id   UUID,
    p_content_hash  BYTEA,
    p_provenance_id UUID DEFAULT NULL
) RETURNS TABLE (version_id UUID, is_new BOOLEAN)
LANGUAGE plpgsql
AS $$
DECLARE
    v_unchanged UUID;
    v_new       UUID;
BEGIN
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

    IF v_new IS NULL THEN
        RAISE EXCEPTION 'no such document %', p_document_id;
    END IF;

    RETURN QUERY SELECT v_new, TRUE;
END;
$$;


-- 6. Add one chunk to an open version.  is_new is the signal that says "embed
-- this one": FALSE means the content was already stored, under this org, at
-- this hash, and the vector it carries is still the right vector.
--
-- The tenancy of a new chunk is derived from its document rather than passed,
-- because a chunk that disagreed with its document about org or collection
-- would be invisible to retrieval or visible to the wrong tenant.
CREATE FUNCTION pgkg_add_version_chunk(
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

    -- No document_id: this row belongs to whatever versions link it, which is
    -- also what puts it under the content-address unique index.
    --
    -- ON CONFLICT rather than a look-then-insert: two connectors crawling the
    -- same boilerplate concurrently would otherwise race on that index.
    INSERT INTO chunks (text, org_id, collection_id, acl_group_id,
                        provenance_id, asserted_at)
    VALUES (p_text, v_org, v_collection, p_acl_group_id,
            v_provenance, p_asserted_at)
    ON CONFLICT (org_id, content_hash) WHERE document_id IS NULL DO NOTHING
    RETURNING chunks.id INTO v_chunk;

    IF v_chunk IS NULL THEN
        SELECT c.id INTO v_chunk
        FROM chunks c
        WHERE c.org_id = v_org
          AND c.content_hash = v_hash
          AND c.document_id IS NULL;
    ELSE
        v_is_new := TRUE;
    END IF;

    -- Named constraint rather than an inferred one: this function's output
    -- column is called chunk_id too, and an inference list is an expression
    -- context where plpgsql would resolve the name to the variable.
    INSERT INTO document_version_chunks (document_version_id, chunk_id, ord)
    VALUES (p_version_id, v_chunk, p_ord)
    ON CONFLICT ON CONSTRAINT document_version_chunks_pkey DO NOTHING;

    RETURN QUERY SELECT v_chunk, v_is_new;
END;
$$;


-- 7. The flip, in the order D6 specifies.  One call, therefore one transaction
-- unless the caller opened a wider one: a concurrent reader sees the old
-- version or the new one, never both and never neither.
CREATE FUNCTION pgkg_promote_document_version(p_version_id UUID)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_document UUID;
    v_previous UUID;
BEGIN
    SELECT dv.document_id INTO v_document
    FROM document_versions dv WHERE dv.id = p_version_id;

    IF v_document IS NULL THEN
        RAISE EXCEPTION 'no such document version %', p_version_id;
    END IF;

    -- The row lock serialises two promotions of the same document; without it
    -- both would read the same outgoing version and one retire would be lost.
    SELECT d.current_version_id INTO v_previous
    FROM documents d WHERE d.id = v_document FOR UPDATE;

    IF v_previous IS NOT DISTINCT FROM p_version_id THEN
        RETURN;
    END IF;

    -- 1. Retire the outgoing version first: the partial unique index permits
    -- exactly one current version, and it is checked per statement.
    UPDATE document_versions
    SET status = 'retired', retired_at = now()
    WHERE id = v_previous;

    -- 2. Promote the incoming one.
    UPDATE document_versions SET status = 'current' WHERE id = p_version_id;

    -- 3. Flip the pointer retrieval reads.
    UPDATE documents SET current_version_id = p_version_id WHERE id = v_document;

    -- 4. Withdraw the facts derived from chunks this version does not carry.
    -- source_updated, not superseded: nothing replaced these claims, their
    -- source stopped saying them.  An existing reason is never overwritten —
    -- the first withdrawal is the true one (021).
    UPDATE propositions p
    SET invalidated_at = now(), invalidation_reason = 'source_updated'
    WHERE p.invalidated_at IS NULL
      AND p.chunk_id IN (
          SELECT dvc.chunk_id
          FROM document_version_chunks dvc
          WHERE dvc.document_version_id = v_previous
          EXCEPT
          SELECT dvc.chunk_id
          FROM document_version_chunks dvc
          WHERE dvc.document_version_id = p_version_id
      );
END;
$$;


-- 8. Physical reclamation, in two grace-period passes rather than one cascade.
--
-- Retirement is a read-path decision and reclamation is a storage one, so they
-- run on different clocks: dropping a retired version's links the moment it
-- retires would collect content an in-flight reader is still resolving, and
-- pgvector's HNSW leaves tombstones behind either way.  The purge is what makes
-- refcounts fall; the collector only removes what nothing points at.  Both take
-- an org, because a sweep over one partition is the tractable unit — one HNSW
-- rebuild at a time; NULL means every org, which is the scheduled global pass.
CREATE FUNCTION pgkg_purge_retired_versions(
    p_grace  INTERVAL DEFAULT INTERVAL '7 days',
    p_org_id UUID     DEFAULT NULL
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    v_purged BIGINT;
BEGIN
    WITH purged AS (
        DELETE FROM document_versions
        WHERE status = 'retired'
          AND retired_at < now() - p_grace
          AND (p_org_id IS NULL OR org_id = p_org_id)
        RETURNING id
    )
    SELECT COUNT(*) INTO v_purged FROM purged;

    RETURN v_purged;
END;
$$;


-- The collector.  refcount = 0 is the condition D6 names, and the second one is
-- what keeps it from being destructive: propositions.chunk_id cascades, so
-- collecting a chunk a fact was extracted from would silently delete the fact.
-- "Delete this passage" and "delete everything we learned from it" are
-- different decisions, and only one of them is this function's.
CREATE FUNCTION pgkg_gc_chunks(p_org_id UUID DEFAULT NULL) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    v_collected BIGINT;
BEGIN
    WITH collected AS (
        DELETE FROM chunks c
        WHERE c.refcount = 0
          AND (p_org_id IS NULL OR c.org_id = p_org_id)
          AND NOT EXISTS (
              SELECT 1 FROM document_version_chunks dvc
              WHERE dvc.chunk_id = c.id
          )
          AND NOT EXISTS (
              SELECT 1 FROM propositions p WHERE p.chunk_id = c.id
          )
        RETURNING c.id
    )
    SELECT COUNT(*) INTO v_collected FROM collected;

    RETURN v_collected;
END;
$$;


-- 9. Subscription resolution, which 023 left to this phase.
--
-- Two arrays, because the retrieval predicate takes two and both have to widen
-- together.  Widening the org list alone would make every collection in the
-- operator's org visible, not just the one a tenant subscribed to: p_org_ids
-- prunes partitions, p_collection_ids is what says which material inside them
-- was actually published to this tenant.  A NULL collection array means
-- unrestricted, so a caller that widens the org list must pass this one.
--
-- Three conditions, none of them redundant.  enabled is how a tenant turns
-- shared material off without a rebuild.  visibility = 'shared' is what makes a
-- subscription row an authorisation rather than a claim — a row naming a
-- collection nobody published resolves to nothing.  And the caller's own org is
-- always present, so the default answer is one element and prunes to one
-- partition exactly as equality did.
--
-- STABLE and a single SELECT so the planner can fold the result into the
-- ordinary column comparisons the HNSW path needs.
CREATE FUNCTION pgkg_subscribed_orgs(p_org_id UUID) RETURNS UUID[]
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT ARRAY(
        SELECT p_org_id
        UNION
        SELECT c.owner_org_id
        FROM collection_subscriptions s
        JOIN collections c ON c.id = s.collection_id
        WHERE s.org_id = p_org_id
          AND s.enabled
          AND c.visibility = 'shared'
    );
$$;

CREATE FUNCTION pgkg_subscribed_collections(p_org_id UUID) RETURNS UUID[]
LANGUAGE SQL STABLE PARALLEL SAFE
AS $$
    SELECT ARRAY(
        SELECT s.collection_id
        FROM collection_subscriptions s
        JOIN collections c ON c.id = s.collection_id
        WHERE s.org_id = p_org_id
          AND s.enabled
          AND c.visibility = 'shared'
    );
$$;

-- The rule the seam exists to hold, on the seam itself: an ADR nobody opens is
-- not a constraint.  The structural half is 023's CHECK that only the system org
-- can own a shared collection, which is why resolution above can widen to
-- owner_org_id without asking who published it.
COMMENT ON TABLE collection_subscriptions IS
    'Empty by default; nothing is subscribed implicitly, and this is the only '
    'thing that widens a tenant''s org list. Nothing a tenant ingests is ever '
    'promoted into a shared collection: not automatically, not by heuristic, '
    'not as a "contribute back" default. Shared collections are populated by '
    'the operator from operator-licensed sources (ADR 0001, D4).';


-- 10. The application role, again.  020's GRANT ... ON ALL TABLES was a snapshot
-- of the tables that existed then, so a table created here is unreachable by
-- the role a deployment is told to connect as until it is granted by name.
--
-- No row-level security on either table yet, deliberately: document_versions
-- carries org_id and belongs behind a policy, but the policy has to arrive with
-- the isolation test that can tell it from USING (TRUE), and that test module is
-- not part of this change.  Both tables are reached only through documents and
-- chunks, which are policy-protected.
DO $$
BEGIN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON '
            'document_versions, document_version_chunks TO pgkg_app';
EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
    RAISE NOTICE 'pgkg_app not granted on the lifecycle tables (%)', SQLERRM;
END;
$$;
