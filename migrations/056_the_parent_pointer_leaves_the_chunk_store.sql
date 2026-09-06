-- 056  chunks.document_id is dropped, and the content address stays partial
--      (ADR 0001, D1, D6; issue #18).
--
-- WHAT 052 LEFT.  052 moved every semantic use of the pointer onto
-- `chunks.provenance_only`: liveness reads the writer's statement, and so does
-- the content address's partial predicate.  What survived was the pointer
-- itself, as the record of which document a chat-provenance chunk came out of,
-- read by one named bridge trigger that translated a write of it into that
-- statement.  052's own header calls the rest "mechanical: retire the writers,
-- drop the trigger, drop the column".  This is that, and it is the whole of
-- what #18 has left to ask for.
--
-- WHAT THE RECORD IS NOW, WHICH IS THE ONLY QUESTION THE DROP RAISES.  045 is
-- the precedent and its lesson is the constraint: when a purge destroyed the
-- only record that a chunk had ever been carried, liveness could not tell a
-- standalone passage from an orphan, so the record became a column.  Removing
-- the only record of something is exactly the mistake not to repeat, so what it
-- recorded was traced before it went.
--
--   * Nothing in the schema reads it.  052 verified that by reading
--     pg_get_functiondef() for every pgkg_ function, and the guard that pins it
--     is in tests/test_corpus_lifecycle.py.
--   * Nothing in the package reads it.  The extraction path was its only
--     writer, and it wrote provenance_only in the same statement from 052.
--   * The surfaces that might have needed it do not go through it.  A document
--     soft delete matches on external_id and withdraws its claims through
--     document_version_chunks; the extraction path's documents carry no
--     external_id, so that surface never reached them anyway.  Retrieval reads
--     chunks.retrievable.  Erasure goes through provenance.
--   * What it recorded is on the row already, in a better shape.  On the
--     extraction path the document is the turn that ingest wrote, and every
--     chunk of it carries a provenance row of its own naming the ingest run and
--     the span a citation names.  The pointer named the document and nothing
--     else; the derivation record names the ingest, the span and the actor.
--
-- WHY THE PARENTAGE IS NOT MOVED TO document_version_chunks.  #18 as filed
-- asks for exactly that, and it is the wrong move — 052 measured why and this
-- migration does not repeat the measurement, it obeys it.  pgkg_item_scope()
-- buckets a proposition 'corpus' when `EXISTS (document_version_chunks ...)`
-- for the chunk it cites, so linking the extraction path's passages under a
-- version would reclassify every chat-derived fact as corpus material and
-- restore D1's drowning failure mode verbatim — the failure 041 keyed the
-- bucket on structure to escape.  It would also move the provenance record from
-- per-chunk to per-version (049: a shared row cannot say which ingest produced
-- it) and so give up the span.  A pointer with no readers is dropped; it is not
-- re-homed into a shape that changes retrieval.
--
-- WHY THE CONTENT ADDRESS STAYS PARTIAL, RE-MEASURED ON TOP OF THIS DROP.  #18
-- and the implementation notes both say a total address follows once the column
-- is gone.  It does not, and the reason is that 052 re-founded the predicate on
-- a column this migration does not touch: the address is partial on
-- `NOT provenance_only`, and dropping the pointer does not move it.  Measured
-- rather than assumed, twice: against the schema this migration starts from a
-- genuinely total index fails 22 tests, and against the schema it leaves, 21.
-- Every one of the 21 is a unique violation on chunks_content_addressed_key.
-- Nineteen are the extraction path colliding with itself on repeated text — a
-- turn that repeats a paragraph is two rows on purpose, because each carries
-- its own span and its own derivation record — and two pin the partial
-- predicate deliberately.  The one the drop accounts for is a test of the
-- bridge, deleted with it, not a collision that stopped happening.
-- The address is total over the rows content addressing governs, which is what
-- 030 was reaching for; it is permanently partial over the rows it does not.
--
-- FORWARD-ONLY, AND WHY NOTHING BRIDGES THIS ONE.  052's bridge existed because
-- a migration cannot reach the callers and rows written by a writer that still
-- stated the pointer would otherwise have become retrievable content-addressed
-- passages the moment the predicates moved.  There is no such risk here: a
-- write of a dropped column is an error at parse time, not a silently different
-- answer, and the one writer in this package was retired in the same change.
-- The bridge goes first so the column drop does not have to cascade to it.


-- 1. The bridge, whose whole subject is the column below.
DROP TRIGGER pgkg_chunks_provenance_bridge ON chunks;

DROP FUNCTION pgkg_chunks_provenance_bridge();


-- 2. The pointer.  The FOREIGN KEY to documents goes with it, which is the last
-- object in the catalogue that depended on the column.
ALTER TABLE chunks DROP COLUMN document_id;


-- 3. What the chunk store says about parentage now that it has no pointer.
COMMENT ON TABLE chunks IS
    'Content-addressed passages. A passage belongs to a document through '
    'document_version_chunks and to nothing else: the single-parent pointer '
    'that predated the lifecycle was dropped in 056, because a row that can '
    'name only one parent cannot be shared by two documents and that is what '
    'content addressing is for. Whether a row is retrievable content or '
    'provenance for the facts extracted from it is stated by the writer in '
    'provenance_only, never inferred from what the row points at (052). A '
    'passage stored as provenance carries its own derivation record, which is '
    'what records the ingest it came out of and the span a citation names '
    '(ADR 0001, D1, D5, D6; issue #18).';
