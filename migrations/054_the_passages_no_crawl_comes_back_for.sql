-- The passages no crawl comes back for (ADR 0001, D6, D8; issue #22).
--
-- WHY THIS MIGRATION EXISTS.  A corpus ingest can leave a live passage with no
-- vector.  The window is small — phase 2 saw a vectored row at a content
-- address, phase 3 asked the database authoritatively and the row it had seen
-- was gone, so the write phase created a new one — and 78b7533 made the ingest
-- pay for those rows itself, after the transaction, rather than dropping them.
-- What it cannot cover is the run that does not get that far: the version is
-- promoted and committed before the repair is attempted, so an embedder that
-- fails, a process that is killed, or a connection that drops between the two
-- leaves the row committed and unvectored.  Nothing revisits it.  The next
-- crawl of that document short-circuits on the unchanged document hash before
-- it ever reaches a chunk, which is what makes the whole batch pipeline cheap
-- and what makes this permanent: the row keeps `embedding IS NULL` for good,
-- reachable by the keyword arm and invisible to the vector arm and to MMR,
-- with nothing anywhere saying so.
--
-- So the backstop belongs on the timer, as the fifth task of `pgkg maintain`,
-- and what a timer needs that the ingest path does not is added here: a way to
-- find those rows that does not read the whole chunk store, and a way to ask
-- for them one generation at a time.
--
-- WHY THE QUERY IS KEYED ON A GENERATION.  D8: two generations produce
-- incomparable vectors, and `chunks.embedding` is the single inline column the
-- org's PRIMARY generation owns.  A row carries the generation it was written
-- in, and a repair that embedded it with whatever model this process happens to
-- run would either write into the wrong space or restamp the row to say it did
-- — the exact silent-wrong-answer shape of the defect it is fixing, and most
-- likely during a cutover window, when rows of the outgoing generation are what
-- a backlog is made of.  Asking per generation is therefore not a filter for
-- efficiency: it is what lets the caller serve the rows it can embed for and
-- leave the rest alone, visibly, for the generation backfill that owns them.
--
-- WHY THE SCOPE IS `refcount > 0` AND NOT `retrievable`.  The scope is the passages a crawl
-- stranded, and `refcount > 0` is what says a document version links this row.
-- `chunks.retrievable` is the read path's own answer and a wider set: a chunk
-- that belongs to no document at all is retrievable content by 052's first arm,
-- which is every chat turn's passage and every row the pre-lifecycle ingest
-- path wrote — none of them stranded by a crawl, all of them embedded inline by
-- the pipeline that wrote them, and all of them a bill this job has no business
-- running up.  A row whose refcount fell to zero after the fact is one a purge
-- is about to collect, and paying an embedder for it would be paying for a
-- deletion.


-- 1. Finding the stranded rows without reading the chunk store.
--
-- The predicate is the job's own, so the index answers both questions it asks:
-- how many rows are stranded for this org — the number the report has to be
-- able to state honestly every tick, settled or not — and which of them belong
-- to the generation this run can embed for.  Without it, a job on a five-minute
-- timer is a sequential scan of every passage the tenant owns, five minutes
-- apart, forever, and the cheapest outcome of a settled corpus stops being
-- cheap.
CREATE INDEX chunks_unvectored_idx
    ON chunks (org_id, embedder_generation_id, created_at)
    WHERE embedding IS NULL AND refcount > 0;


-- 2. One batch of them, oldest first.
--
-- A function rather than a query in the application, for the reason
-- pgkg_unmatched_chunks() is one: the predicate that decides what "stranded"
-- means is a property of the schema — the liveness rule, the generation column,
-- the NULL — and a copy of it in Python is a copy that drifts from the index
-- that serves it.  STABLE and read-only: the write is the caller's, because the
-- vector between the two is a model call and no transaction may be open across
-- it.
CREATE FUNCTION pgkg_unvectored_chunks(
    p_org_id        UUID,
    p_generation_id UUID,
    p_limit         INT DEFAULT 1000
) RETURNS TABLE (chunk_id UUID, chunk_text TEXT)
LANGUAGE SQL STABLE
AS $$
    SELECT c.id, c.text
    FROM chunks c
    WHERE c.org_id = p_org_id
      AND c.embedding IS NULL
      AND c.refcount > 0
      AND c.embedder_generation_id = p_generation_id
    ORDER BY c.created_at, c.id
    LIMIT GREATEST(p_limit, 0)
$$;

COMMENT ON FUNCTION pgkg_unvectored_chunks(UUID, UUID, INT) IS
    'One batch of the passages a crawl left without a vector, in one embedding '
    'generation. The generation is an argument rather than a column read here '
    'because the caller has to embed with the model that produced it: a vector '
    'from another generation is incomparable, and writing one would restate the '
    'defect this finds (ADR 0001, D8; #22).';
