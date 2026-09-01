-- Entity resolution's fuzzy stage, reachable by the trigram index
-- (ADR 0001, D1 the shared entity hub, D3; issue #17).
--
-- WHY THIS IS NOT 047 AGAIN.  047 was about a qual that could not be PROMOTED
-- to an index condition, and its remedy was a claim about a function.  This one
-- is about a qual that was never a candidate for promotion at all:
-- pgkg_link_entity()'s second stage filters with
--
--     similarity(name, p_name) > 0.6
--
-- which is pg_trgm's similarity() FUNCTION over a column.  pg_trgm's index
-- support is declared on the `%` operator (similarity_op), so a plain function
-- call reaches entities_name_trgm_idx for no role — the table owner included.
-- 047 marking similarity_op leakproof neither helped nor harmed this, because
-- the operator never appeared in the statement.  047's header recorded the
-- finding and deliberately left it; this is the fix.
--
-- pgkg_link_entity() runs once per entity per proposition on the ingest path,
-- and stage 2 is the stage reached by every name that is NOT already an exact
-- hit — that is, by every genuinely new or misspelled name, which is the
-- population entity resolution exists for.  Measured on 40,001 entities in one
-- org and namespace, against the same statement, as the table owner and with
-- enable_seqscan untouched:
--
--   before  Seq Scan on entities, 78.4 ms, 949 buffers, 40,001 rows discarded
--   after   Bitmap Index Scan on entities_name_trgm_idx, 0.65 ms, 70 buffers
--
-- The `ORDER BY (embedding <=> p_embedding) LIMIT 1` above the scan is
-- unchanged, and so is which row it picks.  (The issue text says the
-- similarity() value is used for ordering; it is not — 024 orders by cosine
-- distance and similarity only ever gated the candidate set.  Recorded here
-- because the difference is what makes this fix a safe one.)
--
-- WHY THE THRESHOLD IS BOUND ON THE FUNCTION AND NOT LEFT TO THE SESSION.
-- `%` and `similarity() > 0.6` are not the same predicate: `%` compares
-- against pg_trgm.similarity_threshold, a GUC that defaults to 0.3 and that
-- any caller may SET.  Swapping one for the other unqualified would hand the
-- definition of "the same entity" to session state — dedup that silently
-- widens to 0.3 and merges two entities that merely rhyme is corruption of the
-- graph, not a slow query, and it would arrive with no error and no way to tell
-- afterwards which merges were meant.
--
-- So the GUC is pinned in the function's own definition.  A proconfig SET is
-- saved on entry and restored on exit, which is the only one of the three
-- options that is both unaffected by the caller and invisible to it:
--
--   * set_limit() and SET LOCAL inside the body would leave the caller's
--     transaction holding this function's threshold after it returns, so
--     pgkg_link_entity() would silently change the meaning of every later `%`
--     in the same transaction — the gazetteer's fuzzy arm included.
--   * relying on the session's value would make resolution depend on who
--     connected, which is the hazard above.
--
-- WHY THE similarity() CALL STAYS ANYWAY, WHICH IS NOT THE OBVIOUS REASON.
-- `%` is `similarity >= threshold` where `> 0.6` was strict, so the boundary
-- looks like a behaviour change and is not one: similarity() returns float4,
-- the comparison widens it to float8, and no float4 widens to exactly 0.6 in
-- float8, so `>= 0.6` and `> 0.6` cannot disagree on any value the function can
-- produce.  With the threshold pinned at 0.6 the operator alone is therefore
-- already the identical predicate, and the candidate set is unchanged at every
-- value.
--
-- The confirmation is kept for a different reason: it is what makes 0.6 a
-- property of this FUNCTION rather than of GUC state.  Alone, `%` puts the
-- definition of "the same entity" one forgotten line away from pg_trgm's
-- default of 0.3 — a later migration that replaces this function and omits the
-- proconfig line would widen dedup to 0.3 and start merging entities that
-- merely rhyme, with no error and nothing afterwards to say which merges were
-- meant.  With the confirmation present, losing the pin can only ever cost
-- matches, never invent them, which is the direction a silent failure is
-- survivable in.  That asymmetry is also why the two 0.6s must not drift apart
-- in the other direction: a proconfig value looser than the body's costs speed
-- only, a tighter one drops rows the body would have accepted.
--
-- The operator is therefore the candidate GENERATOR — the only qual that can
-- read the index — and similarity() is the confirmation, which is 047's fuzzy
-- arm pattern for a related reason.
--
-- WHAT IS DELIBERATELY UNCHANGED.  Stages 1 and 3, the org and namespace
-- predicates, the conflict target, the recovery read, the signature and
-- SECURITY INVOKER are exactly as 024 left them.  0.6 is not promoted to a
-- parameter: p_threshold already means the embedding threshold, and a second
-- one would have to be pinned into proconfig, which a parameter cannot be.

CREATE OR REPLACE FUNCTION pgkg_link_entity(
    p_namespace  TEXT,
    p_name       TEXT,
    p_type       TEXT,
    p_embedding  halfvec,
    p_threshold  REAL DEFAULT 0.85
) RETURNS UUID
LANGUAGE plpgsql
SECURITY INVOKER
SET pg_trgm.similarity_threshold = 0.6
AS $$
DECLARE
    v_id  UUID;
    v_org UUID := pgkg_current_org();
BEGIN
    -- 1. Exact name + type match within namespace, inside this org
    SELECT id INTO v_id
    FROM entities
    WHERE org_id = v_org
      AND namespace = p_namespace
      AND name = p_name
      AND (type = p_type OR (type IS NULL AND p_type IS NULL))
    LIMIT 1;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    -- 2. Trigram + embedding similarity match, inside this org.
    --
    -- `name % p_name` is the candidate generator and the only qual that can
    -- read entities_name_trgm_idx; it compares against the threshold pinned in
    -- this function's proconfig, not the caller's.  The similarity() call below
    -- it is the confirmation that keeps the strict inequality 024 had.
    IF p_embedding IS NOT NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE org_id = v_org
          AND namespace = p_namespace
          AND name % p_name
          AND similarity(name, p_name) > 0.6
          AND (1 - (embedding <=> p_embedding)) > p_threshold
        ORDER BY (embedding <=> p_embedding)
        LIMIT 1;
    END IF;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    -- 3. Create the entity, or adopt the one a concurrent writer created.
    -- DO NOTHING blocks on the unique index until the competitor commits or
    -- rolls back; on commit it returns no row and the re-read below — a new
    -- snapshot, because each plpgsql statement takes one — finds the winner.
    INSERT INTO entities (name, type, embedding, namespace, org_id)
    VALUES (p_name, p_type, p_embedding, p_namespace, v_org)
    ON CONFLICT (org_id, namespace, name, COALESCE(type, '')) DO NOTHING
    RETURNING id INTO v_id;

    IF v_id IS NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE org_id = v_org
          AND namespace = p_namespace
          AND name = p_name
          AND (type = p_type OR (type IS NULL AND p_type IS NULL))
        LIMIT 1;
    END IF;

    RETURN v_id;
END;
$$;
