-- The gazetteer's three arms, reachable under the role the policies are for
-- (ADR 0001, D2, D4; issue #10).
--
-- WHY THIS IS 043 AGAIN, ON ANOTHER TABLE.  A qual whose function is not
-- leakproof may not be promoted to an index condition on a table with a row
-- security policy: the policy's qual is a security qual at a lower level, and
-- Postgres will not let a higher-level qual be asked about rows the policy
-- hides (indxpath.c, restriction_is_securely_promotable).  043 fixed that for
-- `tsvector @@ tsquery` and left the operators the gazetteer probes with
-- untouched, while `entities` has been under row security since 020.  So every
-- arm of pgkg_match_entity_mentions() was a sequential scan of the whole entity
-- table under pgkg_app, and only under pgkg_app.
--
-- This is the ingest hot path, not a query-time cost that a cache can absorb:
-- the matcher runs per chunk and probes once per candidate phrase, so the miss
-- multiplies by phrases per passage and by passages in the corpus.  Measured
-- here on 40,001 entities in one org, as pgkg_app against the same statement as
-- the table owner:
--
--   name arm   Seq Scan, 42.6 ms, 920 buffers   ->  Index Scan,        4 buffers
--   alias arm  Seq Scan, 186.6 ms, 923 buffers  ->  Bitmap Index Scan, 5 buffers
--   fuzzy arm  Seq Scan, 74.3 ms, 920 buffers   ->  Bitmap Index Scan, 70 buffers
--
-- and after this migration, as pgkg_app: 0.052 ms / 4 buffers on the name arm,
-- which is now an Index Only Scan, 0.032 ms / 5 on the alias arm and 0.089 ms /
-- 70 on the fuzzy arm.  Nothing about which rows any role may read changes:
-- leakproofness decides when a qual may run, not whether the policy filters the
-- row, and the policy's own qual is still in every plan above.
--
-- WHY similarity_op IS MARKED LEAKPROOF.  `text % text` is pg_trgm's
-- similarity_op, and it is the candidate generator for the fuzzy arm — the only
-- thing that can read entities_name_trgm_idx.  The claim leakproofness asks for
-- is that the function cannot reveal its arguments other than through its
-- return value, and similarity_op counts trigrams in two strings and compares
-- the ratio against pg_trgm.similarity_threshold: no data-dependent error, no
-- message carrying either argument, no side effect.  The boolean is not
-- observable either, because the policy qual still filters the row before
-- anything is returned; what the planner gains is permission to ask the index
-- the question first.  This is exactly 043's argument for ts_match_vq, on the
-- operator the other index needs.
--
-- WHY arraycontains IS MARKED LEAKPROOF, AND THE ONE RESERVATION.  `anyarray @>
-- anyarray` is arraycontains, and it is what reads the alias index.  It raises
-- nothing that depends on a value — its only error names a type that has no
-- default equality operator — emits no element in any message, and has no side
-- effect.  Its residual leak surface is not its own code but the element
-- equality it resolves through the type's default btree operator class, and
-- that is the reservation worth stating rather than burying: the function is
-- polymorphic, so the marking is database-wide and covers arrays of every
-- element type, including one whose equality function a later deployment
-- defines.  For the only containment this schema promotes — text[] against
-- text[] — that equality is texteq, which Postgres already ships leakproof.  A
-- deployment that adds a type with a leaky equality and relies on array
-- containment of it should reconsider this line; it is a DO block, and
-- declining it costs the alias arm its index and nothing else.
--
-- WHY THE NORMALISER IS NOT MARKED, AND WHAT THE NAME AND ALIAS ARMS GET
-- INSTEAD.  pgkg_gazetteer_key() is the third non-leakproof function in these
-- quals, and marking it is both ineffective and the wrong kind of claim.
-- Ineffective, measured: the planner inlines a simple SQL function's body
-- before it judges the qual, so the marking on the wrapper is never consulted
-- and the arm still sequential-scans with it set.  What the qual actually
-- contains after inlining is lower(), regexp_replace() and btrim(), none of
-- them leakproof, so honouring the claim would mean marking those three — a
-- claim over every qual in the database that lowercases or trims a column,
-- made to speed up one join.  And a marking on a function this schema owns
-- outlives the review of its body: the next migration to rewrite the
-- normaliser inherits a leakproofness claim nobody re-examined.
--
-- So the normalisation moves to write time.  The two keys become stored
-- generated columns, and the arms compare columns: text equality, which is
-- texteq and already leakproof, and array containment, which is the operator
-- marked above.  Generated rather than trigger-maintained because a generated
-- column cannot drift from the function that defines it, which is the only
-- property that makes a stored key a valid substitute for calling the
-- normaliser.  The name arm gets something the expression index could not
-- give it either — an Index Only Scan, because the key it probes is now a
-- column the index carries.
--
-- The cost is a table rewrite of `entities` and a rebuild of its indexes, paid
-- once here, and two more stored columns per entity row.  The indexes are
-- replaced rather than added because an expression index cannot serve a column
-- qual: keeping both would maintain a second index on every entity write that
-- nothing can use under the role the policies are written for.
--
-- WHAT IS DELIBERATELY UNCHANGED.  The fuzzy arm still generates candidates
-- from the trigram index on the raw `name`, and still confirms them against
-- the threshold, so no phrase matches anything it did not match before; the
-- confirmation reads the stored key instead of recomputing it per row.  And
-- one thing this migration does NOT fix, so it is not mistaken for fixed:
-- pgkg_link_entity()'s second stage filters on `similarity(name, p_name) >
-- 0.6`, the function and not the `%` operator, and a function call over a
-- column reaches no trigram index for any role.  That is a separate finding
-- about that function's own shape, and marking similarity_op leakproof neither
-- helps nor harms it.


-- 1. The two operators the index conditions are made of.
--
-- Both need ownership of the function, which managed Postgres often will not
-- grant, so each degrades to a NOTICE the way 020's role provisioning and 043's
-- own marking do: a deployment that cannot mark them keeps a correct, slow
-- gazetteer.
DO $$
BEGIN
    EXECUTE 'ALTER FUNCTION similarity_op(text, text) LEAKPROOF';
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE
        'similarity_op not marked leakproof (%); the fuzzy gazetteer arm stays '
        'correct but cannot reach entities_name_trgm_idx under a role with row '
        'security', SQLERRM;
END;
$$;

DO $$
BEGIN
    EXECUTE 'ALTER FUNCTION arraycontains(anyarray, anyarray) LEAKPROOF';
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE
        'arraycontains not marked leakproof (%); the alias gazetteer arm stays '
        'correct but cannot reach the alias key index under a role with row '
        'security', SQLERRM;
END;
$$;


-- 2. The gazetteer keys, normalised at write time so the probe is an equality.
ALTER TABLE entities
    ADD COLUMN gazetteer_name_key TEXT
        GENERATED ALWAYS AS (pgkg_gazetteer_key(name)) STORED,
    ADD COLUMN gazetteer_alias_keys TEXT[]
        GENERATED ALWAYS AS (pgkg_gazetteer_keys(aliases)) STORED;

COMMENT ON COLUMN entities.gazetteer_name_key IS
    'pgkg_gazetteer_key(name), stored so the matcher''s probe is an equality on '
    'a column rather than a call over one: a non-leakproof function in the qual '
    'cannot become an index condition on a table with a policy, and this table '
    'has had one since 020 (issue #10).';

COMMENT ON COLUMN entities.gazetteer_alias_keys IS
    'pgkg_gazetteer_keys(aliases), stored for the same reason as '
    'gazetteer_name_key. Keys shorter than three characters are dropped by the '
    'normaliser, so an alias of "AI" is absent here rather than unmatched.';

DROP INDEX entities_gazetteer_name_idx;
DROP INDEX entities_gazetteer_alias_idx;

CREATE INDEX entities_gazetteer_name_idx
    ON entities (org_id, gazetteer_name_key);

CREATE INDEX entities_gazetteer_alias_idx
    ON entities USING gin (gazetteer_alias_keys);


-- 3. The matcher, probing the stored keys.
--
-- Only the three arms of `hit` change: the phrase machinery, the offsets, the
-- one-row-per-pair rule and the watermark are exactly as 040 left them.  The
-- phrase side still calls the normaliser, because a phrase is computed from the
-- passage in the query and there is nothing to store it on.
CREATE OR REPLACE FUNCTION pgkg_match_entity_mentions(
    p_chunk_ids UUID[],
    p_max_words INT  DEFAULT 5,
    p_threshold REAL DEFAULT 0.9
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    v_added BIGINT;
BEGIN
    WITH

    target AS (
        SELECT c.id, c.org_id, lower(c.text) AS lowered
        FROM chunks c
        WHERE c.id = ANY(p_chunk_ids)
    ),

    parts AS (
        SELECT
            t.id, t.org_id, t.lowered, m.ord,
            m.piece[1] AS sep,
            m.piece[2] AS word
        FROM target t
        CROSS JOIN LATERAL regexp_matches(
            t.lowered, '([^[:alnum:]]*)([[:alnum:]]+)', 'g'
        ) WITH ORDINALITY AS m(piece, ord)
    ),

    positioned AS (
        SELECT
            id, org_id, lowered, ord,
            (SUM(length(sep) + length(word)) OVER w - length(word))::INT AS w_start,
            (SUM(length(sep) + length(word)) OVER w)::INT AS w_end
        FROM parts
        WINDOW w AS (PARTITION BY id ORDER BY ord)
    ),

    phrases AS (
        SELECT DISTINCT
            a.id AS chunk_id,
            a.org_id,
            a.w_start AS span_start,
            b.w_end   AS span_end,
            pgkg_gazetteer_key(
                substr(a.lowered, a.w_start + 1, b.w_end - a.w_start)
            ) AS phrase
        FROM positioned a
        JOIN positioned b
          ON b.id = a.id
         AND b.ord BETWEEN a.ord AND a.ord + GREATEST(p_max_words, 1) - 1
    ),

    -- A key shorter than three characters matches half the language, and a
    -- gazetteer that fires on "AI" or "US" is noise the ranking cannot undo.
    usable AS (
        SELECT * FROM phrases WHERE length(phrase) >= 3
    ),

    hit AS (
        SELECT u.chunk_id, u.org_id, e.id AS entity_id,
               u.span_start, u.span_end, 'name'::TEXT AS match_kind
        FROM usable u
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.gazetteer_name_key = u.phrase

        UNION ALL

        SELECT u.chunk_id, u.org_id, e.id,
               u.span_start, u.span_end, 'alias'
        FROM usable u
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.gazetteer_alias_keys @> ARRAY[u.phrase]

        UNION ALL

        SELECT u.chunk_id, u.org_id, e.id,
               u.span_start, u.span_end, 'fuzzy'
        FROM usable u
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.name % u.phrase
         AND similarity(e.gazetteer_name_key, u.phrase) >= p_threshold
        WHERE e.gazetteer_name_key <> u.phrase
    ),

    -- One row per pair.  Certainty outranks position: a phrase that IS the name
    -- is better evidence than one that merely resembles it, wherever each sits.
    best AS (
        SELECT DISTINCT ON (chunk_id, entity_id)
            chunk_id, entity_id, org_id, span_start, span_end, match_kind
        FROM hit
        ORDER BY chunk_id, entity_id,
                 CASE match_kind WHEN 'name' THEN 0 WHEN 'alias' THEN 1 ELSE 2 END,
                 span_start,
                 span_end - span_start DESC
    ),

    inserted AS (
        INSERT INTO entity_mentions
            (entity_id, chunk_id, org_id, span_start, span_end, match_kind)
        SELECT entity_id, chunk_id, org_id, span_start, span_end, match_kind
        FROM best
        ON CONFLICT (entity_id, chunk_id) DO NOTHING
        RETURNING 1
    )

    SELECT COUNT(*) INTO v_added FROM inserted;

    UPDATE chunks SET mentions_matched_at = now()
    WHERE id = ANY(p_chunk_ids)
      AND mentions_matched_at IS NULL;

    RETURN v_added;
END;
$$;


-- 4. The reverse direction, reading the same stored keys.
--
-- Its probe into `entities` is by primary key and was never the sequential scan
-- this migration is about; it reads the columns because recomputing a key the
-- row already carries is work, and because two definitions of "the key of this
-- entity" is how the two directions of the gazetteer come to disagree.
CREATE OR REPLACE FUNCTION pgkg_match_chunk_mentions(
    p_entity_ids UUID[],
    p_max_chunks INT  DEFAULT 1000,
    p_max_words  INT  DEFAULT 5,
    p_threshold  REAL DEFAULT 0.9
) RETURNS BIGINT
LANGUAGE SQL
AS $$
    SELECT pgkg_match_entity_mentions(
        ARRAY(
            SELECT DISTINCT c.id
            FROM (
                SELECT e.org_id, k.key
                FROM entities e
                CROSS JOIN LATERAL (
                    SELECT e.gazetteer_name_key AS key
                    UNION
                    SELECT a.key FROM unnest(e.gazetteer_alias_keys) AS a(key)
                ) k
                WHERE e.id = ANY(p_entity_ids)
                  AND length(k.key) >= 3
            ) n
            CROSS JOIN LATERAL (
                SELECT c.id
                FROM chunks c
                WHERE c.org_id = n.org_id
                  AND c.tsv @@ plainto_tsquery('english', n.key)
                LIMIT GREATEST(p_max_chunks, 0)
            ) c
        ),
        p_max_words, p_threshold
    );
$$;
