-- The mention edge, and the bridge into shared entity space (ADR 0001, D2/D4).
--
-- WHY THIS EDGE IS THE POINT OF THE WHOLE DESIGN.  D2 keeps the corpus out of
-- the extractor, which leaves two pools of retrievable material that meet only
-- topically: a query either sounds like the passage or it does not.  Chats,
-- meanwhile, generate facts that NAME things — the Helios migration, the Acme
-- renewal — and the corpus is where those things are defined.  One edge from an
-- entity to the passages that mention it turns a retrieved chat fact into a
-- seed for the document that explains it, even when the query wording matches
-- that document not at all.  That is the claim D2 makes for the architecture,
-- and nothing else in the schema can make it.
--
-- WHY A GAZETTEER AND NOT A MODEL.  D2 prices the alternative: extracting the
-- corpus is hundreds of dollars per pass, recurring on every prompt_version
-- bump and multiplied by tenant.  Matching the org's OWN entity names against
-- each passage is an index probe per candidate phrase, and the indexes are
-- already there — 002 gave entities a trigram GIN index on name and a GIN index
-- on aliases, and neither has ever been read.  The gazetteer is what reads
-- them.  It also means the edge is only ever as good as the entity list, which
-- is the right failure mode: no hallucinated relations, no silent cost.
--
-- WHY THE MATCHING RUNS BOTH WAYS.  A chunk can only mention entities that
-- exist when it is ingested, and the entity a chat fact creates is usually
-- created afterwards.  Matching at chunk-ingest time alone would therefore miss
-- exactly the entities the graph is learning, so the reverse direction — these
-- new names, against the passages already stored — is a first-class operation
-- rather than a full re-sweep.  The two use different indexes because the
-- selective side differs: phrases of one chunk probe the name index, while a
-- handful of names probe the chunk tsvector index.
--
-- WHY THE SPAN IS THE FIRST OCCURRENCE AND ONE ROW PER PAIR.  A mention row is
-- a graph edge, not an annotation layer: expansion asks "does this passage
-- mention this entity", never "how often".  One row per (entity, chunk) keeps
-- the edge table proportional to the graph rather than to the prose, and the
-- span is carried so a citation can point at the words rather than at the
-- passage.
--
-- WHY entity_links IS NOT A MERGE.  D4 is explicit: "don't merge shared
-- entities into per-org entity space — that is a per-org copy of the shared
-- graph".  Copying is what a tenant-scoped unique key on (org, namespace, name)
-- would force, and the copy immediately stops tracking its original.  A link
-- table bridges instead: the org keeps its own entity, the operator keeps
-- theirs, and expansion crosses the bridge at query time — where the visibility
-- predicate still decides what the crossing is allowed to reach.  A trigger
-- enforces the direction, because a rule stated only in an ADR is not a
-- constraint.
--
-- SECURITY.  Both directions of expansion re-filter every row they reach
-- through pgkg_visible(), the same predicate the seed passed.  D3 accepts that
-- entity NAMES cross user boundaries inside an org and accepts nothing else, so
-- a walk from a shared entity into another user's private fact — or now, into
-- another user's private passage — is the leak this migration must not open.


-- 1. The gazetteer key: what a name and a phrase have to agree on to match.
--
-- Case, punctuation and run-length of whitespace are the three ways the same
-- name is written differently in prose and in an entity row, and none of them
-- carries meaning here.  IMMUTABLE so it can be indexed, which is the whole
-- reason the equality arm below is an index probe rather than a scan.
CREATE FUNCTION pgkg_gazetteer_key(p_text TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
AS $$
    SELECT btrim(regexp_replace(lower(p_text), '[^[:alnum:]]+', ' ', 'g'))
$$;

CREATE FUNCTION pgkg_gazetteer_keys(p_texts TEXT[]) RETURNS TEXT[]
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
AS $$
    SELECT ARRAY(
        SELECT pgkg_gazetteer_key(t)
        FROM unnest(p_texts) AS t
        WHERE length(pgkg_gazetteer_key(t)) >= 3
    )
$$;

-- The two probes the gazetteer makes, over the org it is confined to.
CREATE INDEX entities_gazetteer_name_idx
    ON entities (org_id, pgkg_gazetteer_key(name));

CREATE INDEX entities_gazetteer_alias_idx
    ON entities USING gin (pgkg_gazetteer_keys(aliases));


-- 2. The edge itself.
--
-- org_id is denormalised onto the row rather than reached through the entity,
-- because the row-level policy has to be an equality on this table: a policy
-- that joins is a policy the planner cannot prune with.  It is always the org
-- of BOTH endpoints — the matcher never crosses the boundary — so the two
-- statements of it cannot disagree.
CREATE TABLE entity_mentions (
    entity_id  UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id   UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    org_id     UUID NOT NULL DEFAULT pgkg_current_org() REFERENCES orgs(id),
    span_start INT  NOT NULL,
    span_end   INT  NOT NULL,
    match_kind TEXT NOT NULL DEFAULT 'name',
    matched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, chunk_id),
    CONSTRAINT entity_mentions_match_kind_check
        CHECK (match_kind IN ('name', 'alias', 'fuzzy')),
    CONSTRAINT entity_mentions_span_check CHECK (span_end > span_start)
);

COMMENT ON TABLE entity_mentions IS
    'Which passages name which entities, found by gazetteer matching and never '
    'by a model. The edge that lets a retrieved chat fact pull in the document '
    'that defines what it names (ADR 0001, D2).';

COMMENT ON COLUMN entity_mentions.span_start IS
    'Offset of the first occurrence in chunks.text. One row per pair: an edge '
    'answers whether the passage names the entity, never how often.';

-- Both directions of the expansion below drive off one of these.
CREATE INDEX entity_mentions_chunk_idx ON entity_mentions (chunk_id);
CREATE INDEX entity_mentions_entity_idx ON entity_mentions (entity_id);

-- NO ROW-LEVEL SECURITY YET, DELIBERATELY, and for the reason 030 gave when it
-- deferred the lifecycle tables' policies: a policy has to arrive with the
-- isolation test that can tell it from USING (TRUE), and that test module is
-- not part of this change.  The column the policy will read is here, and it is
-- the org of BOTH endpoints because the matcher never crosses the boundary —
-- so the eventual policy is an equality on this table and not a join, which is
-- what a prunable predicate needs.  Until it lands, the boundary on the read
-- path is pgkg_visible(), re-applied to every row expansion reaches.


-- 3. The watermark.  Without it a sweep re-scans every settled passage forever,
-- and "a passage that mentions nothing" is indistinguishable from "a passage
-- nobody has looked at".  Stamped by the matcher, so the two cannot drift.
ALTER TABLE chunks ADD COLUMN mentions_matched_at TIMESTAMPTZ;

CREATE INDEX chunks_mentions_pending_idx ON chunks (org_id)
    WHERE mentions_matched_at IS NULL;

COMMENT ON COLUMN chunks.mentions_matched_at IS
    'When the gazetteer last ran over this passage. NULL is the sweep queue; a '
    'new entity does not reset it, because the reverse matcher handles that '
    'case without re-reading the corpus.';


-- 4. Chunk to entities: the phrases of one passage, probed against the names.
--
-- The passage is tokenised once and its contiguous word runs up to p_max_words
-- become candidate phrases.  Each phrase is one equality probe on the gazetteer
-- key index, one containment probe on the alias index, and — only when a
-- threshold is asked for — one trigram probe on the index 002 built and nothing
-- has used since.  Cost is therefore linear in the passage and independent of
-- the size of the entity table, which is what makes this affordable on every
-- ingest.
--
-- Offsets are carried through the tokenisation rather than recovered by
-- searching for the phrase afterwards: a running sum over the separator-plus-
-- word split gives an exact offset into the original text, where a later
-- strpos() would silently miss any name written with a newline inside it.
CREATE FUNCTION pgkg_match_entity_mentions(
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

    -- Each match is one separator run plus the word that ends it, in order, so
    -- the cumulative length of the pair is the offset just past that word.
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
         AND pgkg_gazetteer_key(e.name) = u.phrase

        UNION ALL

        SELECT u.chunk_id, u.org_id, e.id,
               u.span_start, u.span_end, 'alias'
        FROM usable u
        JOIN entities e
          ON e.org_id = u.org_id
         AND pgkg_gazetteer_keys(e.aliases) @> ARRAY[u.phrase]

        UNION ALL

        SELECT u.chunk_id, u.org_id, e.id,
               u.span_start, u.span_end, 'fuzzy'
        FROM usable u
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.name % u.phrase
         AND similarity(pgkg_gazetteer_key(e.name), u.phrase) >= p_threshold
        WHERE pgkg_gazetteer_key(e.name) <> u.phrase
    ),

    -- One row per pair.  Certainty outranks position: a phrase that IS the name
    -- is better evidence than one that merely resembles it, wherever each sits.
    -- Among equally certain matches the earliest occurrence wins, and at equal
    -- offsets the longer one does — "Helios Migration Programme" is a better
    -- citation than "Helios".
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


-- 5. Entities to chunks: a handful of names, probed against the passages.
--
-- One gazetteer, two candidate generators.  This direction cannot n-gram the
-- corpus again — that is the full re-sweep it exists to avoid — so it asks the
-- chunk tsvector index 030 built which passages could possibly contain each
-- name, and hands that short list to the matcher above.  The tsvector is
-- stemmed and unordered, so it can only propose; the phrase machinery is still
-- what decides, which is why there is no second definition of a match here.
--
-- Matching the proposed passages against EVERY entity rather than only the ones
-- asked about is deliberate: the set is small, the insert is idempotent, and a
-- passage the corpus never swept gets its whole edge set in the same pass.
CREATE FUNCTION pgkg_match_chunk_mentions(
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
                    SELECT pgkg_gazetteer_key(e.name) AS key
                    UNION
                    SELECT a.key FROM unnest(pgkg_gazetteer_keys(e.aliases)) AS a(key)
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


-- 6. The sweep queue.  Live passages only: a chunk that is a chat turn's
-- provenance, or one carried by a version nobody can resolve, is not something
-- retrieval will ever reach through a mention either.
CREATE FUNCTION pgkg_unmatched_chunks(
    p_org_id UUID,
    p_limit  INT DEFAULT 1000
) RETURNS SETOF UUID
LANGUAGE SQL STABLE
AS $$
    SELECT c.id
    FROM chunks c
    WHERE c.org_id = p_org_id
      AND c.mentions_matched_at IS NULL
      AND pgkg_chunk_live(c.id, c.refcount)
    ORDER BY c.created_at, c.id
    LIMIT GREATEST(p_limit, 0)
$$;


-- 7. The bridge into shared entity space (D4).
--
-- "Don't merge shared entities into per-org entity space — that is a per-org
-- copy of the shared graph."  A copy is what a tenant-scoped unique key on
-- (org, namespace, name) forces, and it stops tracking its original the moment
-- it is made: the operator corrects the shared entity and every tenant keeps
-- the stale duplicate.  A link table bridges instead.  The org keeps its own
-- entity, the operator keeps theirs, and expansion crosses at query time —
-- where the visibility predicate is still what decides what the crossing may
-- reach, so a link is a way to reach subscribed material and never a way to
-- subscribe.
--
-- confidence discounts the crossing rather than gating it, because entity
-- resolution across two independently curated name spaces is a similarity
-- judgement and the honest place for that is the score.
CREATE TABLE entity_links (
    org_entity_id    UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    shared_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    org_id           UUID NOT NULL DEFAULT pgkg_current_org() REFERENCES orgs(id),
    confidence       REAL NOT NULL DEFAULT 1.0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (org_entity_id, shared_entity_id),
    CONSTRAINT entity_links_confidence_check
        CHECK (confidence > 0.0 AND confidence <= 1.0)
);

COMMENT ON TABLE entity_links IS
    'Which of an org''s entities correspond to which entities of a shared '
    'collection. A bridge, never a merge: nothing is copied into per-org '
    'entity space (ADR 0001, D4).';

CREATE INDEX entity_links_shared_idx ON entity_links (shared_entity_id);


-- The direction, enforced.  A CHECK cannot read another table, so the rule that
-- makes this a bridge rather than a merge needs a trigger: the shared side must
-- live in the operator's org and the org side must not.  Without it the table
-- degrades into an arbitrary alias graph inside one tenant, which is the merge
-- under a different name.
CREATE FUNCTION pgkg_entity_links_direction() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_org_side    UUID;
    v_shared_side UUID;
BEGIN
    SELECT org_id INTO v_org_side FROM entities WHERE id = NEW.org_entity_id;
    SELECT org_id INTO v_shared_side FROM entities WHERE id = NEW.shared_entity_id;

    IF v_shared_side IS DISTINCT FROM pgkg_system_org() THEN
        RAISE EXCEPTION
            'entity_links bridges into shared entity space: % is owned by %, '
            'not by the system org', NEW.shared_entity_id, v_shared_side;
    END IF;

    IF v_org_side IS NOT DISTINCT FROM pgkg_system_org() THEN
        RAISE EXCEPTION
            'entity_links bridges from an org entity: % is already shared',
            NEW.org_entity_id;
    END IF;

    NEW.org_id := v_org_side;
    RETURN NEW;
END;
$$;

CREATE TRIGGER pgkg_entity_links_direction
    BEFORE INSERT OR UPDATE ON entity_links
    FOR EACH ROW
    EXECUTE FUNCTION pgkg_entity_links_direction();

-- Its policy is deferred alongside entity_mentions', and for the same reason.


-- 8. Expansion, in both directions, over one arm.
--
-- The seed set is no longer proposition-shaped.  A fused candidate list holds
-- rows from both stores (031 made the item ids resolvable rather than tagged),
-- so a seed contributes entities through propositions.subject_id/object_id when
-- it is a fact and through entity_mentions when it is a passage — and the
-- neighbours it reaches are drawn from both tables for the same reason.  That
-- is what makes the walk bidirectional without a second function: fact to
-- entity to passage, and passage to entity to fact, are the same two joins read
-- in opposite orders.
--
-- EVERY ROW REACHED IS RE-FILTERED.  D3 accepts that entity NAMES cross user
-- boundaries inside an org — that is where most of the graph's value lives —
-- and accepts nothing else.  A seed entity is therefore a bridge that any user
-- of the org can stand on, and both arms below re-apply pgkg_visible() with the
-- caller's own scope, exactly as the arm that produced the seed did.  The chunk
-- arm additionally re-applies pgkg_chunk_live(), because a passage retrieval
-- would not return is not a passage expansion may resurrect.
--
-- The fan-out cap stays per seed entity and is now shared by the two arms: a
-- hub entity with a hundred mentions cannot spend the whole budget any more
-- than a hub entity with a hundred edges could.
--
-- pgkg_search() calls this arm too and joins its output back to propositions,
-- so the passages this now emits simply drop out there.  That surface is
-- proposition-shaped by contract; pgkg_retrieve() is the one D1 defines over
-- both stores, and it resolves whichever store an item id belongs to.
CREATE OR REPLACE FUNCTION pgkg_graph_candidates(
    p_seeds          pgkg_candidate[],
    p_namespace      TEXT   DEFAULT 'default',
    k_seed_entities  INT    DEFAULT 20,
    k_per_seed       INT    DEFAULT 10,
    k_total          INT    DEFAULT 100,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
WITH

seeds AS (
    SELECT s.item_id AS seed_id, s.raw_score AS score
    FROM unnest(p_seeds) AS s(item_id, kind, cand_rank, raw_score)
),

-- Entities named by the seeds, whichever store each seed came from.
named AS (
    SELECT p.subject_id AS entity_id, s.score
    FROM seeds s
    JOIN propositions p ON p.id = s.seed_id
    WHERE p.subject_id IS NOT NULL

    UNION ALL

    SELECT p.object_id, s.score
    FROM seeds s
    JOIN propositions p ON p.id = s.seed_id
    WHERE p.object_id IS NOT NULL

    UNION ALL

    SELECT m.entity_id, s.score
    FROM seeds s
    JOIN entity_mentions m ON m.chunk_id = s.seed_id
),

-- Plus whatever those entities are bridged to in shared space, discounted by
-- the confidence of the correspondence.
bridged AS (
    SELECT entity_id, score FROM named

    UNION ALL

    SELECT el.shared_entity_id, (n.score * el.confidence)::REAL
    FROM named n
    JOIN entity_links el ON el.org_entity_id = n.entity_id
),

seed_entities AS (
    SELECT entity_id
    FROM (
        SELECT entity_id, MAX(score) AS best_score
        FROM bridged
        GROUP BY entity_id
    ) deduped
    ORDER BY best_score DESC
    LIMIT k_seed_entities
),

-- How many of the seed entities each candidate passage names. A passage that
-- ties several of them together is better evidence than one that mentions a
-- single name in passing, and it is the only relevance signal a mention edge
-- carries.
mention_weight AS (
    SELECT m.chunk_id, COUNT(DISTINCT m.entity_id) AS seeds_named
    FROM entity_mentions m
    JOIN seed_entities se ON se.entity_id = m.entity_id
    GROUP BY m.chunk_id
),

-- Facts about each seed entity, by both routes into them.  edges carries a
-- weight and is what a manually curated relation lands in, but it only ever
-- holds entity-to-entity claims: a fact whose object is a literal — which is
-- most of what a chat produces — has a subject_id and no edge row at all.
-- Expanding from a passage into "the facts about what it mentions" has to see
-- those, so subject_id and object_id are a second route to the same set and the
-- weight is whatever the edge route found, or nothing.
fact_route AS (
    SELECT u.entity_id, u.cand_id, MAX(u.weight) AS weight
    FROM (
        SELECT se.entity_id, e.proposition_id AS cand_id,
               COALESCE(e.weight, 0.0) AS weight
        FROM seed_entities se
        JOIN edges e
          ON e.src_entity = se.entity_id
          OR e.dst_entity = se.entity_id

        UNION ALL

        SELECT se.entity_id, np.id, 0.0
        FROM seed_entities se
        JOIN propositions np
          ON np.subject_id = se.entity_id
          OR np.object_id = se.entity_id
    ) u
    GROUP BY u.entity_id, u.cand_id
),

visible_facts AS (
    SELECT fr.entity_id, fr.cand_id, fr.weight
    FROM fact_route fr
    JOIN propositions np ON np.id = fr.cand_id
    WHERE np.namespace = p_namespace
      AND pgkg_temporal_visible(
            np.invalidated_at, np.valid_from, np.valid_to,
            COALESCE(p_valid_at, now())
          )
      AND NOT EXISTS (SELECT 1 FROM seeds s WHERE s.seed_id = np.id)
      AND pgkg_visible(
            np.org_id, np.collection_id, np.visibility,
            np.owner_user_id, np.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
),

-- Numbered within the seed entity, so the cap below is per seed and the two
-- stores share one budget: a hub with a hundred mentions can no more spend the
-- whole allowance than a hub with a hundred edges could.
per_seed AS (
    SELECT
        vf.cand_id,
        ROW_NUMBER() OVER (
            PARTITION BY vf.entity_id ORDER BY vf.weight DESC, vf.cand_id
        ) AS seed_rank
    FROM visible_facts vf

    UNION ALL

    SELECT
        c.id,
        ROW_NUMBER() OVER (
            PARTITION BY m.entity_id
            ORDER BY mw.seeds_named DESC, c.id
        )
    FROM seed_entities se
    JOIN entity_mentions m ON m.entity_id = se.entity_id
    JOIN chunks c ON c.id = m.chunk_id
    JOIN mention_weight mw ON mw.chunk_id = c.id
    WHERE NOT EXISTS (SELECT 1 FROM seeds s WHERE s.seed_id = c.id)
      AND pgkg_chunk_live(c.id, c.refcount)
      AND pgkg_visible(
            c.org_id, c.collection_id, c.visibility,
            c.owner_user_id, c.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
),

capped AS (
    SELECT cand_id, MIN(seed_rank) AS best_seed_rank
    FROM per_seed
    WHERE seed_rank <= k_per_seed
    GROUP BY cand_id
)

SELECT
    capped.cand_id,
    'graph'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY capped.best_seed_rank, capped.cand_id))::INT,
    COALESCE((SELECT MIN(score) FROM seeds), 0.0)::REAL
FROM capped
ORDER BY capped.best_seed_rank, capped.cand_id
LIMIT k_total;
$$;


-- 9. The application role, again: 020's GRANT was a snapshot of the tables that
-- existed then, so a table created here is unreachable by the role a deployment
-- connects as until it is granted by name.
DO $$
BEGIN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON '
            'entity_mentions, entity_links TO pgkg_app';
EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
    RAISE NOTICE 'pgkg_app not granted on the mention tables (%)', SQLERRM;
END;
$$;
