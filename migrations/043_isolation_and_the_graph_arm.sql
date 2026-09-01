-- Isolation that the planner, the seam and the graph arm can all live with
-- (ADR 0001, D1, D3, D4).
--
-- WHY THE KEYWORD MATCH IS MARKED LEAKPROOF.  `tsvector @@ tsquery` is
-- ts_match_vq, and its proleakproof is false.  When a table has row-level
-- security the policy's own qual is a security qual at a lower security level
-- than the query's quals, and Postgres may not use a qual of a higher level as
-- an index condition (indxpath.c, restriction_is_securely_promotable): a
-- non-leakproof qual could be evaluated against rows the policy hides.  So
-- under the role the policies were written for, and only under that role, `@@`
-- was demoted from an Index Cond to a Filter and every BM25 arm degraded to a
-- sequential scan whose cost grows with the whole table.  Measured here on
-- 4,003 chunks: as the owner a Bitmap Index Scan on chunk_tsv_idx; as pgkg_app
-- a Seq Scan removing 4,000 rows by filter, and the GIN index unreachable.
--
-- Marking the operator's function leakproof is the whole fix, and it is a claim
-- about that function rather than a concession: leakproofness asks whether the
-- function can reveal its arguments other than through its return value, and
-- ts_match_vq raises no data-dependent error, emits no message carrying either
-- argument, and has no side effect.  The boolean it returns is not observable
-- either, because the policy qual still filters the row before anything is
-- returned; what the planner gains is permission to ask the index the question
-- first.  It needs superuser, so it degrades to a NOTICE the same way 020's own
-- role provisioning does — a deployment that cannot mark it keeps a correct,
-- slow keyword arm.
--
-- WHY THE READ SIDE OF THE POLICIES WIDENS TO THE SYSTEM ORG.  D3's retrieval
-- predicate reads `org_id = ANY([tenant_org] or [tenant_org, SYSTEM_ORG])`, and
-- D4 ships that sharing seam early precisely because it sits in the hot-path
-- predicate.  Every policy said `org_id = pgkg_current_org()` — one org — so
-- under pgkg_app the second element of that array could never match anything:
-- a tenant reading with include_system_org set got its own rows and silently
-- nothing else, and the seam was demonstrated only by asserting what a Python
-- dataclass returns.  023 had already widened collections' read side for this
-- reason; the tables the shared row itself lives in had not followed.
--
-- The widening is a read, never a write.  WITH CHECK stays a single-org
-- equality on every table, which is the schema-level statement of D4's first
-- hard rule: nothing a tenant ingests is ever promoted into a shared
-- collection.  A tenant may read the operator's shelf and may not put anything
-- on it.
--
-- Which tables widen is decided by whether the read path resolves rows of
-- another org in the caller's read scope, not by whether the table looks
-- shared: propositions and chunks because they are what the predicate scans;
-- entities because 040's bridge points at shared entity space, which lives in
-- the system org by trigger; entity_mentions because it is the passage side of
-- that bridge; documents and document_versions because the context window
-- resolves a shared passage's neighbours through them; corpus_stats and
-- lexeme_df because 041's rule is that the statistics domain mirrors the scan's
-- scope term for term, and a domain narrower than the scan is a scoring defect;
-- org_embedders because 041's vector arm resolves each scanned org's primary
-- generation from it.  users, tenant_shards, ingest_jobs,
-- collection_subscriptions and provenance stay single-org: no read path names
-- another org's rows in them.
--
-- WHAT THE WIDENING DELIBERATELY DOES NOT DO is gate on a subscription.  D4
-- makes subscriptions empty by default and D3 makes the collection list "own +
-- subscribed, resolved by the app", so the subscription is a term in the
-- retrieval predicate and RLS remains what it is called there: the second line,
-- enforcing the org boundary.  Expressing it here would also mean a correlated
-- EXISTS in the policy of the two largest tables, evaluated for every row of
-- every other org that a scan touches, and two of the tables above carry no
-- collection_id to gate on at all.
--
-- WHY FOUR TABLES WITH AN ORG COLUMN GAIN A POLICY NOW.  040 shipped
-- entity_mentions and entity_links with an org column, a GRANT to pgkg_app and
-- row security deliberately deferred — "a policy has to arrive with the
-- isolation test that can tell it from USING (TRUE), and that test module is
-- not part of this change".  041 added an org column to corpus_stats and
-- lexeme_df for the same kind of reason.  The test module is part of this
-- change, and it now enumerates the tables that carry an org rather than the
-- tables that carry a policy, so the next table to ship without one fails a
-- test instead of passing every test.  entity_links keeps a single-org read:
-- its org_id is the org side of the bridge by trigger and is never the system
-- org, so widening it would name a row that cannot exist.
--
-- WHY THE EXTRACTION CACHE GAINS AN ORG.  proposition_cache was keyed on
-- hash(chunk_text, extractor_model, prompt_version) alone, with no org, no
-- policy and no restriction on what may enter it, and phase 2 wired it into the
-- corpus path for private collections.  D4 restricts embedding_cache to
-- operator-licensed public material because "for a confidential document a
-- cache hit would confirm another tenant holds it" — and this cache does not
-- merely confirm the holding, it hands over the extracted facts.  So the row
-- gains the org that paid for the extraction, and the operator's org is the
-- sharing seam: an entry the operator writes under the system org is readable
-- by every tenant, which is exactly the public_source rule expressed in the
-- column the rest of the schema already uses to say who owns a row.  Existing
-- rows are discarded rather than attributed, because a key computed without an
-- org cannot say whose extraction it holds, and a cache is the one thing in the
-- schema that may be thrown away.
--
-- The primary key stays on cache_key: 040-era Python names it in ON CONFLICT.
-- One consequence is deliberate and worth stating — the second org to hold a
-- private passage now misses the cache rather than reading the first org's
-- extraction, and pays its extractor again.  That is the direction D4 chooses
-- when it says the boundary is deliberate.
--
-- WHY THE GRAPH ARM ASKS WHICH STORES THE CALLER WANTS.  040 taught
-- pgkg_graph_candidates() to emit passages as well as facts and noted that
-- pgkg_search() "joins its output back to propositions, so the passages this
-- now emits simply drop out there".  They drop out after they have taken places
-- in k_total: with mentions present, pgkg_search('zorbulon', ...) returned 70
-- facts where it had returned 120, and returned nothing in place of the 50 it
-- evicted.  A budget spent on rows the caller will discard is not a contract
-- either function can fix alone, so the caller states which stores it can
-- resolve — the same shape 031 and 041 give the keyword and vector arms with
-- p_source.  pgkg_retrieve() is D1's two-store surface and keeps both, which is
-- the default; pgkg_search() is proposition-shaped by contract and asks for
-- one, so its budget buys only rows it can return.  The filter is applied
-- before the per-seed cap, not after, because a cap is the budget.
--
-- The redeclaration also reads chunks.retrievable directly where 040 called
-- pgkg_chunk_live(c.id, c.refcount).  041 made that function a lookup of this
-- column, and the arm already has the chunk row joined, so the column is the
-- same test without the sublink that stops the function inlining.


-- 1. The keyword index, reachable under the role the policies are for.
DO $$
BEGIN
    EXECUTE 'ALTER FUNCTION ts_match_vq(tsvector, tsquery) LEAKPROOF';
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE
        'ts_match_vq not marked leakproof (%); the keyword arms stay correct '
        'but cannot reach the GIN index under a role with row security', SQLERRM;
END;
$$;


-- 2. The read side of the org boundary, widened to the operator's org.
ALTER POLICY propositions_org_isolation ON propositions
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

ALTER POLICY chunks_org_isolation ON chunks
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

ALTER POLICY documents_org_isolation ON documents
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

ALTER POLICY entities_org_isolation ON entities
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

ALTER POLICY document_versions_org_isolation ON document_versions
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

ALTER POLICY org_embedders_org_isolation ON org_embedders
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

COMMENT ON POLICY propositions_org_isolation ON propositions IS
    'Reads widen to the operator''s shared org, writes never do: a tenant may '
    'read the shared shelf and may not put anything on it. Which shared '
    'collections a tenant actually retrieves from is the subscription term in '
    'the retrieval predicate, not this policy (ADR 0001, D3, D4).';


-- 3. The tables that shipped with an org column and no policy.
ALTER TABLE entity_mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_links    ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus_stats    ENABLE ROW LEVEL SECURITY;
ALTER TABLE lexeme_df       ENABLE ROW LEVEL SECURITY;

CREATE POLICY entity_mentions_org_isolation ON entity_mentions
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY entity_links_org_isolation ON entity_links
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY corpus_stats_org_isolation ON corpus_stats
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY lexeme_df_org_isolation ON lexeme_df
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

COMMENT ON POLICY corpus_stats_org_isolation ON corpus_stats IS
    'The statistics domain is readable exactly where the candidate scan is '
    'readable, because 041''s rule is that the statistics filter mirrors the '
    'scan''s scope term for term. A full rebuild across orgs is therefore an '
    'owner operation (ADR 0001, D4).';


-- 4. The extraction cache, attributed to the org that paid for it.
DELETE FROM proposition_cache;

ALTER TABLE proposition_cache
    ADD COLUMN org_id UUID NOT NULL DEFAULT pgkg_current_org()
        REFERENCES orgs(id) ON DELETE CASCADE;

-- For the CASCADE, not for the lookup: entries are found by their key, and an
-- unindexed referencing column makes deleting an org a sequential scan of the
-- cache.
CREATE INDEX prop_cache_org_idx ON proposition_cache (org_id);

ALTER TABLE proposition_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY proposition_cache_org_isolation ON proposition_cache
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

COMMENT ON COLUMN proposition_cache.org_id IS
    'Who paid for the extraction. An entry under the system org is the '
    'operator''s, shared with every tenant the way embedding_cache shares '
    'public_source material; anything else is one tenant''s and is readable by '
    'that tenant alone, because the cached payload is the extracted facts and a '
    'hit on a confidential passage would hand them over (ADR 0001, D4).';


-- 5. The graph arm, spending its budget on stores the caller can resolve.
DROP FUNCTION pgkg_graph_candidates(
    pgkg_candidate[], TEXT, INT, INT, INT, UUID[], UUID[], UUID, UUID[],
    TIMESTAMPTZ
);

CREATE FUNCTION pgkg_graph_candidates(
    p_seeds          pgkg_candidate[],
    p_namespace      TEXT   DEFAULT 'default',
    k_seed_entities  INT    DEFAULT 20,
    k_per_seed       INT    DEFAULT 10,
    k_total          INT    DEFAULT 100,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL,
    p_sources        TEXT[] DEFAULT ARRAY['propositions', 'chunks']
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
    WHERE 'chunks' = ANY(p_sources)
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
        WHERE 'propositions' = ANY(p_sources)

        UNION ALL

        SELECT se.entity_id, np.id, 0.0
        FROM seed_entities se
        JOIN propositions np
          ON np.subject_id = se.entity_id
          OR np.object_id = se.entity_id
        WHERE 'propositions' = ANY(p_sources)
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
    WHERE 'chunks' = ANY(p_sources)
      AND NOT EXISTS (SELECT 1 FROM seeds s WHERE s.seed_id = c.id)
      AND c.retrievable
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

COMMENT ON FUNCTION pgkg_graph_candidates(
    pgkg_candidate[], TEXT, INT, INT, INT, UUID[], UUID[], UUID, UUID[],
    TIMESTAMPTZ, TEXT[]
) IS
    'One bidirectional expansion over both stores, re-filtering every row it '
    'reaches through the caller''s own visibility predicate. p_sources names '
    'the stores the caller can resolve, and is applied before the per-seed cap '
    'so the budget is never spent on rows the caller will discard '
    '(ADR 0001, D1, D3).';


-- 6. pgkg_search(), asking the graph arm for the one store it can return.
--
-- Only the `expanded` CTE changes.  Retrieval, fusion, the memory profile, the
-- column list and the ordering are exactly as 022 left them.
DROP FUNCTION pgkg_search(
    TEXT, halfvec, INT, INT, TEXT, TEXT, REAL, BOOLEAN, INT,
    UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, pgkg_gen_query[]
);

CREATE FUNCTION pgkg_search(
    q_text                 TEXT,
    q_embedding            halfvec,
    k_retrieve             INT     DEFAULT 100,
    k_initial              INT     DEFAULT 200,
    p_namespace            TEXT    DEFAULT 'default',
    p_session_id           TEXT    DEFAULT NULL,
    recency_half_life_days REAL    DEFAULT 30.0,
    expand_graph           BOOLEAN DEFAULT TRUE,
    rrf_k                  INT     DEFAULT 60,
    p_org_ids              UUID[]  DEFAULT NULL,
    p_collection_ids       UUID[]  DEFAULT NULL,
    p_user_id              UUID    DEFAULT NULL,
    p_acl_groups           UUID[]  DEFAULT NULL,
    p_valid_at             TIMESTAMPTZ DEFAULT NULL,
    p_gen_queries          pgkg_gen_query[] DEFAULT NULL
) RETURNS TABLE (
    proposition_id UUID,
    text           TEXT,
    embedding      vector,
    rrf_score      REAL,
    adjusted_score REAL,
    source_kind    TEXT,
    chunk_id       UUID,
    subject_id     UUID,
    predicate      TEXT,
    object_id      UUID,
    asserted_at    TIMESTAMPTZ
)
LANGUAGE SQL STABLE
AS $$
WITH

retrieved AS (
    SELECT
        ARRAY(
            SELECT (b.item_id, b.kind, b.rank, b.raw_score)::pgkg_candidate
            FROM pgkg_bm25_candidates(
                q_text, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at
            ) b
        )
        ||
        ARRAY(
            SELECT (v.item_id, v.kind, v.rank, v.raw_score)::pgkg_candidate
            FROM pgkg_vector_candidates(
                q_embedding, p_namespace, p_session_id, k_initial,
                p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                p_valid_at
            ) v
        )
        ||
        ARRAY(
            SELECT (g.item_id, g.kind, g.rank, g.raw_score)::pgkg_candidate
            FROM unnest(COALESCE(p_gen_queries, '{}'::pgkg_gen_query[]))
                 AS gq(generation_id, q_embedding),
                 LATERAL pgkg_generation_candidates(
                     gq.generation_id, gq.q_embedding,
                     p_namespace, p_session_id, k_initial,
                     p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
                     p_valid_at
                 ) g
        ) AS candidates
),

-- Seeds for graph expansion are the fused lexical+vector candidates; their
-- fused scores set both the seed-entity ordering and the neighbour floor.
seeds AS (
    SELECT ARRAY(
        SELECT (f.item_id, 'fused'::TEXT, 0, f.fused_score)::pgkg_candidate
        FROM retrieved r, pgkg_fuse(r.candidates, rrf_k) f
    ) AS candidates
),

expanded AS (
    SELECT ARRAY(
        SELECT (g.item_id, g.kind, g.rank, g.raw_score)::pgkg_candidate
        FROM pgkg_graph_candidates(
            CASE WHEN expand_graph THEN s.candidates ELSE '{}'::pgkg_candidate[] END,
            p_namespace, 20, 10, 100,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups,
            p_valid_at, ARRAY['propositions']
        ) g
    ) AS candidates
    FROM seeds s
),

fused AS (
    SELECT f.*
    FROM retrieved r, expanded x, pgkg_fuse(r.candidates || x.candidates, rrf_k) f
),

profiled AS (
    SELECT ap.*
    FROM pgkg_apply_profile(
        ARRAY(
            SELECT (f.item_id, 'fused'::TEXT, 0, f.fused_score)::pgkg_candidate
            FROM fused f
        ),
        recency_half_life_days
    ) ap
)

SELECT
    fused.item_id,
    p.text,
    p.embedding::vector,
    fused.fused_score,
    profiled.adjusted_score,
    CASE
        WHEN fused.in_kw AND fused.in_vec THEN 'both'
        WHEN fused.in_kw                  THEN 'kw'
        WHEN fused.in_vec                 THEN 'vec'
        ELSE                                   'graph'
    END,
    p.chunk_id,
    p.subject_id,
    p.predicate,
    p.object_id,
    p.asserted_at
FROM fused
JOIN propositions p ON p.id = fused.item_id
JOIN profiled ON profiled.item_id = fused.item_id
ORDER BY profiled.adjusted_score DESC
LIMIT k_retrieve;
$$;
