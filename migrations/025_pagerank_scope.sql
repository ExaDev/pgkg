-- Centrality is computed per tenant, over that tenant's own subgraph.
--
-- D4's hard rule: "Ranking signals are never computed globally over shared
-- content.  pgkg_recompute_pagerank() must run per subscriber over that
-- subscriber's visible subgraph, or over the shared subgraph alone with org
-- edges excluded.  Global centrality over shared entities is a real
-- cross-tenant inference channel."
--
-- The numerator was already confined to the namespace, but the out-degree
-- subquery aggregated every edge in the table with no predicate at all.
-- Out-degree is the divisor, so a tenant's score depended on how many edges
-- other tenants had drawn out of the same entity: measurably, not marginally —
-- one foreign edge out of a two-entity subgraph moved a score by a quarter.
-- That is a channel that carries information the reader cannot otherwise see,
-- and it runs in the direction that matters, from other tenants into this one.
--
-- Two changes.  The org becomes part of the subgraph definition, because
-- namespace alone does not separate tenants once org_id is the isolation
-- boundary — two orgs sharing a namespace shared a divisor.  And out-degree
-- counts only edges that stay inside the subgraph, which is what makes this
-- PageRank over that subgraph rather than a projection of a global one.
--
-- The parameter is last and defaults to the request scope, so a one-argument
-- call keeps working and resolves to the same org every RLS policy and every
-- column default already resolves to.  Adding a parameter changes a function's
-- identity, so the old definition goes first; nothing references it.

DROP FUNCTION pgkg_recompute_pagerank(TEXT, INT, REAL);

CREATE FUNCTION pgkg_recompute_pagerank(
    p_namespace TEXT,
    iterations  INT  DEFAULT 20,
    damping     REAL DEFAULT 0.85,
    p_org_id    UUID DEFAULT NULL
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    n     INT;
    base  REAL;
    iter  INT;
    v_org UUID := COALESCE(p_org_id, pgkg_current_org());
BEGIN
    -- 1. The subgraph: this org's entities in this namespace, and nothing else.
    SELECT COUNT(*) INTO n
    FROM entities
    WHERE namespace = p_namespace AND org_id = v_org;

    IF n = 0 THEN RETURN; END IF;

    base := (1.0 - damping) / n;

    -- 2. Seed uniformly.
    INSERT INTO entity_pagerank (entity_id, score, computed_at)
    SELECT id, 1.0 / n, now()
    FROM entities
    WHERE namespace = p_namespace AND org_id = v_org
    ON CONFLICT (entity_id) DO UPDATE
        SET score = EXCLUDED.score,
            computed_at = EXCLUDED.computed_at;

    -- 3. Power iterations.  Both endpoints of every counted edge are in the
    -- subgraph, in the numerator and in the divisor alike: an edge leaving the
    -- subgraph is not a vote inside it, and counting it in out-degree alone
    -- would dilute every real neighbour by an amount set outside the tenant.
    FOR iter IN 1 .. iterations LOOP
        UPDATE entity_pagerank ep
        SET score = base + damping * (
            SELECT COALESCE(SUM(ep2.score * e.weight / out_deg.total), 0.0)
            FROM edges e
            JOIN entity_pagerank ep2 ON ep2.entity_id = e.src_entity
            JOIN entities src_ent   ON src_ent.id = e.src_entity
                                   AND src_ent.namespace = p_namespace
                                   AND src_ent.org_id = v_org
            JOIN (
                SELECT e2.src_entity, SUM(e2.weight) AS total
                FROM edges e2
                JOIN entities s ON s.id = e2.src_entity
                               AND s.namespace = p_namespace
                               AND s.org_id = v_org
                JOIN entities d ON d.id = e2.dst_entity
                               AND d.namespace = p_namespace
                               AND d.org_id = v_org
                GROUP BY e2.src_entity
            ) out_deg ON out_deg.src_entity = e.src_entity
            WHERE e.dst_entity = ep.entity_id
        )
        WHERE ep.entity_id IN (
            SELECT id FROM entities
            WHERE namespace = p_namespace AND org_id = v_org
        );
    END LOOP;

    -- 4. Stamp computation time.
    UPDATE entity_pagerank ep
    SET computed_at = now()
    WHERE ep.entity_id IN (
        SELECT id FROM entities
        WHERE namespace = p_namespace AND org_id = v_org
    );
END;
$$;
