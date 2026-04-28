-- Offline pagerank recomputation via power-iteration over edges
CREATE OR REPLACE FUNCTION pgkg_recompute_pagerank(
    p_namespace TEXT,
    iterations  INT DEFAULT 20,
    damping     REAL DEFAULT 0.85
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    n        INT;
    base     REAL;
    iter     INT;
BEGIN
    -- Count entities in namespace
    SELECT COUNT(*) INTO n FROM entities WHERE namespace = p_namespace;
    IF n = 0 THEN RETURN; END IF;

    base := (1.0 - damping) / n;

    -- Initialise / reset scores uniformly
    INSERT INTO entity_pagerank (entity_id, score, computed_at)
    SELECT id, 1.0 / n, now()
    FROM entities
    WHERE namespace = p_namespace
    ON CONFLICT (entity_id) DO UPDATE
        SET score = EXCLUDED.score,
            computed_at = EXCLUDED.computed_at;

    -- Power iterations
    FOR iter IN 1 .. iterations LOOP
        UPDATE entity_pagerank ep
        SET score = base + damping * (
            SELECT COALESCE(SUM(ep2.score * e.weight / out_deg.total), 0.0)
            FROM edges e
            JOIN entity_pagerank ep2 ON ep2.entity_id = e.src_entity
            JOIN entities src_ent   ON src_ent.id = e.src_entity
                                   AND src_ent.namespace = p_namespace
            JOIN (
                SELECT src_entity, SUM(weight) AS total
                FROM edges
                GROUP BY src_entity
            ) out_deg ON out_deg.src_entity = e.src_entity
            WHERE e.dst_entity = ep.entity_id
        )
        WHERE ep.entity_id IN (
            SELECT id FROM entities WHERE namespace = p_namespace
        );
    END LOOP;

    -- Stamp computation time
    UPDATE entity_pagerank ep
    SET computed_at = now()
    WHERE ep.entity_id IN (
        SELECT id FROM entities WHERE namespace = p_namespace
    );
END;
$$;
