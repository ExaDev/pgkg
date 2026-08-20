-- Entity resolution has to be idempotent under concurrency.
--
-- pgkg_link_entity() reads before it inserts. Phase 0 made ingest one
-- transaction, which widened the gap between that read and that insert from a
-- single statement to a whole document: two ingests that first mention the same
-- entity now routinely interleave, and the loser's INSERT hits
-- entities_namespace_name_type_key. The caller cannot fix this without
-- replaying the whole document, so the function absorbs it: the create attempt
-- yields to a concurrent winner and re-reads the row that won.
--
-- Only stage 3 changes. Stages 1 and 2 (exact match, then trigram + cosine) are
-- untouched, as are the signature, the return type and every observable result
-- in the uncontended case.
--
-- Note on NULL types: entities_namespace_name_type_key treats NULL types as
-- distinct, so an untyped entity is not protected by the index and cannot
-- conflict. Stage 1 still dedupes it for a sequential caller. Closing that gap
-- means changing the constraint, which is a schema semantics change and belongs
-- with the tenancy migration, not here.

CREATE OR REPLACE FUNCTION pgkg_link_entity(
    p_namespace  TEXT,
    p_name       TEXT,
    p_type       TEXT,
    p_embedding  halfvec,
    p_threshold  REAL DEFAULT 0.85
) RETURNS UUID
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    v_id UUID;
BEGIN
    -- 1. Exact name + type match within namespace
    SELECT id INTO v_id
    FROM entities
    WHERE namespace = p_namespace
      AND name = p_name
      AND (type = p_type OR (type IS NULL AND p_type IS NULL))
    LIMIT 1;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    -- 2. Trigram + embedding similarity match
    IF p_embedding IS NOT NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE namespace = p_namespace
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
    INSERT INTO entities (name, type, embedding, namespace)
    VALUES (p_name, p_type, p_embedding, p_namespace)
    ON CONFLICT (namespace, name, type) DO NOTHING
    RETURNING id INTO v_id;

    IF v_id IS NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE namespace = p_namespace
          AND name = p_name
          AND (type = p_type OR (type IS NULL AND p_type IS NULL))
        LIMIT 1;
    END IF;

    RETURN v_id;
END;
$$;
