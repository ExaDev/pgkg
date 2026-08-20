-- Entity resolution is confined to one org.
--
-- 020 gave entities an org_id and an RLS policy but left the unique key at
-- (namespace, name, type), and migration 013 had already noted that closing
-- the NULL-type gap "belongs with the tenancy migration".  The two halves
-- never met, and the result is worse than either: entity resolution reads and
-- writes without naming the org, while the constraint it relies on and the
-- policy that hides rows from it disagree about what a duplicate is.
--
-- Under RLS that combination loses data silently.  A second tenant naming an
-- entity a first tenant already named finds nothing in stages 1 and 2 (the
-- policy hides the row), conflicts on the unique index in stage 3, takes the
-- DO NOTHING path, re-reads, is hidden from the winner again, and returns
-- NULL.  ingest stores that NULL as a proposition with no subject, so every
-- fact mentioning a name another tenant reached first drops out of the graph
-- with no error anywhere.  Without RLS the same gap runs the other way and the
-- second tenant adopts the first tenant's entity row outright.
--
-- The fix is to put the org in the key and in the lookups.  D3 accepts that
-- entity NAMES are shared between users inside one org, because that is where
-- most of the graph's value lives; it accepts nothing across orgs, which is the
-- hard isolation boundary.
--
-- COALESCE(type, '') rather than a bare column list: the old constraint treated
-- NULL types as distinct, so an untyped entity was not protected by the index
-- and stage 3 had nothing to block on.  An expression index is also what
-- ON CONFLICT can name, and it works on every version the project supports,
-- which NULLS NOT DISTINCT does not.

ALTER TABLE entities
    DROP CONSTRAINT entities_namespace_name_type_key;

CREATE UNIQUE INDEX entities_org_namespace_name_type_key
    ON entities (org_id, namespace, name, COALESCE(type, ''));

CREATE INDEX entities_org_namespace_idx ON entities (org_id, namespace);


-- The signature does not change: the org comes from the same STABLE GUC
-- function that entities.org_id already defaults to and that every RLS policy
-- reads, so a caller that set the request scope gets org-correct resolution
-- without passing anything, and one that did not keeps landing in the default
-- org exactly as before.  Stages 1 and 2 filter on org_id explicitly rather
-- than leaning on RLS, because RLS is inert for the table owner and resolution
-- must not depend on which role happens to be connected.
--
-- Only the org predicate, the conflict target and the recovery read change.
-- Stage 2's trigram-plus-cosine arm is otherwise untouched, and every
-- observable result inside a single org is what it was.
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

    -- 2. Trigram + embedding similarity match, inside this org
    IF p_embedding IS NOT NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE org_id = v_org
          AND namespace = p_namespace
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
    -- The competitor can now only be inside this org, which is what makes the
    -- recovery read able to see it.
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
