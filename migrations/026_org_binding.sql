-- Two invariants that were being maintained by remembering.
--
-- 1. Every org needs a primary embedder generation.  022 bound every org that
-- existed when it ran and left later ones to the application, which meant a
-- schema invariant held only for rows written down one code path: a fixture, a
-- direct INSERT or a migration in another branch each had to know.  The
-- consequence is not a constraint violation but a silent one — recall resolves
-- no generation for the org and embeds the query with nothing.  D8 makes the
-- generation a property of the org, so the org row is where the binding
-- belongs.
--
-- The trigger seeds and then yields: ON CONFLICT DO NOTHING means an operator
-- who re-roles an org onto another generation is not fighting it, and it stays
-- correct if a later migration inserts orgs of its own.
--
-- 2. A transitional generation's side table needs the same grant every other
-- table has.  pgkg_create_generation_storage() builds it at run time, long
-- after 020's GRANT ON ALL TABLES took its one-time snapshot, so the arm would
-- have failed with permission denied for the exact role the RLS policies are
-- written for — during a cutover window, on the path D8 describes as the
-- graceful one.

CREATE FUNCTION pgkg_bind_org_to_primary_generation() RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO org_embedders (org_id, generation_id, role)
    SELECT NEW.id, g.id, 'primary'
    FROM embedder_generations g
    WHERE g.id = pgkg_generation_1()
    ON CONFLICT (org_id, generation_id) DO NOTHING;

    RETURN NEW;
END;
$$;

CREATE TRIGGER pgkg_orgs_bind_generation
    AFTER INSERT ON orgs
    FOR EACH ROW
    EXECUTE FUNCTION pgkg_bind_org_to_primary_generation();

-- Any org created between 022 and this migration.
INSERT INTO org_embedders (org_id, generation_id, role)
SELECT o.id, pgkg_generation_1(), 'primary'
FROM orgs o
ON CONFLICT (org_id, generation_id) DO NOTHING;


-- Only the GRANT is added; every other statement is 022's.
CREATE OR REPLACE FUNCTION pgkg_create_generation_storage(
    p_generation_id UUID,
    p_source        TEXT DEFAULT 'prop'
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_dim          INT;
    v_source_table TEXT := CASE p_source
                               WHEN 'prop'  THEN 'propositions'
                               WHEN 'chunk' THEN 'chunks'
                           END;
    v_table        TEXT := pgkg_generation_table(p_source, p_generation_id);
BEGIN
    IF v_source_table IS NULL THEN
        RAISE EXCEPTION 'unknown embedding source %', p_source;
    END IF;

    SELECT g.dim INTO v_dim
    FROM embedder_generations g WHERE g.id = p_generation_id;

    IF v_dim IS NULL THEN
        RAISE EXCEPTION 'no such embedder generation %', p_generation_id;
    END IF;

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I ('
        '  item_id UUID PRIMARY KEY REFERENCES %I(id) ON DELETE CASCADE,'
        '  vec halfvec(%s) NOT NULL)',
        v_table, v_source_table, v_dim
    );

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (vec halfvec_cosine_ops)',
        v_table || '_vec_idx', v_table
    );

    BEGIN
        EXECUTE format(
            'GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO pgkg_app', v_table
        );
    EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
        RAISE NOTICE 'pgkg_app not granted on % (%)', v_table, SQLERRM;
    END;

    RETURN v_table;
END;
$$;
