-- The gazetteer normaliser under a restricted search_path (issue #28).
--
-- WHY THIS MIGRATION EXISTS.  pgkg_gazetteer_keys called pgkg_gazetteer_key
-- without a schema.  PostgreSQL 17 evaluates an index expression, during CREATE
-- INDEX, REINDEX, VACUUM, ANALYZE and CLUSTER, with search_path pinned to
-- `pg_catalog, pg_temp`, and there the unqualified call does not resolve — so
-- 040 could not build its alias index and a fresh install on 17 or 18 stopped
-- at that migration.  040 is fixed in place, because a fresh install is the one
-- that runs it.  This file is for the installs that ran it before the fix: the
-- runner records a migration by filename, so the corrected 040 is never re-read
-- on a database that has already applied it, and that database is still
-- running the unqualified body.
--
-- WHY IT STILL MATTERS WITH THE EXPRESSION INDEX GONE.  047 replaced both
-- expression indexes with stored generated columns, so an install upgraded in
-- place on 16 has no index for a later pg_upgrade to 17 to trip over.  The
-- generated columns call the same function on every write, though, and a write
-- is not always made under the application's search_path: pg_restore sets it to
-- '' before loading data and recomputes a stored generated column for every row
-- it copies in, so the unqualified body is a dump of `entities` that will not
-- restore — on any version.
--
-- WHY QUALIFIED AND NOT `SET search_path`.  A SQL function with a SET clause is
-- never inlined, and these are the functions the gazetteer's phrase side calls
-- per candidate phrase.  The qualifier is the install schema, substituted by
-- the runner (issue #30).
--
-- The bodies are 040's, character for character but for the qualification.
-- CREATE OR REPLACE keeps the function's OID, so the generated columns that
-- reference it need no rebuild.
CREATE OR REPLACE FUNCTION pgkg_gazetteer_key(p_text TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
AS $$
    SELECT btrim(regexp_replace(lower(p_text), '[^[:alnum:]]+', ' ', 'g'))
$$;

CREATE OR REPLACE FUNCTION pgkg_gazetteer_keys(p_texts TEXT[]) RETURNS TEXT[]
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
AS $$
    SELECT ARRAY(
        SELECT @pgkg_schema@.pgkg_gazetteer_key(t)
        FROM unnest(p_texts) AS t
        WHERE length(@pgkg_schema@.pgkg_gazetteer_key(t)) >= 3
    )
$$;
