-- 058  pgkg_app exists, is granted everything, and its creator can assume it
--      (issue #31).
--
-- WHAT 020 LEFT.  020 created pgkg_app best-effort: without CREATEROLE it
-- raised a NOTICE and carried on, and so did every later GRANT to the role
-- (021, 023, 026, 030, 032, 040), each catching undefined_object.  An install
-- migrated that way has every RLS policy written for a role that does not
-- exist, which protects nothing for the caller that was meant to assume it,
-- and no migration after 020 could notice.  020 now stops instead, so this is
-- only for installs that already ran it.
--
-- WHAT IT DOES, IN THE ORDER A DAMAGED INSTALL NEEDS IT.
--
--   * Creates the role if it is missing, or stops with the same error 020 now
--     raises.  `pgkg migrate` checks the same thing before applying anything,
--     so the runner refuses before a half-applied run; the check here is for a
--     runner that is not `pgkg migrate`.
--   * Grants the role to the session that migrates, if that session could not
--     SET ROLE to it and the migrator holds ADMIN on it.  From PG16 CREATEROLE
--     confers ADMIN on a created role and nothing else — no SET, no INHERIT,
--     with createrole_self_grant empty by default — so an install 020 migrated
--     as a CREATEROLE non-superuser has a role its own login cannot become.
--     Before PG16 a CREATEROLE creator had no membership at all, but could
--     grant any non-superuser role, which is what the version branch reads.  A
--     role an administrator provisioned, and grants membership in themselves,
--     is left alone: the migrator holds no ADMIN on it and needs none.
--   * Re-grants on every table.  A role created late missed every grant from
--     020 onwards, and a GRANT ... ON ALL TABLES that finds them all granted
--     changes nothing, so the repair and the no-op are one statement.
--     tests/test_app_role.py pins the invariant it restores.
--
-- Not an opt-out.  A deployment that means to connect as the owner, for whom
-- every policy is inert anyway, can still have the role created for it; there
-- is no configuration in which the role's absence is the safe answer.
DO $$
DECLARE
    v_session_can_set BOOLEAN;
    v_migrator_can_grant BOOLEAN;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pgkg_app') THEN
        BEGIN
            CREATE ROLE pgkg_app NOLOGIN;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE EXCEPTION
                'pgkg_app does not exist and % cannot create it (%)',
                current_user, SQLERRM
            USING ERRCODE = 'insufficient_privilege',
                  HINT = 'An administrator must run CREATE ROLE pgkg_app NOLOGIN, '
                         'then re-run the migrations; they grant to the role '
                         'and do not need to create it.';
        END;
    END IF;

    IF current_setting('server_version_num')::INT >= 160000 THEN
        v_session_can_set := pg_has_role(session_user, 'pgkg_app', 'SET');
        v_migrator_can_grant :=
            pg_has_role(current_user, 'pgkg_app', 'MEMBER WITH ADMIN OPTION');
    ELSE
        v_session_can_set := pg_has_role(session_user, 'pgkg_app', 'MEMBER');
        v_migrator_can_grant :=
            pg_has_role(current_user, 'pgkg_app', 'MEMBER WITH ADMIN OPTION')
            OR (SELECT rolcreaterole FROM pg_roles WHERE rolname = current_user);
    END IF;

    IF NOT v_session_can_set AND v_migrator_can_grant THEN
        GRANT pgkg_app TO SESSION_USER;
    END IF;

    GRANT USAGE ON SCHEMA public TO pgkg_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO pgkg_app;
END;
$$;
