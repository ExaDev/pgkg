-- 059  The keyword arm on a Postgres that will not mark its operators, and a
--      state nobody has to catch in a NOTICE (ADR 0001, D1, D3; issue #32).
--
-- WHAT 043, 046 AND 047 CANNOT DO ON MANAGED POSTGRES.  Each marks a function
-- behind an index condition LEAKPROOF — ts_match_vq and ts_match_qv for the
-- keyword arms' `@@`, similarity_op and arraycontains for the gazetteer's `%`
-- and `@>` — because a qual that is not leakproof may not become an index
-- condition on a table with a policy.  ALTER FUNCTION on a built-in needs
-- ownership of it, which in practice means a real superuser, and Cloud SQL's
-- cloudsqlsuperuser and RDS's rds_superuser are not one.  So on exactly the
-- deployments most likely to run pgkg in production each ALTER falls into its
-- exception handler, a NOTICE scrolls past during the migration, and under
-- pgkg_app `tsv @@ query` is a Filter over the tenant's btree range rather than
-- an Index Cond.  Measured on PG18 with a 40k-row tenant: 0.13-0.15 ms as the
-- owner, 10.7-11.3 ms under the policy with 39,674 rows removed by filter
-- (~80x), 0.16 ms under the policy once a superuser sets the mark.  The vector
-- arm is unaffected, and so is pgkg_visible(), whose expanded operators ship
-- leakproof.  The arm stays correct; it grows linearly with the tenant.
--
-- The alternatives the issue weighs, and why only one is buildable.  A
-- BYPASSRLS owner is not available: only a BYPASSRLS role may create one, and
-- managed Postgres does not hand that out either.  Making the cost explicit is
-- always available, and is the first half of this migration.  An owner-rights
-- keyword arm is the second half, opt-in, for the reasons set out below.
--
-- WHY THE STATE BECOMES A WARNING AND A QUERY.  046 already reports the two
-- `@@` functions through pgkg_keyword_match_leakproof(), which /health reads.
-- It named two of the four operators this schema marks, so a deployment whose
-- gazetteer lost its trigram index was indistinguishable from one that had
-- not, and the migration itself still said so at NOTICE — below the default
-- client_min_messages of every tool that shows a WARNING.  pgkg_leakproof_state()
-- is the one list: all four signatures, what each serves, whether it is marked,
-- and the statement a superuser runs to mark it.  pgkg_keyword_match_leakproof()
-- is restated over it so the list an operator monitors and the list this file
-- names cannot drift apart, and the migration raises a WARNING per unmarked
-- operator.  `pgkg migrate` and `pgkg check` print the same thing after the
-- fact, because a warning emitted once during a migration is still gone by the
-- time anyone asks why retrieval is slow.
--
-- WHY AN OWNER-RIGHTS ARM, AND WHY IT IS SAFE.  The index is unreachable
-- because the planner will not ask a non-leakproof qual about rows the policy
-- hides.  A SECURITY DEFINER function owned by the table owner runs without the
-- policy at all — the owner is exempt from row security on its own tables
-- unless the table is FORCE ROW LEVEL SECURITY, which nothing in pgkg sets — so
-- inside it `tsv @@ q` is an ordinary qual and the GIN index is back (0.72 ms on
-- the 40k tenant).  What that costs is that every row it reads is one no policy
-- looked at, so the function has to restate every predicate the policies would
-- have applied.  That is feasible here, and small, because of what the policies
-- on the four tables the arm reads actually say:
--
--   * propositions, chunks, corpus_stats, lexeme_df each carry exactly one
--     policy, permissive, FOR ALL, reading
--     `org_id = pgkg_current_org() OR org_id = pgkg_system_org()` (020,
--     widened by 043).  That is the whole of what row security enforces on the
--     keyword arm, and the body below restates it on every one of the four
--     reads — the statistics included, because a BM25 score computed over
--     another org's document frequencies is a leak of that org's vocabulary.
--     tests/test_managed_postgres.py pins the policy list against this
--     restatement, so a fifth policy fails a test naming this function.
--   * Collection scope, private rows, ACL groups (048's write side guarantees
--     a group-bounded row carries one) and the bitemporal filter are not
--     policies.  They are pgkg_visible() and pgkg_temporal_visible() over the
--     caller's arguments, in the arm's own body, and they are carried over
--     verbatim.  Both paths trust those arguments identically: RLS never
--     checked them, so the owner arm widens nothing by also not checking them.
--   * The org is read from the GUC, never from an argument, exactly as
--     pgkg_current_org() reads it — unset and blank both mean the backfill org.
--     p_org_ids stays what it is on the policy path, a narrowing the caller
--     chooses; naming an org the session may not read returns nothing.  The
--     GUC is the application's to set on both paths, so this is the trust model
--     row security already has, not a weaker one.
--
-- pgkg_current_org() itself is not called: its body names pgkg_default_org()
-- unqualified, and a SQL function body is parsed against the caller's path at
-- inlining time, which here is pg_catalog alone.  The expression is restated
-- with every name qualified, and the test above compares the two for the
-- unset, blank and set GUC.
--
-- The hardening a SECURITY DEFINER function needs.  `SET search_path =
-- pg_catalog, pg_temp`: nothing the caller can create is on the path, pg_temp
-- is named last so it is not implicitly searched first, and every pgkg object
-- in the body is named `public.` — the schema 020 grants on.  The helpers that
-- inline into it (pgkg_visible, pgkg_temporal_visible, pgkg_default_org,
-- pgkg_system_org) name only their arguments and pg_catalog, so they parse
-- under that path too; a test calls the function from an empty search_path to
-- prove it.  The pinned path is acceptable here because this is a set-returning
-- candidate function, not a predicate meant to inline into a caller's index
-- condition: it never inlines anyway, being SECURITY DEFINER.  EXECUTE is
-- revoked from PUBLIC and granted to pgkg_app alone.
--
-- What it does not remove.  A GIN scan under the owner consults index entries
-- of every org before the org predicate filters them, so its cost varies with
-- how often a term occurs in other tenants — the same timing channel a
-- LEAKPROOF mark opens, and the one 043 accepted for `@@`.  It returns item
-- ids, a rank and a score, and raises no error carrying another org's data.
--
-- WHY IT IS OPT-IN, AND WHERE THE SWITCH IS.  A security-definer path is a
-- second statement of the policies, and a second statement is a thing that can
-- fall behind the first; a deployment that can set LEAKPROOF should, and should
-- not carry the second statement at all.  So pgkg_bm25_candidates() — the name
-- pgkg_retrieve() and pgkg_search() both call — becomes a dispatch on the
-- `pgkg.keyword_arm` GUC: 'owner' takes the owner-rights arm, and anything
-- else, including unset, takes the policy path unchanged.  The application sets
-- the GUC as a connection startup option from PGKG_KEYWORD_ARM, for the reason
-- pgkg/db.py gives for hnsw.iterative_scan.  The policy-path body is renamed,
-- not restated, so it is still 041's text and the plan-shape tests still see
-- the same plan: the dispatcher is a single SELECT and inlines, and so does
-- the body beneath it.  The owner branch goes through a PL/pgSQL gate (section
-- 5) so that a role without EXECUTE on the owner arm is refused only when it
-- selects that arm, never on the default path.
--
-- The opt-in is a choice of plan, not a security boundary.  The owner arm is
-- installed and granted to pgkg_app on every deployment, and pgkg_app may set
-- the GUC or call the function directly.  That is acceptable only because the
-- arm returns what the policy path returns; the tests that pin that equality
-- are what the grant rests on.
--
-- FORCE ROW LEVEL SECURITY.  If a deployment forces row security on these
-- tables, the owner is under the policy inside the function as well.  The arm
-- stays correct — the policy and its restatement both apply — and loses the
-- index again.  pgkg_owner_arm_bypasses_policy() says which it is, from the
-- function's owner and each table's owner and FORCE flag, and `pgkg check`
-- warns when the owner arm is selected but cannot deliver the index.


-- 1. Every operator a policy would demote, and what fixes it.
CREATE FUNCTION pgkg_leakproof_state()
RETURNS TABLE (signature TEXT, serves TEXT, leakproof BOOLEAN, fix TEXT)
LANGUAGE SQL STABLE
AS $$
SELECT
    s.sig,
    s.serves,
    p.proleakproof,
    format('ALTER FUNCTION %I.%s LEAKPROOF', n.nspname, s.sig)
FROM (VALUES
    ('ts_match_vq(tsvector,tsquery)',    'keyword'),
    ('ts_match_qv(tsquery,tsvector)',    'keyword'),
    ('similarity_op(text,text)',         'gazetteer'),
    ('arraycontains(anyarray,anyarray)', 'gazetteer')
) AS s(sig, serves)
LEFT JOIN pg_proc p ON p.oid = to_regprocedure(s.sig)
LEFT JOIN pg_namespace n ON n.oid = p.pronamespace
ORDER BY s.serves DESC, s.sig;
$$;

COMMENT ON FUNCTION pgkg_leakproof_state() IS
    'The LEAKPROOF state of every function behind an index condition that a '
    'row-security policy would otherwise demote to a filter: the keyword arms '
    '(tsvector @@ tsquery, both operand orders) and the gazetteer (text % text, '
    'anyarray @> anyarray). NULL where this server lacks the function. fix is '
    'the statement a superuser runs to mark it (migrations 043, 046, 047, 059).';

CREATE OR REPLACE FUNCTION pgkg_keyword_match_leakproof()
RETURNS TABLE (signature TEXT, leakproof BOOLEAN)
LANGUAGE SQL STABLE
AS $$
SELECT s.signature, s.leakproof
FROM pgkg_leakproof_state() s
WHERE s.serves = 'keyword';
$$;


-- 2. Said once, loudly, at the moment it becomes true.
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT * FROM pgkg_leakproof_state() WHERE leakproof IS NOT TRUE
    LOOP
        RAISE WARNING
            '% is not leakproof, so the % arm cannot reach its index under a '
            'role with row security and scans the whole tenant instead',
            r.signature, r.serves
        USING HINT = format(
            'As a superuser: %s. For the keyword arm without one, see '
            'PGKG_KEYWORD_ARM=owner in the README (Known limitations).',
            COALESCE(r.fix, 'install the function first'));
    END LOOP;
END;
$$;


-- 3. The policy path, under a name of its own.  Renamed rather than restated,
-- so its body stays 041's.
ALTER FUNCTION pgkg_bm25_candidates(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
) RENAME TO pgkg_bm25_candidates_under_policy;


-- 4. The owner-rights path.  041's body, with every pgkg name qualified and
-- the read policies of the four tables it touches restated on each read.
CREATE FUNCTION pgkg_bm25_candidates_as_owner(
    q_text           TEXT,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL,
    p_source         TEXT   DEFAULT 'propositions'
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
WITH

-- The orgs row security would have admitted: the session's, read from the GUC
-- as pgkg_current_org() reads it, and the operator's shared org.  Every table
-- read below is restricted to these, because none of them is under its policy
-- here.
readable_orgs AS (
    SELECT ARRAY[
        COALESCE(
            NULLIF(pg_catalog.current_setting('pgkg.org_id', TRUE), '')::UUID,
            public.pgkg_default_org()
        ),
        public.pgkg_system_org()
    ] AS orgs
),

query_lexemes AS (
    SELECT trim(BOTH '''' FROM t.lexeme) AS lexeme
    FROM unnest(
        string_to_array(plainto_tsquery('english', q_text)::text, ' & ')
    ) AS t(lexeme)
    WHERE q_text IS NOT NULL
      AND q_text <> ''
      AND trim(BOTH '''' FROM t.lexeme) <> ''
),

query_terms AS (
    SELECT COALESCE(array_agg(lexeme), ARRAY[]::TEXT[]) AS terms
    FROM query_lexemes
),

query_or AS (
    SELECT to_tsquery('simple', string_agg(lexeme, ' | ')) AS q
    FROM query_lexemes
),

stats AS (
    SELECT
        GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8 AS n_total,
        GREATEST(
            COALESCE(SUM(cs.total_len), 0)::FLOAT8
            / GREATEST(COALESCE(SUM(cs.n_total), 1), 1)::FLOAT8,
            1.0
        ) AS avgdl
    FROM public.corpus_stats cs
    CROSS JOIN readable_orgs ro
    WHERE cs.kind = CASE p_source
                        WHEN 'propositions' THEN 'proposition'
                        WHEN 'chunks'       THEN 'chunk'
                    END
      AND cs.namespace = CASE p_source WHEN 'chunks' THEN '' ELSE p_namespace END
      AND cs.org_id = ANY(ro.orgs)
      AND (p_org_ids IS NULL OR cs.org_id = ANY(p_org_ids))
      AND (p_collection_ids IS NULL OR cs.collection_id = ANY(p_collection_ids))
),

term_df AS (
    SELECT ld.lexeme, SUM(ld.df)::FLOAT8 AS df
    FROM public.lexeme_df ld
    CROSS JOIN query_terms qt
    CROSS JOIN readable_orgs ro
    WHERE ld.kind = CASE p_source
                        WHEN 'propositions' THEN 'proposition'
                        WHEN 'chunks'       THEN 'chunk'
                    END
      AND ld.namespace = CASE p_source WHEN 'chunks' THEN '' ELSE p_namespace END
      AND ld.lexeme = ANY(qt.terms)
      AND ld.org_id = ANY(ro.orgs)
      AND (p_org_ids IS NULL OR ld.org_id = ANY(p_org_ids))
      AND (p_collection_ids IS NULL OR ld.collection_id = ANY(p_collection_ids))
    GROUP BY ld.lexeme
),

idf AS (
    SELECT
        ql.lexeme,
        LN(
            (s.n_total - COALESCE(d.df, 0.0) + 0.5)
            / (COALESCE(d.df, 0.0) + 0.5)
            + 1.0
        ) AS idf_val
    FROM query_lexemes ql
    CROSS JOIN stats s
    LEFT JOIN term_df d ON d.lexeme = ql.lexeme
),

candidates AS (
    SELECT p.id AS cand_id, p.tsv AS tsv, p.doc_len AS doc_len
    FROM public.propositions p
    CROSS JOIN query_or
    CROSS JOIN readable_orgs ro
    WHERE p_source = 'propositions'
      AND q_text IS NOT NULL
      AND q_text <> ''
      AND p.org_id = ANY(ro.orgs)
      AND p.namespace = p_namespace
      AND public.pgkg_temporal_visible(
            p.invalidated_at, p.valid_from, p.valid_to,
            COALESCE(p_valid_at, now())
          )
      AND query_or.q IS NOT NULL
      AND p.tsv @@ query_or.q
      AND (
            p_session_id IS NULL
            OR p.session_id = p_session_id
            OR p.session_id IS NULL
          )
      AND public.pgkg_visible(
            p.org_id, p.collection_id, p.visibility,
            p.owner_user_id, p.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )

    UNION ALL

    SELECT c.id, c.tsv, c.doc_len
    FROM public.chunks c
    CROSS JOIN query_or
    CROSS JOIN readable_orgs ro
    WHERE p_source = 'chunks'
      AND q_text IS NOT NULL
      AND q_text <> ''
      AND c.org_id = ANY(ro.orgs)
      AND query_or.q IS NOT NULL
      AND c.tsv @@ query_or.q
      AND c.retrievable
      AND public.pgkg_visible(
            c.org_id, c.collection_id, c.visibility,
            c.owner_user_id, c.acl_group_id,
            p_org_ids, p_collection_ids, p_user_id, p_acl_groups
          )
),

scored AS (
    SELECT
        cd.cand_id,
        SUM(
            i.idf_val
            * (COALESCE(array_length(u.positions, 1), 0)::FLOAT8 * 2.2)
            / (COALESCE(array_length(u.positions, 1), 0)::FLOAT8
               + 1.2 * (1.0 - 0.75 + 0.75 * cd.doc_len::FLOAT8 / s.avgdl))
        ) AS bm25_score
    FROM candidates cd
    CROSS JOIN stats s
    CROSS JOIN LATERAL unnest(cd.tsv) AS u(lexeme, positions, weights)
    JOIN idf i ON i.lexeme = u.lexeme
    GROUP BY cd.cand_id
)

SELECT
    sc.cand_id,
    'kw'::TEXT,
    (ROW_NUMBER() OVER (ORDER BY sc.bm25_score DESC))::INT,
    sc.bm25_score::REAL
FROM scored sc
WHERE sc.bm25_score > 0.0
ORDER BY sc.bm25_score DESC
LIMIT k_initial;
$$;

COMMENT ON FUNCTION pgkg_bm25_candidates_as_owner(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
) IS
    'The keyword arm with its owner''s rights, for a Postgres where the @@ '
    'functions cannot be marked LEAKPROOF. It bypasses row security for its '
    'owner, so it restates the read policy of every table it reads — the '
    'session org from the GUC, and the system org — and must follow any change '
    'to those policies. Selected by pgkg.keyword_arm = owner (migration 059).';

REVOKE ALL ON FUNCTION pgkg_bm25_candidates_as_owner(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
) FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pgkg_app') THEN
        GRANT EXECUTE ON FUNCTION pgkg_bm25_candidates_as_owner(
            TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
        ) TO pgkg_app;
    ELSE
        RAISE WARNING
            'pgkg_app does not exist, so no application role may call '
            'pgkg_bm25_candidates_as_owner(): the policy path is unaffected, and '
            'PGKG_KEYWORD_ARM=owner is refused until the role that selects it '
            'is granted EXECUTE on the function';
    END IF;
END;
$$;


-- 5. The gate the dispatcher reaches the owner arm through.
--
-- Postgres checks EXECUTE on every function in a FROM list when it
-- initialises the plan, including one under a One-Time Filter that will never
-- let it run.  Named directly in the dispatcher, the owner arm would therefore
-- deny every keyword call — on the default path, GUC unset — to any role
-- without the grant: an application role an operator made by hand, which 020
-- supports, a read-only role, or every role of a deployment where pgkg_app did
-- not exist when this ran.  A PL/pgSQL body is planned only when it runs, so
-- behind this gate the owner arm's EXECUTE check happens if and only if the
-- owner branch is taken, and an ungranted role that selects it is refused by
-- name.  The gate is SECURITY INVOKER and PUBLIC may execute it: it confers
-- nothing, since what it calls still checks the caller.
CREATE FUNCTION pgkg_bm25_candidates_owner_gate(
    q_text           TEXT,
    p_namespace      TEXT,
    p_session_id     TEXT,
    k_initial        INT,
    p_org_ids        UUID[],
    p_collection_ids UUID[],
    p_user_id        UUID,
    p_acl_groups     UUID[],
    p_valid_at       TIMESTAMPTZ,
    p_source         TEXT
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT o.item_id, o.kind, o.rank, o.raw_score
    FROM public.pgkg_bm25_candidates_as_owner(
        q_text, p_namespace, p_session_id, k_initial, p_org_ids,
        p_collection_ids, p_user_id, p_acl_groups, p_valid_at, p_source
    ) o;
END;
$$;


-- 6. The name every keyword caller uses, dispatching on the setting.  Unset,
-- or anything but 'owner', is the policy path.  Ordered at the top, because a
-- UNION ALL promises no order and callers read the arm as the single ordered
-- SELECT it was; ordering by rank is the arm's own order, and it still inlines.
CREATE FUNCTION pgkg_bm25_candidates(
    q_text           TEXT,
    p_namespace      TEXT   DEFAULT 'default',
    p_session_id     TEXT   DEFAULT NULL,
    k_initial        INT    DEFAULT 200,
    p_org_ids        UUID[] DEFAULT NULL,
    p_collection_ids UUID[] DEFAULT NULL,
    p_user_id        UUID   DEFAULT NULL,
    p_acl_groups     UUID[] DEFAULT NULL,
    p_valid_at       TIMESTAMPTZ DEFAULT NULL,
    p_source         TEXT   DEFAULT 'propositions'
) RETURNS TABLE (
    item_id   UUID,
    kind      TEXT,
    rank      INT,
    raw_score REAL
)
LANGUAGE SQL STABLE
AS $$
SELECT b.item_id, b.kind, b.rank, b.raw_score
FROM public.pgkg_bm25_candidates_under_policy(
    q_text, p_namespace, p_session_id, k_initial, p_org_ids, p_collection_ids,
    p_user_id, p_acl_groups, p_valid_at, p_source
) b
WHERE pg_catalog.current_setting('pgkg.keyword_arm', TRUE) IS DISTINCT FROM 'owner'

UNION ALL

SELECT o.item_id, o.kind, o.rank, o.raw_score
FROM public.pgkg_bm25_candidates_owner_gate(
    q_text, p_namespace, p_session_id, k_initial, p_org_ids, p_collection_ids,
    p_user_id, p_acl_groups, p_valid_at, p_source
) o
WHERE pg_catalog.current_setting('pgkg.keyword_arm', TRUE) = 'owner'

ORDER BY 3;
$$;

COMMENT ON FUNCTION pgkg_bm25_candidates(
    TEXT, TEXT, TEXT, INT, UUID[], UUID[], UUID, UUID[], TIMESTAMPTZ, TEXT
) IS
    'The keyword arm. The policy path (pgkg_bm25_candidates_under_policy) '
    'unless pgkg.keyword_arm = owner, which selects the owner-rights path for a '
    'Postgres that cannot mark the @@ functions LEAKPROOF (migration 059).';


-- 7. Whether the owner-rights path actually escapes the policy, which is the
-- only thing it is for.
CREATE FUNCTION pgkg_owner_arm_bypasses_policy()
RETURNS BOOLEAN
LANGUAGE SQL STABLE
AS $$
SELECT COALESCE(bool_and(
           r.rolsuper
        OR r.rolbypassrls
        OR (pg_has_role(p.proowner, c.relowner, 'USAGE')
            AND NOT c.relforcerowsecurity)
       ), FALSE)
FROM pg_proc p
JOIN pg_roles r ON r.oid = p.proowner
CROSS JOIN pg_class c
WHERE p.oid = to_regprocedure(
          'public.pgkg_bm25_candidates_as_owner(text, text, text, integer, '
          'uuid[], uuid[], uuid, uuid[], timestamp with time zone, text)')
  AND c.oid IN (
          to_regclass('public.propositions'), to_regclass('public.chunks'),
          to_regclass('public.corpus_stats'), to_regclass('public.lexeme_df'));
$$;

COMMENT ON FUNCTION pgkg_owner_arm_bypasses_policy() IS
    'Whether pgkg_bm25_candidates_as_owner() runs free of row security on every '
    'table it reads: its owner is a superuser or BYPASSRLS, or has the rights '
    'of each table''s owner and the table is not FORCE ROW LEVEL SECURITY. '
    'False means the owner arm is correct and no faster than the policy path.';
