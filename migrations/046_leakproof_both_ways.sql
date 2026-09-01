-- The keyword match, leakproof in both operand orders and visible to an
-- operator (ADR 0001, D1, D3).
--
-- WHY THE REVERSED OPERAND ORDER IS MARKED TOO.  043 marked
-- ts_match_vq(tsvector, tsquery) because a qual whose function is not
-- leakproof may not become an index condition on a table with a policy, and
-- without the mark every BM25 arm degraded to a sequential scan under the one
-- role the policies are written for.  `@@` is two functions, not one:
-- `tsquery @@ tsvector` is ts_match_qv, and the planner asks about
-- leakproofness of the clause as written, before it considers commuting it
-- through the operator's commutator.  So the same query with the operands the
-- other way round still cannot reach the GIN index.  Measured here as pgkg_app
-- on 4,003 chunks with `to_tsquery('simple','unobtainium') @@ c.tsv`: a Seq
-- Scan removing 4,000 rows by filter, which is 043's defect reached by a
-- different keystroke.
--
-- The issue this closes offered the alternative of asserting that no retrieval
-- function writes the reversed form.  The operator is marked instead, because
-- the guard belongs on the thing that is wrong rather than on the style of the
-- SQL that reaches it: a text search for `@@` cannot see a query built in
-- application Python, in a bench script, or in the next migration written by
-- someone who never read this one, and a test that passes because nobody has
-- yet typed the operands in the losing order is a test of the codebase's
-- current habits, not of the database's behaviour.  Marking costs one
-- statement and makes both forms correct by construction.
--
-- The leakproofness claim is the same claim 043 argues for the forward form,
-- and it is the same function underneath: ts_match_qv swaps its arguments and
-- performs the identical match.  It raises no data-dependent error, emits no
-- message carrying either argument, and has no side effect.  It needs
-- superuser, so it degrades to a NOTICE exactly as 043 does — a deployment
-- that cannot mark it keeps a correct, slow keyword arm.
--
-- WHY THE STATE IS READABLE AS A FUNCTION.  043's NOTICE is the whole record
-- that the mark was not applied, and a NOTICE emitted during a migration is
-- gone by the time anyone asks why retrieval is slow.  The symptom is an
-- order-of-magnitude regression that appears only at scale and only under the
-- application role, so a deployment that missed the fix is indistinguishable
-- from one that got it — including to the test suite, which migrates as a
-- superuser and can therefore only ever observe the marked state.  The fix
-- cannot be enforced, because a managed Postgres will not hand over ownership
-- of a built-in function; what it can be is a fact something monitors.  So the
-- state becomes a query, which /health reports next to the embedder registry.
--
-- The signatures live here rather than in Python because this is the file that
-- makes the claim about them: the list an operator monitors and the list the
-- migrations mark are the same list, and a second copy in the API layer could
-- only ever fall out of step with this one.  EXECUTE is PUBLIC by default and
-- stays that way — pg_proc is world-readable, so the function tells a caller
-- nothing it could not already select, and the health endpoint connects as
-- pgkg_app.


-- 1. The other half of `@@`.
DO $$
BEGIN
    EXECUTE 'ALTER FUNCTION ts_match_qv(tsquery, tsvector) LEAKPROOF';
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE
        'ts_match_qv not marked leakproof (%); a keyword qual written with the '
        'tsquery on the left stays correct but cannot reach the GIN index '
        'under a role with row security', SQLERRM;
END;
$$;


-- 2. What was actually marked, for whoever has to answer for the plan.
CREATE FUNCTION pgkg_keyword_match_leakproof()
RETURNS TABLE (signature TEXT, leakproof BOOLEAN)
LANGUAGE SQL STABLE
AS $$
SELECT
    s.sig,
    (SELECT p.proleakproof FROM pg_proc p WHERE p.oid = to_regprocedure(s.sig))
FROM unnest(ARRAY[
    'ts_match_vq(tsvector,tsquery)',
    'ts_match_qv(tsquery,tsvector)'
]) AS s(sig);
$$;

COMMENT ON FUNCTION pgkg_keyword_match_leakproof() IS
    'The leakproof state of both functions behind `tsvector @@ tsquery`, which '
    'is what decides whether the keyword arms can reach the GIN index under a '
    'role with row security. NULL for a signature this server does not have. '
    'False anywhere means the arms are correct and grow linearly with the '
    'table (ADR 0001, D1, D3; migrations 043 and 046).';
