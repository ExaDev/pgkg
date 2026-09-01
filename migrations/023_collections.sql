-- Collections: the row behind collection_id, and the sharing seam.
--
-- 020 put collection_id on every retrievable row and pointed it at a reserved
-- constant, deliberately leaving the table for later.  Later is now: a UUID
-- with no table behind it is the stringly-typed namespace again wearing a
-- different type, and D3 rejected that string for having "no referential
-- integrity, no indexable way to express 'this user's private facts plus the
-- org's shared corpus', nothing for the planner to prune on".  The first of
-- those is a foreign key, and this migration adds it.
--
-- The collection is where a decay profile and a claim scope belong (D6).  One
-- constant cannot serve chat, policy and vendor documentation, and the choice
-- is a property of the material rather than of the row: every proposition
-- extracted from the handbook decays the same way.  Only the columns land here.
-- Reading them in pgkg_apply_profile() is phase 2, which is where the ADR puts
-- the three profiles.
--
-- The subscription seam ships now and stays empty.  It sits in the hot-path
-- predicate (D4), so the shape has to exist before anything is built on top of
-- it; resolving subscriptions *inside* the retrieval predicate is phase 2.
-- rrf_weight is what lets a tenant turn shared material down, or off, without
-- a rebuild.
--
-- BACKFILL.  The default collection is seeded with the id 020's column default
-- already wrote into every pre-existing row, so the foreign key validates
-- without touching a single row and an un-scoped caller keeps working.

CREATE TABLE collections (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    owner_org_id  UUID NOT NULL REFERENCES orgs(id),
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'chat',
    visibility    TEXT NOT NULL DEFAULT 'private',
    claim_scope   TEXT NOT NULL DEFAULT 'org',
    decay_profile TEXT NOT NULL DEFAULT 'conversational',
    acl_mode      TEXT NOT NULL DEFAULT 'none',
    extract_propositions BOOLEAN NOT NULL DEFAULT FALSE,
    public_source BOOLEAN NOT NULL DEFAULT FALSE,
    licence       TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    CONSTRAINT collections_kind_check
        CHECK (kind IN ('chat', 'corpus', 'mixed')),
    CONSTRAINT collections_visibility_check
        CHECK (visibility IN ('private', 'shared')),
    CONSTRAINT collections_claim_scope_check
        CHECK (claim_scope IN ('world', 'org', 'user')),
    CONSTRAINT collections_decay_profile_check
        CHECK (decay_profile IN ('conversational', 'timeless', 'perishable')),
    CONSTRAINT collections_acl_mode_check
        CHECK (acl_mode IN ('none', 'group')),
    -- Only the operator publishes.  Nothing a tenant ingests is ever promoted
    -- into a shared collection, so a tenant-owned collection cannot be shared.
    CONSTRAINT collections_only_system_org_shares
        CHECK (visibility = 'private' OR owner_org_id = pgkg_system_org()),
    -- The content-hash embedding cache is probe-able, so it is restricted to
    -- crawled or licensed public material: for a confidential document a cache
    -- hit would confirm another tenant holds it (D4).
    CONSTRAINT collections_public_source_is_operator_owned
        CHECK (NOT public_source OR owner_org_id = pgkg_system_org())
);

COMMENT ON TABLE collections IS
    'Carries decay profile, claim scope and ACL mode for every row that names '
    'it. Nothing a tenant ingests is ever promoted into a shared collection: '
    'not automatically, not by heuristic, not as a "contribute back" default. '
    'Shared collections are populated by the operator from operator-licensed '
    'sources (ADR 0001, D4).';

COMMENT ON COLUMN collections.decay_profile IS
    'conversational keys on asserted_at with a ~30 day half-life and the '
    'frequency boost; timeless is a flat factor of 1.0; perishable keys on the '
    'publication date with a 12-36 month half-life. The frequency boost is off '
    'for both corpus profiles: on reference material it is a popularity '
    'feedback loop, and on shared material it carries usage across tenants '
    '(ADR 0001, D6).';

COMMENT ON COLUMN collections.extract_propositions IS
    'Off by default: the corpus is not proposition-extracted by default (D2).';

INSERT INTO collections (id, org_id, owner_org_id, name, kind)
VALUES (
    pgkg_default_collection(), pgkg_default_org(), pgkg_default_org(),
    'default', 'mixed'
);


-- The seam, empty by default: nothing is subscribed implicitly.  Pruning still
-- works — two partitions when subscribed, one when not, which is the default.
CREATE TABLE collection_subscriptions (
    org_id        UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    rrf_weight    REAL NOT NULL DEFAULT 1.0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, collection_id),
    CONSTRAINT collection_subscriptions_weight_check
        CHECK (rrf_weight >= 0.0)
);

COMMENT ON TABLE collection_subscriptions IS
    'Empty by default; nothing is subscribed implicitly. Resolution into the '
    'retrieval predicate is phase 2 (ADR 0001, D4).';

CREATE INDEX collection_subscriptions_collection_idx
    ON collection_subscriptions (collection_id);


-- The foreign keys 020 could not add.  Every existing row already names the
-- seeded default collection, so each of these validates without a rewrite.
ALTER TABLE propositions
    ADD CONSTRAINT propositions_collection_id_fkey
        FOREIGN KEY (collection_id) REFERENCES collections(id);
ALTER TABLE chunks
    ADD CONSTRAINT chunks_collection_id_fkey
        FOREIGN KEY (collection_id) REFERENCES collections(id);
ALTER TABLE documents
    ADD CONSTRAINT documents_collection_id_fkey
        FOREIGN KEY (collection_id) REFERENCES collections(id);


-- Reads widen to the operator's shared collections so a subscriber can resolve
-- one; writes never widen, which is the schema half of "only the operator
-- publishes".
ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE collection_subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY collections_org_isolation ON collections
    USING (org_id = pgkg_current_org() OR org_id = pgkg_system_org())
    WITH CHECK (org_id = pgkg_current_org());

CREATE POLICY collection_subscriptions_org_isolation ON collection_subscriptions
    USING (org_id = pgkg_current_org())
    WITH CHECK (org_id = pgkg_current_org());


-- GRANT ... ON ALL TABLES in 020 was a snapshot of the tables that existed
-- then, not a standing rule, so every table created since has to be granted
-- again.  022's registry was missed: pgkg_app — the role a deployment is told
-- to connect as, and the role org_embedders' policy is written for — could not
-- read the embedder registry at all, so resolving the live generations on the
-- recall path failed with permission denied rather than with a wrong answer.
DO $$
BEGIN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON '
            'collections, collection_subscriptions, '
            'embedder_generations, org_embedders TO pgkg_app';
EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
    RAISE NOTICE 'pgkg_app not granted on the collection and registry tables (%)',
                 SQLERRM;
END;
$$;
