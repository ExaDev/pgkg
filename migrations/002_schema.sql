CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source      TEXT,
    namespace   TEXT NOT NULL DEFAULT 'default',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    span_start  INT,
    span_end    INT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS entities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    type        TEXT,
    embedding   vector(1024),
    aliases     TEXT[] DEFAULT '{}',
    namespace   TEXT NOT NULL DEFAULT 'default',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(namespace, name, type)
);

CREATE INDEX IF NOT EXISTS entities_embedding_idx ON entities
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS entities_name_trgm_idx ON entities
    USING gin (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS entities_aliases_idx ON entities
    USING gin (aliases);

CREATE TABLE IF NOT EXISTS propositions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text             TEXT NOT NULL,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding        vector(1024),
    subject_id       UUID REFERENCES entities(id),
    predicate        TEXT,
    object_id        UUID REFERENCES entities(id),
    object_literal   TEXT,
    chunk_id         UUID REFERENCES chunks(id) ON DELETE CASCADE,
    namespace        TEXT NOT NULL DEFAULT 'default',
    session_id       TEXT,
    confidence       REAL DEFAULT 1.0,
    created_at       TIMESTAMPTZ DEFAULT now(),
    last_accessed_at TIMESTAMPTZ DEFAULT now(),
    access_count     INT DEFAULT 0,
    superseded_by    UUID REFERENCES propositions(id),
    metadata         JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS prop_tsv_idx ON propositions USING gin (tsv);
CREATE INDEX IF NOT EXISTS prop_emb_idx ON propositions USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS prop_ns_session_idx ON propositions (namespace, session_id);
CREATE INDEX IF NOT EXISTS prop_subject_idx ON propositions (subject_id);
CREATE INDEX IF NOT EXISTS prop_object_idx ON propositions (object_id);
CREATE INDEX IF NOT EXISTS prop_active_idx ON propositions (namespace)
    WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS edges (
    src_entity     UUID NOT NULL REFERENCES entities(id),
    dst_entity     UUID NOT NULL REFERENCES entities(id),
    relation       TEXT NOT NULL,
    weight         REAL DEFAULT 1.0,
    proposition_id UUID NOT NULL REFERENCES propositions(id) ON DELETE CASCADE,
    PRIMARY KEY (src_entity, dst_entity, relation, proposition_id)
);

CREATE INDEX IF NOT EXISTS edges_src_idx ON edges (src_entity);
CREATE INDEX IF NOT EXISTS edges_dst_idx ON edges (dst_entity);

CREATE TABLE IF NOT EXISTS entity_pagerank (
    entity_id   UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    score       REAL NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT now()
);
