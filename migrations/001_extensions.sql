-- The extensions pgkg depends on.  Created in the schema the runner names for
-- each (issue #30): the one it already lives in, if the database has it, or
-- else PGKG_EXTENSION_SCHEMA (default public).  The function bodies in every
-- later migration name them through the same placeholder, so they resolve
-- whatever the caller's search_path holds.
CREATE EXTENSION IF NOT EXISTS vector SCHEMA @extschema:vector@;
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA @extschema:pg_trgm@;
CREATE EXTENSION IF NOT EXISTS pgcrypto SCHEMA @extschema:pgcrypto@;
