-- Migration 005: Proposition extraction cache
-- Avoids redundant LLM calls when the same chunk is re-ingested with the same
-- extractor model and prompt version.

CREATE TABLE IF NOT EXISTS proposition_cache (
  cache_key       TEXT PRIMARY KEY,
  chunk_hash      TEXT NOT NULL,
  extractor_model TEXT NOT NULL,
  prompt_version  TEXT NOT NULL,
  propositions    JSONB NOT NULL,           -- list of {text, subject, predicate, object, object_is_literal}
  created_at      TIMESTAMPTZ DEFAULT now(),
  hit_count       INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS prop_cache_model_idx
    ON proposition_cache (extractor_model, prompt_version);
