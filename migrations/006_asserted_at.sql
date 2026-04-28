-- Add nullable assertion timestamp to propositions and chunks.
-- Populated when the source has a real timestamp (chat turn, email, commit);
-- null otherwise (README, pasted text, code comments).

ALTER TABLE propositions
    ADD COLUMN IF NOT EXISTS asserted_at TIMESTAMPTZ;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS asserted_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS prop_asserted_at_idx ON propositions (asserted_at)
    WHERE asserted_at IS NOT NULL;
