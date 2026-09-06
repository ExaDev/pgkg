-- What the two thresholds in pgkg_link_entity() stage 2 actually decide
-- (issue #23; follows 051, which pinned the trigram one).
--
-- THIS MIGRATION CHANGES NO BEHAVIOUR.  It adds a COMMENT, and it exists
-- because 051's header put a spotlight on `0.6` and a reader will reasonably
-- conclude that 0.6 is what "the same entity" means.  It is not, for most of
-- the population that reaches stage 2.  The finding is recorded here rather
-- than in 051, which is committed and forward-only.
--
-- WHAT WAS OBSERVED.  Two entities coexisted as separate rows in one org and
-- namespace:
--
--     'Helios migration'  and  'Helios migration ships'
--
-- at trigram similarity 0.739, comfortably above the 0.6 that stage 2's
-- candidate generator pins.  The probe database was gone before anyone asked
-- what their embeddings were, so #23 filed it as an observation and not a
-- defect.
--
-- WHAT REPRODUCING IT SHOWED.  Stage 2 is an AND:
--
--     name % p_name                                   -- pinned at 0.6
--     AND similarity(name, p_name) > 0.6              -- 051's confirmation
--     AND (1 - (embedding <=> p_embedding)) > 0.85    -- p_threshold's default
--
-- Measured with the shipped embedder (BAAI/bge-m3, L2-normalised, the default
-- `embed_model`) over the bare entity names, and with pg_trgm's similarity()
-- on the same strings:
--
--     pair                                        trgm     cos    outcome
--     'Helios migration' / '... ships'           0.7391  0.8069   two rows
--     'William Shakespeare' / '... Shakespear'   0.8571  0.8198   two rows
--     'New York City' / 'New York Citty'         0.8125  0.6786   two rows
--     'Acme Corp' / 'Acme Corporation'           0.5000  0.9589   two rows
--     'Helios' / 'Helios project'                0.4667  0.8602   two rows
--     'Postgres' / 'PostgreSQL'                  0.6667  0.8933   ONE row
--     'Helios migration' / 'Helios migrations'   0.8421  0.9659   ONE row
--
-- So the observation is exactly what the code says, and 0.8069 is the number
-- that decided it.  It is a consequence of the AND, not a defect: the trigram
-- predicate offered the pair as a candidate and the cosine rejected it.
--
-- WHICH ONE GOVERNS.  Neither, always — but not symmetrically.  On a labelled
-- probe of 40 pairs (20 the same entity under a different surface form, 20
-- genuinely different entities with similar names):
--
--     trigram > 0.6 alone   merges 18/20 same,  5/20 different
--     cosine  > 0.85 alone  merges 15/20 same,  2/20 different
--     the AND, as shipped   merges 13/20 same,  1/20 different
--
-- Of the 12 pairs where the two predicates disagree, the cosine is the one
-- that rejects in 9.  Among same-entity pairs — the population entity
-- resolution exists to merge, and the population that reaches stage 2 at all,
-- since anything spelled identically was already answered by stage 1 — the
-- cosine rejects 5 and the trigram 2.  #23's claim is therefore right about
-- the near-spelling case and wrong as a universal: 'Acme Corp' /
-- 'Acme Corporation' is split by the trigram at 0.5000 while its cosine is
-- 0.9589, so tuning either number alone moves dedup.  A test that asserted
-- "no two entities in one org are within 0.6 trigrams of each other" would be
-- asserting something false; tests/test_entity_linking.py pins the truth
-- table instead.
--
-- WHY 0.85 IS NOT MOVED HERE, THOUGH IT IS NOT A GOOD THRESHOLD.  An entity
-- name is a two- or three-word string, and cosine similarity over strings that
-- short does not separate the two populations at any cutoff.  On the probe set
-- the same-entity cosines span 0.6786 to 0.9883 and the different-entity ones
-- span 0.3989 to 0.8889 — overlapping across the whole region a threshold
-- would have to sit in.  0.85 splits 'William Shakespeare' from
-- 'William Shakespear', which is the canonical near-duplicate this function
-- exists for, and 'New York City' from 'New York Citty', which is one doubled
-- letter; the best single cutoff available on the set is 0.859, worth 75%
-- recall at 5% false merges, which is not a threshold problem that a better
-- threshold fixes.
--
-- What makes 0.85 defensible anyway is the direction it fails in.  A missed
-- merge leaves two nodes that a later pass, a gazetteer sweep or an operator
-- can still join; a wrong merge fuses two entities' propositions into one row
-- and there is nothing afterwards to say which edges came from which.  The AND
-- of two conservative predicates fails toward splitting, and that is the same
-- asymmetry 051's header invokes for the pin.  'version 1.2.0' /
-- 'version 2.1.0' — trigram 1.0000, cosine 0.8889, two different entities —
-- is what a looser gate buys, and it is already through.
--
-- The real fix is not a number: it is that the vector being compared is an
-- embedding of the bare name.  Embedding the name with its type, or with the
-- mention context it was extracted from, is what would make a cosine
-- meaningful at this length.  That changes how the whole graph merges and is
-- not a change to make from one observation, so it is proposed and not taken.

COMMENT ON FUNCTION pgkg_link_entity(TEXT, TEXT, TEXT, halfvec, REAL) IS
$c$Idempotent entity resolution, per org and namespace.

Stage 1 is exact (name, type).  Stage 2 is an AND of a trigram predicate
pinned at 0.6 (the candidate generator, 051) and a name-embedding cosine
above p_threshold, default 0.85.  Both must hold, so 0.6 is not what "the
same entity" means: for near-identical spellings — the population that
reaches stage 2 at all — the cosine is usually what decides, and it splits
pairs the trigram accepts ('Helios migration' / 'Helios migration ships':
trigram 0.739, cosine 0.807, two rows).  The trigram also decides on its own
in the other direction ('Acme Corp' / 'Acme Corporation': trigram 0.500,
cosine 0.959, two rows).  Stage 2 therefore merges strictly less than either
threshold read alone suggests, deliberately: it fails toward two rows, which
is recoverable, rather than toward a fused entity, which is not.  See
migration 055 for the measurements.$c$;
