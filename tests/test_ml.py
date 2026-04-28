"""Tests for pgkg/ml.py — all offline, no API keys or GPU needed."""
from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# test_mmr_picks_diverse
# ---------------------------------------------------------------------------

def test_mmr_picks_diverse():
    """MMR should prefer diverse vectors over redundant ones."""
    from pgkg.ml import mmr

    # query points along dim 0; doc 1 is nearly identical to doc 0;
    # doc 2 has moderate similarity to query but is diverse from doc 0.
    query = [1.0, 0.0, 0.0]
    doc_embs = [
        [1.0, 0.0, 0.0],   # 0 — identical to query, picked first
        [1.0, 0.01, 0.0],  # 1 — almost identical to doc 0, high redundancy
        [0.6, 0.8, 0.0],   # 2 — less similar to query but diverse
    ]

    # With lambda=0.5, doc 2 gains more from diversity than doc 1
    # For doc 1 after selecting doc 0:
    #   rel  = 0.5 * cos(doc1, query) ≈ 0.5 * 1.0    = 0.5
    #   red  = 0.5 * cos(doc1, doc0)  ≈ 0.5 * 1.0    = 0.5  → score ≈ 0
    # For doc 2 after selecting doc 0:
    #   rel  = 0.5 * cos(doc2, query) = 0.5 * 0.6    = 0.3
    #   red  = 0.5 * cos(doc2, doc0)  = 0.5 * 0.6    = 0.3  → score = 0
    # Both tie at 0 because doc 2 has cosine = 0.6 both ways.
    # Use lambda_=0.7 to weight relevance more, breaking the tie in favour of
    # still picking the more diverse doc when scores differ.
    # Better: use a doc 2 that is truly orthogonal but give it query sim via dim 1.
    query = [1.0, 0.0, 0.0]
    doc_embs = [
        [1.0, 0.0, 0.0],   # 0 — identical to query
        [0.98, 0.2, 0.0],  # 1 — very close to doc 0
        [0.5, 0.0, 0.866], # 2 — different dimension; cos with query = 0.5, cos with doc0 = 0.5
    ]
    # After doc 0 selected:
    # doc1: rel = 0.5*cos(1,q) ≈ 0.5*0.98 = 0.49,  red = 0.5*cos(1,0) ≈ 0.5*0.98 = 0.49, score≈0
    # doc2: rel = 0.5*0.5 = 0.25,                  red = 0.5*0.5 = 0.25,             score=0
    # Still a tie! Use lambda < 0.5 to penalise redundancy more.
    selected = mmr(query, doc_embs, k=2, lambda_=0.3)
    # With lambda=0.3:
    # doc1: rel=0.3*0.98=0.294, red=0.7*0.98=0.686, score≈-0.39
    # doc2: rel=0.3*0.5 =0.15,  red=0.7*0.5 =0.35,  score≈-0.20
    # doc2 wins (less negative)
    assert len(selected) == 2
    assert selected[0] == 0
    assert selected[1] == 2


def test_mmr_empty_docs():
    from pgkg.ml import mmr
    assert mmr([1.0, 0.0], [], k=5) == []


def test_mmr_k_larger_than_docs():
    from pgkg.ml import mmr
    docs = [[1.0, 0.0], [0.0, 1.0]]
    result = mmr([1.0, 0.0], docs, k=10)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# test_extract_propositions_offline_mode
# ---------------------------------------------------------------------------

def test_extract_propositions_offline_mode(monkeypatch):
    """With PGKG_OFFLINE_EXTRACT=1 extract_propositions returns deterministic fallback."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    from pgkg.ml import extract_propositions
    props = extract_propositions("The sky is blue and the grass is green.")

    assert len(props) == 1
    p = props[0]
    assert p.subject == "?"
    assert p.predicate == "states"
    assert p.object_is_literal is True
    assert "The sky is blue" in p.object


# ---------------------------------------------------------------------------
# test_embed_returns_normalized
# ---------------------------------------------------------------------------

def test_embed_returns_normalized(monkeypatch):
    """embed() should return L2-normalized vectors (norm ≈ 1.0)."""
    import pgkg.ml as ml_module

    class FakeST:
        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True):
            # Return raw un-normalized vectors; ml.embed must normalize
            arr = np.array([[3.0, 4.0]] * len(texts))
            return arr

    monkeypatch.setattr(ml_module, "_embed_model", FakeST())

    vecs = ml_module.embed(["hello", "world"])
    assert len(vecs) == 2
    for v in vecs:
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm≈1, got {norm}"


def test_embed_empty():
    from pgkg.ml import embed
    assert embed([]) == []


# ---------------------------------------------------------------------------
# test_rerank_orders
# ---------------------------------------------------------------------------

def test_rerank_orders(monkeypatch):
    """rerank() should return scores in document order, not sorted."""
    import pgkg.ml as ml_module

    raw_scores = [0.9, 0.1, 0.5]

    class FakeCE:
        def predict(self, pairs):
            return raw_scores

    monkeypatch.setattr(ml_module, "_rerank_model", FakeCE())

    docs = ["doc A", "doc B", "doc C"]
    scores = ml_module.rerank("query", docs)

    assert len(scores) == 3
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.1)
    assert scores[2] == pytest.approx(0.5)


def test_rerank_empty(monkeypatch):
    from pgkg.ml import rerank
    assert rerank("q", []) == []
