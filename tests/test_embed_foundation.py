"""Tests for the embedding foundation (paperscope/embed/).

All nine semantic analyses sit on ``cosine_sim`` + ``embed_texts``, so this
file pins down:

- cosine-similarity correctness on tiny known vectors (shape, known angles,
  scale invariance, zero-vector safety);
- ``embed_texts`` edge cases (empty corpus, single chunk);
- the TF-IDF fallback path — the advertised "runs anywhere" design decision —
  forced by making the sentence-transformers loader unavailable, and checked
  to produce usable similarity *rankings* on a synthetic corpus with an
  obvious nearest neighbour.

No test here touches the network or loads a real sentence-transformers model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paperscope.embed import cosine_sim, embed_texts
from paperscope.embed import embed_claims


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------


class TestCosineSim:
    def test_identical_vectors_score_one(self):
        v = np.array([[3.0, 4.0]])
        assert cosine_sim(v, v)[0, 0] == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert cosine_sim(a, b)[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[-1.0, -2.0]])
        assert cosine_sim(a, b)[0, 0] == pytest.approx(-1.0)

    def test_known_45_degree_value(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        assert cosine_sim(a, b)[0, 0] == pytest.approx(1.0 / math.sqrt(2.0))

    def test_returns_m_by_n_matrix(self):
        a = np.ones((3, 4))
        b = np.ones((2, 4))
        assert cosine_sim(a, b).shape == (3, 2)

    def test_scale_invariance(self):
        a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
        b = np.array([[2.0, 0.0, 1.0]])
        assert np.allclose(cosine_sim(7.0 * a, 0.3 * b), cosine_sim(a, b))

    def test_zero_vector_yields_zero_not_nan(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        out = cosine_sim(a, b)
        assert np.all(np.isfinite(out))
        assert out[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual_computation(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(4, 6))
        b = rng.normal(size=(3, 6))
        expected = np.empty((4, 3))
        for i in range(4):
            for j in range(3):
                expected[i, j] = np.dot(a[i], b[j]) / (
                    np.linalg.norm(a[i]) * np.linalg.norm(b[j])
                )
        assert np.allclose(cosine_sim(a, b), expected, atol=1e-9)

    def test_values_bounded_in_unit_interval(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(5, 8))
        b = rng.normal(size=(6, 8))
        out = cosine_sim(a, b)
        assert np.all(out <= 1.0 + 1e-9)
        assert np.all(out >= -1.0 - 1e-9)


# ---------------------------------------------------------------------------
# embed_texts: edge cases
# ---------------------------------------------------------------------------


class TestEmbedTextsEdges:
    def test_empty_corpus_returns_empty_matrix(self):
        emb, info = embed_texts([])
        assert emb.shape == (0, 1)
        assert info["backend"] == "empty"

    def test_empty_tuple_treated_like_empty_list(self):
        emb, info = embed_texts(())
        assert emb.shape == (0, 1)
        assert info["backend"] == "empty"


# ---------------------------------------------------------------------------
# embed_texts: primary (sentence-transformers) path via an injected model
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stands in for a SentenceTransformer; returns deterministic vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.encoded = []

    def encode(self, texts, show_progress_bar=False, batch_size=64):
        self.encoded.append(list(texts))
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), self.dim))


class _ExplodingModel:
    def encode(self, texts, show_progress_bar=False, batch_size=64):
        raise RuntimeError("model backend exploded (forced by test)")


class TestEmbedTextsWithProvidedModel:
    def test_uses_injected_model_and_reports_backend(self):
        model = _FakeModel(dim=8)
        texts = ["alpha", "beta", "gamma"]
        emb, info = embed_texts(texts, model=model, show_progress=False)
        assert emb.shape == (3, 8)
        assert info["backend"] == "sentence-transformers"
        assert info["n_items"] == 3
        assert info["dim"] == 8
        assert model.encoded == [texts]

    def test_empty_corpus_short_circuits_before_model(self):
        model = _FakeModel()
        emb, info = embed_texts([], model=model)
        assert info["backend"] == "empty"
        assert model.encoded == []


# ---------------------------------------------------------------------------
# embed_texts: TF-IDF fallback (the "runs anywhere" design decision)
# ---------------------------------------------------------------------------


@pytest.fixture
def force_tfidf_fallback(monkeypatch):
    """Simulate an environment without sentence-transformers installed."""

    def _unavailable(*args, **kwargs):
        raise ImportError("forced by test: sentence-transformers unavailable")

    monkeypatch.setattr(embed_claims, "load_model", _unavailable)


class TestTfidfFallback:
    def test_fallback_engages_and_reports_metadata(self, force_tfidf_fallback):
        texts = ["cats purr softly", "markets rallied sharply"]
        emb, info = embed_texts(texts, show_progress=False)
        assert info["backend"] == "tfidf"
        assert info["n_items"] == 2
        assert info["dim"] == info["vocabulary_size"]
        assert "sentence-transformers unavailable" in info["fallback_reason"]
        assert emb.shape == (2, info["dim"])

    def test_single_chunk_corpus(self, force_tfidf_fallback):
        emb, info = embed_texts(["just one lonely chunk"], show_progress=False)
        assert info["backend"] == "tfidf"
        assert emb.shape == (1, info["dim"])
        assert cosine_sim(emb, emb)[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_fallback_produces_usable_nearest_neighbour_ranking(
        self, force_tfidf_fallback
    ):
        # Query + corpus must be embedded in ONE call: the TF-IDF vectorizer
        # is fitted per embed_texts() call, so vectors from separate calls
        # are not comparable.
        query = "feline cats purr kittens"
        corpus = [
            "cats and kittens purr softly feline companions",  # obvious NN
            "stock markets rallied while inflation eased sharply",
            "quantum entanglement violates classical locality bounds",
            "volcanic eruptions reshape remote oceanic islands",
        ]
        emb, info = embed_texts([query] + corpus, show_progress=False)
        assert info["backend"] == "tfidf"
        sims = cosine_sim(emb[:1], emb[1:])[0]
        assert int(np.argmax(sims)) == 0
        assert sims[0] > 0.2  # clearly related
        # Vocabulary-disjoint documents score (essentially) zero.
        assert np.all(sims[1:] < 0.05)
        # And the margin over the runner-up is decisive, not marginal.
        assert sims[0] - float(np.max(sims[1:])) > 0.15

    def test_model_encode_failure_also_falls_back(self):
        # The fallback must catch runtime encode failures too, not just a
        # missing package: `except Exception` is the documented contract.
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        emb, info = embed_texts(texts, model=_ExplodingModel(), show_progress=False)
        assert info["backend"] == "tfidf"
        assert "exploded" in info["fallback_reason"]
        assert emb.shape[0] == 2
