"""Tests for EmbeddingProvider."""

import numpy as np
import pytest

from kimix.memory.embedding import EmbeddingProvider


class TestEmbeddingProvider:
    def test_embed_dimensions(self):
        provider = EmbeddingProvider(dim=384)
        vec = provider.embed("hello world")
        assert len(vec) == 384

    def test_embed_consistency(self):
        provider = EmbeddingProvider(dim=384)
        vec1 = provider.embed("test text")
        vec2 = provider.embed("test text")
        assert np.array_equal(vec1, vec2)  # Cached / deterministic

    def test_embed_different_texts(self):
        provider = EmbeddingProvider(dim=384)
        vec1 = provider.embed("text one")
        vec2 = provider.embed("text two")
        assert not np.array_equal(vec1, vec2)

    def test_similarity_same_vector(self):
        provider = EmbeddingProvider(dim=384)
        vec = provider.embed("same")
        sim = provider.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_similarity_range(self):
        provider = EmbeddingProvider(dim=384)
        vec1 = provider.embed("hello world")
        vec2 = provider.embed("goodbye world")
        sim = provider.similarity(vec1, vec2)
        assert -1.0 <= sim <= 1.0

    def test_similarity_orthogonal(self):
        provider = EmbeddingProvider(dim=2)
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = provider.similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_similarity_zero_norm(self):
        provider = EmbeddingProvider(dim=2)
        sim = provider.similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0


