"""Performance benchmarks for EmbeddingProvider.

All timings are assert-based so the file doubles as a regression test.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from kimix.memory.embedding import EmbeddingProvider

# -----------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _lorem(words: int) -> str:
    """Return a deterministic pseudo-text of *words* length."""
    pool = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim "
        "ad minim veniam quis nostrud exercitation ullamco laboris nisi ut "
        "aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit "
        "in voluptate velit esse cillum dolore eu fugiat nulla pariatur "
        "excepteur sint occaecat cupidatat non proident sunt in culpa qui "
        "officia deserunt mollit anim id est laborum"
    ).split()
    return " ".join(pool[i % len(pool)] for i in range(words))


# -----------------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------------------

class TestInitBenchmark:
    def test_init_default(self) -> None:
        start = time.perf_counter()
        for _ in range(10_000):
            EmbeddingProvider()
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_init_custom_dim(self) -> None:
        start = time.perf_counter()
        for _ in range(10_000):
            EmbeddingProvider(dim=768, max_cache_size=8192)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0


# -----------------------------------------------------------------------------
# _normalize_key
# ------------------------------------------------------------------------------

class TestNormalizeKeyBenchmark:
    def test_normalize_short(self) -> None:
        provider = EmbeddingProvider()
        start = time.perf_counter()
        for _ in range(100_000):
            provider._normalize_key("hello world")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_normalize_long(self) -> None:
        provider = EmbeddingProvider()
        text = _lorem(500)
        start = time.perf_counter()
        for _ in range(50_000):
            provider._normalize_key(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0


# -----------------------------------------------------------------------------
# _hash_token
# ------------------------------------------------------------------------------

class TestHashTokenBenchmark:
    def test_hash_short(self) -> None:
        provider = EmbeddingProvider()
        start = time.perf_counter()
        for _ in range(100_000):
            provider._hash_token("hello")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_hash_long(self) -> None:
        provider = EmbeddingProvider()
        token = "a" * 1_000
        start = time.perf_counter()
        for _ in range(50_000):
            provider._hash_token(token)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0

    def test_hash_utf8(self) -> None:
        provider = EmbeddingProvider()
        token = "你好世界" * 50
        start = time.perf_counter()
        for _ in range(50_000):
            provider._hash_token(token)
        elapsed = time.perf_counter() - start
        assert elapsed < 4.0


# -----------------------------------------------------------------------------
# _compute
# ------------------------------------------------------------------------------

class TestComputeBenchmark:
    def test_compute_empty(self) -> None:
        provider = EmbeddingProvider(dim=384)
        start = time.perf_counter()
        for _ in range(10_000):
            provider._compute("")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_compute_short(self) -> None:
        provider = EmbeddingProvider(dim=384)
        start = time.perf_counter()
        for _ in range(10_000):
            provider._compute("hello world")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_compute_medium(self) -> None:
        provider = EmbeddingProvider(dim=384)
        text = _lorem(100)
        start = time.perf_counter()
        for _ in range(5_000):
            provider._compute(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_compute_long(self) -> None:
        provider = EmbeddingProvider(dim=384)
        text = _lorem(1_000)
        start = time.perf_counter()
        for _ in range(1_000):
            provider._compute(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0

    def test_compute_various_dims(self) -> None:
        for dim in (64, 128, 384, 768):
            provider = EmbeddingProvider(dim=dim)
            start = time.perf_counter()
            for _ in range(5_000):
                provider._compute("benchmark text with dimension " + str(dim))
            elapsed = time.perf_counter() - start
            assert elapsed < 2.0


# -----------------------------------------------------------------------------
# embed (single)
# ------------------------------------------------------------------------------

class TestEmbedBenchmark:
    def test_embed_short_cold(self) -> None:
        provider = EmbeddingProvider(dim=384)
        start = time.perf_counter()
        for i in range(1_000):
            provider.embed(f"unique text {i}")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_embed_short_hot(self) -> None:
        provider = EmbeddingProvider(dim=384)
        text = "hot cached text"
        provider.embed(text)  # warm
        start = time.perf_counter()
        for _ in range(50_000):
            provider.embed(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_embed_medium_cold(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = [_lorem(100) + f" {i}" for i in range(500)]
        start = time.perf_counter()
        for t in texts:
            provider.embed(t)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0

    def test_embed_long_cold(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = [_lorem(1_000) + f" {i}" for i in range(100)]
        start = time.perf_counter()
        for t in texts:
            provider.embed(t)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0


# -----------------------------------------------------------------------------
# embed_batch
# ------------------------------------------------------------------------------

class TestEmbedBatchBenchmark:
    def test_batch_small_all_miss(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = [f"text {i}" for i in range(100)]
        start = time.perf_counter()
        provider.embed_batch(texts)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_batch_small_all_hit(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = ["hit"] * 1_000
        provider.embed_batch(texts[:1])  # warm cache
        start = time.perf_counter()
        provider.embed_batch(texts)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_batch_mixed_hit_miss(self) -> None:
        provider = EmbeddingProvider(dim=384)
        for i in range(500):
            provider.embed(f"cached {i}")
        texts = [f"cached {i}" for i in range(250)] + [f"new {i}" for i in range(250)]
        start = time.perf_counter()
        provider.embed_batch(texts)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_batch_large(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = [f"large batch item {i}" for i in range(5_000)]
        start = time.perf_counter()
        provider.embed_batch(texts)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0

    def test_batch_duplicate_heavy(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = ["dup"] * 10_000
        start = time.perf_counter()
        provider.embed_batch(texts)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0


# -----------------------------------------------------------------------------
# similarity
# ------------------------------------------------------------------------------

class TestSimilarityBenchmark:
    def test_similarity_numpy_arrays(self) -> None:
        provider = EmbeddingProvider(dim=384)
        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)
        start = time.perf_counter()
        for _ in range(100_000):
            provider.similarity(a, b)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_similarity_list_input(self) -> None:
        provider = EmbeddingProvider(dim=384)
        a = [float(i) for i in range(384)]
        b = [float(384 - i) for i in range(384)]
        start = time.perf_counter()
        for _ in range(50_000):
            provider.similarity(a, b)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_similarity_same_vector(self) -> None:
        provider = EmbeddingProvider(dim=384)
        v = np.random.randn(384).astype(np.float32)
        start = time.perf_counter()
        for _ in range(100_000):
            provider.similarity(v, v)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_similarity_zero_norm(self) -> None:
        provider = EmbeddingProvider(dim=384)
        zero = np.zeros(384, dtype=np.float32)
        v = np.random.randn(384).astype(np.float32)
        start = time.perf_counter()
        for _ in range(100_000):
            provider.similarity(zero, v)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_similarity_various_dims(self) -> None:
        for dim in (64, 128, 384, 768):
            provider = EmbeddingProvider(dim=dim)
            a = np.random.randn(dim).astype(np.float32)
            b = np.random.randn(dim).astype(np.float32)
            start = time.perf_counter()
            for _ in range(50_000):
                provider.similarity(a, b)
            elapsed = time.perf_counter() - start
            assert elapsed < 2.0


# -----------------------------------------------------------------------------
# Cache behaviour
# ------------------------------------------------------------------------------

class TestCacheBenchmark:
    def test_cache_eviction_stress(self) -> None:
        provider = EmbeddingProvider(dim=384, max_cache_size=128)
        start = time.perf_counter()
        for i in range(10_000):
            provider.embed(f"evict stress {i}")
        elapsed = time.perf_counter() - start
        assert len(provider._cache) == 128
        assert elapsed < 5.0

    def test_cache_lru_stress(self) -> None:
        provider = EmbeddingProvider(dim=384, max_cache_size=256)
        # warm
        for i in range(256):
            provider.embed(f"lru {i}")
        start = time.perf_counter()
        for _ in range(10_000):
            for i in range(128):
                provider.embed(f"lru {i}")  # touch first half repeatedly
            for j in range(256, 384):
                provider.embed(f"lru {j}")  # insert new
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0


# -----------------------------------------------------------------------------
# Thread-safety / concurrency
# ------------------------------------------------------------------------------

class TestConcurrencyBenchmark:
    def test_concurrent_embed(self) -> None:
        provider = EmbeddingProvider(dim=384)
        texts = [f"concurrent {i}" for i in range(200)]

        def worker():
            for t in texts:
                provider.embed(t)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0

    def test_concurrent_batch_embed(self) -> None:
        provider = EmbeddingProvider(dim=384, max_cache_size=1_000)
        batches = [[f"batch {i} item {j}" for j in range(50)] for i in range(20)]

        def worker(batch):
            provider.embed_batch(batch)

        threads = [threading.Thread(target=worker, args=(b,)) for b in batches]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0

    def test_concurrent_similarity(self) -> None:
        provider = EmbeddingProvider(dim=384)
        vecs = [provider.embed(f"vec {i}") for i in range(100)]
        results = []

        def worker():
            for i in range(len(vecs)):
                for j in range(i, len(vecs)):
                    results.append(provider.similarity(vecs[i], vecs[j]))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        assert len(results) == 4 * 5_050
        assert elapsed < 5.0


# -----------------------------------------------------------------------------
# End-to-end workload
# ------------------------------------------------------------------------------

class TestEndToEndBenchmark:
    def test_retrieval_like_workload(self) -> None:
        """Simulate a tiny retrieval pipeline: embed corpus + score queries."""
        provider = EmbeddingProvider(dim=384, max_cache_size=2_000)
        corpus = [f"document about topic number {i}" for i in range(1_000)]
        queries = [f"query regarding {i}" for i in range(50)]

        start = time.perf_counter()
        corpus_vecs = provider.embed_batch(corpus)
        query_vecs = provider.embed_batch(queries)
        scores = []
        for qv in query_vecs:
            for cv in corpus_vecs:
                scores.append(provider.similarity(qv, cv))
        elapsed = time.perf_counter() - start
        assert len(scores) == 50_000
        assert elapsed < 15.0

    @pytest.mark.slow
    def test_large_corpus_embedding(self) -> None:
        provider = EmbeddingProvider(dim=384, max_cache_size=10_000)
        corpus = [_lorem(50) + f" doc {i}" for i in range(5_000)]
        start = time.perf_counter()
        vecs = provider.embed_batch(corpus)
        elapsed = time.perf_counter() - start
        assert len(vecs) == 5_000
        assert elapsed < 30.0
