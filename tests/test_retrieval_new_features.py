"""Tests for new retrieval features: similarities, SimHash, MMR, RM3, LambdaMART, QPP."""

from __future__ import annotations

import math

import pytest

from kimix.retrieval import (
    BM25Scorer,
    InvertedIndex,
    LambdaMART,
    NgramTokenizer,
    QueryPerformancePredictor,
    RM3Expander,
    Searcher,
    SimHash,
    jaccard_similarity_tokens,
    jaro_similarity,
    jaro_winkler_similarity,
    mmr_rerank,
    ngram_overlap,
    sorensen_dice_coefficient,
)


class TestStringSimilarity:
    def test_jaro_exact(self) -> None:
        assert jaro_similarity("hello", "hello") == pytest.approx(1.0)

    def test_jaro_empty(self) -> None:
        assert jaro_similarity("", "abc") == 0.0
        assert jaro_similarity("abc", "") == 0.0

    def test_jaro_typical(self) -> None:
        assert jaro_similarity("MARTHA", "MARHTA") > 0.9

    def test_jaro_winkler_prefix_boost(self) -> None:
        jw = jaro_winkler_similarity("hello", "helo")
        j = jaro_similarity("hello", "helo")
        assert jw > j

    def test_dice_exact(self) -> None:
        assert sorensen_dice_coefficient("hello", "hello") == pytest.approx(1.0)

    def test_dice_partial(self) -> None:
        d = sorensen_dice_coefficient("night", "nacht")
        assert 0.0 < d < 1.0

    def test_ngram_overlap_exact(self) -> None:
        assert ngram_overlap("hello", "hello") == pytest.approx(1.0)

    def test_jaccard_tokens(self) -> None:
        assert jaccard_similarity_tokens({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)


class TestSimHash:
    def test_exact_duplicate(self) -> None:
        h1 = SimHash("the quick brown fox")
        h2 = SimHash("the quick brown fox")
        assert h1.is_near_duplicate(h2)
        assert h1.distance(h2) == 0

    def test_near_duplicate(self) -> None:
        h1 = SimHash("the quick brown fox")
        h2 = SimHash("the quick brown fox jumps")
        # Small edit should be near duplicate
        assert h1.distance(h2) <= 5

    def test_different_texts(self) -> None:
        h1 = SimHash("lorem ipsum dolor sit amet")
        h2 = SimHash("completely different content here")
        assert not h1.is_near_duplicate(h2, threshold=3)


class TestMMR:
    def _make_index(self) -> InvertedIndex:
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async", "programming"])
        idx.add_document(1, ["python", "threading", "programming"])
        idx.add_document(2, ["java", "async", "programming"])
        idx.finalize(stop_threshold=1.0)
        return idx

    def test_mmr_pure_relevance(self) -> None:
        idx = self._make_index()
        results = [(0, 1.0), (1, 0.9), (2, 0.8)]
        ranked = mmr_rerank(results, idx, lambda_param=1.0, top_k=2)
        assert len(ranked) == 2
        assert ranked[0][0] == 0
        assert ranked[1][0] == 1

    def test_mmr_diversity(self) -> None:
        idx = self._make_index()
        results = [(0, 1.0), (1, 0.9), (2, 0.8)]
        ranked = mmr_rerank(results, idx, lambda_param=0.1, top_k=2)
        assert len(ranked) == 2
        # With strong diversity emphasis, first pick is still most relevant (0),
        # second pick should differ from 0
        ids = [r[0] for r in ranked]
        assert ids[0] == 0
        assert ids[1] != 0

    def test_mmr_empty(self) -> None:
        assert mmr_rerank([], InvertedIndex()) == []


class TestRM3Expander:
    def test_expand_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async", "programming"])
        idx.add_document(1, ["python", "threading", "programming"])
        idx.add_document(2, ["java", "async", "programming"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RM3Expander(idx, scorer, fb_docs=2, fb_terms=2, alpha=0.5)
        expanded = expander.expand(["python"])
        assert len(expanded) > 0
        assert "python" in expanded

    def test_expand_no_results(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        expander = RM3Expander(idx, scorer)
        assert expander.expand(["missing"]) == ["missing"]

    def test_expand_empty_query(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RM3Expander(idx, scorer)
        assert expander.expand([]) == []


class TestLambdaMART:
    def test_fit_predict(self) -> None:
        model = LambdaMART(n_iterations=20, learning_rate=0.1)
        # Two queries, 3 docs each, 2 features
        X = [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]],
        ]
        y = [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
        model.fit(X, y)
        scores = model.predict([[1.0, 0.0], [0.0, 1.0]])
        assert len(scores) == 2

    def test_rank(self) -> None:
        model = LambdaMART(n_iterations=20, learning_rate=0.1)
        X = [[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]]
        y = [[2.0, 1.0, 0.0]]
        model.fit(X, y)
        ranked = model.rank([(0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.5, 0.5])])
        assert len(ranked) == 3
        # Best doc should have highest score
        assert ranked[0][1] >= ranked[1][1]

    def test_empty_fit(self) -> None:
        model = LambdaMART()
        model.fit([], [])
        assert model.predict([[1.0, 0.0]]) == [0.0]


class TestQueryPerformancePredictor:
    def test_avg_idf(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(1, ["hello", "foo"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        avg = qpp.avg_idf(["hello"])
        assert avg > 0.0

    def test_query_scope(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.add_document(1, ["world"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        scope = qpp.query_scope(["hello"])
        assert scope == pytest.approx(0.5)

    def test_is_hard_query(self) -> None:
        idx = InvertedIndex()
        for i in range(100):
            idx.add_document(i, ["common", f"word{i}"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        # "common" appears in all docs -> low IDF -> hard query
        assert qpp.is_hard_query(["common"], avg_idf_threshold=5.0)
