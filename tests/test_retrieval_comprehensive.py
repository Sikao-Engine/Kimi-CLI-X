"""Comprehensive tests for previously untested parts of kimix.retrieval."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kimix.retrieval import (
    BM25Scorer,
    CoordinateAscent,
    InvertedIndex,
    LambdaMART,
    LevenshteinAutomaton,
    MinHash,
    NgramTokenizer,
    NoisyChannelSpeller,
    QueryPerformancePredictor,
    RM3Expander,
    RankBoost,
    RankSVM,
    RocchioExpander,
    Searcher,
    SimHash,
    clarity_score,
    cosine_similarity_tfidf,
    hamming_distance,
    i_match_fingerprint,
    jaccard_similarity_tokens,
    metaphone,
    mmr_rerank,
    ngram_overlap,
    porter_stem,
    scq,
    sorensen_dice_coefficient,
    soundex,
    xquad_rerank,
)


# ---------------------------------------------------------------------------
# Phonetic / stemming helpers
# ---------------------------------------------------------------------------


class TestSoundex:
    def test_empty(self) -> None:
        assert soundex("") == ""

    def test_basic(self) -> None:
        assert soundex("Robert") == "R163"
        assert soundex("Rupert") == "R163"
        assert soundex("Rubin") == "R150"
        assert soundex("Ashcraft") == "A226"

    def test_single_char(self) -> None:
        assert soundex("A") == "A000"

    def test_case_insensitive(self) -> None:
        assert soundex("robert") == soundex("ROBERT")


class TestMetaphone:
    def test_empty(self) -> None:
        assert metaphone("") == ""

    def test_basic(self) -> None:
        assert metaphone("Knight") == "KNT"
        assert metaphone("Robert") == "RBRT"
        assert metaphone(" gx ") == "KS"

    def test_vowel_start(self) -> None:
        # Vowel at start is kept
        assert metaphone("Apple").startswith("A")


class TestPorterStem:
    def test_short_words(self) -> None:
        assert porter_stem("a") == "a"
        assert porter_stem("ab") == "ab"

    def test_step1a(self) -> None:
        assert porter_stem("cats") == "cat"
        assert porter_stem("sses") == "ss"
        assert porter_stem("ies") == "i"

    def test_step1b(self) -> None:
        assert porter_stem("running") == "run"
        assert porter_stem("bled") == "bled"

    def test_step2(self) -> None:
        # Simplified stemmer: "relate" loses final 'e' in step 5a
        assert porter_stem("relational") == "relat"
        # "digitizer" -> "digitize" in step2, then step4 drops "ize" -> "digit"
        assert porter_stem("digitizer") == "digit"

    def test_step3(self) -> None:
        # Simplified stemmer replaces "iciti" with "ic", not removes it
        assert porter_stem("electriciti") == "electric"
        assert porter_stem("hopeful") == "hope"

    def test_step4(self) -> None:
        assert porter_stem("revival") == "reviv"

    def test_step5(self) -> None:
        assert porter_stem("probate") == "probat"
        assert porter_stem("rate") == "rate"


# ---------------------------------------------------------------------------
# Distance / similarity helpers
# ---------------------------------------------------------------------------


class TestHammingDistance:
    def test_equal(self) -> None:
        assert hamming_distance("abc", "abc") == 0

    def test_one_off(self) -> None:
        assert hamming_distance("abc", "abd") == 1

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            hamming_distance("ab", "abc")


class TestCosineSimilarityTfidf:
    def test_identical(self) -> None:
        vec = {"a": 1.0, "b": 2.0}
        assert cosine_similarity_tfidf(vec, vec) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert cosine_similarity_tfidf({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_empty(self) -> None:
        assert cosine_similarity_tfidf({}, {"a": 1.0}) == 0.0
        assert cosine_similarity_tfidf({}, {}) == 0.0

    def test_partial(self) -> None:
        a = {"a": 1.0, "b": 1.0}
        b = {"a": 1.0, "c": 1.0}
        assert 0.0 < cosine_similarity_tfidf(a, b) < 1.0


class TestNgramOverlap:
    def test_exact(self) -> None:
        assert ngram_overlap("hello", "hello") == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert ngram_overlap("", "hello") == 0.0
        assert ngram_overlap("hello", "") == 0.0

    def test_partial(self) -> None:
        assert 0.0 < ngram_overlap("hello", "helo") < 1.0


class TestSorensenDiceCoefficient:
    def test_exact(self) -> None:
        assert sorensen_dice_coefficient("hello", "hello") == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert sorensen_dice_coefficient("", "abc") == 0.0
        assert sorensen_dice_coefficient("", "") == pytest.approx(1.0)


class TestJaccardSimilarityTokens:
    def test_exact(self) -> None:
        s = {"a", "b"}
        assert jaccard_similarity_tokens(s, s) == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert jaccard_similarity_tokens(set(), {"a"}) == 0.0
        assert jaccard_similarity_tokens(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


class TestMinHash:
    def test_empty(self) -> None:
        mh = MinHash("")
        assert len(mh.signature) == 128

    def test_same_text(self) -> None:
        mh1 = MinHash("hello world")
        mh2 = MinHash("hello world")
        assert mh1.jaccard(mh2) == pytest.approx(1.0)

    def test_different_text(self) -> None:
        mh1 = MinHash("hello world")
        mh2 = MinHash("completely different content")
        assert 0.0 <= mh1.jaccard(mh2) < 1.0

    def test_mismatch_num_perm_raises(self) -> None:
        mh1 = MinHash("a", num_perm=64)
        mh2 = MinHash("b", num_perm=128)
        with pytest.raises(ValueError, match="same num_perm"):
            mh1.jaccard(mh2)

    def test_approximate_jaccard(self) -> None:
        # Two strings with ~50% overlap should give moderate similarity
        mh1 = MinHash("abcdefgh", num_perm=256)
        mh2 = MinHash("abcdxyzw", num_perm=256)
        sim = mh1.jaccard(mh2)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Clarity / SCQ / Collection LM
# ---------------------------------------------------------------------------


class TestClarityScore:
    def test_empty_query(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        assert clarity_score(idx, []) == 0.0

    def test_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(1, ["hello", "foo"])
        idx.finalize(stop_threshold=1.0)
        score = clarity_score(idx, ["hello"])
        assert score >= 0.0

    def test_no_terms(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        assert clarity_score(idx, ["a"]) == 0.0


class TestScq:
    def test_empty_query(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize()
        assert scq(idx, []) == 0.0

    def test_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(1, ["hello", "foo"])
        idx.finalize(stop_threshold=1.0)
        score = scq(idx, ["hello"])
        assert score > 0.0


# ---------------------------------------------------------------------------
# NoisyChannelSpeller
# ---------------------------------------------------------------------------


class TestNoisyChannelSpeller:
    def test_exact_in_dict(self) -> None:
        speller = NoisyChannelSpeller({"hello": 10})
        assert speller.correct("hello") == "hello"

    def test_one_edit(self) -> None:
        speller = NoisyChannelSpeller({"hello": 10, "hallo": 5})
        assert speller.correct("helo") == "hello"

    def test_no_candidate(self) -> None:
        speller = NoisyChannelSpeller({"abc": 1})
        assert speller.correct("xyz") == "xyz"

    def test_candidates(self) -> None:
        speller = NoisyChannelSpeller({"hello": 10, "help": 5}, max_edits=1)
        cands = speller._candidates("helo")
        assert "hello" in cands or "help" in cands


# ---------------------------------------------------------------------------
# RocchioExpander
# ---------------------------------------------------------------------------


class TestRocchioExpander:
    def test_expand_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async"])
        idx.add_document(1, ["python", "threading"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RocchioExpander(idx, scorer, alpha=1.0, beta=0.75, gamma=0.15)
        result = expander.expand(["python"])
        assert "python" in result

    def test_expand_no_results(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        expander = RocchioExpander(idx, scorer)
        assert expander.expand(["missing"]) == ["missing"]

    def test_expand_empty_query(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RocchioExpander(idx, scorer)
        assert expander.expand([]) == []

    def test_expand_with_non_rel(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async"])
        idx.add_document(1, ["python", "threading"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RocchioExpander(idx, scorer, alpha=1.0, beta=0.75, gamma=0.15)
        result = expander.expand(["python"], non_rel_docs={1})
        assert "python" in result


# ---------------------------------------------------------------------------
# RankSVM
# ---------------------------------------------------------------------------


class TestRankSVM:
    def test_fit_predict(self) -> None:
        model = RankSVM(learning_rate=0.01, n_iterations=100)
        X = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        y = [2.0, 1.0, 0.0]
        model.fit(X, y)
        scores = model.predict(X)
        assert len(scores) == 3
        # Higher relevance should generally score higher
        assert scores[0] > scores[2]

    def test_rank(self) -> None:
        model = RankSVM()
        X = [[1.0, 0.0], [0.0, 1.0]]
        y = [1.0, 0.0]
        model.fit(X, y)
        ranked = model.rank([(0, [1.0, 0.0]), (1, [0.0, 1.0])])
        assert ranked[0][0] == 0

    def test_empty_fit(self) -> None:
        model = RankSVM()
        model.fit([], [])
        assert model.predict([[1.0, 0.0]]) == [0.0]


# ---------------------------------------------------------------------------
# RankBoost
# ---------------------------------------------------------------------------


class TestRankBoost:
    def test_fit_predict(self) -> None:
        model = RankBoost(n_iterations=20)
        X = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        y = [2.0, 1.0, 0.0]
        model.fit(X, y)
        scores = model.predict(X)
        assert len(scores) == 3

    def test_rank(self) -> None:
        model = RankBoost(n_iterations=20)
        X = [[1.0, 0.0], [0.0, 1.0]]
        y = [1.0, 0.0]
        model.fit(X, y)
        ranked = model.rank([(0, [1.0, 0.0]), (1, [0.0, 1.0])])
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]

    def test_empty_fit(self) -> None:
        model = RankBoost()
        model.fit([], [])
        assert model.predict([[1.0]]) == [0.0]

    def test_single_feature_value(self) -> None:
        model = RankBoost(n_iterations=5)
        X = [[1.0], [1.0], [0.0]]
        y = [1.0, 1.0, 0.0]
        model.fit(X, y)
        scores = model.predict(X)
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# xQuAD
# ---------------------------------------------------------------------------


class TestXquadRerank:
    def test_basic(self) -> None:
        results = [(0, 1.0), (1, 0.9), (2, 0.8)]
        aspects = {0: {"a", "b"}, 1: {"a", "c"}, 2: {"d"}}
        ranked = xquad_rerank(results, aspects, lambda_param=0.5, top_k=2)
        assert len(ranked) == 2
        assert ranked[0][0] == 0

    def test_empty(self) -> None:
        assert xquad_rerank([], {}) == []

    def test_no_aspects(self) -> None:
        results = [(0, 1.0), (1, 0.5)]
        ranked = xquad_rerank(results, {}, lambda_param=0.0, top_k=2)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# I-Match
# ---------------------------------------------------------------------------


class TestIMatchFingerprint:
    def test_basic(self) -> None:
        tokens = ["the", "quick", "brown", "fox", "the"]
        fp = i_match_fingerprint(tokens)
        # No stopwords provided, so "the" is kept
        assert fp == "brown fox quick the"

    def test_with_stopwords(self) -> None:
        tokens = ["the", "quick", "brown"]
        fp = i_match_fingerprint(tokens, stopwords={"the"})
        assert fp == "brown quick"

    def test_empty(self) -> None:
        assert i_match_fingerprint([]) == ""


# ---------------------------------------------------------------------------
# InvertedIndex save/load with forward index
# ---------------------------------------------------------------------------


class TestInvertedIndexForwardIndex:
    def test_save_load_with_forward_index(self, tmp_path: Path) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb", "aa"])
        idx.add_document(1, ["bb", "cc"])
        idx.finalize(stop_threshold=1.0)

        path = tmp_path / "index_fwd.bin"
        idx.save(path, include_forward_index=True)

        idx2 = InvertedIndex()
        idx2.load(path)
        assert idx2.N == idx.N
        assert idx2.avgdl == idx.avgdl
        assert sorted(idx2.terms()) == sorted(idx.terms())
        # Forward index should be restored
        assert len(idx2._doc_term_freqs) == 2
        assert idx2._doc_term_freqs[0].get("aa") == 2


# ---------------------------------------------------------------------------
# BM25Scorer buffer / edge cases
# ---------------------------------------------------------------------------


class TestBM25ScorerEdgeCases:
    def test_ensure_buffers(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        buf1, buf2 = scorer._ensure_buffers(10)
        assert len(buf1) >= 10
        assert len(buf2) >= 10

    def test_prepare_candidates(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        assert scorer._prepare_candidates(None) is None
        arr = scorer._prepare_candidates({2, 1, 0})
        assert list(arr) == [0, 1, 2]

    def test_score_candidate_docs_empty_set(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        assert scorer.score(["a"], candidate_docs=set()) == {}

    def test_score_topk_sparse_small_topk(self) -> None:
        idx = InvertedIndex()
        for i in range(200):
            idx.add_document(i, ["token"] if i % 2 == 0 else ["other"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        top = scorer.score_topk(["token"], top_k=2)
        assert len(top) == 2

    def test_score_topk_partition_mask(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.add_document(1, ["b"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        top = scorer.score_topk(["a"], top_k=5)
        assert len(top) == 1
        assert top[0][0] == 0


# ---------------------------------------------------------------------------
# QueryPerformancePredictor full coverage
# ---------------------------------------------------------------------------


class TestQueryPerformancePredictorFull:
    def test_max_idf(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.add_document(1, ["world"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        assert qpp.max_idf(["hello"]) > 0.0

    def test_avg_idf_missing_terms(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        assert qpp.avg_idf(["missing"]) == 0.0

    def test_query_scope_empty(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        assert qpp.query_scope(["a"]) == 0.0

    def test_is_hard_query_true(self) -> None:
        idx = InvertedIndex()
        for i in range(100):
            idx.add_document(i, ["common"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        qpp = QueryPerformancePredictor(idx, scorer)
        assert qpp.is_hard_query(["common"], avg_idf_threshold=5.0) is True


# ---------------------------------------------------------------------------
# LevenshteinAutomaton edge cases
# ---------------------------------------------------------------------------


class TestLevenshteinAutomatonEdgeCases:
    def test_freq_lower_bound_long_pattern(self) -> None:
        auto = LevenshteinAutomaton("abcdefghijklmnopqrstuvwxyz", max_edits=2)
        # exact match
        assert auto._freq_lower_bound("abcdefghijklmnopqrstuvwxyz") == 0

    def test_freq_lower_bound_one_off_long(self) -> None:
        auto = LevenshteinAutomaton("abcdefghijklmnopqrstuvwxyz", max_edits=2)
        # one substitution should still have low lower bound
        assert auto._freq_lower_bound("abcdefghijklmnopqrstuvwxYz") <= 1

    def test_match_without_any_optimizations(self) -> None:
        # Plain list has no _terms_by_length or _symmetric_delete_index
        auto = LevenshteinAutomaton("hello", max_edits=1)
        result = auto.match(["hello", "hallo", "world", "hell"], max_expansions=50)
        assert "hello" in result
        assert "hallo" in result

    def test_match_prefix_length_zero(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "jello"])
        idx.finalize(stop_threshold=1.0)
        auto = LevenshteinAutomaton("hello", max_edits=1, prefix_length=0)
        result = auto.match(idx, max_expansions=50)
        assert "hello" in result
        assert "jello" in result  # prefix_length=0 means no prefix filter


# ---------------------------------------------------------------------------
# Searcher edge cases
# ---------------------------------------------------------------------------


class TestSearcherEdgeCases:
    def test_search_min_should_match_zero(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, min_should_match=0.0)
        # min_match is max(1, int(tokens * 0.0)) = 1, so at least 1 token must match
        results = searcher.search("aa")
        assert len(results) == 1

    def test_search_all_tokens_missing(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, min_should_match=1.0)
        results = searcher.search("bb cc")
        assert results == []

    def test_expand_token_cache(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, fuzziness=1)
        r1 = searcher._expand_token("helo")
        r2 = searcher._expand_token("helo")
        assert r1 == r2

    def test_search_with_fuzzy_and_min_match(self) -> None:
        tokenizer = NgramTokenizer(n=3)
        idx = InvertedIndex()
        idx.add_document(0, tokenizer.tokenize("hello world"))
        idx.add_document(1, tokenizer.tokenize("foo bar"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer, fuzziness=1, min_should_match=0.5)
        results = searcher.search("helo wrld", top_k=5)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# CoordinateAscent alias
# ---------------------------------------------------------------------------


class TestCoordinateAscent:
    def test_alias(self) -> None:
        assert CoordinateAscent is LambdaMART

    def test_basic_usage(self) -> None:
        model = CoordinateAscent(n_iterations=10, learning_rate=0.1)
        X = [[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]]
        y = [[2.0, 1.0, 0.0]]
        model.fit(X, y)
        assert len(model.predict([[1.0, 0.0]])) == 1


# ---------------------------------------------------------------------------
# SimHash additional coverage
# ---------------------------------------------------------------------------


class TestSimHashAdditional:
    def test_distance_symmetry(self) -> None:
        h1 = SimHash("foo bar")
        h2 = SimHash("baz qux")
        assert h1.distance(h2) == h2.distance(h1)

    def test_is_near_duplicate_threshold(self) -> None:
        h1 = SimHash("foo")
        h2 = SimHash("foo")
        assert h1.is_near_duplicate(h2, threshold=0) is True


# ---------------------------------------------------------------------------
# MMR additional coverage
# ---------------------------------------------------------------------------


class TestMmrAdditional:
    def test_mmr_top_k_none(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a", "b"])
        idx.add_document(1, ["a", "c"])
        idx.finalize(stop_threshold=1.0)
        results = [(0, 1.0), (1, 0.5)]
        ranked = mmr_rerank(results, idx, lambda_param=0.5, top_k=None)
        assert len(ranked) == 2

    def test_mmr_forward_index_fallback(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        # doc_id beyond forward index length
        results = [(0, 1.0), (999, 0.5)]
        ranked = mmr_rerank(results, idx, lambda_param=0.5, top_k=2)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# LambdaMART additional coverage
# ---------------------------------------------------------------------------


class TestLambdaMARTAdditional:
    def test_empty_query_list(self) -> None:
        model = LambdaMART()
        model.fit([], [])
        assert model.predict([[1.0]]) == [0.0]

    def test_rank_empty(self) -> None:
        model = LambdaMART()
        assert model.rank([]) == []


# ---------------------------------------------------------------------------
# InvertedIndex empty / edge cases
# ---------------------------------------------------------------------------


class TestInvertedIndexAdditional:
    def test_add_document_with_stop_ngram(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["..", "!!"])
        idx.finalize(stop_threshold=1.0)
        assert idx.has_term("..") is False
        assert idx.has_term("!!") is False

    def test_get_postings_auto_finalize(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a", "b"])
        idx.add_document(1, ["c", "d"])
        # "a" appears in 1 of 2 docs -> not pruned with default threshold 0.5
        postings = idx.get_postings("a")
        assert postings is not None

    def test_doc_freq_missing(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        assert idx.doc_freq("missing") == 0

    def test_finalize_idempotent(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        idx.finalize(stop_threshold=1.0)
        assert idx._finalized is True


# ---------------------------------------------------------------------------
# NgramTokenizer edge cases
# ---------------------------------------------------------------------------


class TestNgramTokenizerAdditional:
    def test_detect_n_all_cjk(self) -> None:
        t = NgramTokenizer(n=3)
        text = "你好世界"
        assert t._detect_n(text) == 2

    def test_detect_n_no_cjk(self) -> None:
        t = NgramTokenizer(n=2)
        text = "hello"
        assert t._detect_n(text) == 3

    def test_tokenize_whitespace_only(self) -> None:
        t = NgramTokenizer()
        assert t.tokenize("   \n\t  ") == []

    def test_is_cjk_hangul(self) -> None:
        assert NgramTokenizer._is_cjk("가") is True


# ---------------------------------------------------------------------------
# RM3Expander additional coverage
# ---------------------------------------------------------------------------


class TestRM3ExpanderAdditional:
    def test_expand_total_tokens_zero(self) -> None:
        # If feedback docs have no terms (empty forward index after prune)
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        expander = RM3Expander(idx, scorer, fb_docs=1, fb_terms=1)
        result = expander.expand(["a"])
        assert "a" in result


# ---------------------------------------------------------------------------
# RocchioExpander additional coverage
# ---------------------------------------------------------------------------


class TestRocchioExpanderAdditional:
    def test_expand_no_rel_docs(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        expander = RocchioExpander(idx, scorer)
        assert expander.expand(["a"]) == ["a"]


# ---------------------------------------------------------------------------
# RankSVM / RankBoost rank empty
# ---------------------------------------------------------------------------


class TestRankersRankEmpty:
    def test_ranksvm_rank_empty(self) -> None:
        model = RankSVM()
        assert model.rank([]) == []

    def test_rankboost_rank_empty(self) -> None:
        model = RankBoost()
        assert model.rank([]) == []
