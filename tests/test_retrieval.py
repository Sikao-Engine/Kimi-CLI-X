"""Comprehensive tests for kimix.retrieval."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kimix.retrieval import (
    BM25Scorer,
    InvertedIndex,
    LevenshteinAutomaton,
    NgramTokenizer,
    Searcher,
)


class TestNgramTokenizer:
    def test_normalize_lowercase(self) -> None:
        t = NgramTokenizer()
        assert t.normalize("Hello World") == "hello world"

    def test_normalize_nfkc(self) -> None:
        t = NgramTokenizer()
        # Full-width latin characters -> half-width
        assert t.normalize("\uff21\uff22\uff23") == "abc"

    def test_is_cjk(self) -> None:
        t = NgramTokenizer()
        assert t._is_cjk("\u4e00") is True
        assert t._is_cjk("\u3042") is True
        assert t._is_cjk("\u30a2") is True
        assert t._is_cjk("\uac00") is True
        assert t._is_cjk("a") is False
        assert t._is_cjk("1") is False

    def test_detect_n_empty(self) -> None:
        t = NgramTokenizer(n=2)
        assert t._detect_n("") == 2

    def test_detect_n_cjk(self) -> None:
        t = NgramTokenizer(n=3)
        # More than 30% CJK -> bigram
        text = "\u4e00\u4e01\u4e02abcd"
        assert t._detect_n(text) == 2

    def test_detect_n_mixed(self) -> None:
        t = NgramTokenizer(n=3)
        # Less than 30% CJK -> trigram (or default n if >=3)
        text = "\u4e00abcde"
        assert t._detect_n(text) == 3

    def test_detect_n_latin(self) -> None:
        t = NgramTokenizer(n=2)
        text = "abcdef"
        assert t._detect_n(text) == 3  # because self.n < 3 triggers return 3

    def test_tokenize_empty(self) -> None:
        t = NgramTokenizer()
        assert t.tokenize("") == []
        assert t.tokenize("   ") == []

    def test_tokenize_short(self) -> None:
        t = NgramTokenizer(n=3)
        assert t.tokenize("ab") == ["ab"]

    def test_tokenize_latin_trigram(self) -> None:
        t = NgramTokenizer(n=3)
        result = t.tokenize("hello")
        assert result == ["hel", "ell", "llo"]

    def test_tokenize_cjk_bigram(self) -> None:
        t = NgramTokenizer(n=3)
        result = t.tokenize("\u4e00\u4e01\u4e02")
        assert result == ["\u4e00\u4e01", "\u4e01\u4e02"]

    def test_tokenize_explicit_n(self) -> None:
        t = NgramTokenizer(n=2)
        result = t.tokenize("hello", n=2)
        assert result == ["he", "el", "ll", "lo"]


class TestInvertedIndex:
    def test_add_and_get_postings(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb", "aa"])
        idx.add_document(1, ["bb", "cc"])
        idx.finalize(stop_threshold=1.0)

        docs, tfs = idx.get_postings("aa")
        assert list(docs) == [0]
        assert list(tfs) == [2]

        docs, tfs = idx.get_postings("bb")
        assert list(docs) == [0, 1]
        assert list(tfs) == [1, 1]

        assert idx.get_postings("zz") is None

    def test_doc_freq_and_has_term(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.finalize(stop_threshold=1.0)
        assert idx.doc_freq("aa") == 1
        assert idx.has_term("aa") is True
        assert idx.has_term("zz") is False
        assert sorted(idx.terms()) == ["aa", "bb"]

    def test_add_after_finalize_raises(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.finalize()
        with pytest.raises(RuntimeError, match="Cannot add documents after finalize"):
            idx.add_document(1, ["bb"])

    def test_finalize_empty(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        assert idx.N == 0
        assert idx.avgdl == 0.0
        assert idx.doc_lengths == []

    def test_stop_ngram_removal(self) -> None:
        idx = InvertedIndex()
        # ".." appears in all docs -> stop ngram
        idx.add_document(0, ["aa", ".."])
        idx.add_document(1, ["bb", ".."])
        idx.finalize(stop_threshold=0.5)
        assert idx.has_term("..") is False
        assert idx.has_term("aa") is True

    def test_prune_df(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.add_document(1, ["aa"])
        idx.add_document(2, ["aa"])
        idx.add_document(3, ["bb"])
        idx.finalize(prune_df=2)
        assert idx.has_term("aa") is False
        assert idx.has_term("bb") is True

    def test_properties(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a", "b", "c"])
        idx.add_document(1, ["d", "e"])
        idx.finalize(stop_threshold=1.0)
        assert idx.N == 2
        assert idx.avgdl == 2.5
        assert idx.doc_lengths == [3, 2]
        assert len(idx.doc_lengths_arr) == 2

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb", "aa"])
        idx.add_document(1, ["bb", "cc"])
        idx.finalize(stop_threshold=1.0)

        path = tmp_path / "index.bin"
        idx.save(path)

        idx2 = InvertedIndex()
        idx2.load(path)

        assert idx2.N == idx.N
        assert idx2.avgdl == idx.avgdl
        assert idx2.doc_lengths == idx.doc_lengths
        assert sorted(idx2.terms()) == sorted(idx.terms())

        docs, tfs = idx2.get_postings("aa")
        assert list(docs) == [0]
        assert list(tfs) == [2]

    def test_load_invalid_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.bin"
        path.write_bytes(b"BAD\x00")
        with pytest.raises(ValueError, match="Invalid file format"):
            InvertedIndex().load(path)

    def test_load_unsupported_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.bin"
        data = b"KIMX\x63" + struct.pack("<I", 0) * 2 + struct.pack("<d", 0.0) + struct.pack("<I", 0)
        path.write_bytes(data)
        with pytest.raises(ValueError, match="Unsupported version"):
            InvertedIndex().load(path)

    def test_generate_deletes(self) -> None:
        assert InvertedIndex._generate_deletes("abc", 0) == {"abc"}
        assert InvertedIndex._generate_deletes("abc", 1) == {"abc", "ab", "ac", "bc"}

    def test_is_stop_ngram_punctuation(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["xx"])
        assert idx._is_stop_ngram("!!", 1, threshold=1.0) is True
        assert idx._is_stop_ngram("ab", 1, threshold=1.0) is False


class TestBM25Scorer:
    def test_idf(self) -> None:
        scorer = BM25Scorer(InvertedIndex())
        # Lucene variant: log(1 + (N - df + 0.5) / (df + 0.5))
        N, df = 10, 2
        expected = np.log(1 + (N - df + 0.5) / (df + 0.5))
        assert scorer._idf(df, N) == pytest.approx(expected)

    def test_score_empty_index(self) -> None:
        idx = InvertedIndex()
        idx.finalize()
        scorer = BM25Scorer(idx)
        assert scorer.score(["aa"]) == {}

    def test_score_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb", "cc"])
        idx.add_document(1, ["aa", "bb", "dd"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        scores = scorer.score(["aa"])
        assert len(scores) == 2
        assert scores[0] > 0
        assert scores[1] > 0

    def test_score_candidate_docs(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa", "bb"])
        idx.add_document(1, ["aa", "cc"])
        idx.add_document(2, ["bb", "cc"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        scores = scorer.score(["aa"], candidate_docs={0, 2})
        assert 0 in scores
        assert 2 not in scores  # doc 2 has no "aa"
        assert 1 not in scores

    def test_score_topk_basic(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"] * 10)
        idx.add_document(1, ["aa"] * 5)
        idx.add_document(2, ["bb"] * 10)
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        top = scorer.score_topk(["aa"], top_k=2)
        assert len(top) == 2
        assert top[0][0] == 0  # doc 0 should win
        assert top[0][1] >= top[1][1]

    def test_score_topk_zero_k(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        assert scorer.score_topk(["aa"], top_k=0) == []

    def test_score_topk_all_docs(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["aa"])
        idx.add_document(1, ["bb"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        top = scorer.score_topk(["aa"], top_k=10)
        assert len(top) == 1

    def test_accumulate_vs_sparse_consistency(self) -> None:
        idx = InvertedIndex()
        for i in range(100):
            idx.add_document(i, ["aa", "bb"] if i % 2 == 0 else ["bb", "cc"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        dense = scorer._accumulate(["aa"], None)
        sparse = scorer._accumulate_sparse(["aa"], None)
        nonzero = {int(i): float(dense[i]) for i in np.flatnonzero(dense)}
        assert nonzero == pytest.approx(sparse)

    def test_large_n_uses_sparse(self) -> None:
        idx = InvertedIndex()
        for i in range(60000):
            idx.add_document(i, ["aa"] if i % 2 == 0 else ["bb"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        # score() should use sparse path when N > 50000
        scores = scorer.score(["aa"])
        assert len(scores) == 30000

    def test_score_topk_sparse_path(self) -> None:
        idx = InvertedIndex()
        for i in range(60000):
            idx.add_document(i, ["token"] if i % 2 == 0 else ["other"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        top = scorer.score_topk(["token"], top_k=5)
        assert len(top) == 5


class TestLevenshteinAutomaton:
    def test_damerau_levenshtein_empty(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("", "") == 0
        assert LevenshteinAutomaton._damerau_levenshtein("a", "") == 1
        assert LevenshteinAutomaton._damerau_levenshtein("", "a") == 1

    def test_damerau_levenshtein_exact(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("abc", "abc") == 0

    def test_damerau_levenshtein_substitution(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("abc", "axc") == 1

    def test_damerau_levenshtein_transposition(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("ab", "ba") == 1
        assert LevenshteinAutomaton._damerau_levenshtein("abc", "acb") == 1

    def test_damerau_levenshtein_deletion(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("abc", "ac") == 1

    def test_damerau_levenshtein_insertion(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("ac", "abc") == 1

    def test_damerau_levenshtein_2x2(self) -> None:
        assert LevenshteinAutomaton._damerau_levenshtein("ab", "cd") == 2
        assert LevenshteinAutomaton._damerau_levenshtein("ab", "ba") == 1

    def test_auto_fuzziness(self) -> None:
        assert LevenshteinAutomaton.auto_fuzziness("") == 0
        assert LevenshteinAutomaton.auto_fuzziness("ab") == 0
        assert LevenshteinAutomaton.auto_fuzziness("abc") == 1
        assert LevenshteinAutomaton.auto_fuzziness("abcde") == 1
        assert LevenshteinAutomaton.auto_fuzziness("abcdef") == 2

    def test_freq_lower_bound_exact(self) -> None:
        auto = LevenshteinAutomaton("abc", max_edits=1)
        assert auto._freq_lower_bound("abc") == 0

    def test_freq_lower_bound_one_off(self) -> None:
        auto = LevenshteinAutomaton("abc", max_edits=1)
        # "abd" has one substitution -> lower bound should be <= 1
        assert auto._freq_lower_bound("abd") <= 1

    def test_match_exact(self) -> None:
        auto = LevenshteinAutomaton("hello", max_edits=1)
        assert auto.match(["hello", "world"]) == ["hello"]

    def test_match_fuzzy(self) -> None:
        auto = LevenshteinAutomaton("hello", max_edits=1)
        result = auto.match(["hello", "hallo", "world"])
        assert "hello" in result
        assert "hallo" in result
        assert "world" not in result

    def test_match_max_expansions(self) -> None:
        auto = LevenshteinAutomaton("a", max_edits=1)
        dictionary = ["a", "b", "c", "d", "e", "f"]
        result = auto.match(dictionary, max_expansions=3)
        assert len(result) <= 3

    def test_match_with_inverted_index(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "hallo", "hillo"])
        idx.finalize(stop_threshold=1.0)
        auto = LevenshteinAutomaton("hello", max_edits=1, prefix_length=1)
        result = auto.match(idx, max_expansions=50)
        assert "hello" in result
        assert "hallo" in result
        assert "hillo" in result

    def test_match_prefix_filter(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "jello"])
        idx.finalize(stop_threshold=1.0)
        auto = LevenshteinAutomaton("hello", max_edits=1, prefix_length=1)
        result = auto.match(idx, max_expansions=50)
        assert "hello" in result
        assert "jello" not in result  # different prefix

    def test_match_symmetric_delete_path(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["abcde"])
        idx.finalize(stop_threshold=1.0)
        auto = LevenshteinAutomaton("abde", max_edits=1, prefix_length=1)
        result = auto.match(idx, max_expansions=50)
        assert "abcde" in result

    def test_match_length_bounds(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["a", "abcdefgh"])
        idx.finalize(stop_threshold=1.0)
        auto = LevenshteinAutomaton("abc", max_edits=1, prefix_length=0)
        result = auto.match(idx, max_expansions=50)
        assert "a" not in result  # too short
        assert "abcdefgh" not in result  # too long


class TestSearcher:
    def _make_searcher(
        self,
        docs: list[str],
        tokenizer: NgramTokenizer | None = None,
    ) -> Searcher:
        idx = InvertedIndex()
        tokenizer = tokenizer or NgramTokenizer()
        for i, text in enumerate(docs):
            idx.add_document(i, tokenizer.tokenize(text))
        idx.finalize(stop_threshold=1.0)
        return Searcher(idx, tokenizer=tokenizer)

    def test_search_empty_index(self) -> None:
        searcher = Searcher(InvertedIndex())
        assert searcher.search("hello") == []

    def test_search_empty_query(self) -> None:
        searcher = self._make_searcher(["hello"])
        assert searcher.search("") == []
        assert searcher.search("   ") == []

    def test_search_basic(self) -> None:
        searcher = self._make_searcher([
            "hello world",
            "hello there",
            "foo bar",
        ])
        results = searcher.search("hello")
        assert len(results) == 2
        doc_ids = [r[0] for r in results]
        assert 0 in doc_ids
        assert 1 in doc_ids

    def test_search_top_k(self) -> None:
        searcher = self._make_searcher([
            "hello " * 10,
            "hello " * 5,
            "foo bar",
        ])
        results = searcher.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_min_should_match(self) -> None:
        searcher = self._make_searcher([
            "aa bb",
            "aa cc",
            "bb cc",
        ])
        # query "aa bb" with min_should_match=0.5 -> at least 1 of 3 ngram tokens
        results = searcher.search("aa bb")
        doc_ids = {r[0] for r in results}
        assert 0 in doc_ids
        assert 1 in doc_ids or 2 in doc_ids

    def test_min_should_match_strict(self) -> None:
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        idx.add_document(0, tokenizer.tokenize("aa bb"))
        idx.add_document(1, tokenizer.tokenize("aa cc"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer, min_should_match=1.0)
        results = searcher.search("aa bb")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_fuzziness_auto(self) -> None:
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        idx.add_document(0, tokenizer.tokenize("hello"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer, fuzziness="AUTO")
        results = searcher.search("hallo")  # 1 edit away, length 5 -> max_edits=1
        assert len(results) == 1
        assert results[0][0] == 0

    def test_fuzziness_zero(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, fuzziness=0)
        results = searcher.search("hallo")
        assert results == []

    def test_expand_token_cjk(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["\u4e00\u4e01"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx)
        assert searcher._expand_token("\u4e00\u4e01") == ["\u4e00\u4e01"]

    def test_expand_token_missing(self) -> None:
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, fuzziness=0)
        assert searcher._expand_token("missing") == []

    def test_is_latin_token(self) -> None:
        assert Searcher._is_latin_token("hello") is True
        assert Searcher._is_latin_token("") is False
        assert Searcher._is_latin_token("\u4e00") is False
        assert Searcher._is_latin_token("h\u00e9llo") is False  # é is >127

    def test_search_with_candidate_docs_filter(self) -> None:
        idx = InvertedIndex()
        for i in range(100):
            idx.add_document(i, ["aa"] if i < 50 else ["bb"])
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx)
        results = searcher.search("aa", top_k=50)
        assert len(results) == 50
        assert all(r[0] < 50 for r in results)

    def test_search_no_matches(self) -> None:
        searcher = self._make_searcher(["hello", "world"])
        results = searcher.search("zzz")
        assert results == []

    def test_search_cjk(self) -> None:
        searcher = self._make_searcher([
            "\u4e00\u4e01\u4e02\u4e03",
            "\u4e00\u4e01\u4e04\u4e05",
        ])
        results = searcher.search("\u4e00\u4e01")
        assert len(results) == 2

    def test_prefix_length_filter(self) -> None:
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        idx.add_document(0, tokenizer.tokenize("abcde"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer, fuzziness=1, prefix_length=1, max_expansions=50)
        results = searcher.search("abde")
        assert len(results) == 1
        assert results[0][0] == 0


class TestIntegration:
    def test_end_to_end(self, tmp_path: Path) -> None:
        idx = InvertedIndex()
        docs = [
            "the quick brown fox",
            "the lazy dog",
            "the quick dog",
            "a fox is quick",
        ]
        tokenizer = NgramTokenizer(n=3)
        for i, text in enumerate(docs):
            idx.add_document(i, tokenizer.tokenize(text))
        idx.finalize()

        searcher = Searcher(idx, tokenizer=tokenizer)
        results = searcher.search("quick fox", top_k=2)
        assert len(results) == 2
        # doc 0 has "quick" and "fox"
        assert results[0][0] == 0

        # Save and reload
        path = tmp_path / "index.bin"
        idx.save(path)
        idx2 = InvertedIndex()
        idx2.load(path)
        searcher2 = Searcher(idx2, tokenizer=tokenizer)
        results2 = searcher2.search("quick fox", top_k=2)
        assert results2 == results

    def test_fuzzy_search_integration(self) -> None:
        idx = InvertedIndex()
        tokenizer = NgramTokenizer(n=3)
        idx.add_document(0, tokenizer.tokenize("hello world"))
        idx.add_document(1, tokenizer.tokenize("hallo there"))
        idx.finalize()

        searcher = Searcher(idx, tokenizer=tokenizer, fuzziness="AUTO")
        results = searcher.search("hello")
        assert any(r[0] == 0 for r in results)
        # "hallo" is 1 edit from "hello" -> should also match
        assert any(r[0] == 1 for r in results)

class TestNgramTokenizerMemory:
    def test_tokenize_basic(self):
        t = NgramTokenizer(n=2)
        tokens = t.tokenize("hello", n=2)
        assert tokens == ["he", "el", "ll", "lo"]

    def test_tokenize_short_text(self):
        t = NgramTokenizer(n=3)
        tokens = t.tokenize("ab")
        assert tokens == ["ab"]

    def test_normalize(self):
        assert NgramTokenizer.normalize("HELLO") == "hello"

    def test_detect_n_cjk(self):
        t = NgramTokenizer(n=3)
        assert t._detect_n("你好世界") == 2

    def test_detect_n_latin(self):
        t = NgramTokenizer(n=2)
        assert t._detect_n("hello world") == 3

    def test_empty_text(self):
        t = NgramTokenizer(n=2)
        assert t.tokenize("") == []
        assert t.tokenize("   ") == []


class TestInvertedIndexMemory:
    def test_add_and_get_postings(self):
        idx = InvertedIndex()
        idx.add_document(0, ["alpha", "beta", "gamma"])
        idx.add_document(1, ["alpha", "beta"])
        idx.finalize(stop_threshold=1.0)
        postings = idx.get_postings("alpha")
        assert postings is not None
        docs, tfs = postings
        assert len(docs) == 2

    def test_doc_freq(self):
        idx = InvertedIndex()
        idx.add_document(0, ["alpha", "beta"])
        idx.add_document(1, ["alpha"])
        idx.finalize(stop_threshold=1.0)
        assert idx.doc_freq("alpha") == 2
        assert idx.doc_freq("beta") == 1
        assert idx.doc_freq("gamma") == 0

    def test_save_and_load(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(1, ["hello"])
        idx.finalize(stop_threshold=1.0)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            idx2 = InvertedIndex()
            idx2.load(path)
            assert idx2.N == idx.N
            assert idx2.doc_freq("hello") == 2
        finally:
            import os
            os.unlink(path)

    def test_cannot_add_after_finalize(self):
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize()
        with pytest.raises(RuntimeError):
            idx.add_document(1, ["b"])


class TestBM25ScorerMemory:
    def test_basic_score(self):
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async", "programming"])
        idx.add_document(1, ["python", "threading"])
        idx.finalize(stop_threshold=0.6)
        scorer = BM25Scorer(idx)
        scores = scorer.score(["async"])
        assert len(scores) == 1
        assert scores[0] > 0

    def test_empty_index(self):
        idx = InvertedIndex()
        scorer = BM25Scorer(idx)
        scores = scorer.score(["test"])
        assert scores == {}

    def test_candidate_docs_filter(self):
        idx = InvertedIndex()
        idx.add_document(0, ["alpha", "beta"])
        idx.add_document(1, ["alpha", "gamma"])
        idx.finalize(stop_threshold=1.0)
        scorer = BM25Scorer(idx)
        scores = scorer.score(["alpha"], candidate_docs={0})
        assert len(scores) == 1
        assert 0 in scores
        assert 1 not in scores


class TestLevenshteinAutomatonMemory:
    def test_exact_match(self):
        auto = LevenshteinAutomaton("hello", max_edits=0)
        assert auto.match(["hello"]) == ["hello"]

    def test_fuzzy_match(self):
        auto = LevenshteinAutomaton("hello", max_edits=1)
        results = auto.match(["hello", "hell", "hallo", "world"])
        assert "hello" in results
        assert "hell" in results
        assert "hallo" in results
        assert "world" not in results

    def test_auto_fuzziness(self):
        assert LevenshteinAutomaton.auto_fuzziness("ab") == 0
        assert LevenshteinAutomaton.auto_fuzziness("hello") == 1
        assert LevenshteinAutomaton.auto_fuzziness("hello world") == 2

    def test_damerau_levenshtein(self):
        auto = LevenshteinAutomaton("", max_edits=0)
        assert auto._damerau_levenshtein("hello", "hello") == 0
        assert auto._damerau_levenshtein("hello", "hell") == 1
        assert auto._damerau_levenshtein("ab", "ba") == 1  # transposition

    def test_max_expansions(self):
        auto = LevenshteinAutomaton("a", max_edits=1)
        dictionary = ["a", "b", "c", "d", "e"]
        results = auto.match(dictionary, max_expansions=2)
        assert len(results) <= 2

    def test_prefix_filter(self):
        auto = LevenshteinAutomaton("hello", max_edits=2, prefix_length=2)
        # 'ha' != 'he', so 'hallo' is filtered by prefix
        results = auto.match(["hallo", "world"])
        assert "hallo" not in results
        assert "world" not in results
        # 'helo' matches prefix 'he' and is within 1 edit
        results = auto.match(["hello", "helo", "world"])
        assert "hello" in results
        assert "helo" in results
        assert "world" not in results


class TestSearcherMemory:
    def test_basic_search(self):
        tokenizer = NgramTokenizer(n=3)
        idx = InvertedIndex()
        idx.add_document(0, tokenizer.tokenize("python async programming"))
        idx.add_document(1, tokenizer.tokenize("python threading model"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx)
        results = searcher.search("async programming", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0
        assert results[0][0] in {0, 1}

    def test_empty_index(self):
        idx = InvertedIndex()
        searcher = Searcher(idx)
        results = searcher.search("test")
        assert results == []

    def test_empty_query(self):
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.finalize()
        searcher = Searcher(idx)
        results = searcher.search("")
        assert results == []

    def test_min_should_match(self):
        idx = InvertedIndex()
        idx.add_document(0, ["python", "async"])
        idx.add_document(1, ["python", "threading"])
        idx.finalize()
        searcher = Searcher(idx, min_should_match=1.0)
        results = searcher.search("python nonexistent")
        assert len(results) == 0  # Only one token matches, need 100%

    def test_fuzzy_search(self):
        tokenizer = NgramTokenizer(n=3)
        idx = InvertedIndex()
        idx.add_document(0, tokenizer.tokenize("hello world"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, fuzziness=1)
        results = searcher.search("helo wrld")  # typo
        assert len(results) > 0

class TestNgramTokenizerEdgeCases:
    def test_tokenize_pure_cjk(self):
        t = NgramTokenizer()
        tokens = t.tokenize("中文测试")
        assert len(tokens) > 0
        assert all(len(tok) == 2 for tok in tokens)

    def test_tokenize_mixed_cjk_latin(self):
        t = NgramTokenizer()
        tokens = t.tokenize("Python编程")
        # Should auto-detect bigram for CJK-heavy text
        assert len(tokens) > 0

    def test_tokenize_single_char(self):
        t = NgramTokenizer()
        tokens = t.tokenize("a")
        assert tokens == ["a"]

    def test_tokenize_empty(self):
        t = NgramTokenizer()
        assert t.tokenize("") == []
        assert t.tokenize("   ") == []

    def test_normalize_unicode(self):
        t = NgramTokenizer()
        # NFKC should normalize full-width letters
        result = t.normalize("ＡＢＣ")
        assert result == "abc"


class TestInvertedIndexEdgeCases:
    def test_empty_index_finalize(self):
        idx = InvertedIndex()
        idx.finalize()
        assert idx.N == 0
        assert idx.avgdl == 0.0

    def test_add_document_empty_tokens(self):
        idx = InvertedIndex()
        idx.add_document(0, [])
        idx.finalize()
        assert idx.N == 1
        assert idx.avgdl == 0.0

    def test_get_postings_missing_term(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.finalize()
        assert idx.get_postings("missing") is None

    def test_doc_freq_missing_term(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        idx.finalize()
        assert idx.doc_freq("missing") == 0


class TestBM25ScorerEdgeCases:
    def test_empty_query(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.finalize()
        scorer = BM25Scorer(idx)
        assert scorer.score([]) == {}

    def test_single_document(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world", "hello"])
        idx.add_document(1, ["foo", "bar"])
        idx.finalize()
        scorer = BM25Scorer(idx)
        scores = scorer.score(["hello"])
        assert 0 in scores
        assert scores[0] > 0

    def test_candidate_docs_filter_excludes_all(self):
        idx = InvertedIndex()
        idx.add_document(0, ["a"])
        idx.add_document(1, ["b"])
        idx.finalize()
        scorer = BM25Scorer(idx)
        scores = scorer.score(["a"], candidate_docs={1})  # doc 1 has no 'a'
        assert scores == {}


class TestLevenshteinAutomatonEdgeCases:
    def test_zero_max_edits_exact_only(self):
        automaton = LevenshteinAutomaton("test", max_edits=0)
        results = automaton.match(["test", "tent", "best"])
        assert results == ["test"]

    def test_auto_fuzziness_short(self):
        assert LevenshteinAutomaton.auto_fuzziness("ab") == 0

    def test_auto_fuzziness_medium(self):
        assert LevenshteinAutomaton.auto_fuzziness("hello") == 1
        assert LevenshteinAutomaton.auto_fuzziness("abcdef") == 2

    def test_transposition_distance(self):
        dl = LevenshteinAutomaton._damerau_levenshtein
        assert dl("ab", "ba") == 1

    def test_empty_strings(self):
        dl = LevenshteinAutomaton._damerau_levenshtein
        assert dl("", "") == 0
        assert dl("abc", "") == 3


class TestSearcherEdgeCases:
    def test_search_cjk_query(self):
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        # Pure CJK text ensures bigram tokenization
        tokens = tokenizer.tokenize("异步编程指南")
        idx.add_document(0, tokens)
        idx.add_document(1, ["other"])
        idx.finalize()
        searcher = Searcher(idx, tokenizer=tokenizer)
        results = searcher.search("异步", top_k=5)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_search_no_match_due_to_min_should_match(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.finalize()
        searcher = Searcher(idx, min_should_match=1.0)
        results = searcher.search("foo bar", top_k=5)
        assert results == []

    def test_fuzzy_search_expansion(self):
        tokenizer = NgramTokenizer(n=3)
        idx = InvertedIndex()
        idx.add_document(0, tokenizer.tokenize("hello world"))
        idx.add_document(1, tokenizer.tokenize("foo bar"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer, fuzziness=1)
        results = searcher.search("helo", top_k=5)  # typo for hello
        assert len(results) == 1

    def test_search_top_k_larger_than_results(self):
        tokenizer = NgramTokenizer(n=3)
        idx = InvertedIndex()
        idx.add_document(0, tokenizer.tokenize("alpha beta"))
        idx.add_document(1, tokenizer.tokenize("gamma beta"))
        idx.finalize(stop_threshold=1.0)
        searcher = Searcher(idx, tokenizer=tokenizer)
        results = searcher.search("beta", top_k=100)
        assert len(results) == 2
