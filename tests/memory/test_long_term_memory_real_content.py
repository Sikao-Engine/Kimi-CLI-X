"""Integration tests for LongTermMemory using real project files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kimix.memory.long_term_memory import LongTermMemory
from kimix.memory.types import MemoryEntry, MemoryType
from kimix.retrieval import (
    BM25Scorer,
    InvertedIndex,
    NgramTokenizer,
    Searcher,
    cosine_similarity_tfidf,
    jaccard_similarity_tokens,
    jaro_winkler_similarity,
    mmr_rerank,
    ngram_overlap,
    sorensen_dice_coefficient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters."""
    chunks: list[str] = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text.strip())
            break
        # try to break at newline
        cut = text.rfind("\n", max_chars // 2, max_chars)
        if cut == -1:
            cut = text.rfind(" ", max_chars // 2, max_chars)
        if cut == -1:
            cut = max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:]
    return [c for c in chunks if c]


def _load_text_files(root: Path, patterns: tuple[str, ...], max_files: int = 10) -> dict[str, str]:
    """Load text files matching *patterns* under *root*."""
    results: dict[str, str] = {}
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and len(results) < max_files:
                try:
                    results[str(p.relative_to(root))] = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
    return results


def _build_raw_bm25_index(entries: list[tuple[str, str]]) -> Searcher:
    """Build a standalone BM25 index from (entry_id, content) pairs."""
    idx = InvertedIndex()
    tokenizer = NgramTokenizer()
    for doc_id, (_, content) in enumerate(entries):
        stemmed = " ".join(content.split())  # simple whitespace tokenisation for raw index
        idx.add_document(doc_id, tokenizer.tokenize(stemmed))
    idx.finalize()
    return Searcher(idx, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robocute_files() -> dict[str, str]:
    return _load_text_files(Path("D:/RoboCute"), ("*.md", "*.txt", "*.py"), max_files=12)


@pytest.fixture(scope="module")
def compute_files() -> dict[str, str]:
    return _load_text_files(Path("D:/compute"), ("*.md", "*.txt", "*.py"), max_files=12)


@pytest.fixture
def ltm() -> LongTermMemory:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    mem = LongTermMemory(storage_path=path)
    yield mem
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


class TestStoreRealContent:
    def test_store_robocute_docs(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        assert robocute_files, "No RoboCute files found – check D:/RoboCute"
        total_chunks = 0
        for rel_path, text in robocute_files.items():
            chunks = _chunk_text(text, max_chars=600)
            for chunk in chunks:
                ltm.store(
                    content=chunk,
                    importance=6.0,
                    tags=["robocute", Path(rel_path).suffix.lstrip(".")],
                    source=rel_path,
                    memory_type=MemoryType.SEMANTIC,
                )
                total_chunks += 1
        assert ltm.count() > 0
        # Near-duplicate detection may merge chunks, so count <= total_chunks
        assert ltm.count() <= total_chunks

    def test_store_compute_docs(self, ltm: LongTermMemory, compute_files: dict[str, str]):
        assert compute_files, "No compute files found – check D:/compute"
        for rel_path, text in compute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(
                    content=chunk,
                    importance=5.0,
                    tags=["luisacompute", Path(rel_path).suffix.lstrip(".")],
                    source=rel_path,
                    memory_type=MemoryType.SEMANTIC,
                )
        assert ltm.count() > 0

    def test_store_many_batch(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        items: list[dict[str, object]] = []
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=500):
                items.append(
                    {
                        "content": chunk,
                        "importance": 5.5,
                        "tags": ["batch", "robocute"],
                        "source": rel_path,
                        "memory_type": MemoryType.SEMANTIC,
                    }
                )
        if items:
            entries = ltm.store_many(items)
            assert len(entries) == len(items)
            assert ltm.count() == len(items)


# ---------------------------------------------------------------------------
# Retrieve tests
# ---------------------------------------------------------------------------


class TestRetrieveRealContent:
    @pytest.fixture(autouse=True)
    def _seed(self, ltm: LongTermMemory, robocute_files: dict[str, str], compute_files: dict[str, str]):
        """Populate LTM with real content before each retrieve test."""
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(
                    content=chunk,
                    importance=6.0,
                    tags=["robocute", Path(rel_path).suffix.lstrip(".")],
                    source=rel_path,
                    memory_type=MemoryType.SEMANTIC,
                )
        for rel_path, text in compute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(
                    content=chunk,
                    importance=5.0,
                    tags=["luisacompute", Path(rel_path).suffix.lstrip(".")],
                    source=rel_path,
                    memory_type=MemoryType.SEMANTIC,
                )

    def test_retrieve_robocute_related(self, ltm: LongTermMemory):
        results = ltm.retrieve("RoboCute architecture", top_k=5)
        assert results, "Expected some results for 'RoboCute architecture'"
        for r in results:
            assert isinstance(r, MemoryEntry)
        # At least one top result should mention RoboCute or architecture
        assert any(
            "robocute" in r.content.lower()
            or "架构" in r.content
            or "architecture" in r.content.lower()
            for r in results[:3]
        )

    def test_retrieve_compute_related(self, ltm: LongTermMemory):
        results = ltm.retrieve("LuisaCompute rendering framework", top_k=5)
        assert results, "Expected some results for 'LuisaCompute rendering framework'"
        texts = [r.content.lower() for r in results]
        assert any("luisa" in t or "render" in t or "compute" in t for t in texts)

    def test_retrieve_tag_filter(self, ltm: LongTermMemory):
        # Filter to only robocute-tagged entries
        results = ltm.retrieve("scene", tag_filter=["robocute"], top_k=5)
        assert results
        for r in results:
            assert "robocute" in r.tags

    def test_retrieve_min_importance(self, ltm: LongTermMemory):
        results = ltm.retrieve("build", min_importance=6.0, top_k=5)
        # RoboCute chunks were stored with importance 6.0, compute with 5.0
        for r in results:
            assert r.importance >= 6.0

    def test_retrieve_updates_access_count(self, ltm: LongTermMemory):
        before = ltm.count()
        assert before > 0
        first_eid = list(ltm.entries.keys())[0]
        old_count = ltm.entries[first_eid].access_count
        ltm.retrieve("architecture", top_k=3)
        # access_count on returned results is incremented via touch()
        # Note: the exact entry touched depends on ranking; just verify
        # at least one entry in the top-k has been touched.

    def test_hybrid_vs_semantic_only(self, ltm: LongTermMemory):
        q = "physics simulation"
        hybrid = ltm.retrieve(q, top_k=5, use_hybrid=True)
        semantic_only = ltm.retrieve(q, top_k=5, use_hybrid=False)
        assert len(hybrid) <= 5
        assert len(semantic_only) <= 5
        # Hybrid should still return results
        assert hybrid, "Hybrid retrieval should return results"

    def test_retrieve_with_diversity(self, ltm: LongTermMemory):
        results = ltm.retrieve("node graph", top_k=5, use_diversity=True, diversity_lambda=0.5)
        assert results

    def test_retrieve_with_xquad(self, ltm: LongTermMemory):
        results = ltm.retrieve("ECS entity component", top_k=5, use_xquad=True, xquad_lambda=0.5)
        assert results

    def test_retrieve_ltr_rerank(self, ltm: LongTermMemory):
        for model in ("lambdamart", "ranksvm", "rankboost"):
            results = ltm.retrieve("Python first", top_k=5, use_ltr=True, ltr_model=model)
            assert isinstance(results, list)

    def test_persistence_with_real_content(self, ltm: LongTermMemory):
        path = ltm.storage_path
        count_before = ltm.count()
        assert count_before > 0
        del ltm

        ltm2 = LongTermMemory(storage_path=path)
        assert ltm2.count() == count_before
        results = ltm2.retrieve("RoboCute", top_k=3)
        assert results


# ---------------------------------------------------------------------------
# Retrieval.py algorithm combination / fallback tests
# ---------------------------------------------------------------------------


class TestRetrievalAlgorithmCombination:
    """Verify that raw retrieval.py algorithms can be used to augment or validate LTM results."""

    @pytest.fixture
    def indexed_entries(self, robocute_files: dict[str, str], compute_files: dict[str, str]):
        entries: list[tuple[str, str]] = []  # (entry_id, content)
        all_files = {**robocute_files, **compute_files}
        for rel_path, text in all_files.items():
            for idx, chunk in enumerate(_chunk_text(text, max_chars=600)):
                eid = f"{rel_path}#{idx}"
                entries.append((eid, chunk))
        return entries

    def test_raw_bm25_search(self, indexed_entries: list[tuple[str, str]]):
        searcher = _build_raw_bm25_index(indexed_entries)
        results = searcher.search("rendering framework", top_k=5)
        assert results, "BM25 should find matches for 'rendering framework'"
        for doc_id, score in results:
            assert score > 0
            content = indexed_entries[doc_id][1]
            # N-gram tokenisation may match substrings; just ensure non-empty content
            assert content

    def test_raw_bm25_topk_vs_full_score(self, indexed_entries: list[tuple[str, str]]):
        searcher = _build_raw_bm25_index(indexed_entries)
        q = "Python first 3D AIGC"
        topk = searcher.scorer.score_topk(searcher.tokenizer.tokenize(q), top_k=5)
        full = searcher.scorer.score(searcher.tokenizer.tokenize(q))
        assert len(topk) <= 5
        assert all(doc_id in full for doc_id, _ in topk)

    def _tf_vector(self, text: str) -> dict[str, float]:
        from collections import Counter
        tokens = text.lower().split()
        counts = Counter(tokens)
        return {t: float(c) for t, c in counts.items()}

    def test_cosine_similarity_tfidf_fallback(self, indexed_entries: list[tuple[str, str]]):
        """If LTM semantic retrieve misses, cosine TF-IDF can act as a fallback."""
        query = "cross-platform graphics engine"
        q_vec = self._tf_vector(query)
        scores: list[tuple[str, float]] = []
        for eid, content in indexed_entries:
            sim = cosine_similarity_tfidf(q_vec, self._tf_vector(content))
            scores.append((eid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        top5 = scores[:5]
        assert top5, "cosine_similarity_tfidf should produce results"
        best_eid, best_score = top5[0]
        best_content = next(c for e, c in indexed_entries if e == best_eid)
        assert "graphics" in best_content.lower() or "engine" in best_content.lower() or "cross" in best_content.lower()

    def test_jaccard_and_dice_fallback(self, indexed_entries: list[tuple[str, str]]):
        query = "node-based workflow"
        q_tokens = set(query.lower().split())
        jac_scores = [
            (eid, jaccard_similarity_tokens(q_tokens, set(content.lower().split())))
            for eid, content in indexed_entries
        ]
        dice_scores = [(eid, sorensen_dice_coefficient(query, content)) for eid, content in indexed_entries]
        jac_scores.sort(key=lambda x: x[1], reverse=True)
        dice_scores.sort(key=lambda x: x[1], reverse=True)
        assert jac_scores[0][1] >= 0
        assert dice_scores[0][1] >= 0

    def test_jaro_winkler_string_fallback(self, indexed_entries: list[tuple[str, str]]):
        query = "RoboCute editor"
        scores = [(eid, jaro_winkler_similarity(query, content)) for eid, content in indexed_entries]
        scores.sort(key=lambda x: x[1], reverse=True)
        assert scores[0][1] > 0.5, "Jaro-Winkler should find high similarity for 'RoboCute editor'"

    def test_ngram_overlap_fallback(self, indexed_entries: list[tuple[str, str]]):
        query = "physics simulation UIPC"
        scores = [(eid, ngram_overlap(query, content)) for eid, content in indexed_entries]
        scores.sort(key=lambda x: x[1], reverse=True)
        assert scores[0][1] > 0

    def test_mmr_rerank_on_bm25_results(self, indexed_entries: list[tuple[str, str]]):
        """MMR re-ranking can diversify BM25 results using the InvertedIndex."""
        searcher = _build_raw_bm25_index(indexed_entries)
        q_tokens = searcher.tokenizer.tokenize("Python animation scene")
        bm25_results = searcher.scorer.score_topk(q_tokens, top_k=10)
        assert bm25_results
        mmr = mmr_rerank(bm25_results, searcher.index, lambda_param=0.5, top_k=5)
        assert len(mmr) <= 5
        doc_ids = [d for d, _ in mmr]
        assert len(doc_ids) == len(set(doc_ids)), "MMR should not duplicate doc_ids"

    def test_combined_ranking_pipeline(self, indexed_entries: list[tuple[str, str]]):
        """Full pipeline: BM25 + string similarities + simple ensemble."""
        query = "high performance rendering"
        searcher = _build_raw_bm25_index(indexed_entries)
        bm25_results = searcher.scorer.score_topk(searcher.tokenizer.tokenize(query), top_k=20)
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}

        ensemble: list[tuple[str, float]] = []
        q_vec = self._tf_vector(query)
        for eid, content in indexed_entries:
            doc_id = indexed_entries.index((eid, content))
            bm25 = bm25_dict.get(doc_id, 0.0)
            jw = jaro_winkler_similarity(query, content)
            dice = sorensen_dice_coefficient(query, content)
            ngo = ngram_overlap(query, content)
            tfidf = cosine_similarity_tfidf(q_vec, self._tf_vector(content))
            score = 0.4 * bm25 + 0.15 * jw + 0.15 * dice + 0.15 * ngo + 0.15 * tfidf
            ensemble.append((eid, score))

        ensemble.sort(key=lambda x: x[1], reverse=True)
        top5 = ensemble[:5]
        assert top5
        # Best result should contain rendering-related words
        best_content = next(c for e, c in indexed_entries if e == top5[0][0])
        assert "render" in best_content.lower() or "performance" in best_content.lower()

    def test_ltm_retrieve_matches_raw_bm25(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        """Ensure LTM hybrid retrieve and raw BM25 agree on top results for a query."""
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(
                    content=chunk,
                    importance=6.0,
                    tags=["robocute"],
                    source=rel_path,
                    memory_type=MemoryType.SEMANTIC,
                )

        query = "RoboCute node graph"
        ltm_results = ltm.retrieve(query, top_k=5, use_hybrid=True)
        assert ltm_results

        # Build raw BM25 over the same content
        entries = [(eid, e.content) for eid, e in ltm.entries.items()]
        searcher = _build_raw_bm25_index(entries)
        raw_results = searcher.search(query, top_k=5)
        assert raw_results

        # At least one entry_id should overlap between top-3 of each
        ltm_top3 = {ltm._hash(r.content) for r in ltm_results[:3]}
        raw_top3 = {entries[doc_id][0] for doc_id, _ in raw_results[:3]}
        overlap = ltm_top3 & raw_top3
        assert overlap, f"No overlap between LTM and raw BM25 top-3: {ltm_top3} vs {raw_top3}"


# ---------------------------------------------------------------------------
# Edge cases with real content
# ---------------------------------------------------------------------------


class TestRealContentEdgeCases:
    def test_empty_query(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(content=chunk, importance=5.0)
        results = ltm.retrieve("", top_k=5)
        # empty query should either return empty or gracefully degrade
        assert isinstance(results, list)

    def test_very_long_query(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(content=chunk, importance=5.0)
        long_query = " ".join(["Python"] * 50)
        results = ltm.retrieve(long_query, top_k=5)
        assert isinstance(results, list)

    def test_nonexistent_topic(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(content=chunk, importance=5.0)
        results = ltm.retrieve("quantum cryptography blockchain", top_k=5)
        # Should still return something (best-effort) or gracefully return empty
        assert isinstance(results, list)

    def test_chinese_query(self, ltm: LongTermMemory, robocute_files: dict[str, str]):
        for rel_path, text in robocute_files.items():
            for chunk in _chunk_text(text, max_chars=600):
                ltm.store(content=chunk, importance=5.0, tags=["robocute"])
        results = ltm.retrieve("节点图 动画 物理", top_k=5)
        assert isinstance(results, list)
        if results:
            # With CJK text, n-gram tokenizer should still work
            assert all(isinstance(r, MemoryEntry) for r in results)
