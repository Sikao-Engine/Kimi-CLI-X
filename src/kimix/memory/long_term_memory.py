"""Long-term memory: persistent storage with hybrid semantic + BM25 retrieval."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import time
import numpy as np

from kimix.memory.types import MemoryEntry, MemoryType, _DECAY_COEFF
from kimix.memory.embedding import EmbeddingProvider
from kimix.retrieval import (
    BM25Scorer,
    CoordinateAscent,
    InvertedIndex,
    LambdaMART,
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
    jaro_similarity,
    jaro_winkler_similarity,
    metaphone,
    mmr_rerank,
    ngram_overlap,
    porter_stem,
    scq,
    sorensen_dice_coefficient,
    soundex,
    xquad_rerank,
)
from kimix.memory.short_term_memory import ShortTermMemory


class LongTermMemory:
    __slots__ = (
        "storage_path", "dim", "entries", "index", "embedding_provider",
        "_dirty", "_backend", "_agent_id", "_bm25_index", "_bm25_searcher",
        "_doc_id_map", "_bm25_doc_to_entry_id", "_next_doc_id",
        "_simhash_map", "_near_dup_threshold",
        "_minhash_map", "_imatch_map", "_soundex_map", "_metaphone_map",
        "_speller",
    )

    def __init__(
        self,
        storage_path: str | None = None,
        dim: int = 384,
        backend: "kimix.memory.sqlite_backend.SQLiteBackend" | None = None,
        agent_id: str = "default",
    ) -> None:
        self.storage_path = storage_path or "ltm.json"
        if backend is None and (not isinstance(self.storage_path, str) or not self.storage_path):
            raise ValueError("storage_path must be a non-empty string")
        if backend is None:
            parent = os.path.dirname(self.storage_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.dim = dim
        self.entries: dict[str, MemoryEntry] = {}
        self.index: dict[str, set[str]] = {}
        self.embedding_provider = EmbeddingProvider(dim)
        self._dirty = False
        self._backend = backend
        self._agent_id = agent_id

        self._bm25_index: InvertedIndex | None = None
        self._bm25_searcher: Searcher | None = None
        self._doc_id_map: dict[str, int] = {}
        self._bm25_doc_to_entry_id: list[str] = []
        self._next_doc_id = 0
        self._simhash_map: dict[str, SimHash] = {}
        self._near_dup_threshold = 3

        self._minhash_map: dict[str, MinHash] = {}
        self._imatch_map: dict[str, str] = {}
        self._soundex_map: dict[str, set[str]] = {}
        self._metaphone_map: dict[str, set[str]] = {}
        self._speller: NoisyChannelSpeller | None = None

        self._load()

    def _hash(self, content: str) -> str:
        return hashlib.blake2b(content.encode(), digest_size=8).hexdigest()

    def _update_index(self, entry_id: str, entry: MemoryEntry) -> None:
        for tag in entry.tags:
            self.index.setdefault(tag, set()).add(entry_id)

    def _invalidate_bm25(self) -> None:
        self._bm25_index = None
        self._bm25_searcher = None
        self._bm25_doc_to_entry_id = []
        self._doc_id_map = {}
        self._next_doc_id = 0
        self._speller = None

    def _find_near_duplicate(self, content: str) -> str | None:
        """Return entry_id of a near-duplicate entry, or None."""
        if self._backend is not None:
            return None
        h = SimHash(content)
        mh = MinHash(content)
        fp = i_match_fingerprint(content.split())
        sx = soundex(content.split()[0]) if content.split() else ""
        mp = metaphone(content.split()[0]) if content.split() else ""
        for eid, other in self._simhash_map.items():
            if h.is_near_duplicate(other, threshold=self._near_dup_threshold):
                return eid
            # MinHash near-duplicate check
            if eid in self._minhash_map:
                if mh.jaccard(self._minhash_map[eid]) >= 0.8:
                    return eid
            # I-Match exact fingerprint
            if eid in self._imatch_map and self._imatch_map[eid] == fp:
                return eid
            # Phonetic dedup on first word
            if sx and sx in self._soundex_map and eid in self._soundex_map[sx]:
                return eid
            if mp and mp in self._metaphone_map and eid in self._metaphone_map[mp]:
                return eid
        return None

    def _insert_entry(self, entry_id: str, entry: MemoryEntry, *, invalidate_bm25: bool = True) -> None:
        if self._backend is not None:
            self._backend.store(entry, entry_id, dim=self.dim)
        else:
            dup_id = self._find_near_duplicate(entry.content)
            if dup_id is not None:
                existing = self.entries.get(dup_id)
                if existing is not None:
                    existing.importance = min(10.0, existing.importance + entry.importance * 0.5)
                    existing.touch()
                    if entry.tags:
                        existing.tags = list(dict.fromkeys(existing.tags + entry.tags))
                    self._dirty = True
                    return
            self.entries[entry_id] = entry
            self._update_index(entry_id, entry)
            self._simhash_map[entry_id] = SimHash(entry.content)
            self._minhash_map[entry_id] = MinHash(entry.content)
            tokens = entry.content.split()
            self._imatch_map[entry_id] = i_match_fingerprint(tokens)
            first = tokens[0] if tokens else ""
            self._soundex_map.setdefault(soundex(first), set()).add(entry_id)
            self._metaphone_map.setdefault(metaphone(first), set()).add(entry_id)
            self._dirty = True
        if invalidate_bm25:
            self._invalidate_bm25()

    def _build_bm25(self) -> Searcher:
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        self._doc_id_map = {}
        self._bm25_doc_to_entry_id = []
        self._next_doc_id = 0
        if self._backend is not None:
            now = time.time()
            for eid, content, expires_at in self._backend.iter_rows(
                agent_id=self._agent_id, exclude_expired=False
            ):
                if expires_at is not None and expires_at <= now:
                    continue
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._doc_id_map[eid] = doc_id
                self._bm25_doc_to_entry_id.append(eid)
                stemmed = " ".join(porter_stem(w) for w in content.split())
                idx.add_document(doc_id, tokenizer.tokenize(stemmed))
        else:
            for eid, entry in self.entries.items():
                if entry.is_expired():
                    continue
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._doc_id_map[eid] = doc_id
                self._bm25_doc_to_entry_id.append(eid)
                stemmed = " ".join(porter_stem(w) for w in entry.content.split())
                idx.add_document(doc_id, tokenizer.tokenize(stemmed))
        idx.finalize()
        self._bm25_index = idx
        self._bm25_searcher = Searcher(idx, tokenizer=tokenizer)
        return self._bm25_searcher

    def _ensure_bm25(self) -> Searcher:
        if self._bm25_searcher is None:
            return self._build_bm25()
        return self._bm25_searcher

    def _iter_entries(self):
        if self._backend is not None:
            for eid, entry in self._backend.list_all(
                agent_id=self._agent_id,
                exclude_expired=False,
                dim=self.dim,
                include_embedding=True,
            ):
                yield eid, entry
        else:
            for eid, entry in self.entries.items():
                yield eid, entry

    def _get_entry(self, entry_id: str) -> MemoryEntry | None:
        if self._backend is not None:
            entry = self._backend.get(entry_id, dim=self.dim, include_embedding=True)
            if entry is not None and entry.agent_id != self._agent_id:
                return None
            return entry
        return self.entries.get(entry_id)

    @staticmethod
    def _valid(entry: MemoryEntry, now: float, min_importance: float) -> bool:
        return (entry.expires_at is None or entry.expires_at > now) and entry.importance >= min_importance

    def _load(self) -> None:
        if self._backend is not None:
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                entry = MemoryEntry.from_dict(item)
                if entry.agent_id != self._agent_id:
                    continue
                emb = entry.embedding
                if isinstance(emb, list):
                    arr = np.asarray(emb, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    if norm:
                        arr /= norm
                    entry.embedding = arr
                entry_id = self._hash(entry.content)
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
                self._simhash_map[entry_id] = SimHash(entry.content)
                self._minhash_map[entry_id] = MinHash(entry.content)
                tokens = entry.content.split()
                self._imatch_map[entry_id] = i_match_fingerprint(tokens)
                first = tokens[0] if tokens else ""
                self._soundex_map.setdefault(soundex(first), set()).add(entry_id)
                self._metaphone_map.setdefault(metaphone(first), set()).add(entry_id)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        if self._backend is not None:
            return
        if not self._dirty:
            return
        data = [e.to_dict() for e in self.entries.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        self._dirty = False

    def store(
        self,
        content: str,
        importance: float = 5.0,
        tags: list[str] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        source: str = "",
        metadata: dict[str, object] | None = None,
        expires_at: float | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            source=source,
            metadata=metadata or {},
            expires_at=expires_at,
            agent_id=self._agent_id,
        )
        entry.embedding = self.embedding_provider.embed(content)
        entry_id = self._hash(content)
        self._insert_entry(entry_id, entry)
        self._save()
        return entry

    def store_many(
        self,
        items: list[dict[str, object]],
    ) -> list[MemoryEntry]:
        results: list[MemoryEntry] = []
        batch: list[tuple[str, MemoryEntry]] = []
        embed = self.embedding_provider.embed
        for item in items:
            content = str(item["content"])
            entry = MemoryEntry(
                content=content,
                memory_type=item.get("memory_type", MemoryType.SEMANTIC),  # type: ignore[arg-type]
                importance=float(item.get("importance", 5.0)),
                tags=list(item.get("tags", []) or []),
                source=str(item.get("source", "")),
                metadata=dict(item.get("metadata", {}) or {}),
                expires_at=item.get("expires_at"),
                agent_id=self._agent_id,
            )
            entry.embedding = embed(content)
            entry_id = self._hash(content)
            batch.append((entry_id, entry))
            results.append(entry)

        if self._backend is not None and batch:
            self._backend.store_many(batch, dim=self.dim)
        else:
            for entry_id, entry in batch:
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
            self._dirty = True

        self._invalidate_bm25()
        self._save()
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        min_importance: float = 0.0,
        use_hybrid: bool = True,
        bm25_weight: float = 0.3,
        query_vec: np.ndarray | None = None,
        use_diversity: bool = True,
        diversity_lambda: float = 0.5,
        use_rm3: bool = True,
        rm3_fb_docs: int = 3,
        rm3_fb_terms: int = 10,
        rm3_alpha: float = 0.5,
        adaptive_bm25: bool = True,
        use_spelling: bool = True,
        use_stemming: bool = True,
        use_rocchio: bool = True,
        rocchio_alpha: float = 1.0,
        rocchio_beta: float = 0.75,
        rocchio_gamma: float = 0.15,
        use_xquad: bool = True,
        xquad_lambda: float = 0.5,
        use_string_similarity: bool = True,
        use_ltr: bool = True,
        ltr_model: str = "lambdamart",
    ) -> list[MemoryEntry]:
        if self._backend is None and not self.entries:
            return []

        if query_vec is None:
            query_vec = self.embedding_provider.embed(query)
        now = time.time()

        # --- Query preprocessing: spelling + stemming ---
        original_query = query
        if use_spelling:
            if self._speller is None and self._bm25_index is not None:
                term_freqs: dict[str, int] = {}
                for term in self._bm25_index.terms():
                    postings = self._bm25_index.get_postings(term)
                    if postings is not None:
                        term_freqs[term] = int(postings[1].sum())
                if term_freqs:
                    self._speller = NoisyChannelSpeller(term_freqs, max_edits=2)
            if self._speller is not None:
                corrected_words = []
                for w in query.split():
                    cw = self._speller.correct(w)
                    corrected_words.append(cw if cw else w)
                query = " ".join(corrected_words)
        if use_stemming:
            query = " ".join(porter_stem(w) for w in query.split())

        candidates: list[MemoryEntry] = []
        candidate_ids: list[str] = []
        if tag_filter:
            if self._backend is not None:
                raw = self._backend.search_by_tag(
                    tag_filter, agent_id=self._agent_id, dim=self.dim, include_embedding=True
                )
                for eid, entry in raw:
                    if not self._valid(entry, now, min_importance):
                        continue
                    candidates.append(entry)
                    candidate_ids.append(eid)
            else:
                filtered_ids: set[str] | None = None
                for tag in tag_filter:
                    ids = self.index.get(tag)
                    if ids is None:
                        return []
                    if filtered_ids is None:
                        filtered_ids = set(ids)
                    else:
                        filtered_ids.intersection_update(ids)
                if not filtered_ids:
                    return []
                for eid in filtered_ids:
                    entry = self.entries.get(eid)
                    if entry is not None and self._valid(entry, now, min_importance):
                        candidates.append(entry)
                        candidate_ids.append(eid)
        else:
            for eid, entry in self._iter_entries():
                if self._valid(entry, now, min_importance):
                    candidates.append(entry)
                    candidate_ids.append(eid)

        if not candidates or top_k <= 0:
            return []

        n_cand = len(candidates)

        missing = [entry for entry in candidates if entry.embedding is None]
        if missing:
            texts = [e.content for e in missing]
            vecs = self.embedding_provider.embed_batch(texts)
            for entry, vec in zip(missing, vecs):
                entry.embedding = vec

        try:
            embeddings = np.stack([entry.embedding for entry in candidates])
        except (TypeError, ValueError):
            embeddings = np.array([entry.embedding for entry in candidates], dtype=np.float32)

        query_arr = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0:
            semantic_arr = np.zeros(n_cand, dtype=np.float64)
        else:
            dots = embeddings @ query_arr
            norms = np.linalg.norm(embeddings, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(norms == 0, 0.0, dots / (norms * q_norm))

            timestamps = np.empty(n_cand, dtype=np.float64)
            access_counts = np.empty(n_cand, dtype=np.float64)
            importances = np.empty(n_cand, dtype=np.float64)
            for i, entry in enumerate(candidates):
                timestamps[i] = entry.timestamp
                access_counts[i] = entry.access_count
                importances[i] = entry.importance
            recency = np.exp(_DECAY_COEFF * (now - timestamps))
            access_boost = np.minimum(access_counts * 0.1, 2.0)
            eff = importances * recency * (1.0 + access_boost)
            semantic_arr = sims.astype(np.float64) * eff

        bm25_arr = np.zeros(n_cand, dtype=np.float64)
        string_arr = np.zeros(n_cand, dtype=np.float64)
        _bm25_weight = bm25_weight
        if use_hybrid:
            searcher = self._ensure_bm25()
            if adaptive_bm25 and self._bm25_index is not None:
                scorer = BM25Scorer(self._bm25_index)
                qpp = QueryPerformancePredictor(self._bm25_index, scorer)
                q_tokens = searcher.tokenizer.tokenize(query)
                if qpp.is_hard_query(q_tokens, avg_idf_threshold=2.0):
                    _bm25_weight = min(0.6, bm25_weight + 0.2)
                else:
                    _bm25_weight = max(0.1, bm25_weight - 0.1)
                # Additional QPP signals
                if len(q_tokens) > 0:
                    clarity = clarity_score(self._bm25_index, q_tokens)
                    scq_val = scq(self._bm25_index, q_tokens)
                    if clarity > 1.0 or scq_val > 5.0:
                        _bm25_weight = min(0.7, _bm25_weight + 0.1)
                    elif clarity < 0.5:
                        _bm25_weight = max(0.1, _bm25_weight - 0.1)

            # Query expansion: RM3 + Rocchio
            expanded_tokens: list[str] = []
            if self._bm25_index is not None:
                scorer = BM25Scorer(self._bm25_index)
                q_tokens = searcher.tokenizer.tokenize(query)
                if use_rm3:
                    rm3 = RM3Expander(
                        self._bm25_index,
                        scorer,
                        fb_docs=rm3_fb_docs,
                        fb_terms=rm3_fb_terms,
                        alpha=rm3_alpha,
                    )
                    expanded_tokens.extend(rm3.expand(q_tokens, top_k=rm3_fb_docs))
                if use_rocchio:
                    rocchio = RocchioExpander(
                        self._bm25_index,
                        scorer,
                        alpha=rocchio_alpha,
                        beta=rocchio_beta,
                        gamma=rocchio_gamma,
                        fb_docs=rm3_fb_docs,
                        fb_terms=rm3_fb_terms,
                    )
                    expanded_tokens.extend(rocchio.expand(q_tokens))
                if not expanded_tokens:
                    expanded_tokens = q_tokens
                bm25_results = searcher.scorer.score_topk(expanded_tokens, top_k=n_cand)
            else:
                bm25_results = searcher.search(query, top_k=n_cand)

            if bm25_results:
                scores = [score for _, score in bm25_results]
                max_bm25 = max(scores)
                min_bm25 = min(scores)
                bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
                eid_to_idx = {eid: i for i, eid in enumerate(candidate_ids)}
                doc_to_eid = self._bm25_doc_to_entry_id
                for doc_id, score in bm25_results:
                    idx = eid_to_idx.get(doc_to_eid[doc_id])
                    if idx is not None:
                        bm25_arr[idx] = (score - min_bm25) / bm25_range

            # String similarity features
            if use_string_similarity:
                for i, entry in enumerate(candidates):
                    jw = jaro_winkler_similarity(original_query, entry.content)
                    dice = sorensen_dice_coefficient(original_query, entry.content)
                    ngo = ngram_overlap(original_query, entry.content)
                    string_arr[i] = (jw + dice + ngo) / 3.0

        final = (
            (1.0 - _bm25_weight) * semantic_arr
            + _bm25_weight * bm25_arr
            + 0.1 * string_arr
        )
        if top_k * 4 < n_cand:
            top_indices = np.argpartition(final, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(final[top_indices])[::-1]]
        else:
            top_indices = np.argsort(final)[::-1][:top_k]
        top_indices = top_indices.tolist()

        # --- LTR re-ranking ---
        if use_ltr and n_cand >= 2:
            features: list[list[float]] = []
            labels: list[float] = []
            for i in top_indices:
                feat = [
                    float(semantic_arr[i]),
                    float(bm25_arr[i]),
                    float(string_arr[i]),
                    candidates[i].importance / 10.0,
                    math.exp(_DECAY_COEFF * (now - candidates[i].timestamp)),
                    min(candidates[i].access_count * 0.1, 2.0) / 2.0,
                ]
                features.append(feat)
                labels.append(float(final[i]))
            if features:
                doc_features = [(i, f) for i, f in zip(top_indices, features)]
                if ltr_model == "lambdamart":
                    model: LambdaMART | RankSVM | RankBoost = LambdaMART(n_iterations=10, learning_rate=0.05)
                elif ltr_model == "ranksvm":
                    model = RankSVM(learning_rate=0.01, n_iterations=50)
                else:
                    model = RankBoost(n_iterations=20)
                # Fit on a single query with synthetic labels
                if ltr_model == "lambdamart":
                    model.fit([features], [labels])  # type: ignore[arg-type]
                else:
                    model.fit(features, labels)  # type: ignore[arg-type]
                ranked = model.rank(doc_features)  # type: ignore[arg-type]
                top_indices = [idx for idx, _ in ranked]

        # --- Diversification: xQuAD + MMR ---
        results: list[MemoryEntry]
        if use_xquad and self._bm25_index is not None:
            eid_to_doc_id = self._doc_id_map
            doc_to_eid = self._bm25_doc_to_entry_id
            ranked_docs = []
            for i in top_indices:
                eid = candidate_ids[i]
                doc_id = eid_to_doc_id.get(eid)
                if doc_id is not None:
                    ranked_docs.append((doc_id, float(final[i])))
            if ranked_docs:
                aspects: dict[int, set[str]] = {}
                for i in top_indices:
                    eid = candidate_ids[i]
                    doc_id = eid_to_doc_id.get(eid)
                    if doc_id is not None:
                        entry = candidates[i]
                        aspects[doc_id] = set(entry.tags)
                xq_results = xquad_rerank(
                    ranked_docs, aspects, lambda_param=xquad_lambda, top_k=top_k
                )
                xq_eids = [doc_to_eid[doc_id] for doc_id, _ in xq_results if doc_id < len(doc_to_eid)]
                id_to_entry = {candidate_ids[i]: candidates[i] for i in top_indices}
                results = [id_to_entry[eid] for eid in xq_eids if eid in id_to_entry]
            else:
                results = [candidates[i] for i in top_indices]
        elif use_diversity and self._bm25_index is not None:
            eid_to_doc_id = self._doc_id_map
            doc_to_eid = self._bm25_doc_to_entry_id
            ranked_docs = []
            for i in top_indices:
                eid = candidate_ids[i]
                doc_id = eid_to_doc_id.get(eid)
                if doc_id is not None:
                    ranked_docs.append((doc_id, float(final[i])))
            if ranked_docs:
                mmr_results = mmr_rerank(
                    ranked_docs, self._bm25_index, lambda_param=diversity_lambda, top_k=top_k
                )
                mmr_eids = [doc_to_eid[doc_id] for doc_id, _ in mmr_results if doc_id < len(doc_to_eid)]
                id_to_entry = {candidate_ids[i]: candidates[i] for i in top_indices}
                results = [id_to_entry[eid] for eid in mmr_eids if eid in id_to_entry]
            else:
                results = [candidates[i] for i in top_indices]
        else:
            results = [candidates[i] for i in top_indices]

        if self._backend is not None:
            eids = [candidate_ids[i] for i in top_indices]
            self._backend.update_access_many(eids)
        for entry in results:
            entry.touch(now)

        return results

    def consolidate(
        self,
        short_term: ShortTermMemory,
        threshold: float = 7.0,
    ) -> None:
        if not isinstance(short_term, ShortTermMemory):
            raise TypeError("short_term must be a ShortTermMemory instance")

        to_migrate = [
            entry for entry in short_term.buffer
            if entry.get_effective_importance() >= threshold and not entry.is_expired()
        ]
        if not to_migrate:
            return

        batch: list[tuple[str, MemoryEntry]] = []
        for entry in to_migrate:
            entry_id = self._hash(entry.content)
            if entry.embedding is None:
                entry.embedding = self.embedding_provider.embed(entry.content)
            entry.agent_id = self._agent_id
            batch.append((entry_id, entry))

        if self._backend is not None and batch:
            self._backend.store_many(batch, dim=self.dim)
        else:
            for entry_id, entry in batch:
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
            self._dirty = True

        self._invalidate_bm25()

        migrate_ids = {id(entry) for entry in to_migrate}
        short_term.buffer = [e for e in short_term.buffer if id(e) not in migrate_ids]

        self._save()

    def _remove_from_dedup_maps(self, entry_id: str, entry: MemoryEntry) -> None:
        self._simhash_map.pop(entry_id, None)
        self._minhash_map.pop(entry_id, None)
        self._imatch_map.pop(entry_id, None)
        tokens = entry.content.split()
        first = tokens[0] if tokens else ""
        sx = soundex(first)
        mp = metaphone(first)
        if sx and sx in self._soundex_map:
            self._soundex_map[sx].discard(entry_id)
            if not self._soundex_map[sx]:
                del self._soundex_map[sx]
        if mp and mp in self._metaphone_map:
            self._metaphone_map[mp].discard(entry_id)
            if not self._metaphone_map[mp]:
                del self._metaphone_map[mp]

    def forget(self, entry_id: str) -> None:
        if self._backend is not None:
            entry = self._backend.get(entry_id, dim=self.dim)
            if entry is None:
                return
            entry.importance *= 0.5
            if entry.importance < 0.1:
                self._backend.delete(entry_id)
            else:
                self._backend.store(entry, entry_id, dim=self.dim)
            self._invalidate_bm25()
            return

        entry = self.entries.get(entry_id)
        if entry is None:
            return
        entry.importance *= 0.5
        if entry.importance < 0.1:
            del self.entries[entry_id]
            for tag in entry.tags:
                tag_set = self.index.get(tag)
                if tag_set is not None:
                    tag_set.discard(entry_id)
                    if not tag_set:
                        del self.index[tag]
            self._remove_from_dedup_maps(entry_id, entry)
        self._dirty = True
        self._invalidate_bm25()
        self._save()

    def forget_many(self, entry_ids: list[str]) -> None:
        if self._backend is not None:
            for entry_id in entry_ids:
                entry = self._backend.get(entry_id, dim=self.dim)
                if entry is None:
                    continue
                entry.importance *= 0.5
                if entry.importance < 0.1:
                    self._backend.delete(entry_id)
                else:
                    self._backend.store(entry, entry_id, dim=self.dim)
            self._invalidate_bm25()
            return

        changed = False
        for entry_id in entry_ids:
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            entry.importance *= 0.5
            if entry.importance < 0.1:
                del self.entries[entry_id]
                for tag in entry.tags:
                    tag_set = self.index.get(tag)
                    if tag_set is not None:
                        tag_set.discard(entry_id)
                        if not tag_set:
                            del self.index[tag]
                self._remove_from_dedup_maps(entry_id, entry)
            changed = True

        if changed:
            self._dirty = True
            self._invalidate_bm25()
            self._save()

    def count(self) -> int:
        if self._backend is not None:
            return self._backend.count(agent_id=self._agent_id)
        return len(self.entries)
