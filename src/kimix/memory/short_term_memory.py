"""Short-term memory: detailed current session records with temporal validity."""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider
from kimix.retrieval import jaro_winkler_similarity


class ShortTermMemory:
    __slots__ = ("max_size", "ttl", "buffer", "_evict_margin")

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.buffer: list[MemoryEntry] = []
        self._evict_margin = max(1, max_size // 10)

    def add(self, entry: MemoryEntry) -> None:
        entry.memory_type = MemoryType.EPISODIC
        self.buffer.append(entry)
        if len(self.buffer) >= self.max_size + self._evict_margin:
            self._evict_to_size()

    def _evict_to_size(self) -> None:
        buf = self.buffer
        excess = len(buf) - self.max_size
        if excess <= 0:
            return
        now = time.time()
        coeff = -0.1 / 86400.0
        vals = []
        append = vals.append
        for e in buf:
            delta = now - e.timestamp
            decay = 1.0 + coeff * delta if delta < 60.0 else math.exp(coeff * delta)
            boost = e.access_count * 0.1 if e.access_count < 20 else 2.0
            append(e.importance * decay * (1.0 + boost))
        if excess == 1:
            min_idx = vals.index(min(vals))
            buf[min_idx] = buf[-1]
            buf.pop()
        else:
            idxs = heapq.nsmallest(excess, range(len(vals)), key=vals.__getitem__)
            idxs.sort(reverse=True)
            for idx in idxs:
                buf[idx] = buf[-1]
                buf.pop()

    def _active_buffer(self, now: float | None = None) -> list[MemoryEntry]:
        if now is None:
            now = time.time()
        cutoff = now - self.ttl
        return [
            e for e in self.buffer
            if e.timestamp > cutoff and (e.expires_at is None or e.expires_at > now)
        ]

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        top_k: int = 5,
        query_vec: np.ndarray | None = None,
        use_string_fallback: bool = False,
    ) -> list[MemoryEntry]:
        now = time.time()
        active = self._active_buffer(now)
        if not active:
            return []

        if query_vec is None:
            query_vec = embedding_provider.embed(query)

        missing = [(i, entry.content) for i, entry in enumerate(active) if entry.embedding is None]
        if missing:
            indices, texts = zip(*missing)
            embeddings = embedding_provider.embed_batch(texts)
            for i, emb in zip(indices, embeddings):
                active[i].embedding = emb

        # Fast-path: embeddings are unit-normalized float32; use raw dot product.
        coeff = -0.1 / 86400.0
        qv = query_vec
        scored = []
        for i, entry in enumerate(active):
            emb = entry.embedding
            if emb is None:
                sim = 0.0
            else:
                sim = float(np.dot(qv, emb))
            delta = now - entry.timestamp
            recency = 1.0 + coeff * delta if delta < 3600.0 else math.exp(coeff * delta)
            boost = entry.access_count * 0.1 if entry.access_count < 20 else 2.0
            eff = entry.importance * recency * (1.0 + boost)
            scored.append((sim * eff, i, entry))
        results = [entry for _, _, entry in heapq.nlargest(top_k, scored)]

        # If all semantic scores are near-zero, optionally fall back to string similarity
        if use_string_fallback and results:
            max_semantic = max((s for s, _, _ in scored), default=0.0)
            if max_semantic < 0.1:
                string_scored = []
                for i, entry in enumerate(active):
                    delta = now - entry.timestamp
                    recency = 1.0 + coeff * delta if delta < 3600.0 else math.exp(coeff * delta)
                    boost = entry.access_count * 0.1 if entry.access_count < 20 else 2.0
                    eff = entry.importance * recency * (1.0 + boost)
                    string_scored.append((jaro_winkler_similarity(query, entry.content) * eff, i, entry))
                results = [entry for _, _, entry in heapq.nlargest(top_k, string_scored)]

        for entry in results:
            entry.touch(now)
        return results

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        now = time.time()
        cutoff = now - self.ttl
        return [e for _, _, e in heapq.nlargest(
            n,
            ((e.timestamp, i, e) for i, e in enumerate(self.buffer) if e.timestamp > cutoff and (e.expires_at is None or e.expires_at > now)),
        )]

    def _evict_least_valuable(self) -> None:
        """Alias for _evict_to_size for backward compatibility."""
        self._evict_to_size()

    def clear_expired(self) -> None:
        now = time.time()
        self.buffer[:] = self._active_buffer(now)
