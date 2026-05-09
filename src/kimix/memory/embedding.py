"""Deterministic hashing-trick embedding vectors."""

import functools
import hashlib
import struct
import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np

__all__ = ("EmbeddingProvider",)


class EmbeddingProvider:
    __slots__ = ("dim", "_cache", "_max_cache_size", "_lock", "_local")

    def __init__(self, dim: int = 384, max_cache_size: int = 4096) -> None:
        self.dim = dim
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._lock = threading.Lock()
        self._local = threading.local()

    def _normalize_key(self, text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    @functools.lru_cache(maxsize=65_536)
    def _hash_token(token: str) -> int:
        # C-accelerated hash: MD5 first 8 bytes as little-endian uint64
        return struct.unpack("<Q", hashlib.md5(token.encode("utf-8")).digest()[:8])[0]

    def _compute(self, text: str) -> np.ndarray:
        try:
            vec = self._local.buf
        except AttributeError:
            vec = np.zeros(self.dim, dtype=np.float32)
            self._local.buf = vec
        vec.fill(0.0)
        tokens = text.split()

        # 1. Feature hashing from unigrams
        for token in tokens:
            h = self._hash_token(token)
            idx = h % self.dim
            sign = 1 if (h >> 32) & 1 else -1
            vec[idx] += sign

        # 2. Feature hashing from bigrams (lower weight for locality)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            h = self._hash_token(bigram)
            idx = h % self.dim
            sign = 1 if (h >> 32) & 1 else -1
            vec[idx] += sign * 0.5

        # 3. Statistical features
        if text:
            # Prefix / suffix
            prefix = text[:3]
            suffix = text[-3:]
            vec[self._hash_token(prefix) % self.dim] += 0.3
            vec[self._hash_token(suffix) % self.dim] += 0.3

            # Length and word count
            vec[self._hash_token("len:" + str(len(text))) % self.dim] += 0.2
            vec[self._hash_token("words:" + str(len(tokens))) % self.dim] += 0.2

            # Single-pass character statistics
            digits = 0
            puncts = 0
            total_letters = 0
            hist = [0] * 26
            for c in text:
                if c.isdigit():
                    digits += 1
                elif "a" <= c <= "z":
                    total_letters += 1
                    hist[ord(c) - 97] += 1
                elif not c.isspace():
                    puncts += 1

            vec[self._hash_token("dig:" + str(digits)) % self.dim] += 0.15
            vec[self._hash_token("pun:" + str(puncts)) % self.dim] += 0.15

            if total_letters:
                for i, count in enumerate(hist):
                    if count:
                        ch = chr(97 + i)
                        idx = self._hash_token("hist:" + ch) % self.dim
                        vec[idx] += (count / total_letters) * 0.1

        # Normalize
        norm = np.sqrt(np.dot(vec, vec))
        if norm:
            vec /= norm

        # Return a read-only copy to prevent silent cache corruption
        result = vec.copy()
        result.setflags(write=False)
        return result

    def embed(self, text: str) -> np.ndarray:
        key = self._normalize_key(text)
        with self._lock:
            vec = self._cache.get(key)
            if vec is not None:
                self._cache.move_to_end(key)
                return vec

        vec = self._compute(key)

        with self._lock:
            self._cache[key] = vec
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        results: list[np.ndarray | None] = [None] * len(texts)
        keys = [self._normalize_key(t) for t in texts]

        with self._lock:
            n = len(keys)
            missing_keys: list[str] = [""] * n
            missing_indices: list[int] = [0] * n
            count = 0
            for i, key in enumerate(keys):
                vec = self._cache.get(key)
                if vec is not None:
                    self._cache.move_to_end(key)
                    results[i] = vec
                else:
                    missing_keys[count] = key
                    missing_indices[count] = i
                    count += 1
            missing_keys = missing_keys[:count]
            missing_indices = missing_indices[:count]

        if missing_keys:
            # Deduplicate while preserving order for deterministic testing
            unique_keys = list(dict.fromkeys(missing_keys))
            computed_map = {k: self._compute(k) for k in unique_keys}
            with self._lock:
                for key, idx in zip(missing_keys, missing_indices):
                    vec = computed_map[key]
                    results[idx] = vec
                    self._cache[key] = vec
                # Evict exactly the required amount in one shot
                excess = len(self._cache) - self._max_cache_size
                if excess > 0:
                    for _ in range(excess):
                        self._cache.popitem(last=False)

        return results  # type: ignore[return-value]

    def similarity(self, vec1: Sequence[float] | np.ndarray, vec2: Sequence[float] | np.ndarray) -> float:
        if vec1 is vec2:
            return 1.0
        v1 = vec1 if isinstance(vec1, np.ndarray) and vec1.dtype == np.float32 else np.asarray(vec1, dtype=np.float32)
        v2 = vec2 if isinstance(vec2, np.ndarray) and vec2.dtype == np.float32 else np.asarray(vec2, dtype=np.float32)
        dot = float(np.dot(v1, v2))
        if abs(dot) < 1e-12:
            return 0.0
        n1 = float(np.sqrt(np.dot(v1, v1)))
        n2 = float(np.sqrt(np.dot(v2, v2)))
        norms = n1 * n2
        if norms < 1e-12:
            return 0.0
        return dot / norms

    def similarity_batch(self, query: Sequence[float] | np.ndarray, vectors: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
        """Cosine similarities between *query* and many *vectors* (matrix multiplication)."""
        q = query if isinstance(query, np.ndarray) and query.dtype == np.float32 else np.asarray(query, dtype=np.float32)
        if not vectors:
            return np.array([], dtype=np.float32)
        mat = vectors if isinstance(vectors, np.ndarray) and vectors.dtype == np.float32 else np.asarray(vectors, dtype=np.float32)
        dots = mat @ q
        q_norm = np.sqrt(np.dot(q, q))
        if q_norm < 1e-12:
            return np.zeros(len(vectors), dtype=np.float32)
        v_norms = np.sqrt(np.einsum("ij,ij->i", mat, mat))
        with np.errstate(invalid="ignore"):
            sims = dots / (v_norms * q_norm)
        return np.where(np.isfinite(sims), sims, 0.0)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_cache_size,
            }
