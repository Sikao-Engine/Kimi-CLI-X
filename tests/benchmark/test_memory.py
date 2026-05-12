"""Comprehensive stress-test benchmarks for the kimix memory subsystem.

Stages:
  1. MemoryEntry  – creation, expiry, importance decay, serialization
  2. WorkingMemory – add, get_context, summarize under capacity pressure
  3. ShortTermMemory – add/evict, search, get_recent, clear_expired
  4. LongTermMemory (file) – store, store_many, retrieve, forget, consolidate
  5. LongTermMemory (sqlite) – same ops backed by SQLiteBackend
  6. SQLiteBackend – raw store/get/list/tag_search/update_access heavy load
  7. AgentMemorySystem – end-to-end perceive/recall/remember/reflect cycle
  8. Concurrency – thread-safety of SQLite backend and EmbeddingProvider cache

All timings are assert-based so the file doubles as a regression test.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import string
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kimix.memory.embedding import EmbeddingProvider
from kimix.memory.long_term_memory import LongTermMemory
from kimix.memory.short_term_memory import ShortTermMemory
from kimix.memory.sqlite_backend import SQLiteBackend
from kimix.memory.system import AgentMemorySystem
from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.working_memory import WorkingMemory

# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def _section(title: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n{title}\n{sep}")


def _report(stage: str, ops: int, elapsed: float, unit: str = "ops") -> dict[str, Any]:
    throughput = ops / elapsed if elapsed > 0 else float("inf")
    avg_ms = (elapsed / ops) * 1000 if ops > 0 else 0.0
    info = {
        "stage": stage,
        "ops": ops,
        "elapsed_s": elapsed,
        "avg_ms": avg_ms,
        "throughput": throughput,
    }
    print(
        f"  [{stage:30s}] {ops:8d} {unit:8s} | "
        f"{elapsed:8.4f}s total | {avg_ms:10.6f}ms avg | {throughput:10.2f} {unit}/s"
    )
    return info


def _report_header() -> None:
    print(
        f"  {'Stage':32s} {'Ops':>8s} {'Unit':>8s} | "
        f"{'Total':>8s} | {'Avg ms':>10s} | {'Throughput':>12s}"
    )
    print("  " + "-" * 90)


# -----------------------------------------------------------------------------
# Text generators
# -----------------------------------------------------------------------------

_LOREM_POOL = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim "
    "ad minim veniam quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit "
    "in voluptate velit esse cillum dolore eu fugiat nulla pariatur "
    "excepteur sint occaecat cupidatat non proident sunt in culpa qui "
    "officia deserunt mollit anim id est laborum"
).split()


def _lorem(words: int) -> str:
    return " ".join(_LOREM_POOL[i % len(_LOREM_POOL)] for i in range(words))


def _random_words(n: int, length: int = 5, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(string.ascii_lowercase, k=length)) for _ in range(n)]


def _random_content(rng: random.Random, min_words: int = 5, max_words: int = 30) -> str:
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(_LOREM_POOL) for _ in range(n))


# -----------------------------------------------------------------------------
# Stage 1: MemoryEntry
# -----------------------------------------------------------------------------


class TestMemoryEntryBenchmark:
    def test_creation_bulk(self) -> None:
        _section("Stage 1a: MemoryEntry creation")
        _report_header()
        n = 100_000
        start = time.perf_counter()
        for i in range(n):
            MemoryEntry(
                content=f"entry number {i}",
                memory_type=MemoryType.SEMANTIC,
                importance=5.0,
                tags=["tag_a", "tag_b"],
            )
        elapsed = time.perf_counter() - start
        _report("creation", n, elapsed)
        assert elapsed < 10.0

    def test_is_expired_bulk(self) -> None:
        now = time.time()
        entries = [
            MemoryEntry(content="ok", memory_type=MemoryType.SEMANTIC),
            MemoryEntry(content="exp", memory_type=MemoryType.SEMANTIC, expires_at=now - 1),
        ] * 50_000
        start = time.perf_counter()
        for e in entries:
            e.is_expired(now)
        elapsed = time.perf_counter() - start
        _report("is_expired", len(entries), elapsed)
        assert elapsed < 2.0

    def test_get_effective_importance_bulk(self) -> None:
        now = time.time()
        entries = [
            MemoryEntry(
                content=f"entry {i}",
                memory_type=MemoryType.SEMANTIC,
                importance=rng.random() * 10,
                timestamp=now - rng.random() * 86400 * 30,
                access_count=rng.randint(0, 50),
            )
            for i, rng in enumerate((random.Random(i) for i in range(100_000)))
        ]
        start = time.perf_counter()
        for e in entries:
            e.get_effective_importance(now)
        elapsed = time.perf_counter() - start
        _report("effective_importance", len(entries), elapsed)
        assert elapsed < 3.0

    def test_touch_bulk(self) -> None:
        entries = [MemoryEntry(content=f"e{i}", memory_type=MemoryType.SEMANTIC) for i in range(100_000)]
        start = time.perf_counter()
        for e in entries:
            e.touch()
        elapsed = time.perf_counter() - start
        _report("touch", len(entries), elapsed)
        assert elapsed < 2.0

    def test_to_dict_bulk(self) -> None:
        entries = [MemoryEntry(content=f"e{i}", memory_type=MemoryType.SEMANTIC) for i in range(100_000)]
        start = time.perf_counter()
        for e in entries:
            e.to_dict()
        elapsed = time.perf_counter() - start
        _report("to_dict", len(entries), elapsed)
        assert elapsed < 3.0

    def test_from_dict_bulk(self) -> None:
        data = [MemoryEntry(content=f"e{i}", memory_type=MemoryType.SEMANTIC).to_dict() for i in range(50_000)]
        start = time.perf_counter()
        for d in data:
            MemoryEntry.from_dict(d)
        elapsed = time.perf_counter() - start
        _report("from_dict", len(data), elapsed)
        assert elapsed < 3.0


# -----------------------------------------------------------------------------
# Stage 2: WorkingMemory
# -----------------------------------------------------------------------------


class TestWorkingMemoryBenchmark:
    def test_add_under_capacity(self) -> None:
        _section("Stage 2a: WorkingMemory add (under capacity)")
        _report_header()
        wm = WorkingMemory(max_items=10)
        n = 100_000
        start = time.perf_counter()
        for i in range(n):
            wm.add(MemoryEntry(content=f"item {i}", memory_type=MemoryType.WORKING))
        elapsed = time.perf_counter() - start
        _report("add_under_cap", n, elapsed)
        assert len(wm.items) == 10
        assert elapsed < 2.0

    def test_get_context_large(self) -> None:
        wm = WorkingMemory(max_items=10_000)
        for i in range(10_000):
            wm.add(MemoryEntry(content=f"item {i}", memory_type=MemoryType.WORKING))
        start = time.perf_counter()
        for _ in range(10_000):
            wm.get_context(n=50)
        elapsed = time.perf_counter() - start
        _report("get_context", 10_000, elapsed)
        assert elapsed < 3.0

    def test_summarize_large(self) -> None:
        wm = WorkingMemory(max_items=10_000)
        for i in range(10_000):
            wm.add(MemoryEntry(content=f"item {i}", memory_type=MemoryType.WORKING))
        start = time.perf_counter()
        for _ in range(10_000):
            wm.summarize()
        elapsed = time.perf_counter() - start
        _report("summarize", 10_000, elapsed)
        assert elapsed < 3.0


# -----------------------------------------------------------------------------
# Stage 3: ShortTermMemory
# -----------------------------------------------------------------------------


class TestShortTermMemoryBenchmark:
    def test_add_evict_pressure(self) -> None:
        _section("Stage 3a: ShortTermMemory add + evict")
        _report_header()
        stm = ShortTermMemory(max_size=1_000, ttl_seconds=3600)
        n = 50_000
        rng = random.Random(42)
        start = time.perf_counter()
        for i in range(n):
            entry = MemoryEntry(
                content=_random_content(rng),
                memory_type=MemoryType.EPISODIC,
                importance=rng.random() * 10,
            )
            stm.add(entry)
        elapsed = time.perf_counter() - start
        _report("add_evict", n, elapsed)
        assert len(stm.buffer) == 1_000
        assert elapsed < 30.0

    def test_search_warm(self) -> None:
        stm = ShortTermMemory(max_size=500, ttl_seconds=3600)
        provider = EmbeddingProvider(dim=384)
        rng = random.Random(42)
        for i in range(500):
            entry = MemoryEntry(
                content=f"topic {i % 10} " + _random_content(rng, 5, 15),
                memory_type=MemoryType.EPISODIC,
                importance=5.0,
            )
            stm.add(entry)
        queries = [f"topic {i % 10}" for i in range(1_000)]
        start = time.perf_counter()
        for q in queries:
            stm.search(q, provider, top_k=5)
        elapsed = time.perf_counter() - start
        _report("search_warm", len(queries), elapsed)
        assert elapsed < 40.0

    def test_get_recent(self) -> None:
        stm = ShortTermMemory(max_size=1_000, ttl_seconds=3600)
        rng = random.Random(42)
        for i in range(1_000):
            stm.add(MemoryEntry(content=f"item {i}", memory_type=MemoryType.EPISODIC, importance=5.0))
        start = time.perf_counter()
        for _ in range(10_000):
            stm.get_recent(n=20)
        elapsed = time.perf_counter() - start
        _report("get_recent", 10_000, elapsed)
        assert elapsed < 3.0

    def test_clear_expired(self) -> None:
        stm = ShortTermMemory(max_size=1_000, ttl_seconds=1)
        rng = random.Random(42)
        for i in range(1_000):
            stm.add(
                MemoryEntry(
                    content=f"item {i}",
                    memory_type=MemoryType.EPISODIC,
                    importance=5.0,
                    timestamp=time.time() - 10,
                )
            )
        time.sleep(1.1)
        start = time.perf_counter()
        for _ in range(1_000):
            stm.clear_expired()
        elapsed = time.perf_counter() - start
        _report("clear_expired", 1_000, elapsed)
        assert elapsed < 3.0


# -----------------------------------------------------------------------------
# Stage 4: LongTermMemory (file-backed)
# -----------------------------------------------------------------------------


class TestLongTermMemoryFileBenchmark:
    @pytest.fixture(scope="function")
    def ltm_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ltm.json")
            yield path

    def test_store_many_small(self, ltm_file: str) -> None:
        _section("Stage 4a: LTM (file) store_many small")
        _report_header()
        ltm = LongTermMemory(storage_path=ltm_file, dim=384)
        items = [{"content": f"doc {i}", "importance": 5.0} for i in range(1_000)]
        start = time.perf_counter()
        ltm.store_many(items)
        elapsed = time.perf_counter() - start
        _report("store_many_1k", len(items), elapsed)
        assert ltm.count() == 1_000
        assert elapsed < 20.0

    def test_store_many_medium(self, ltm_file: str) -> None:
        ltm = LongTermMemory(storage_path=ltm_file, dim=384)
        rng = random.Random(42)
        items = [{"content": _random_content(rng, 10, 30), "importance": rng.random() * 10} for _ in range(5_000)]
        start = time.perf_counter()
        ltm.store_many(items)
        elapsed = time.perf_counter() - start
        _report("store_many_5k", len(items), elapsed)
        assert ltm.count() == 5_000
        assert elapsed < 60.0

    def test_retrieve_cold_then_warm(self, ltm_file: str) -> None:
        ltm = LongTermMemory(storage_path=ltm_file, dim=384)
        rng = random.Random(42)
        items = [
            {
                "content": f"topic {i % 20} " + _random_content(rng, 5, 20),
                "importance": rng.random() * 10,
                "tags": [f"tag_{i % 10}"],
            }
            for i in range(2_000)
        ]
        ltm.store_many(items)
        queries = [f"topic {i % 20}" for i in range(200)]

        # Cold retrieval (BM25 index build)
        start = time.perf_counter()
        for q in queries:
            ltm.retrieve(q, top_k=5, use_hybrid=True)
        cold_elapsed = time.perf_counter() - start
        _report("retrieve_cold", len(queries), cold_elapsed)

        # Warm retrieval
        start = time.perf_counter()
        for q in queries:
            ltm.retrieve(q, top_k=5, use_hybrid=True)
        warm_elapsed = time.perf_counter() - start
        _report("retrieve_warm", len(queries), warm_elapsed)

        assert cold_elapsed < 120.0
        assert warm_elapsed < 60.0

    def test_forget_many(self, ltm_file: str) -> None:
        ltm = LongTermMemory(storage_path=ltm_file, dim=384)
        rng = random.Random(42)
        items = [{"content": f"forgettable {i} " + _random_content(rng, 3, 10), "importance": 5.0} for i in range(2_000)]
        ltm.store_many(items)
        ids = list(ltm.entries.keys())
        to_forget = ids[:1_000]
        start = time.perf_counter()
        ltm.forget_many(to_forget)
        elapsed = time.perf_counter() - start
        _report("forget_many_1k", len(to_forget), elapsed)
        assert elapsed < 30.0

    def test_consolidate_from_stm(self, ltm_file: str) -> None:
        ltm = LongTermMemory(storage_path=ltm_file, dim=384)
        stm = ShortTermMemory(max_size=500, ttl_seconds=3600)
        rng = random.Random(42)
        for i in range(500):
            entry = MemoryEntry(
                content=f"consolidate {i} " + _random_content(rng, 5, 15),
                memory_type=MemoryType.EPISODIC,
                importance=8.0,
            )
            stm.add(entry)
        start = time.perf_counter()
        ltm.consolidate(stm, threshold=6.0)
        elapsed = time.perf_counter() - start
        _report("consolidate", 500, elapsed)
        assert elapsed < 15.0


# -----------------------------------------------------------------------------
# Stage 5: LongTermMemory (SQLite-backed)
# -----------------------------------------------------------------------------


class TestLongTermMemorySQLiteBenchmark:
    @pytest.fixture(scope="function")
    def ltm_sqlite(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "memory.db")
            backend = SQLiteBackend(db_path)
            ltm = LongTermMemory(storage_path="", dim=384, backend=backend, agent_id="bench")
            try:
                yield ltm
            finally:
                backend.close()

    def test_sqlite_store_many(self, ltm_sqlite: LongTermMemory) -> None:
        _section("Stage 5a: LTM (SQLite) store_many")
        _report_header()
        rng = random.Random(42)
        items = [{"content": _random_content(rng, 10, 30), "importance": rng.random() * 10} for _ in range(5_000)]
        start = time.perf_counter()
        ltm_sqlite.store_many(items)
        elapsed = time.perf_counter() - start
        _report("sqlite_store_many", len(items), elapsed)
        assert ltm_sqlite.count() == 5_000
        assert elapsed < 30.0

    def test_sqlite_retrieve(self, ltm_sqlite: LongTermMemory) -> None:
        rng = random.Random(42)
        items = [
            {
                "content": f"topic {i % 20} " + _random_content(rng, 5, 20),
                "importance": rng.random() * 10,
                "tags": [f"tag_{i % 10}"],
            }
            for i in range(2_000)
        ]
        ltm_sqlite.store_many(items)
        queries = [f"topic {i % 20}" for i in range(200)]
        start = time.perf_counter()
        for q in queries:
            ltm_sqlite.retrieve(q, top_k=5, use_hybrid=True)
        elapsed = time.perf_counter() - start
        _report("sqlite_retrieve", len(queries), elapsed)
        assert elapsed < 60.0

    def test_sqlite_tag_filter(self, ltm_sqlite: LongTermMemory) -> None:
        rng = random.Random(42)
        items = [
            {"content": f"tagged doc {i}", "importance": 5.0, "tags": [f"tag_{i % 5}"]}
            for i in range(2_000)
        ]
        ltm_sqlite.store_many(items)
        start = time.perf_counter()
        for i in range(500):
            ltm_sqlite.retrieve("doc", top_k=5, tag_filter=[f"tag_{i % 5}"])
        elapsed = time.perf_counter() - start
        _report("sqlite_tag_filter", 500, elapsed)
        assert elapsed < 60.0


# -----------------------------------------------------------------------------
# Stage 6: SQLiteBackend raw ops
# -----------------------------------------------------------------------------


class TestSQLiteBackendBenchmark:
    @pytest.fixture(scope="function")
    def backend(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "raw.db")
            be = SQLiteBackend(db_path)
            try:
                yield be
            finally:
                be.close()

    def _make_entry(self, i: int) -> tuple[str, MemoryEntry]:
        entry = MemoryEntry(
            content=f"raw entry {i}",
            memory_type=MemoryType.SEMANTIC,
            importance=5.0,
            tags=["bench", f"tag_{i % 10}"],
        )
        eid = hashlib.blake2b(entry.content.encode(), digest_size=8).hexdigest()
        return eid, entry

    def test_store_single_heavy(self, backend: SQLiteBackend) -> None:
        _section("Stage 6a: SQLiteBackend store (single) heavy")
        _report_header()
        n = 5_000
        start = time.perf_counter()
        for i in range(n):
            eid, entry = self._make_entry(i)
            backend.store(entry, eid)
        elapsed = time.perf_counter() - start
        _report("store_single", n, elapsed)
        assert elapsed < 20.0

    def test_store_many_bulk(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        start = time.perf_counter()
        backend.store_many(batch)
        elapsed = time.perf_counter() - start
        _report("store_many", len(batch), elapsed)
        assert elapsed < 20.0

    def test_get_random_access(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        backend.store_many(batch)
        ids = [eid for eid, _ in batch]
        rng = random.Random(42)
        sample = [rng.choice(ids) for _ in range(5_000)]
        start = time.perf_counter()
        for eid in sample:
            backend.get(eid)
        elapsed = time.perf_counter() - start
        _report("get_random", len(sample), elapsed)
        assert elapsed < 15.0

    def test_list_all(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        backend.store_many(batch)
        start = time.perf_counter()
        for _ in range(100):
            backend.list_all(agent_id="default", include_embedding=True)
        elapsed = time.perf_counter() - start
        _report("list_all", 100, elapsed, unit="calls")
        assert elapsed < 30.0

    def test_search_by_tag(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        backend.store_many(batch)
        start = time.perf_counter()
        for i in range(500):
            backend.search_by_tag([f"tag_{i % 10}"], agent_id="default")
        elapsed = time.perf_counter() - start
        _report("search_by_tag", 500, elapsed)
        assert elapsed < 20.0

    def test_update_access_many(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        backend.store_many(batch)
        ids = [eid for eid, _ in batch]
        start = time.perf_counter()
        for _ in range(100):
            backend.update_access_many(ids)
        elapsed = time.perf_counter() - start
        _report("update_access_many", 100, elapsed, unit="calls")
        assert elapsed < 20.0

    def test_count(self, backend: SQLiteBackend) -> None:
        batch = [self._make_entry(i) for i in range(10_000)]
        backend.store_many(batch)
        start = time.perf_counter()
        for _ in range(10_000):
            backend.count(agent_id="default")
        elapsed = time.perf_counter() - start
        _report("count", 10_000, elapsed)
        assert elapsed < 5.0

    def test_purge_expired(self, backend: SQLiteBackend) -> None:
        now = time.time()
        entries = [
            MemoryEntry(
                content=f"expired {i}",
                memory_type=MemoryType.SEMANTIC,
                importance=5.0,
                expires_at=now - 1,
            )
            for i in range(5_000)
        ]
        batch = []
        for i, entry in enumerate(entries):
            eid = hashlib.blake2b(entry.content.encode(), digest_size=8).hexdigest()
            batch.append((eid, entry))
        backend.store_many(batch)
        start = time.perf_counter()
        deleted = backend.purge_expired(now=now + 1)
        elapsed = time.perf_counter() - start
        _report("purge_expired", deleted, elapsed, unit="rows")
        assert deleted == 5_000
        assert elapsed < 10.0


# -----------------------------------------------------------------------------
# Stage 7: AgentMemorySystem end-to-end
# -----------------------------------------------------------------------------


class TestAgentMemorySystemBenchmark:
    @pytest.fixture(scope="function")
    def system(self):
        with tempfile.TemporaryDirectory() as td:
            ltm_path = os.path.join(td, "ltm.json")
            db_path = os.path.join(td, "memory.db")
            sys = AgentMemorySystem(
                dim=384,
                ltm_path=ltm_path,
                agent_id="bench",
                use_sqlite=True,
                db_path=db_path,
            )
            try:
                yield sys
            finally:
                if sys.long_term._backend is not None:
                    sys.long_term._backend.close()

    def test_perceive_heavy(self, system: AgentMemorySystem) -> None:
        _section("Stage 7a: AgentMemorySystem perceive")
        _report_header()
        n = 2_000
        rng = random.Random(42)
        start = time.perf_counter()
        for i in range(n):
            system.perceive(
                observation=_random_content(rng, 5, 20),
                importance=rng.random() * 10,
                tags=[f"tag_{i % 10}"],
            )
        elapsed = time.perf_counter() - start
        _report("perceive", n, elapsed)
        assert len(system.short_term.buffer) <= system.short_term.max_size
        assert elapsed < 30.0

    def test_recall_mixed(self, system: AgentMemorySystem) -> None:
        rng = random.Random(42)
        for i in range(500):
            system.perceive(
                observation=f"topic {i % 20} " + _random_content(rng, 5, 15),
                importance=5.0,
                tags=[f"tag_{i % 10}"],
            )
        for i in range(500):
            system.remember(
                fact=f"knowledge {i % 20} " + _random_content(rng, 5, 15),
                importance=8.0,
                tags=[f"tag_{i % 10}"],
            )
        queries = [f"topic {i % 20}" for i in range(200)]
        start = time.perf_counter()
        for q in queries:
            system.recall(q, context_size=5)
        elapsed = time.perf_counter() - start
        _report("recall", len(queries), elapsed)
        assert elapsed < 60.0

    def test_remember_bulk(self, system: AgentMemorySystem) -> None:
        rng = random.Random(42)
        n = 1_000
        start = time.perf_counter()
        for i in range(n):
            system.remember(
                fact=_random_content(rng, 10, 30),
                importance=rng.random() * 10,
                tags=["learning"],
            )
        elapsed = time.perf_counter() - start
        _report("remember_bulk", n, elapsed)
        assert system.long_term.count() == n
        assert elapsed < 30.0

    def test_get_context_for_llm(self, system: AgentMemorySystem) -> None:
        rng = random.Random(42)
        for i in range(200):
            system.perceive(_random_content(rng, 5, 20), importance=5.0)
        for i in range(200):
            system.remember(_random_content(rng, 10, 30), importance=8.0)
        queries = [f"query {i}" for i in range(200)]
        start = time.perf_counter()
        for q in queries:
            system.get_context_for_llm(q, max_tokens=2000)
        elapsed = time.perf_counter() - start
        _report("get_context_llm", len(queries), elapsed)
        assert elapsed < 60.0

    def test_self_reflect(self, system: AgentMemorySystem) -> None:
        rng = random.Random(42)
        for i in range(500):
            system.remember(_random_content(rng, 5, 15), importance=5.0)
        start = time.perf_counter()
        for _ in range(100):
            system.self_reflect()
        elapsed = time.perf_counter() - start
        _report("self_reflect", 100, elapsed)
        assert elapsed < 10.0

    def test_reflect(self, system: AgentMemorySystem) -> None:
        rng = random.Random(42)
        for i in range(100):
            system.perceive(_random_content(rng, 5, 15), importance=5.0)
        start = time.perf_counter()
        for _ in range(1_000):
            system.reflect()
        elapsed = time.perf_counter() - start
        _report("reflect", 1_000, elapsed)
        assert elapsed < 5.0


# -----------------------------------------------------------------------------
# Stage 8: Concurrency stress
# -----------------------------------------------------------------------------


class TestConcurrencyBenchmark:
    def test_sqlite_concurrent_reads(self) -> None:
        _section("Stage 8a: SQLiteBackend concurrent reads")
        _report_header()
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "concurrent.db")
            backend = SQLiteBackend(db_path)
            batch = []
            for i in range(5_000):
                entry = MemoryEntry(content=f"concurrent {i}", memory_type=MemoryType.SEMANTIC, importance=5.0)
                eid = hashlib.blake2b(entry.content.encode(), digest_size=8).hexdigest()
                batch.append((eid, entry))
            backend.store_many(batch)

            ids = [eid for eid, _ in batch]
            results: list[Any] = []
            lock = threading.Lock()

            def worker():
                local = []
                for eid in ids:
                    local.append(backend.get(eid))
                with lock:
                    results.extend(local)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            start = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.perf_counter() - start
            backend.close()
            _report("sqlite_concurrent_reads", len(results), elapsed)
            assert len(results) == 4 * 5_000
            assert elapsed < 20.0

    def test_embedding_cache_concurrent(self) -> None:
        provider = EmbeddingProvider(dim=384, max_cache_size=1_024)
        texts = [f"concurrent cache text {i}" for i in range(500)]

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
        _report("embed_cache_concurrent", 8 * len(texts), elapsed)
        assert elapsed < 10.0

    def test_agent_memory_system_concurrent_perceive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ltm_path = os.path.join(td, "ltm.json")
            db_path = os.path.join(td, "memory.db")
            system = AgentMemorySystem(
                dim=384,
                ltm_path=ltm_path,
                agent_id="concurrent",
                use_sqlite=True,
                db_path=db_path,
            )
            rng = random.Random(42)
            texts = [_random_content(rng, 5, 15) for _ in range(200)]
            errors: list[Exception] = []
            lock = threading.Lock()

            def worker():
                try:
                    for t in texts:
                        system.perceive(t, importance=5.0)
                except Exception as e:
                    with lock:
                        errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            start = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.perf_counter() - start
            if system.long_term._backend is not None:
                system.long_term._backend.close()
            _report("ams_concurrent_perceive", 4 * len(texts), elapsed)
            assert not errors
            assert elapsed < 20.0


# -----------------------------------------------------------------------------
# Stage 9: Mixed workload / endurance
# -----------------------------------------------------------------------------


class TestEnduranceBenchmark:
    def test_endurance_mixed_workload(self) -> None:
        _section("Stage 9: Endurance mixed workload")
        _report_header()
        with tempfile.TemporaryDirectory() as td:
            ltm_path = os.path.join(td, "ltm.json")
            db_path = os.path.join(td, "memory.db")
            system = AgentMemorySystem(
                dim=384,
                ltm_path=ltm_path,
                agent_id="endurance",
                use_sqlite=True,
                db_path=db_path,
            )
            rng = random.Random(42)
            n_cycles = 500

            start = time.perf_counter()
            for cycle in range(n_cycles):
                # Perceive 5 observations
                for _ in range(5):
                    system.perceive(_random_content(rng, 5, 20), importance=rng.random() * 10)
                # Remember 2 facts
                for _ in range(2):
                    system.remember(_random_content(rng, 10, 30), importance=rng.random() * 10)
                # Recall 3 queries
                for q in range(3):
                    system.recall(f"topic {q}", context_size=3)
                # Periodic reflect
                if cycle % 50 == 0:
                    system.self_reflect()
            elapsed = time.perf_counter() - start

            total_ops = n_cycles * (5 + 2 + 3)  # perceive + remember + recall
            _report("endurance_mixed", total_ops, elapsed)
            print(f"    Final state: WM={len(system.working.items)} STM={len(system.short_term.buffer)} LTM={system.long_term.count()}")
            if system.long_term._backend is not None:
                system.long_term._backend.close()
            assert elapsed < 240.0
