"""Tests for ShortTermMemory."""

import time

import pytest

from kimix.memory.short_term_memory import ShortTermMemory
from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider


class TestShortTermMemory:
    def test_add(self):
        stm = ShortTermMemory(max_size=5)
        entry = MemoryEntry(content="test", memory_type=MemoryType.EPISODIC)
        stm.add(entry)
        assert len(stm.buffer) == 1

    def test_eviction_when_full(self):
        stm = ShortTermMemory(max_size=3)
        for i in range(5):
            stm.add(MemoryEntry(content=f"item{i}", memory_type=MemoryType.EPISODIC, importance=float(i)))
        assert len(stm.buffer) == 3

    def test_evict_least_valuable(self):
        stm = ShortTermMemory(max_size=3)
        stm.add(MemoryEntry(content="high", memory_type=MemoryType.EPISODIC, importance=10.0))
        stm.add(MemoryEntry(content="mid", memory_type=MemoryType.EPISODIC, importance=5.0))
        stm.add(MemoryEntry(content="low", memory_type=MemoryType.EPISODIC, importance=1.0))
        stm.add(MemoryEntry(content="new", memory_type=MemoryType.EPISODIC, importance=8.0))
        contents = [e.content for e in stm.buffer]
        assert "low" not in contents

    def test_search(self):
        stm = ShortTermMemory(max_size=10)
        provider = EmbeddingProvider(dim=384)
        stm.add(MemoryEntry(content="python async programming", memory_type=MemoryType.EPISODIC))
        stm.add(MemoryEntry(content="python threading model", memory_type=MemoryType.EPISODIC))
        stm.add(MemoryEntry(content="flask web framework", memory_type=MemoryType.EPISODIC))
        results = stm.search("async python", provider, top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_search_empty(self):
        stm = ShortTermMemory(max_size=10)
        provider = EmbeddingProvider(dim=384)
        results = stm.search("anything", provider)
        assert results == []

    def test_get_recent(self):
        stm = ShortTermMemory(max_size=10)
        old_time = time.time() - 100
        stm.add(MemoryEntry(content="old", memory_type=MemoryType.EPISODIC, timestamp=old_time))
        stm.add(MemoryEntry(content="new", memory_type=MemoryType.EPISODIC))
        recent = stm.get_recent(1)
        assert len(recent) == 1
        assert recent[0].content == "new"

    def test_clear_expired(self):
        stm = ShortTermMemory(max_size=10, ttl_seconds=1)
        old_time = time.time() - 10
        stm.add(MemoryEntry(content="expired", memory_type=MemoryType.EPISODIC, timestamp=old_time))
        stm.add(MemoryEntry(content="fresh", memory_type=MemoryType.EPISODIC))
        stm.clear_expired()
        assert len(stm.buffer) == 1
        assert stm.buffer[0].content == "fresh"

    def test_memory_type_set_to_episodic(self):
        stm = ShortTermMemory(max_size=5)
        entry = MemoryEntry(content="test", memory_type=MemoryType.SEMANTIC)
        stm.add(entry)
        assert entry.memory_type == MemoryType.EPISODIC


