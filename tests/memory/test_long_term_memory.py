"""Tests for LongTermMemory."""

import json
import os
import tempfile

import pytest

from kimix.memory.long_term_memory import LongTermMemory
from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.short_term_memory import ShortTermMemory


class TestLongTermMemory:
    def test_store_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            entry = ltm.store("python asyncio guide", importance=8.0, tags=["python", "async"])
            assert entry.content == "python asyncio guide"
            results = ltm.retrieve("asyncio")
            assert len(results) > 0
        finally:
            os.unlink(path)

    def test_tag_filter(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            ltm.store("item1", tags=["a", "b"])
            ltm.store("item2", tags=["b", "c"])
            ltm.store("item3", tags=["c", "d"])
            results = ltm.retrieve("item", tag_filter=["a"])
            assert len(results) == 1
            assert results[0].content == "item1"
        finally:
            os.unlink(path)

    def test_min_importance_filter(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            ltm.store("high", importance=8.0)
            ltm.store("low", importance=2.0)
            results = ltm.retrieve("high low", min_importance=5.0)
            assert len(results) == 1
            assert results[0].content == "high"
        finally:
            os.unlink(path)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm1 = LongTermMemory(storage_path=path)
            ltm1.store("persist me", tags=["test"])
            del ltm1

            ltm2 = LongTermMemory(storage_path=path)
            assert len(ltm2.entries) == 1
            results = ltm2.retrieve("persist")
            assert len(results) == 1
        finally:
            os.unlink(path)

    def test_forget(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            entry = ltm.store("forget me", importance=1.0)
            entry_id = ltm._hash("forget me")
            # 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 (< 0.1 removed)
            for _ in range(4):
                ltm.forget(entry_id)
                if entry_id not in ltm.entries:
                    break
            assert entry_id not in ltm.entries
        finally:
            os.unlink(path)

    def test_consolidate(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            stm = ShortTermMemory(max_size=10)
            stm.add(MemoryEntry(content="important event", memory_type=MemoryType.EPISODIC, importance=9.0))
            stm.add(MemoryEntry(content="routine event", memory_type=MemoryType.EPISODIC, importance=3.0))
            ltm.consolidate(stm, threshold=7.0)
            assert len(stm.buffer) == 1
            assert stm.buffer[0].content == "routine event"
            assert len(ltm.entries) == 1
        finally:
            os.unlink(path)

    def test_empty_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            results = ltm.retrieve("anything")
            assert results == []
        finally:
            os.unlink(path)

    def test_retrieve_updates_access(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ltm = LongTermMemory(storage_path=path)
            ltm.store("access test", importance=5.0)
            assert ltm.entries[list(ltm.entries.keys())[0]].access_count == 0
            ltm.retrieve("access")
            assert ltm.entries[list(ltm.entries.keys())[0]].access_count == 1
        finally:
            os.unlink(path)


