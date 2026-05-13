"""Comprehensive tests for file_builder.py."""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pytest

from kimix.tools.skill.searching.file_builder import (
    FileBuilder,
    FileReader,
    formatted_print,
)


class TestFileReader:
    def test_is_text_file_true(self, tmp_path: Path) -> None:
        fr = FileReader([tmp_path], tmp_path / "out.json")
        text_file = tmp_path / "hello.txt"
        text_file.write_text("hello world", encoding="utf-8")
        assert fr._is_text_file(text_file) is True

    def test_is_text_file_binary(self, tmp_path: Path) -> None:
        fr = FileReader([tmp_path], tmp_path / "out.json")
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        assert fr._is_text_file(bin_file) is False

    def test_hash_file(self, tmp_path: Path) -> None:
        fr = FileReader([tmp_path], tmp_path / "out.json")
        f = tmp_path / "data.txt"
        content = b"abc123"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert fr._hash_file(f) == expected

    def test_process_file_text(self, tmp_path: Path) -> None:
        fr = FileReader([tmp_path], tmp_path / "out.json")
        f = tmp_path / "text.txt"
        content = b"some text"
        f.write_bytes(content)
        result = fr._process_file("text.txt", f)
        assert result is not None
        assert result[0] == "text.txt"
        assert result[1] == hashlib.sha256(content).hexdigest()

    def test_process_file_binary(self, tmp_path: Path) -> None:
        fr = FileReader([tmp_path], tmp_path / "out.json")
        f = tmp_path / "bin.bin"
        f.write_bytes(b"\x00\x01")
        assert fr._process_file("bin.bin", f) is None

    def test_collect_files_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "single.txt"
        f.write_text("x")
        fr = FileReader([f], tmp_path / "out.json")
        collected = fr._collect_files()
        assert len(collected) == 1
        assert collected[0][1] == f

    def test_collect_files_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.txt").write_text("a")
        (d / "b.txt").write_text("b")
        fr = FileReader([d], tmp_path / "out.json")
        collected = fr._collect_files()
        assert len(collected) == 2

    def test_scan_and_write(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.txt").write_text("beta")
        out = tmp_path / "out.json"
        fr = FileReader([tmp_path], out)
        assert len(fr._mapping) == 2
        assert out.exists()
        data = out.read_bytes()
        import orjson
        mapping = orjson.loads(data)
        assert "a.txt" in mapping
        assert "b.txt" in mapping

    def test_update_no_change(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        out = tmp_path / "out.json"
        fr = FileReader([d], out)
        assert fr.update() is False

    def test_update_file_modified(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("alpha")
        out = tmp_path / "out.json"
        fr = FileReader([tmp_path], out)
        f.write_text("alpha_modified")
        assert fr.update() is True
        assert fr._mapping["a.txt"] == hashlib.sha256(b"alpha_modified").hexdigest()

    def test_update_new_file(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha")
        out = tmp_path / "out.json"
        fr = FileReader([tmp_path], out)
        (tmp_path / "c.txt").write_text("charlie")
        assert fr.update() is True
        assert "c.txt" in fr._mapping

    def test_update_deleted_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("alpha")
        out = tmp_path / "out.json"
        fr = FileReader([tmp_path], out)
        f.unlink()
        assert fr.update() is True
        assert "a.txt" not in fr._mapping

    def test_scan_skips_binary(self, tmp_path: Path) -> None:
        (tmp_path / "text.txt").write_text("hello")
        (tmp_path / "binary.bin").write_bytes(b"\x00\x00")
        fr = FileReader([tmp_path], tmp_path / "out.json")
        assert "text.txt" in fr._mapping
        assert "binary.bin" not in fr._mapping


class TestFileBuilder:
    def test_build_and_search_basic(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "code.py").write_text("def hello():\n    print('world')\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("hello")
        assert len(results) > 0
        assert any(r["path"].endswith("code.py") for r in results)

    def test_search_no_match(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "code.py").write_text("def foo():\n    pass\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("nonexistent_xyz")
        assert results == []

    def test_search_top_k(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("alpha beta gamma\n")
        (d / "b.py").write_text("alpha delta epsilon\n")
        (d / "c.py").write_text("alpha zeta eta\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("alpha", top_k=2)
        assert len(results) <= 2

    def test_search_with_diversify(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("searchable content one\n")
        (d / "b.py").write_text("searchable content two\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("searchable", top_k=2, diversify=True)
        assert len(results) >= 0

    def test_search_disable_spelling(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("hello", use_spelling=False)
        assert len(results) > 0

    def test_search_disable_stemming(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("running runner runs\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("running", use_stemming=False)
        assert len(results) > 0

    def test_search_disable_string_similarity(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("unique string here\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("unique", use_string_similarity=False)
        assert len(results) > 0

    def test_search_disable_ltr(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("ltr test content\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("ltr", use_ltr=False)
        assert len(results) > 0

    def test_search_disable_adaptive_scoring(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("adaptive scoring test\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("adaptive", use_adaptive_scoring=False)
        assert len(results) > 0

    def test_search_disable_xquad(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("xquad test content\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("xquad", use_xquad=False)
        assert len(results) > 0

    def test_update_rebuilds(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("old content\n")
        fb = FileBuilder([d], tmp_path / "hashes.json")
        old_results = fb.search("old")
        assert len(old_results) > 0
        (d / "a.py").write_text("new shiny content\n")
        fb.update()
        new_results = fb.search("shiny")
        assert len(new_results) > 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        fb = FileBuilder([d], tmp_path / "hashes.json")
        assert fb._search is None
        assert fb.search("anything") == []

    def test_single_file_path(self, tmp_path: Path) -> None:
        f = tmp_path / "script.py"
        f.write_text("def main():\n    return 42\n")
        fb = FileBuilder([f], tmp_path / "hashes.json")
        results = fb.search("main")
        assert len(results) > 0
        assert results[0]["path"] == "script.py"

    def test_doc_info_line_index(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("line one\nline two\nline three\n")
        # Prevent stale caches from polluting this test
        cache_path = tmp_path / "hashes.index_cache.pkl"
        build_cache_path = tmp_path / "hashes.build_cache.pkl"
        for p in [cache_path, build_cache_path]:
            p.unlink(missing_ok=True)
        fb = FileBuilder([d], tmp_path / "hashes.json")
        results = fb.search("three")
        assert len(results) > 0
        # line_index is 1-based in output
        assert results[0]["line_index"] == 3

    def test_index_cache_created(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)
        assert fb._cache_path.exists()

    def test_build_cache_created(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)
        assert fb._build_cache_path.exists()

    def test_index_cache_content(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)
        with fb._cache_path.open("rb") as f:
            cache = pickle.load(f)
        assert "mapping" in cache
        assert "doc_info" in cache
        assert "searcher" in cache
        assert cache["mapping"] == fb.file_reader._mapping

    def test_index_cache_loads_on_reinit(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        out = tmp_path / "hashes.json"
        fb1 = FileBuilder([d], out)
        assert fb1.search("hello")

        fb2 = FileBuilder([d], out)
        assert fb2._doc_info == fb1._doc_info
        assert fb2.search("hello")

    def test_index_cache_invalidated_when_file_changes(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        out = tmp_path / "hashes.json"
        fb1 = FileBuilder([d], out)
        old_doc_info = fb1._doc_info.copy()

        (d / "a.py").write_text("goodbye world\n")
        fb2 = FileBuilder([d], out)
        assert fb2._doc_info != old_doc_info
        assert fb2.search("goodbye")

    def test_build_cache_skips_unchanged_files(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        (d / "b.py").write_text("foo bar\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)

        call_count = 0
        original_process = fb._process_file_lines

        def counting_process(
            rel: str, abs_path: Path, tokenizer: Any
        ) -> list[dict[str, Any]] | None:
            nonlocal call_count
            call_count += 1
            return original_process(rel, abs_path, tokenizer)

        fb._process_file_lines = counting_process
        fb._build()
        assert call_count == 0

    def test_build_cache_processes_modified_file(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        (d / "b.py").write_text("foo bar\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)

        (d / "a.py").write_text("hello modified world\n")
        fb.file_reader.update()

        call_count = 0
        original_process = fb._process_file_lines

        def counting_process(
            rel: str, abs_path: Path, tokenizer: Any
        ) -> list[dict[str, Any]] | None:
            nonlocal call_count
            call_count += 1
            return original_process(rel, abs_path, tokenizer)

        fb._process_file_lines = counting_process
        fb._build()
        assert call_count == 1

    def test_build_cache_evicts_deleted_file(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello world\n")
        (d / "b.py").write_text("foo bar\n")
        out = tmp_path / "hashes.json"
        fb = FileBuilder([d], out)

        cache = fb._load_build_cache()
        assert "a.py" in cache
        assert "b.py" in cache

        (d / "b.py").unlink()
        fb.file_reader.update()
        fb._build()

        cache = fb._load_build_cache()
        assert "a.py" in cache
        assert "b.py" not in cache


class TestFormattedPrint:
    def test_empty_results(self) -> None:
        assert formatted_print([]) == "No results found."

    def test_single_result(self) -> None:
        results = [{"path": "a.py", "line_index": 5, "score": 0.9876}]
        output = formatted_print(results)
        assert "[1] a.py (line 5)" in output
        assert "score=0.9876" in output

    def test_multiple_results(self) -> None:
        results = [
            {"path": "a.py", "line_index": 1, "score": 0.9},
            {"path": "b.py", "line_index": 2, "score": 0.8},
        ]
        output = formatted_print(results)
        lines = output.split("\n")
        assert len(lines) == 2
        assert "[1] a.py (line 1)" in lines[0]
        assert "[2] b.py (line 2)" in lines[1]
