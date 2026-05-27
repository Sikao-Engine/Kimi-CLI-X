"""Tests for adaptive_preserve_depth heuristic in compaction.py."""

from __future__ import annotations

from kosong.message import Message

from kimi_cli.soul.compaction import SimpleCompaction, adaptive_preserve_depth
from kimi_cli.wire.types import TextPart, ThinkPart


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextPart(text=text)])


def _msg_with_think(role: str, text: str, think: str) -> Message:
    return Message(role=role, content=[TextPart(text=text), ThinkPart(think=think)])


class TestAdaptivePreserveDepth:
    def test_empty_history_returns_min(self):
        assert adaptive_preserve_depth([], min_preserved=1, max_preserved=5) == 1

    def test_no_signals_returns_min(self):
        messages = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 1

    def test_error_keyword_boosts_depth(self):
        messages = [_msg("user", "There was an error in the build")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 2

    def test_exception_keyword_boosts_depth(self):
        messages = [_msg("assistant", "A RuntimeException occurred")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 2

    def test_failed_keyword_boosts_depth(self):
        messages = [_msg("user", "The test failed")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 2

    def test_think_part_boosts_depth(self):
        messages = [_msg_with_think("assistant", "Let me think", "deep reasoning...")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 2

    def test_file_edits_boosts_depth(self):
        messages = [_msg("assistant", "Edited file: foo.py and bar.md and baz.py")]
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 2

    def test_combined_signals(self):
        messages = [
            _msg_with_think(
                "assistant",
                "There was an error while editing file: a.py, b.py, c.py",
                "reasoning...",
            )
        ]
        # error (+1) + think (+1) + file refs (+1) = min(1) + 3 = 4, capped at max(5)
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 4

    def test_respects_max_cap(self):
        messages = [
            _msg_with_think(
                "assistant",
                "error exception failed file: a.py b.py c.py d.py",
                "think...",
            )
        ]
        # All signals fire but capped at max_preserved
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=3) == 3

    def test_respects_min_floor(self):
        messages = [_msg("user", "plain chat")]
        assert adaptive_preserve_depth(messages, min_preserved=2, max_preserved=5) == 2

    def test_only_inspects_last_user_or_assistant(self):
        messages = [
            _msg("user", "error here"),
            _msg("assistant", "all good"),
        ]
        # Last turn has no signals
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 1

    def test_skips_system_and_tool_roles(self):
        messages = [
            Message(role="system", content=[TextPart(text="error here")]),
            _msg("user", "all good"),
        ]
        # system message is ignored; last user/assistant is "all good"
        assert adaptive_preserve_depth(messages, min_preserved=1, max_preserved=5) == 1


class TestSimpleCompactionWithPreserveDepth:
    """Test SimpleCompaction with callable preserve_depth."""

    def test_callable_preserve_depth(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
            Message(role="assistant", content=[TextPart(text="New reply")]),
        ]

        def _depth(msgs):
            return 3

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=_depth)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 4

    def test_int_preserve_depth(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
        ]

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=2)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 3

    def test_none_preserve_depth_uses_default(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
        ]

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=None)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 2
