"""Tests for the ResilienceReminderProvider dynamic injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.dynamic_injection import DynamicInjection
from kimi_cli.soul.dynamic_injections.resilience_reminder import (
    _RESILIENCE_REMINDER_TEMPLATE,
    _RESILIENCE_REMINDER_TYPE,
    _contains_give_up_language,
    ResilienceReminderProvider,
)
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.types import TextPart, ThinkPart


def _make_soul(runtime: Runtime, tmp_path: Path, toolset: Any = None) -> KimiSoul:
    """Create a KimiSoul with the given (or empty) toolset."""
    if toolset is None:
        toolset = EmptyToolset()
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


def _assistant_message(text: str) -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _thinking_message(think_text: str) -> Message:
    return Message(role="assistant", content=[ThinkPart(think=think_text)])


def _mixed_message(text: str, think_text: str) -> Message:
    return Message(role="assistant", content=[TextPart(text=text), ThinkPart(think=think_text)])


async def test_no_injection_without_assistant_message() -> None:
    """No injection when history contains no assistant message."""
    provider = ResilienceReminderProvider()
    soul = MagicMock()
    soul.is_subagent = False
    soul._current_step_no = 1
    history = [Message(role="user", content=[TextPart(text="hello")])]

    assert await provider.get_injections(history, soul) == []


async def test_no_injection_when_text_has_no_give_up_language(
    runtime: Runtime, tmp_path: Path
) -> None:
    """No injection when the assistant text lacks resignation language."""
    soul = _make_soul(runtime, tmp_path)
    provider = ResilienceReminderProvider()
    history = [_assistant_message("I will start working on this now.")]

    assert await provider.get_injections(history, soul) == []


async def test_injection_when_text_contains_give_up_language(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Injection fires when the latest assistant text contains give-up language."""
    soul = _make_soul(runtime, tmp_path)
    soul._current_step_no = 1
    provider = ResilienceReminderProvider()
    history = [_assistant_message("This is impossible to implement.")]

    injections = await provider.get_injections(history, soul)

    assert len(injections) == 1
    assert injections[0].type == _RESILIENCE_REMINDER_TYPE
    assert injections[0].content == _RESILIENCE_REMINDER_TEMPLATE


async def test_injection_when_think_part_contains_give_up_language(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Injection fires when a ThinkPart (reasoning block) contains give-up language."""
    soul = _make_soul(runtime, tmp_path)
    soul._current_step_no = 1
    provider = ResilienceReminderProvider()
    history = [_thinking_message("Actually, this approach is impossible.")]

    injections = await provider.get_injections(history, soul)

    assert len(injections) == 1
    assert injections[0].type == _RESILIENCE_REMINDER_TYPE


async def test_injection_when_mixed_text_and_think_part_contains_give_up_language(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Injection fires when either TextPart or ThinkPart contains resignation language."""
    soul = _make_soul(runtime, tmp_path)
    soul._current_step_no = 1
    provider = ResilienceReminderProvider()
    history = [_mixed_message("I will keep trying.", "This is actually impossible.")]

    injections = await provider.get_injections(history, soul)

    assert len(injections) == 1
    assert injections[0].type == _RESILIENCE_REMINDER_TYPE


async def test_subagent_is_skipped(runtime: Runtime, tmp_path: Path) -> None:
    """The provider does not inject reminders for subagent sessions."""
    soul = _make_soul(runtime, tmp_path)
    soul._agent.runtime.role = "subagent"
    provider = ResilienceReminderProvider()
    history = [_assistant_message("This is impossible.")]

    assert await provider.get_injections(history, soul) == []


async def test_disabled_config_returns_empty() -> None:
    """A disabled provider never returns injections."""
    provider = ResilienceReminderProvider(enabled=False)
    soul = MagicMock()
    soul.is_subagent = False
    soul._current_step_no = 1
    history = [_assistant_message("This is impossible.")]

    assert await provider.get_injections(history, soul) == []


async def test_cooldown_prevents_duplicate_injection(runtime: Runtime, tmp_path: Path) -> None:
    """Throttling prevents re-injecting for the same message or too soon."""
    soul = _make_soul(runtime, tmp_path)
    provider = ResilienceReminderProvider(cooldown_steps=3)

    # First assistant message at step 1 -> injects.
    soul._current_step_no = 1
    history1 = [_assistant_message("This is impossible.")]
    injections = await provider.get_injections(history1, soul)
    assert len(injections) == 1

    # Same message again -> no injection (same assistant index).
    injections = await provider.get_injections(history1, soul)
    assert injections == []

    # New assistant message at step 2, within cooldown -> no injection.
    soul._current_step_no = 2
    history2 = [
        _assistant_message("This is impossible."),
        Message(role="user", content=[TextPart(text="ok")]),
        _assistant_message("Still impossible."),
    ]
    injections = await provider.get_injections(history2, soul)
    assert injections == []

    # New assistant message after cooldown has passed -> injects.
    soul._current_step_no = 5
    history3 = [
        _assistant_message("This is impossible."),
        Message(role="user", content=[TextPart(text="ok")]),
        _assistant_message("Still impossible."),
        Message(role="user", content=[TextPart(text="keep going")]),
        _assistant_message("Maybe it is not feasible."),
    ]
    injections = await provider.get_injections(history3, soul)
    assert len(injections) == 1


async def test_on_context_compacted_and_afk_reset_throttling(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Reset hooks clear throttling state so the provider can fire again."""
    soul = _make_soul(runtime, tmp_path)
    provider = ResilienceReminderProvider()

    soul._current_step_no = 1
    history = [_assistant_message("This is impossible.")]
    assert len(await provider.get_injections(history, soul)) == 1

    # Re-running the same history would normally be throttled by assistant index.
    await provider.on_context_compacted()
    injections = await provider.get_injections(history, soul)
    assert len(injections) == 1

    await provider.on_afk_changed(True)
    injections = await provider.get_injections(history, soul)
    assert len(injections) == 1


@pytest.mark.parametrize(
    "phrase",
    [
        # Explicit defeat
        "I give up",
        "giving up",
        "This can't be done",
        "impossible",
        "not possible",
        "no viable approach",
        "no viable solution",
        "cannot be expressed",
        "cannot be implemented",
        "architectural limitations",
        "fundamental limitation",
        "requires significant compiler changes",
        "require inline SPIR-V",
        "not feasible",
        "not workable",
        "cannot fix",
        "cannot address",
        "cannot resolve",
        "won't fix",
        "will not fix",
        "not fixable",
        "unfixable",
        "irreconcilable",
        "intractable",
        "insurmountable",
        # Pragmatic resignation / documentation fallback
        "pragmatic approach",
        "pragmatically",
        "document what works and what doesn't",
        "document the limitation",
        "step back",
        "given the time I've spent",
        "let me finalize",
        "let me update the report and finalize",
        "accept that these operations can't",
        "the only viable approaches are",
        "given the constraints",
        "given the time",
        "not worth the effort",
        "diminishing returns",
        "not practical",
        "not realistic",
        "too complex to",
        "too complicated to",
        "too risky to",
        # External / pre-existing excuses
        "pre-existing",
        "preexisting",
        "pre-existing issue",
        "pre-existing limitation",
        "existing limitation",
        "existing issue",
        "legacy issue",
        "legacy limitation",
        "inherited limitation",
        "inherited problem",
        "by design",
        "works as designed",
        "out of scope",
        "not in scope",
        "not my responsibility",
        "not our responsibility",
        "upstream issue",
        "upstream limitation",
        "third-party limitation",
        "third-party issue",
        "external dependency",
        "blocked by",
        "blocked on",
        "waiting for",
        "depends on",
        "beyond the scope",
        "outside the scope",
        # Abandonment framing
        "I think we should stop",
        "we should stop",
        "let's stop",
        "it is time to stop",
        "call it done",
        "call it complete",
        "declare victory",
        "declare defeat",
        "throw in the towel",
        "cut our losses",
        "move on",
        "let's move on",
        "we should move on",
        "park this",
        "shelve this",
        "table this",
        "put this aside",
        "put this on hold",
        "on hold",
        "punt on",
        "punt this",
        "defer this",
        "defer indefinitely",
        # Concessive framing
        "there is no way",
        "there's no way",
        "no way to",
        "we have to accept",
        "must accept",
        "this is problematic",
        "I realize this approach is problematic",
    ],
)
async def test_detection_phrases(runtime: Runtime, tmp_path: Path, phrase: str) -> None:
    """Each documented resignation phrase triggers the resilience reminder."""
    soul = _make_soul(runtime, tmp_path)
    soul._current_step_no = 1
    provider = ResilienceReminderProvider()
    history = [_assistant_message(f"Hmm, {phrase}.")]

    injections = await provider.get_injections(history, soul)
    assert len(injections) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I give up.", True),
        ("This is impossible.", True),
        ("We should stop here.", True),
        ("It is a pre-existing limitation.", True),
        ("Let's move on.", True),
        ("There's no way to fix this.", True),
        ("I realize this is problematic.", True),
        ("I will keep trying.", False),
        ("This works well.", False),
        ("Next step is to add tests.", False),
        ("Scope includes everything.", False),
    ],
)
def test_contains_give_up_language(text: str, expected: bool) -> None:
    assert _contains_give_up_language(text) is expected
