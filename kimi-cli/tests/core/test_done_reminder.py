"""Tests for the DoneReminderProvider dynamic injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kosong.message import Message
from kosong.tooling import ToolReturnValue
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.dynamic_injection import DynamicInjection
from kimi_cli.soul.dynamic_injections.done_reminder import (
    _DONE_REMINDER_TEMPLATE,
    _DONE_REMINDER_TYPE,
    _contains_completion_keyword,
    DoneReminderProvider,
)
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.todo import TodoList
from kimi_cli.wire.types import TextPart, ThinkPart


def _make_soul(runtime: Runtime, tmp_path: Path, toolset: Any = None) -> KimiSoul:
    """Create a KimiSoul with the given (or a TodoList-equipped) toolset."""
    if toolset is None:
        toolset = KimiToolset()
        toolset.add(TodoList(runtime))
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


async def test_no_injection_without_assistant_message(
    runtime: Runtime, tmp_path: Path
) -> None:
    """No injection when history contains no assistant message."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [Message(role="user", content=[TextPart(text="hello")])]

    assert await provider.get_injections(history, soul) == []


async def test_no_injection_when_text_has_no_completion_word(
    runtime: Runtime, tmp_path: Path
) -> None:
    """No injection when the assistant text lacks any completion keyword."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [_assistant_message("I will start working on this now.")]

    assert await provider.get_injections(history, soul) == []


@pytest.mark.parametrize(
    "phrase",
    [
        # Single-word markers
        "done",
        "Done.",
        "It is finished",
        "completed",
        "complete",
        "resolved",
        "fixed",
        "closed",
        "verified",
        "approved",
        "shipped",
        "delivered",
        "merged",
        "deployed",
        "released",
        "published",
        "finalized",
        "finalised",
        "concluded",
        "accomplished",
        "addressed",
        "handled",
        "cleared",
        "settled",
        # Multi-word phrases
        "all done",
        "all set",
        "wrapped up",
        "taken care of",
        "good to go",
        "ready to go",
        "completed successfully",
        "successfully finished",
        "task complete",
        "tasks complete",
        "work complete",
        "implementation complete",
        "is complete",
        "are complete",
        "now complete",
        "is done",
        "are done",
        "has been done",
        "have been done",
        "no further action",
        "no more steps",
        "nothing left",
        "it works",
        "working as expected",
        "completed the task",
        "finished the task",
        "fixed the bug",
        "resolved the issue",
        "done with it",
        "done all",
        "finished all",
        "completed all",
        "resolved all",
        "fixed all",
    ],
)
async def test_injection_for_various_completion_phrases(
    runtime: Runtime, tmp_path: Path, phrase: str
) -> None:
    """A pending todo plus any completion phrase triggers the reminder."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [_assistant_message(f"Great, the work is {phrase}.")]

    with patch(
        "kimi_cli.soul.dynamic_injections.done_reminder.TodoList",
        return_value=AsyncMock(
            return_value=ToolReturnValue(
                is_error=False,
                output="Current todo list:\n- [pending] Write tests",
                message="",
                display=[],
            )
        ),
    ):
        injections = await provider.get_injections(history, soul)

    assert len(injections) == 1
    assert injections[0].type == _DONE_REMINDER_TYPE
    assert injections[0].content == _DONE_REMINDER_TEMPLATE


async def test_injection_when_text_contains_done_and_pending_todos(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Injection fires when completion language is present and todos are pending."""
    from kimi_cli.session_state import TodoItemState, save_session_state

    runtime.session.state.todos = [
        TodoItemState(title="Implement feature", status="pending"),
        TodoItemState(title="Write tests", status="in_progress"),
    ]
    save_session_state(runtime.session.state, runtime.session.dir)

    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [_assistant_message("I am done with the implementation.")]

    injections = await provider.get_injections(history, soul)

    assert len(injections) == 1
    assert injections[0].type == _DONE_REMINDER_TYPE


async def test_no_injection_when_all_todos_done(
    runtime: Runtime, tmp_path: Path
) -> None:
    """No injection when every todo is already marked done."""
    from kimi_cli.session_state import TodoItemState, save_session_state

    runtime.session.state.todos = [
        TodoItemState(title="Implement feature", status="done"),
    ]
    save_session_state(runtime.session.state, runtime.session.dir)

    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [_assistant_message("I am done with the implementation.")]

    assert await provider.get_injections(history, soul) == []


async def test_no_injection_when_todolist_tool_unavailable(
    runtime: Runtime, tmp_path: Path
) -> None:
    """No injection when the agent's toolset does not expose TodoList."""
    soul = _make_soul(runtime, tmp_path, toolset=EmptyToolset())
    provider = DoneReminderProvider()
    history = [_assistant_message("I am done.")]

    assert await provider.get_injections(history, soul) == []


async def test_no_injection_for_thinking_block_only(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Thinking/reasoning content must not count as assistant text."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()
    history = [_thinking_message("I am done with the implementation.")]

    with patch(
        "kimi_cli.soul.dynamic_injections.done_reminder.TodoList",
        return_value=AsyncMock(
            return_value=ToolReturnValue(
                is_error=False,
                output="Current todo list:\n- [pending] Write tests",
                message="",
                display=[],
            )
        ),
    ):
        assert await provider.get_injections(history, soul) == []


async def test_cooldown_prevents_duplicate_injection(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Throttling prevents re-injecting for the same message or too soon."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider(cooldown_steps=1)

    pending_output = "Current todo list:\n- [pending] Write tests"
    with patch(
        "kimi_cli.soul.dynamic_injections.done_reminder.TodoList",
        return_value=AsyncMock(
            return_value=ToolReturnValue(
                is_error=False,
                output=pending_output,
                message="",
                display=[],
            )
        ),
    ):
        # First assistant message at step 1 -> injects.
        soul._current_step_no = 1
        history1 = [_assistant_message("Done with step one.")]
        injections = await provider.get_injections(history1, soul)
        assert len(injections) == 1

        # Same message again -> no injection (same assistant index).
        injections = await provider.get_injections(history1, soul)
        assert injections == []

        # New assistant message at step 2, within cooldown -> no injection.
        soul._current_step_no = 2
        history2 = [
            _assistant_message("Done with step one."),
            Message(role="user", content=[TextPart(text="ok")]),
            _assistant_message("Also finished step two."),
        ]
        injections = await provider.get_injections(history2, soul)
        assert injections == []

        # New assistant message after cooldown has passed -> injects.
        soul._current_step_no = 3
        history3 = [
            _assistant_message("Done with step one."),
            Message(role="user", content=[TextPart(text="ok")]),
            _assistant_message("Also finished step two."),
            Message(role="user", content=[TextPart(text="great")]),
            _assistant_message("And done with step three."),
        ]
        injections = await provider.get_injections(history3, soul)
        assert len(injections) == 1


async def test_disabled_config_returns_empty() -> None:
    """A disabled provider never returns injections."""
    provider = DoneReminderProvider(enabled=False)
    soul = MagicMock()
    soul.is_subagent = False
    history = [_assistant_message("I am done.")]

    assert await provider.get_injections(history, soul) == []


async def test_subagent_is_skipped(runtime: Runtime, tmp_path: Path) -> None:
    """The provider does not inject reminders for subagent sessions."""
    soul = _make_soul(runtime, tmp_path)
    # Simulate a subagent by overriding the property source.
    soul._agent.runtime.role = "subagent"
    provider = DoneReminderProvider()
    history = [_assistant_message("I am done.")]

    assert await provider.get_injections(history, soul) == []


async def test_on_context_compacted_resets_state(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Reset hooks clear throttling state so the provider can fire again."""
    soul = _make_soul(runtime, tmp_path)
    provider = DoneReminderProvider()

    pending_output = "Current todo list:\n- [pending] Write tests"
    with patch(
        "kimi_cli.soul.dynamic_injections.done_reminder.TodoList",
        return_value=AsyncMock(
            return_value=ToolReturnValue(
                is_error=False,
                output=pending_output,
                message="",
                display=[],
            )
        ),
    ):
        soul._current_step_no = 1
        history = [_assistant_message("Done.")]
        assert len(await provider.get_injections(history, soul)) == 1

        # Re-running the same history would normally be throttled by assistant index.
        await provider.on_context_compacted()
        injections = await provider.get_injections(history, soul)
        assert len(injections) == 1

        # on_afk_changed should also reset state.
        await provider.on_afk_changed(True)
        injections = await provider.get_injections(history, soul)
        assert len(injections) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I am done.", True),
        ("This is finished.", True),
        ("All done here.", True),
        ("Nothing left to do.", True),
        ("It is complete.", True),
        ("No further action needed.", True),
        ("I will start soon.", False),
        ("Pending work remains.", False),
        ("undone is not a completion marker", False),
        ("doneness is not a completion marker", False),
    ],
)
def test_contains_completion_keyword(text: str, expected: bool) -> None:
    assert _contains_completion_keyword(text) is expected
