from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any

import pytest
from kosong.chat_provider import APIStatusError

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float = 0.0
    context_tokens: int = 0


class FakeSession:
    def __init__(self, raises: Exception | None = None) -> None:
        self.status = FakeStatus()
        self._cancel_event = None
        self.cancelled = False
        self.raises = raises

    async def prompt(self, prompt_str: str, *, merge_wire_messages: bool = False) -> Any:
        if self.raises is not None:
            raise self.raises
        # Force this into an async generator that yields nothing.
        if False:
            yield None

    def cancel(self) -> None:
        self.cancelled = True


def _suppress_output(monkeypatch: Any) -> list[str]:
    printed: list[str] = []

    def _capture(text: str, *args: Any, **kwargs: Any) -> None:
        printed.append(text)

    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", _capture)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *args, **kwargs: None)
    return printed


@pytest.mark.asyncio
async def test_run_single_prompt_propagates_api_status_error(monkeypatch: Any) -> None:
    """A 429 from the low-level layer must not trigger the old high-level backoff."""
    printed = _suppress_output(monkeypatch)
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    session = FakeSession(raises=APIStatusError(429, "rate limited"))

    with pytest.raises(APIStatusError, match="rate limited"):
        await prompt_mod._run_single_prompt(
            session, "hi", output_function=None, cancel_callable=None,
            merge_wire_messages=False, info_print=False,
        )

    # No high-level "Rate limited" message or status-code string matching.
    assert not any("Rate limited" in text for text in printed)
    # 429 is handled by the low-level layer; the high-level utility must not sleep.
    assert sleeps == []
    assert session.cancelled


@pytest.mark.asyncio
async def test_run_single_prompt_propagates_non_retryable_400(monkeypatch: Any) -> None:
    """A 400 Bad Request must not be retried at the high level."""
    printed = _suppress_output(monkeypatch)
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    session = FakeSession(raises=APIStatusError(400, "bad request"))

    with pytest.raises(APIStatusError, match="bad request"):
        await prompt_mod._run_single_prompt(
            session, "hi", output_function=None, cancel_callable=None,
            merge_wire_messages=False, info_print=False,
        )

    assert not any("Rate limited" in text for text in printed)
    # 400 is not retried at the high level.
    assert sleeps == []
    assert session.cancelled
