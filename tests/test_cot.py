"""Comprehensive tests for manual Chain-of-Thought (CoT) system."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimix.cot import (
    CoTResult,
    _build_prompt,
    _parse_response,
    cot_prompt,
    cot_prompt_async,
    cot_prompt_with_verification,
    cot_prompt_with_verification_async,
)


@pytest.fixture(autouse=True)
def mock_cot_session():
    with patch("kimix.cot.get_default_session", return_value=MagicMock()):
        yield


class TestBuildPrompt:
    def test_basic_prompt(self) -> None:
        prompt = _build_prompt("What is 2+2?")
        assert "Think step by step" in prompt
        assert "<thinking>" in prompt
        assert "<quit/>" in prompt
        assert "What is 2+2?" in prompt

    def test_self_verify_appended(self) -> None:
        prompt = _build_prompt("What is 2+2?", self_verify=True)
        assert "Review your reasoning" in prompt
        assert "errors" in prompt

    def test_self_verify_false(self) -> None:
        prompt = _build_prompt("What is 2+2?", self_verify=False)
        assert "Review your reasoning" not in prompt

    def test_existing_thinking(self) -> None:
        prior = "I think 2+2 = 4"
        prompt = _build_prompt("What is 2+2?", existing_thinking=prior)
        assert "Continue from the prior thinking" in prompt
        assert prior in prompt
        assert "<thinking>" in prompt

    def test_existing_thinking_and_self_verify(self) -> None:
        prior = "Initial thought"
        prompt = _build_prompt("Q", existing_thinking=prior, self_verify=True)
        assert "Continue from the prior thinking" in prompt
        assert "Initial thought" in prompt
        assert "Review your reasoning" in prompt

    def test_prompt_strips_whitespace(self) -> None:
        prompt = _build_prompt("  hello  ", self_verify=False)
        assert prompt.endswith("hello")


class TestParseResponse:
    def test_full_tags(self) -> None:
        text = (
            "Some preamble\n"
            "<thinking>Step 1: add. Step 2: conclude.</thinking>\n"
            "<quit/>\n"
            "postamble"
        )
        result = _parse_response(text)
        assert result.thinking == "Step 1: add. Step 2: conclude."
        assert result.quit is True

    def test_missing_thinking(self) -> None:
        text = "<quit/>"
        result = _parse_response(text)
        assert result.thinking == ""
        assert result.quit is True

    def test_only_thinking(self) -> None:
        text = "<thinking>just thinking</thinking>"
        result = _parse_response(text)
        assert result.thinking == "just thinking"
        assert result.quit is False

    def test_no_tags(self) -> None:
        text = "Plain text response"
        result = _parse_response(text)
        assert result.thinking == ""
        assert result.quit is False

    def test_case_insensitive_tags(self) -> None:
        text = "<THINKING>Upper</THINKING><QUIT/>"
        result = _parse_response(text)
        assert result.thinking == "Upper"
        assert result.quit is True

    def test_multiline_content(self) -> None:
        text = (
            "<thinking>\nLine 1\nLine 2\n</thinking>\n"
            "<quit/>"
        )
        result = _parse_response(text)
        assert result.thinking == "Line 1\nLine 2"
        assert result.quit is True

    def test_empty_tags(self) -> None:
        text = "<thinking></thinking><quit/>"
        result = _parse_response(text)
        assert result.thinking == ""
        assert result.quit is True

    def test_only_whitespace_in_tags(self) -> None:
        text = "<thinking>   </thinking><quit/>"
        result = _parse_response(text)
        assert result.thinking == ""
        assert result.quit is True


class TestCotPromptAsync:
    @pytest.mark.asyncio
    async def test_basic_async(self) -> None:
        async def mock_prompt(prompt_str: str, session: Any) -> str:
            return (
                f"<thinking>received: {prompt_str[:20]}...</thinking>"
                f"<quit/>"
            )

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async("hello")
        assert isinstance(result, CoTResult)
        assert "received:" in result.thinking
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_self_verify_false(self) -> None:
        calls: list[str] = []

        async def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<thinking>t</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async("q", self_verify=False)
        assert "Review your reasoning" not in calls[0]
        assert result.thinking == "t"
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_self_verify_true(self) -> None:
        calls: list[str] = []

        async def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<thinking>t</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            await cot_prompt_async("q", self_verify=True)
        assert "Review your reasoning" in calls[0]

    @pytest.mark.asyncio
    async def test_existing_thinking_passed(self) -> None:
        async def mock_prompt(prompt_str: str, session: Any) -> str:
            assert "Continue from the prior thinking" in prompt_str
            assert "old thought" in prompt_str
            return "<thinking>new</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async(
                "q", existing_thinking="old thought"
            )
        assert result.thinking == "new"
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_async_mock_callback(self) -> None:
        mock = AsyncMock(return_value="<thinking>t</thinking><quit/>")
        with patch("kimix.cot._prompt_to_text_async", side_effect=mock):
            result = await cot_prompt_async("q")
        mock.assert_awaited_once()
        assert result.thinking == "t"
        assert result.quit is True


class TestCotPromptSync:
    def test_basic_sync(self) -> None:
        def mock_prompt(prompt_str: str, session: Any) -> str:
            return "<thinking>sync t</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            result = cot_prompt("hello")
        assert result.thinking == "sync t"
        assert result.quit is True

    def test_existing_thinking_sync(self) -> None:
        def mock_prompt(prompt_str: str, session: Any) -> str:
            assert "Continue from the prior thinking" in prompt_str
            return "<thinking>t</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            result = cot_prompt("q", existing_thinking="prior")
        assert result.thinking == "t"
        assert result.quit is True

    def test_self_verify_sync(self) -> None:
        calls: list[str] = []

        def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<thinking>t</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            cot_prompt("q", self_verify=True)
        assert "Review your reasoning" in calls[0]


class TestCotPromptWithVerificationAsync:
    @pytest.mark.asyncio
    async def test_two_pass(self) -> None:
        calls: list[str] = []

        async def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            if len(calls) == 1:
                return "<thinking>first</thinking><quit/>"
            return "<thinking>second</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_with_verification_async("q")
        assert len(calls) == 2
        assert "first" in calls[1]
        assert result.thinking == "second"
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_short_circuit_on_empty_thinking(self) -> None:
        calls: list[str] = []

        async def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_with_verification_async("q")
        assert len(calls) == 1
        assert result.thinking == ""
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_uses_existing_thinking(self) -> None:
        calls: list[str] = []

        async def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<thinking>refined</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_with_verification_async(
                "q", existing_thinking="prior"
            )
        assert "prior" in calls[0]
        assert result.thinking == "refined"
        assert result.quit is True


class TestCotPromptWithVerificationSync:
    def test_two_pass_sync(self) -> None:
        calls: list[str] = []

        def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            if len(calls) == 1:
                return "<thinking>init</thinking><quit/>"
            return "<thinking>final</thinking><quit/>"

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            result = cot_prompt_with_verification("q")
        assert len(calls) == 2
        assert result.thinking == "final"
        assert result.quit is True

    def test_short_circuit_sync(self) -> None:
        calls: list[str] = []

        def mock_prompt(prompt_str: str, session: Any) -> str:
            calls.append(prompt_str)
            return "<quit/>"

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            result = cot_prompt_with_verification("q")
        assert len(calls) == 1
        assert result.thinking == ""
        assert result.quit is True


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_response_async(self) -> None:
        async def mock_prompt(prompt_str: str, session: Any) -> str:
            return ""

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async("q")
        assert result.thinking == ""
        assert result.quit is False

    def test_empty_response_sync(self) -> None:
        def mock_prompt(prompt_str: str, session: Any) -> str:
            return ""

        with patch("kimix.cot._prompt_to_text", side_effect=mock_prompt):
            result = cot_prompt("q")
        assert result.thinking == ""
        assert result.quit is False

    @pytest.mark.asyncio
    async def test_multiple_thinking_tags(self) -> None:
        """Only the first occurrence should be captured."""
        async def mock_prompt(prompt_str: str, session: Any) -> str:
            return (
                "<thinking>first</thinking>"
                "<thinking>second</thinking>"
                "<quit/>"
            )

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async("q")
        assert result.thinking == "first"
        assert result.quit is True

    @pytest.mark.asyncio
    async def test_multiple_quit_tags(self) -> None:
        async def mock_prompt(prompt_str: str, session: Any) -> str:
            return (
                "<thinking>t</thinking>"
                "<quit/>"
                "<quit/>"
            )

        with patch("kimix.cot._prompt_to_text_async", side_effect=mock_prompt):
            result = await cot_prompt_async("q")
        assert result.thinking == "t"
        assert result.quit is True
