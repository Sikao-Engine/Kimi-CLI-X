"""Comprehensive tests for system_prompt ToolCallReason integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from kimix.utils.system_prompt import get_system_prompt, SystemPromptType
from kimi_cli.tools.reason import ToolCallReason

_OUTPUT_MD = Path(__file__).with_name("output.md")


def _append_prompt(test_name: str, prompt: str) -> None:
    with _OUTPUT_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n## {test_name}\n\n{prompt}\n")


class _MockParams(BaseModel):
    path: str
    reason: str = ""


class _MockTool:
    name: str = "WriteFile"

    def __init__(self, name: str = "WriteFile") -> None:
        self.name = name


def _make_runtime(tmp_path: Path, custom_data: dict[str, Any] | None = None) -> SimpleNamespace:
    """Build a minimal runtime-like object for get_system_prompt."""
    builtin_args = SimpleNamespace(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=tmp_path,
        KIMI_WORK_DIR_LS="",
        KIMI_AGENTS_MD="",
        KIMI_SKILLS="",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS="Windows",
        KIMI_SHELL="bash",
    )
    session = SimpleNamespace(
        dir=tmp_path,
        id="test-session",
        custom_data=custom_data or {},
    )
    return SimpleNamespace(
        builtin_args=builtin_args,
        session=session,
    )


class TestSystemPromptMemoryCompaction:
    """Test step memory compaction in system prompt."""

    def _write_steps(self, tmp_path: Path, count: int) -> None:
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        steps = []
        for i in range(1, count + 1):
            steps.append({
                "seq": i,
                "time": f"2024-01-{i:02d}T00:00:00+00:00",
                "brief": f"brief {i}",
                "step": f"step text {i}",
                "result": f"result {i}",
                "files": [f"file_{i}.py"],
            })
        (steps_dir / "test-session.json").write_text(json.dumps(steps), encoding="utf-8")

    def test_memory_no_compaction_under_limit(self, tmp_path: Path):
        self._write_steps(tmp_path, 3)
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_memory_no_compaction_under_limit", prompt)

        assert "Memory:" in prompt
        assert "step text 1" in prompt
        assert "step text 2" in prompt
        assert "step text 3" in prompt
        assert "[compacted]" not in prompt

    def test_memory_at_limit_no_compaction(self, tmp_path: Path):
        self._write_steps(tmp_path, 200)
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_memory_at_limit_no_compaction", prompt)

        assert "Memory:" in prompt
        assert "step text 1" in prompt
        assert "step text 200" in prompt
        assert "[compacted]" not in prompt

    def test_memory_one_over_limit_compacts(self, tmp_path: Path):
        self._write_steps(tmp_path, 201)
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_memory_one_over_limit_compacts", prompt)

        assert "Memory:" in prompt
        assert "[compacted]" in prompt
        # First half (first 100) compacted
        assert "step:[compacted] step text 1" in prompt
        assert "step:[compacted] step text 100" in prompt
        # Second half intact
        assert "step:step text 101" in prompt
        assert "step:step text 201" in prompt
        assert "step:[compacted] step text 101" not in prompt

    def test_memory_compacts_over_limit(self, tmp_path: Path):
        self._write_steps(tmp_path, 202)
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_memory_compacts_over_limit", prompt)

        assert "Memory:" in prompt
        # Oldest half (first 101) should be compacted
        assert "[compacted]" in prompt
        # First step should be compacted (step text prefixed with [compacted])
        assert "step:[compacted] step text 1" in prompt
        # Steps around the split should be compacted
        assert "step:[compacted] step text 50" in prompt
        assert "step:[compacted] step text 101" in prompt
        # Steps in the second half should remain intact (no [compacted] prefix)
        assert "step:step text 102" in prompt
        assert "step:step text 202" in prompt
        # Non-compacted steps should not have [compacted] in their step field
        assert "step:[compacted] step text 102" not in prompt
        assert "step:[compacted] step text 202" not in prompt


class TestSystemPromptAgentsMd:
    """Test AGENTS.md size limiting in system prompt."""

    def test_agents_md_too_long(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("x" * 5000, encoding="utf-8")

        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_agents_md_too_long", prompt)

        assert "read AGENTS.md before work" in prompt
        assert "x" * 100 not in prompt

    def test_agents_md_short(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        content = "Short agents file"
        agents_md.write_text(content, encoding="utf-8")

        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_agents_md_short", prompt)

        assert content in prompt
        assert "read AGENTS.md before work" not in prompt


class TestSystemPromptToolCallReason:
    """Test that ToolCallReason changed files appear in the system prompt."""

    def test_prompt_includes_changed_files(self, tmp_path: Path):
        tcr = ToolCallReason()
        tool = _MockTool()
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "a.py"), reason="create a"), tool)
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "b.py"), reason="create b"), tool)

        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": tcr})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_prompt_includes_changed_files", prompt)

        assert "Changed files:" in prompt
        assert str((tmp_path / "a.py").resolve()) in prompt
        assert str((tmp_path / "b.py").resolve()) in prompt
        assert "WriteFile: create a" in prompt
        assert "WriteFile: create b" in prompt

    def test_prompt_no_changed_files_when_empty(self, tmp_path: Path):
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_prompt_no_changed_files_when_empty", prompt)

        assert "Changed files:" not in prompt

    def test_prompt_no_changed_files_when_reason_not_instance(self, tmp_path: Path):
        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": "not an instance"})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_prompt_no_changed_files_when_reason_not_instance", prompt)

        assert "Changed files:" not in prompt

    def test_prompt_changed_files_edit_tool(self, tmp_path: Path):
        tcr = ToolCallReason()
        write_tool = _MockTool("WriteFile")
        edit_tool = _MockTool("EditFile")
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "config.py"), reason="create config"), write_tool)
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "config.py"), reason="update config"), edit_tool)

        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": tcr})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_prompt_changed_files_edit_tool", prompt)

        assert "Changed files:" in prompt
        config_line = [line for line in prompt.splitlines() if "config.py" in line][0]
        assert "WriteFile: create config" in config_line
        assert "EditFile: update config" in config_line

    def test_changed_files_sorted(self, tmp_path: Path):
        tcr = ToolCallReason()
        tool = _MockTool()
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "z.py"), reason="z"), tool)
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "a.py"), reason="a"), tool)

        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": tcr})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_changed_files_sorted", prompt)

        prompt_lines = prompt.splitlines()
        idx = prompt_lines.index("Changed files:")
        file_lines = prompt_lines[idx + 1 : idx + 3]
        assert len(file_lines) == 2
        # Paths should appear in sorted order
        assert "a.py" in file_lines[0]
        assert "z.py" in file_lines[1]

    def test_tool_call_reason_too_long_keeps_latest(self, tmp_path: Path):
        tcr = ToolCallReason()
        tool = _MockTool()
        for i in range(101):
            tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / f"file_{i:03d}.py"), reason=f"reason {i}"), tool)

        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": tcr})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_tool_call_reason_too_long_keeps_latest", prompt)

        assert "Changed files:" in prompt
        # Only the latest file should appear
        assert "file_100.py" in prompt
        assert "file_000.py" not in prompt
        assert "file_050.py" not in prompt
        # Only one file line
        prompt_lines = prompt.splitlines()
        idx = prompt_lines.index("Changed files:")
        file_lines = [line for line in prompt_lines[idx + 1 :] if line.strip().startswith("-")]
        assert len(file_lines) == 1

    def test_tool_call_reason_relative_paths(self, tmp_path: Path, monkeypatch: Any):
        monkeypatch.chdir(tmp_path)
        tcr = ToolCallReason()
        tool = _MockTool()
        tcr.add_tool_call_reason(_MockParams(path=str(tmp_path / "a.py"), reason="create a"), tool)

        runtime = _make_runtime(tmp_path, custom_data={"tool_call_reason": tcr})
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_tool_call_reason_relative_paths", prompt)

        assert "Changed files:" in prompt
        # Path should be relative since cwd == tmp_path
        assert str(tmp_path / "a.py") not in prompt
        assert "a.py" in prompt
