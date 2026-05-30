"""Comprehensive tests for the Bash tool (bash_tool.py) which uses the system bash executable."""

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk
from kimi_cli.session import Session

from kimi_cli.tools import SkipThisTool
from kimix.tools.file.bash import (
    Bash,
    BashParams,
    BASH_COMMANDS,
    WINDOWS_ALIASES,
)
from kimix.tools.file.bash.bash_tool import find_bash
from kimix.tools.background.utils import _pop_task_data


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


# ============================================================================
# find_bash
# ============================================================================

class TestFindBash:
    def test_returns_path_on_this_system(self) -> None:
        path = find_bash()
        assert path is not None
        assert Path(path).exists()

    def test_returns_basename_bash(self) -> None:
        path = find_bash()
        assert path is not None
        assert Path(path).name.lower() in ("bash.exe", "bash")


# ============================================================================
# BashParams
# ============================================================================

class TestBashParams:
    def test_defaults(self) -> None:
        p = BashParams(cmd="ls")
        assert p.cmd == "ls"
        assert p.args == []
        assert p.timeout == 10
        assert p.output_path is None
        assert p.cwd is None

    def test_full(self) -> None:
        p = BashParams(cmd="cat", args=["-n", "file.txt"], timeout=30, output_path="/tmp/out", cwd="/home")
        assert p.cmd == "cat"
        assert p.args == ["-n", "file.txt"]
        assert p.timeout == 30
        assert p.output_path == "/tmp/out"
        assert p.cwd == "/home"

    def test_timeout_min(self) -> None:
        with pytest.raises(Exception):
            BashParams(cmd="ls", timeout=1)

    def test_timeout_max(self) -> None:
        with pytest.raises(Exception):
            BashParams(cmd="ls", timeout=200)


# ============================================================================
# Bash.resolve_command
# ============================================================================

class TestBashResolveCommand:
    def test_known_builtin(self) -> None:
        name, tool = Bash.resolve_command("cat")
        assert name == "cat"
        assert tool is not None
        assert isinstance(tool, CallableTool2)

    def test_windows_alias_dir_to_ls(self) -> None:
        name, tool = Bash.resolve_command("dir")
        assert name == "ls"
        assert tool is not None

    def test_windows_alias_copy_to_cp(self) -> None:
        name, tool = Bash.resolve_command("copy")
        assert name == "cp"
        assert tool is not None

    def test_windows_alias_del_to_rm(self) -> None:
        name, tool = Bash.resolve_command("del")
        assert name == "rm"
        assert tool is not None

    def test_windows_alias_type_to_cat(self) -> None:
        name, tool = Bash.resolve_command("type")
        assert name == "cat"
        assert tool is not None

    def test_windows_alias_Get_ChildItem(self) -> None:
        name, tool = Bash.resolve_command("Get-ChildItem")
        assert name == "ls"
        assert tool is not None

    def test_unknown_command(self) -> None:
        name, tool = Bash.resolve_command("nonexistent_cmd_xyz")
        assert name == "nonexistent_cmd_xyz"
        assert tool is None

    def test_all_builtins_resolvable(self) -> None:
        for cmd_name in BASH_COMMANDS:
            name, tool = Bash.resolve_command(cmd_name)
            expected = WINDOWS_ALIASES.get(cmd_name, cmd_name)
            assert name == expected, f"Command {cmd_name} resolved to {name}, expected {expected}"
            assert tool is not None, f"Command {cmd_name} should resolve"


# ============================================================================
# Bash.__call__
# ============================================================================

class TestBashCall:
    async def test_echo_via_bash(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo", args=["hello"])
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output

    async def test_true_via_bash(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="true")
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_false_via_bash(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="false")
        result = await bash(params)
        assert isinstance(result, ToolError)

    async def test_unknown_command_error(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="no_such_command_12345", timeout=5)
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "command not found" in result.output or "not found" in result.output.lower()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-aliases apply universally but we test 'dir' alias")
    async def test_dir_alias_dispatches_to_ls(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="dir", args=[".", "-a"], timeout=10)
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_space_separated_cmd_args(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello world")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello world" in result.output

    async def test_known_builtin_with_timeout(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo", args=["quick"], timeout=30)
        result = await bash(params)
        assert isinstance(result, ToolOk)

    async def test_cat_builtin(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello cat", encoding="utf-8")
        bash = Bash(session=mock_session)
        params = BashParams(cmd="cat", args=[str(f)])
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "hello cat" in result.output

    async def test_pwd(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="pwd")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_whoami(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="whoami")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_empty_command(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="", timeout=5)
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "Empty command" in result.output

    async def test_timeout(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="sleep", args=["5"], timeout=3)
        result = await bash(params)
        assert isinstance(result, ToolError)
        assert "Timeout" in result.brief

    async def test_with_output_path(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "src.txt"
        out = tmp_path / "dst.txt"
        f.write_text("output_path_test", encoding="utf-8")
        bash = Bash(session=mock_session)
        params = BashParams(cmd="cat", args=[str(f)], output_path=str(out))
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "saved to file" in result.output
        assert "output_path_test" in out.read_text(encoding="utf-8")

    async def test_with_cwd(self, mock_session: MagicMock, tmp_path: Path) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="pwd", cwd=str(tmp_path))
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # Git bash on Windows translates Windows paths to Unix-style paths (e.g. /c/... or /tmp/...)
        assert tmp_path.name in result.output or str(tmp_path).replace("\\", "/") in result.output

    async def test_bash_not_found_fallback(self, mock_session: MagicMock) -> None:
        """When bash is not found, Bash.__init__ raises SkipThisTool."""
        with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=None):
            with pytest.raises(SkipThisTool):
                Bash(session=mock_session)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    async def test_windows_alias_all_known(self) -> None:
        for alias, target in WINDOWS_ALIASES.items():
            assert target in BASH_COMMANDS, f"Alias {alias} -> {target} not in BASH_COMMANDS"

    async def test_command_with_special_chars(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo", args=["hello\tworld"])
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # Tab may be preserved or converted by echo depending on bash version
        assert "hello" in result.output
        assert "world" in result.output

    async def test_command_with_quotes(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo", args=['"quoted text"'])
        result = await bash(params)
        assert isinstance(result, ToolOk)
        assert "quoted text" in result.output
