"""Comprehensive tests for run_bash, Bash tool class, and Run tool class."""

import asyncio
import queue
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.session import Session

from kimix.tools.file.bash import (
    Bash,
    BashParams,
    BASH_COMMANDS,
    WINDOWS_ALIASES,
)
from kimix.tools.file.bash.run_bash import run_bash
from kimix.tools.file.run import Run, RunParams
from kimix.tools.background.utils import (
    BackgroundStream,
    _pop_task_data,
    add_task,
    generate_task_id,
    remove_task_id,
)


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
        with pytest.raises(Exception):  # pydantic validation
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
            # Some bash builtins are shadowed by WINDOWS_ALIASES (e.g. 'type' -> 'cat')
            # In that case resolve_command returns the alias target, not the original name
            expected = WINDOWS_ALIASES.get(cmd_name, cmd_name)
            assert name == expected, f"Command {cmd_name} resolved to {name}, expected {expected}"
            assert tool is not None, f"Command {cmd_name} should resolve"


# ============================================================================
# Bash __call__
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
        assert "command not found" in result.output or "Unknown bash command" in result.output

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-aliases apply universally but we test 'dir' alias")
    async def test_dir_alias_dispatches_to_ls(self, mock_session: MagicMock) -> None:
        bash = Bash(session=mock_session)
        params = BashParams(cmd="dir", args=[".", "-a"], timeout=10)
        result = await bash(params)
        # ls should succeed listing current dir
        assert isinstance(result, ToolOk)

    async def test_space_separated_cmd_args(self, mock_session: MagicMock) -> None:
        """cmd='echo hello' should split and execute echo with arg 'hello'."""
        bash = Bash(session=mock_session)
        params = BashParams(cmd="echo hello world")
        result = await bash(params)
        assert isinstance(result, ToolOk)
        # The output should contain "hello world"
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


# ============================================================================
# run_bash - builtin command path
# ============================================================================

class TestRunBashBuiltin:
    async def test_echo(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="echo", args=["hello_builtin"])
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)
        assert "hello_builtin" in result.output

    async def test_pwd(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="pwd")
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_true(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="true")
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)

    async def test_false(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="false")
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolError)

    async def test_whoami(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="whoami")
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)
        assert len(result.output) > 0

    async def test_windows_alias_dir(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="dir", args=["."])
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)

    async def test_unknown_command(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="not_a_real_cmd_abc123", timeout=5)
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolError)
        assert "Unknown bash command" in result.output

    async def test_timeout_during_builtin(self, mock_session: MagicMock) -> None:
        """Use `sleep 5` with a very short timeout to trigger timeout."""
        params = BashParams(cmd="sleep", args=["5"], timeout=3)
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolError)
        assert "Timeout" in result.brief


# ============================================================================
# RunParams
# ============================================================================

class TestRunParams:
    def test_defaults(self) -> None:
        p = RunParams(command="python")
        assert p.command == "python"
        assert p.timeout == 10
        assert p.output_path is None
        assert p.cwd is None
        assert p.env is None

    def test_full(self) -> None:
        p = RunParams(command="/usr/bin/python -c print(1)", timeout=30,
                      output_path="/tmp/out", cwd="/tmp", env=["FOO=bar", "DEBUG"])
        assert p.command == "/usr/bin/python -c print(1)"
        assert p.env == ["FOO=bar", "DEBUG"]

    def test_timeout_range(self) -> None:
        with pytest.raises(Exception):
            RunParams(command="python", timeout=1)
        with pytest.raises(Exception):
            RunParams(command="python", timeout=901)


# ============================================================================
# Run __call__
# ============================================================================

class TestRunCall:
    async def test_run_python_print(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(command=f'{sys.executable} -c "print(\'run_ok\')"', timeout=15)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "run_ok" in result.output

    async def test_run_with_env(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import os; print(os.environ.get(\'KIMIX_TEST_VAR\', \'\'))"',
            env=["KIMIX_TEST_VAR=hello_env"],
            timeout=15,
        )
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "hello_env" in result.output

    async def test_run_with_env_no_equals(self, mock_session: MagicMock) -> None:
        """env='DEBUG' should set DEBUG=1."""
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import os; print(os.environ.get(\'KIMIX_DEBUG_VAR\', \'NOT_SET\'))"',
            env=["KIMIX_DEBUG_VAR"],
            timeout=15,
        )
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert result.output.strip() == "1"

    async def test_run_with_cwd(self, mock_session: MagicMock, tmp_path: Path) -> None:
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import os; print(os.getcwd())"',
            cwd=str(tmp_path),
            timeout=15,
        )
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert str(tmp_path) in result.output

    async def test_run_with_output_path(self, mock_session: MagicMock, tmp_path: Path) -> None:
        out = tmp_path / "output.txt"
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "print(\'output_file_test\')"',
            output_path=str(out),
            timeout=15,
        )
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "saved to file" in result.output
        assert "output_file_test" in out.read_text(encoding="utf-8")

    async def test_run_nonzero_exit(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import sys; sys.exit(42)"',
            timeout=15,
        )
        result = await tool(params)
        assert isinstance(result, ToolError)

    async def test_run_timeout(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import time; time.sleep(30)"',
            timeout=3,
        )
        result = await tool(params)
        assert isinstance(result, ToolError)
        assert "Timeout" in result.brief

    async def test_run_not_found(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(command="not_a_command_xyz_123", timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolError)

    async def test_run_python_alias(self, mock_session: MagicMock) -> None:
        """`python` path is replaced with sys.executable."""
        tool = Run(session=mock_session)
        params = RunParams(command='python -c "print(\'alias_ok\')"', timeout=15)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "alias_ok" in result.output

    async def test_run_falls_back_to_bash_builtin(self, mock_session: MagicMock) -> None:
        """When not a real process, Run should fall back to bash builtins."""
        tool = Run(session=mock_session)
        params = RunParams(command="echo hello_from_run", timeout=15)
        result = await tool(params)
        assert not result.is_error
        assert "hello_from_run" in result.output

    async def test_run_windows_alias_fallback(self, mock_session: MagicMock) -> None:
        """dir is not a real executable but a Windows alias -> ls."""
        tool = Run(session=mock_session)
        params = RunParams(command="dir .", timeout=10)
        result = await tool(params)
        assert not result.is_error

    async def test_space_separated_path_with_args(self, mock_session: MagicMock) -> None:
        """command='python -c print(1)' splits path and args."""
        tool = Run(session=mock_session)
        params = RunParams(command=f"{sys.executable} -c print('space_split')", timeout=15)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "space_split" in result.output

    async def test_space_separated_path_with_file_lookup(self, mock_session: MagicMock, tmp_path: Path) -> None:
        """Progressive prefix lookup for paths with spaces."""
        if sys.platform == "win32":
            pytest.skip("shlex.split on Windows does not handle spaces in arguments consistently")
        script = tmp_path / "my script.py"
        script.write_text("print('spaces_ok')")
        tool = Run(session=mock_session)
        params = RunParams(command=f"{sys.executable} {script}", timeout=15)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "spaces_ok" in result.output


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases:
    async def test_run_bash_with_RunParams(self, mock_session: MagicMock) -> None:
        """run_bash accepts RunParams (used by Run.__call__ for fallback)."""
        params = RunParams(command="echo hello_runparams")
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)
        assert "hello_runparams" in result.output

    async def test_run_bash_cmd_empty(self, mock_session: MagicMock) -> None:
        params = BashParams(cmd="", timeout=5)
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolError)
        assert "Unknown bash command" in result.output

    async def test_concurrent_runs(self, mock_session: MagicMock) -> None:
        """Multiple concurrent Run calls should be fine with semaphore."""
        tool = Run(session=mock_session)

        async def one_run(i: int) -> ToolReturnValue:
            params = RunParams(
                command=f'{sys.executable} -c "print(\'concurrent_{i}\')"',
                timeout=15,
            )
            return await tool(params)

        results = await asyncio.gather(*(one_run(i) for i in range(4)))
        for i, r in enumerate(results):
            assert isinstance(r, ToolOk), f"Run {i} failed: {r}"
            assert f"concurrent_{i}" in r.output

    async def test_bash_cat_with_output_path(self, mock_session: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "src.txt"
        out = tmp_path / "dst.txt"
        f.write_text("output_path_test", encoding="utf-8")
        params = BashParams(cmd="cat", args=[str(f)], output_path=str(out))
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolOk)
        # Cat with output_path should save to file
        if "saved to file" in result.output:
            assert "output_path_test" in out.read_text()

    async def test_windows_alias_all_known(self) -> None:
        """All WINDOWS_ALIASES map to known BASH_COMMANDS."""
        for alias, target in WINDOWS_ALIASES.items():
            assert target in BASH_COMMANDS, f"Alias {alias} -> {target} not in BASH_COMMANDS"

    async def test_bash_builtin_raises_exception(self, mock_session: MagicMock) -> None:
        """If a builtin raises, it should be caught and returned as error."""
        # Use 'cp' with no args, which requires source/dest operands
        params = BashParams(cmd="cp", args=[])
        result = await run_bash(params, mock_session)
        assert isinstance(result, ToolError)
        assert "missing" in result.message.lower()
