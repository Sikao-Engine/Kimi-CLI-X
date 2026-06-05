"""Comprehensive tests for the PowerShell tool (pwsh_tool.py)."""

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool
from kimix.tools.file.bash import (
    Powershell,
    PowershellParams,
)
from kimix.tools.file.run import find_pwsh
from kimix.tools.background.utils import _pop_task_data


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.custom_data = {}
    session.custom_config = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


@pytest.fixture(autouse=True)
def clear_find_pwsh_cache() -> None:
    """Clear the lru_cache on find_pwsh before each test."""
    find_pwsh.cache_clear()
    yield


# ============================================================================
# find_pwsh
# ============================================================================

class TestFindPwsh:
    def test_returns_path_on_this_system(self) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        path = find_pwsh()
        assert path is not None
        assert Path(path).exists()

    def test_returns_pwsh_or_powershell_basename(self) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        path = find_pwsh()
        assert path is not None
        name = Path(path).name.lower()
        assert name in ("pwsh.exe", "powershell.exe")

    def test_returns_none_on_non_windows(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "linux"):
            assert find_pwsh() is None

    def test_finds_pwsh_first_on_windows(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", side_effect=lambda name: (
                "C:\\Program Files\\PowerShell\\7\\pwsh.exe" if name in ("pwsh", "pwsh.exe") else None
            )):
                path = find_pwsh()
                assert path is not None
                assert Path(path).name.lower() == "pwsh.exe"

    def test_falls_back_to_powershell_when_pwsh_missing(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            def _which(name: str) -> str | None:
                if name in ("powershell", "powershell.exe"):
                    return "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
                return None
            with patch("kimix.tools.file.run.shutil.which", side_effect=_which):
                path = find_pwsh()
                assert path is not None
                assert Path(path).name.lower() == "powershell.exe"

    def test_returns_none_when_neither_found(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", return_value=None):
                with patch("subprocess.run", side_effect=FileNotFoundError):
                    with patch.object(Path, "exists", return_value=False):
                        assert find_pwsh() is None

    def test_uses_where_exe_for_pwsh(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", return_value=None):
                proc = MagicMock()
                proc.stdout = "C:\\Program Files\\PowerShell\\7\\pwsh.exe\n"
                with patch("subprocess.run", return_value=proc):
                    path = find_pwsh()
                    assert path is not None
                    assert "pwsh.exe" in path.lower()

    def test_uses_where_exe_for_powershell_fallback(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", return_value=None):
                pwsh_proc = MagicMock()
                pwsh_proc.check_returncode.side_effect = Exception("not found")
                ps_proc = MagicMock()
                ps_proc.stdout = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\n"

                def _subprocess_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                    if "pwsh" in cmd[1]:
                        raise subprocess.CalledProcessError(1, cmd)
                    return ps_proc

                import subprocess
                with patch("subprocess.run", side_effect=_subprocess_run):
                    path = find_pwsh()
                    assert path is not None
                    assert "powershell.exe" in path.lower()

    def test_common_paths_fallback_pwsh(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", return_value=None):
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["where.exe"])):
                    candidate = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")

                    def _exists(self: Path) -> bool:
                        return str(self) == str(candidate)

                    with patch.object(Path, "exists", _exists):
                        with patch.object(Path, "resolve", lambda self: self):
                            path = find_pwsh()
                            assert path is not None
                            assert "pwsh.exe" in path.lower()

    def test_common_paths_fallback_windows_powershell(self) -> None:
        with patch("kimix.tools.file.run.sys.platform", "win32"):
            with patch("kimix.tools.file.run.shutil.which", return_value=None):
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["where.exe"])):
                    candidate = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")

                    def _exists(self: Path) -> bool:
                        return str(self) == str(candidate)

                    with patch.object(Path, "exists", _exists):
                        with patch.object(Path, "resolve", lambda self: self):
                            path = find_pwsh()
                            assert path is not None
                            assert "powershell.exe" in path.lower()


# ============================================================================
# PowershellParams
# ============================================================================

class TestPowershellParams:
    def test_defaults(self) -> None:
        p = PowershellParams(cmd="Write-Output hello")
        assert p.cmd == "Write-Output hello"
        assert p.timeout == 10

    def test_full(self) -> None:
        p = PowershellParams(cmd="Get-Process", timeout=30)
        assert p.cmd == "Get-Process"
        assert p.timeout == 30

    def test_timeout_min(self) -> None:
        with pytest.raises(Exception):
            PowershellParams(cmd="Write-Output hello", timeout=1)

    def test_timeout_max(self) -> None:
        with pytest.raises(Exception):
            PowershellParams(cmd="Write-Output hello", timeout=901)


# ============================================================================
# Powershell.__init__
# ============================================================================

class TestPowershellInit:
    def test_skips_on_non_windows(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.sys.platform", "linux"):
            with pytest.raises(SkipThisTool):
                Powershell(session=mock_session)

    def test_skips_when_pwsh_not_found(self, mock_session: MagicMock) -> None:
        with patch("kimix.tools.file.bash.pwsh_tool.sys.platform", "win32"):
            with patch("kimix.tools.file.bash.pwsh_tool.find_pwsh", return_value=None):
                with pytest.raises(SkipThisTool):
                    Powershell(session=mock_session)

    def test_initializes_on_windows_when_found(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        assert ps._pwsh is not None
        assert ps.name == "Powershell"


# ============================================================================
# Powershell.__call__
# ============================================================================

class TestPowershellCall:
    async def test_echo_hello(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Write-Output hello")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output

    async def test_echo_with_multiple_args(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Write-Output hello world")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output
        assert "world" in result.output

    async def test_empty_command(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="", timeout=5)
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "Empty command" in result.output

    async def test_timeout(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Start-Sleep -Seconds 5", timeout=3)
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "Timeout" in result.brief

    async def test_error_command(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="throw ' intentional error '")
        result = await ps(params)
        assert isinstance(result, ToolError)

    async def test_pipe_command(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="'hello', 'world' | Select-Object -First 1")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output

    async def test_variable_expansion(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="$env:COMPUTERNAME")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert len(result.output.strip()) > 0

    async def test_forbidden_command_taskkill(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="taskkill /IM notepad.exe")
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "Forbidden command" in result.brief

    async def test_forbidden_command_kill(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="kill 1234")
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "Forbidden command" in result.brief

    async def test_custom_forbidden_command(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        mock_session.custom_config = {"config_json": {"forbidden_commands": ["Remove-Item"]}}
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Remove-Item C:\\some\\file.txt")
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "Forbidden command" in result.brief

    async def test_non_forbidden_command_passes(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        mock_session.custom_config = {"config_json": {"forbidden_commands": ["Remove-Item"]}}
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Write-Output safe")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "safe" in result.output

    async def test_get_location(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Get-Location")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert len(result.output.strip()) > 0

    async def test_complex_pipeline(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="1..5 | ForEach-Object { $_ * 2 } | Select-Object -First 3")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "2" in result.output

    async def test_if_statement(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="if ($true) { Write-Output 'TRUE' } else { Write-Output 'FALSE' }")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "TRUE" in result.output

    async def test_try_catch(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="try { 1 / 0 } catch { Write-Output 'caught' }")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "caught" in result.output


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    async def test_command_with_special_chars(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Write-Output 'hello`tworld'")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "hello" in result.output
        assert "world" in result.output

    async def test_command_with_quotes(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd='Write-Output "quoted text"')
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "quoted text" in result.output

    async def test_very_long_command(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        long_text = "x" * 5000
        params = PowershellParams(cmd=f"Write-Output '{long_text}'")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert long_text in result.output

    async def test_command_with_backslashes_in_path(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Test-Path 'C:\\Windows\\System32'")
        result = await ps(params)
        assert isinstance(result, ToolOk)
        assert "True" in result.output


# ============================================================================
# Background task support
# ============================================================================

class TestBackgroundTasks:
    async def test_long_running_command_returns_task_id(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Start-Sleep -Seconds 10", timeout=3)
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "task_id" in result.output

    async def test_background_task_can_be_retrieved(self, mock_session: MagicMock) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        ps = Powershell(session=mock_session)
        params = PowershellParams(cmd="Start-Sleep -Seconds 10", timeout=3)
        result = await ps(params)
        assert isinstance(result, ToolError)
        assert "task_id" in result.output
        # Verify the task_id is in the expected format
        assert "pwsh" in result.output
