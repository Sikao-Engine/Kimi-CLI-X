"""PowerShell tool that executes commands via the system PowerShell executable."""

import functools
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import kimi_cli
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.display import ShellDisplayBlock
from kimix.tools.file.bash import bash_tool as _bash_tool
from kimix.tools.file.bash.proccess_pwsh import pwsh_transform
from kimix.tools.common import _maybe_export_output_async, _summarize_long_output_async, ProcessTask

if TYPE_CHECKING:
    from kimi_agent_sdk import CallableTool2 as _CallableTool2

def _print_warning(message: str) -> None:
    """Print a yellow WARNING message to stderr."""
    yellow = "\033[33m"
    reset = "\033[0m"
    print(f"{yellow}WARNING: {message}{reset}", file=sys.stderr, flush=True)


def _pwsh_major_version(path: str) -> int | None:
    """Return the major version reported by a PowerShell executable, or None."""
    try:
        output = subprocess.check_output(
            [path, "-NoP", "-NonI", "-C", "$PSVersionTable.PSVersion.Major"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return int(output)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _where_candidates(name: str) -> list[str]:
    """Return candidate paths reported by ``where.exe <name>``."""
    try:
        result = subprocess.run(
            ["where.exe", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@functools.lru_cache(maxsize=1)
def find_pwsh() -> str | None:
    """Find PowerShell 7.x on the current platform.

    Resolution order:
      1. ``pwsh`` / ``pwsh.exe`` on PATH (via ``shutil.which``).
      2. ``where.exe pwsh.exe`` on Windows.
      3. Common fixed installation paths.

    Returns the absolute path to a PowerShell 7+ executable, or ``None`` if
    only Windows PowerShell 5.1 (or no PowerShell) is available.
    """
    candidates: list[str] = []

    if sys.platform == "win32":
        names = ["pwsh.exe", "pwsh"]
    else:
        names = ["pwsh"]

    # 1. PATH
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    # 2. where.exe (Windows only)
    if sys.platform == "win32":
        for name in names:
            candidates.extend(_where_candidates(name))

    # 3. Fixed common install locations
    if sys.platform == "win32":
        candidates.extend(
            [
                r"C:\Program Files\PowerShell\7\pwsh.exe",
                r"C:\Program Files (x86)\PowerShell\7\pwsh.exe",
            ]
        )
    else:
        candidates.extend(
            [
                "/opt/microsoft/powershell/7/pwsh",
                "/usr/local/bin/pwsh",
                "/usr/bin/pwsh",
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        candidate = shutil.which(candidate) or candidate
        if not os.path.exists(candidate):
            continue
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        major = _pwsh_major_version(candidate)
        if major is not None and major >= 7:
            return candidate

    return None


class PowershellParams(BaseModel):
    """Parameters for the Powershell tool — execute a PowerShell command."""

    cmd: str = Field(description="PowerShell command.")
    timeout: int = Field(
        default=10,
        ge=3,
        le=900,
        description="Timeout in seconds."
    )
    max_output_length: int = Field(
        default=65536,
        ge=0,
        description="Output length threshold. Exceeding it sends the output to an anonymous sub-agent for summarization. 0 disables."
    )

class Powershell(CallableTool2[PowershellParams]):

    name: str = "Powershell"
    description: str = "Run a simple powershell command. Prefer Python for complex or stateful tasks. "
    params: type[PowershellParams] = PowershellParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        if not _bash_tool._should_enable_powershell():
            raise SkipThisTool()

        self._pwsh_path = find_pwsh()
        if self._pwsh_path is None:
            _print_warning(
                "PowerShell 7.x not found on this system; falling back to Windows PowerShell 5.1. "
                "PowerShell 7 syntax will be downgraded automatically, which may change command behavior."
            )

        # Pre-normalize forbidden commands once at init time for O(1) per-call lookup.
        # PowerShell is case-insensitive; normalize to lowercase.
        raw_forbidden = self._session.custom_config.get("config_json", {}).get("forbidden_commands", [])
        self._forbidden_keywords: list[str] = []
        seen: set[str] = set()
        for cmd in raw_forbidden:
            if not isinstance(cmd, str) or not cmd:
                continue
            normalized = " ".join(cmd.split()).lower()
            if normalized not in seen:
                seen.add(normalized)
                self._forbidden_keywords.append(normalized)

    async def __call__(self, params: PowershellParams) -> ToolReturnValue:
        """Execute the PowerShell command via the system PowerShell executable.

        Args:
            params: The parameters specifying the command and its arguments.

        Returns:
            ToolOk on success, ToolError on failure or timeout.
        """
        from kimix.tools.background.utils import remove_task_id


        if not params.cmd:
            return ToolError(
                output="Empty command.",
                message="No command specified.",
                brief="Empty command",
            )

        if self._pwsh_path is not None:
            # PowerShell 7 is available: run the command as-is without syntax transforms.
            cmd = params.cmd
            transform_warning = ""
            executable = self._pwsh_path
        else:
            # Fall back to Windows PowerShell 5.1 and downgrade PS7 syntax.
            cmd, transform_warnings = pwsh_transform(params.cmd)
            transform_warning = ""
            if transform_warnings:
                warning_lines = "\n".join(w for w in transform_warnings)
                transform_warning = '\n[WARNING]' + warning_lines
            executable = "powershell"

        if self._forbidden_keywords:
            # PowerShell is case-insensitive: compare lowercased strings.
            normalized_cmd = " ".join(cmd.split()).lower()
            for keyword in self._forbidden_keywords:
                if keyword in normalized_cmd:
                    return ToolError(
                        output="",
                        message=f"`{cmd}` is forbidden by config rule." + transform_warning,
                        brief="Forbidden command",
                    )
        # Refresh PATH/PATHEXT from registry so that tools installed
        # since the last command (e.g. via WinGet) are discoverable.
        if sys.platform == "win32":
            from kimix.utils.windows_env import refresh_env_from_registry
            refresh_env_from_registry()

        # Build the command line to pass to PowerShell -Command
        process_task = ProcessTask(executable, ["-NoP", "-NonI", "-Exec", "Bypass", "-NoL", "-C", "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;$OutputEncoding=[System.Text.Encoding]::UTF8;", cmd], None, None)
        task_id = await process_task.start(self._session, "pwsh")

        await process_task.wait(params.timeout)

        if await process_task.thread_is_alive():
            output = await process_task.stream.get_output() if process_task.stream else ""
            return ToolError(
                output=output,
                message=f"`{cmd}` Running in background. task_id: `{task_id}`. use `TaskOutput` or `Input`." + transform_warning,
                brief="Timeout",
            )

        remove_task_id(self._session, task_id)

        output = await process_task.stream.pop_output() if process_task.stream else ""
        success = await process_task.stream.success() if process_task.stream else False

        if not success:
            if params.max_output_length > 0 and len(output) > params.max_output_length:
                output = await _summarize_long_output_async(self._session, cmd, output)
            return ToolError(output=output, message=f"`{cmd}` failed." + transform_warning, brief="Command execution failed")

        if params.max_output_length > 0 and len(output) > params.max_output_length:
            output = await _summarize_long_output_async(self._session, cmd, output)
        output = await _maybe_export_output_async(output)
        return ToolOk(
            output=output,
            message=f'`{cmd}` success.' + transform_warning,
            brief=f"Command executed successfully",
            display_block=ShellDisplayBlock(language="powershell", command=cmd),
        )
