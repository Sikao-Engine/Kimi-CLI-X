"""Standalone async function for executing Bash built-in commands, with subprocess fallback."""

import queue
import sys
from typing import TYPE_CHECKING

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.session import Session

from kimix.tools.common import _maybe_export_output_async, ProcessTask
from kimix.tools.file.bash import BASH_COMMANDS, WINDOWS_ALIASES, BashParams

if TYPE_CHECKING:
    from kimix.tools.file.run import RunParams


async def run_bash(params: "BashParams | RunParams", session: Session) -> ToolReturnValue:
    """Execute a bash command via built-in Python implementation, with subprocess fallback.

    Args:
        params: The parameters specifying the command and its arguments.
        session: The current session.

    Returns:
        ToolOk on success, ToolError on failure or timeout.
    """
    from kimix.tools.background.utils import (
        BackgroundStream,
        generate_task_id,
        add_task,
        remove_task_id,
    )

    # Normalize: accept both BashParams.cmd and RunParams.path
    cmd: str = getattr(params, 'cmd', None) or getattr(params, 'path', '')

    # Handle space-separated command + args in cmd
    if " " in cmd:
        parts = cmd.split(" ")
        cmd = parts[0]
        remaining = parts[1:]
        if remaining:
            params.args.insert(0, " ".join(remaining))

    # Resolve command - check Windows aliases first, then builtin map
    bash_name = WINDOWS_ALIASES.get(cmd, cmd)
    bash_tool: CallableTool2 | None = BASH_COMMANDS.get(bash_name)

    # --- Subprocess fallback: if not a built-in, try running as a real process ---
    if bash_tool is None:
        import shutil
        import os
        from pathlib import Path

        is_process = False
        # check if using python
        if params.path == 'python':
            params.path = sys.executable
            is_process = True
        elif os.sep in cmd or "/" in cmd:
            is_process = Path(cmd).is_file()
        else:
            is_process = shutil.which(cmd) is not None

        if not is_process:
            return ToolError(
                output=f"Unknown bash command: '{cmd}'",
                message=f"Command '{cmd}' is not a recognized bash builtin.",
                brief="Unknown command",
            )

        # Run as real subprocess via ProcessTask
        task = ProcessTask(cmd, params.args, params.cwd, env=None)
        task_id = await task.start(session, "bash_proc", cmd)

        wait_timeout = params.timeout
        await task.wait(wait_timeout)

        if await task.thread_is_alive():
            output = await task.stream.get_output() if task.stream else ""
            return ToolError(
                output=output or f"Running in background. task_id: `{task_id}`. use `TaskOutput` or `Input`",
                message="Process timeout",
                brief="Timeout",
            )

        remove_task_id(session, task_id)

        output = await task.stream.pop_output() if task.stream else ""
        success = await task.stream.success() if task.stream else False

        if not success:
            return ToolError(
                output=output,
                message="Command execution failed",
                brief="Command execution failed",
            )

        output = await _maybe_export_output_async(output)
        return ToolOk(output=output)

    # --- Built-in command execution ---
    result_holder: list[ToolReturnValue] = []

    async def wrapper(q: queue.Queue[str]) -> bool:
        try:
            result = await bash_tool(params)
            result_holder.append(result)
            output_str = result.output if isinstance(result.output, str) else str(result.output)
            q.put_nowait(output_str)
            return not result.is_error
        except Exception as e:
            q.put_nowait(f"\n[Error: {str(e)}]")
            return False

    stream = BackgroundStream()
    task_id = generate_task_id(session, "bash", cmd)
    await stream.start(wrapper, lambda: None)
    add_task(session, task_id, stream)

    await stream.wait(params.timeout)

    if await stream.thread_is_alive():
        output = await stream.get_output()
        return ToolError(
            output=output or f"Running in background. task_id: `{task_id}`. use `TaskOutput` or `Input`",
            message="Process timeout",
            brief="Timeout",
        )

    remove_task_id(session, task_id)

    if result_holder:
        return result_holder[0]

    output = await stream.pop_output()
    success = await stream.success()
    if not success:
        return ToolError(output=output, message="Command execution failed", brief="Command execution failed")
    output = await _maybe_export_output_async(output)
    return ToolOk(output=output)
