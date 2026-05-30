"""Standalone async function for executing Bash built-in commands, with subprocess fallback."""

import queue
from typing import TYPE_CHECKING

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.session import Session

from kimix.tools.common import _maybe_export_output_async
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

    # Normalize: accept BashParams.cmd, RunParams.executable (legacy), or RunParams.command
    cmd: str = getattr(params, 'cmd', None) or getattr(params, 'executable', None) or ''
    if not cmd:
        # New RunParams with 'command' field — extract first token as cmd, rest as args
        import shlex
        cmd_parts = shlex.split(getattr(params, 'command', ''))
        cmd = cmd_parts[0] if cmd_parts else ''
        # Set args on params so bash builtins can access them
        object.__setattr__(params, 'args', cmd_parts[1:] if len(cmd_parts) > 1 else [])
    # Normalize args: RunParams.args was str (legacy), BashParams.args is list[str]
    if hasattr(params, 'args') and isinstance(params.args, str):
        import shlex
        params.args = shlex.split(params.args) if params.args else []

    # Resolve command - check builtin map first, then try Windows alias
    bash_tool: CallableTool2 | None = BASH_COMMANDS.get(cmd)
    if bash_tool is None:
        bash_name = WINDOWS_ALIASES.get(cmd, cmd)
        bash_tool = BASH_COMMANDS.get(bash_name)

    # --- Subprocess fallback: if not a built-in, try running as a real process ---
    if bash_tool is None:
        return ToolError(
            output=f"Unknown bash command: '{cmd}'",
            message=f"Command '{cmd}' is not a recognized bash builtin.",
            brief="Unknown command",
        )

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
