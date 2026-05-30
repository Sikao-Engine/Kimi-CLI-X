"""grep tool - print lines that match patterns."""
import os

from kimi_agent_sdk import CallableTool2, ToolError, ToolReturnValue
from .params import Params, _is_protected_path



class Grep(CallableTool2[Params]):
    name: str = "Grep"
    description: str = "Print lines that match patterns."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        if params.output_path:
            cwd = params.cwd or os.getcwd()
            is_prot, reason = _is_protected_path(params.output_path, cwd)
            if is_prot:
                return ToolError(message=reason, output=reason, brief="protected path")
        return ToolError(
            message="grep command is not available. use the Grep tool instead.",
            output="",
            brief="use Grep tool",
        )

