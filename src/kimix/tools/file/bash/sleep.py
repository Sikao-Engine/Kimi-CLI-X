"""sleep tool - delay for a specified amount of time."""
import asyncio
import os

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from .params import Params

from kimix.tools.common import _maybe_export_output_async


class Sleep(CallableTool2[Params]):
    name: str = "Sleep"
    description: str = "Delay for a specified amount of time."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            args = [arg for arg in params.args if not arg.startswith("-")]
            if not args:
                return ToolError(message="sleep: missing operand", output="", brief="missing operand")
            total = 0.0
            for arg in args:
                s = arg
                if s.endswith("s"):
                    total += float(s[:-1])
                elif s.endswith("m"):
                    total += float(s[:-1]) * 60
                elif s.endswith("h"):
                    total += float(s[:-1]) * 3600
                elif s.endswith("d"):
                    total += float(s[:-1]) * 86400
                else:
                    total += float(s)
            await asyncio.sleep(total)
            return ToolOk(output="")
        except Exception as e:
            return ToolError(message=str(e), output="", brief="sleep failed")
