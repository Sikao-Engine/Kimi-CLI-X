"""true tool - do nothing, return success."""
import os

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from .params import Params

from kimix.tools.common import _maybe_export_output_async


class True_(CallableTool2[Params]):
    name: str = "True"
    description: str = "Do nothing, return success."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        return ToolOk(output="")
