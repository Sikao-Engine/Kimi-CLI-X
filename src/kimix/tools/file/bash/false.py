"""false tool - do nothing, return failure."""
import os

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from .params import Params

from kimix.tools.common import _maybe_export_output_async


class False_(CallableTool2[Params]):
    name: str = "False"
    description: str = "Do nothing, return failure."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        return ToolError(message="", output="", brief="false")
