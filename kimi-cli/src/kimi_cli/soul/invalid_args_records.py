from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class InvalidArgRecord(BaseModel):
    """A record of an invalid argument tool call stored in `invalid_arguments.jsonl`."""

    model_config = ConfigDict(strict=True)

    role: Literal["_invalid_arg"]
    """Literal tag to distinguish from other context.jsonl record types."""

    timestamp: float
    """Unix epoch seconds (from time.time())."""

    session_id: str
    """The active session ID."""

    tool_name: str
    """Name of the tool that received bad arguments."""

    tool_call_id: str
    """The tool call ID (from ToolCall.id)."""

    arguments: str
    """The raw JSON arguments string received from the LLM."""

    error_type: Literal["parse_error", "validate_error"]
    """Type of failure: JSON parse error or schema validation error."""

    error_message: str
    """The human-readable error message returned to the LLM."""

    turn_id: str | None = None
    """Current turn ID (optional, for traceability)."""

    step_no: int | None = None
    """Current step number (optional)."""
