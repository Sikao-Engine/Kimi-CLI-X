"""Tool call reason tracker.

Records why each tool was called by capturing the `reason` field from tool
parameters together with the tool name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kosong.tooling import CallableTool2
from pydantic import BaseModel


class ToolCallReason:
    """Tracks reasons for tool invocations.

    Each entry stores the tool name and the human-readable reason provided
    in the tool parameters, keyed by file path.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[dict[str, str]]] = {}

    def add_tool_call_reason(self, params: BaseModel, tool: CallableTool2[Any]) -> None:
        """Record a tool call reason for WriteFile or EditFile.

        Args:
            params: Validated parameters for the tool call. Expected to contain
                a ``reason`` attribute of type ``str``.
            tool: The tool instance that was invoked. Must be WriteFile or EditFile.

        Raises:
            ValueError: If ``tool`` is not WriteFile or EditFile.
        """
        if tool.name not in ("WriteFile", "EditFile"):
            raise ValueError(f"Expected WriteFile or EditFile, got {tool.name}")

        reason: str = getattr(params, "reason", "")
        raw_path: str = getattr(params, "path", "")
        if not raw_path:
            raise ValueError("params must contain a non-empty 'path' attribute.")
        path: str = str(Path(raw_path).resolve())
        record: dict[str, str] = {"tool_name": tool.name, "reason": reason}

        self._records.setdefault(path, []).append(record)

    def formatted_print(self, paths: list[str]) -> str:
        """Find the paths' changes and return them as a formatted string.

        Args:
            paths: The file paths to look up. Each will be resolved to an absolute path.

        Returns:
            A formatted string containing all changes for the given paths.
        """
        lines: list[str] = []
        for path in paths:
            abs_path = str(Path(path).resolve())
            records = self._records.get(abs_path)
            if not records:
                lines.append(f"- {abs_path}: no record")
                continue

            parts: list[str] = []
            for rec in records:
                tool_name = rec.get("tool_name", "Unknown")
                reason = rec.get("reason", "")
                if reason:
                    parts.append(f"{tool_name}: {reason}")
                else:
                    parts.append(tool_name)
            lines.append(f"- {abs_path} ({', '.join(parts)})")

        return "\n".join(lines)

    @property
    def changed_files(self) -> list[str]:
        """Return a sorted list of absolute paths that have been recorded."""
        return sorted(self._records.keys())

    def to_markdown(self, cwd: Path | None = None, max_count: int = 100) -> str:
        """Return a dense markdown representation of changed files.

        Args:
            cwd: Optional directory to make paths relative to.
            max_count: Maximum number of records to show per file. Defaults to 100.
        """
        if not self._records:
            return ""
        lines = ["Changed files:"]
        target_paths = sorted(self._records.keys())
        for path in target_paths:
            records = self._records.get(path)
            if not records:
                continue
            parts: list[str] = []
            for rec in records[-max_count:]:
                tool_name = rec.get("tool_name", "Unknown")
                reason = rec.get("reason", None)
                if reason:
                    parts.append(f"{tool_name}: {reason}")
                else:
                    parts.append(tool_name)
            display_path = path
            if cwd is not None:
                try:
                    display_path = str(Path(path).relative_to(cwd.resolve()))
                except ValueError:
                    display_path = path
            lines.append(f"- {display_path} ({', '.join(parts)})")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all recorded reasons."""
        self._records.clear()

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())

    def __bool__(self) -> bool:
        return bool(self._records)
