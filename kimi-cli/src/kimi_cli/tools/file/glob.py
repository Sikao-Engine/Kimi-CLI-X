"""Glob tool implementation."""

import asyncio
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.logging import logger
from kimi_cli.vfs import VFS
from .utils import resolve_vfs
from kimi_cli.utils.path import (
    kaos_path_from_user_input,
)
MAX_MATCHES = 1000
MAX_BYTES = 100 << 10  # 100KB
GLOB_DESC_PATH = Path(__file__).parent / "glob.md"
WINDOWS_PATH_HINT = (
    "Windows: `directory` accepts native (`C:\\Users\\foo`) and POSIX-style "
    "(`/c/Users/foo`) paths. Results use backslashes — convert to forward "
    "slashes for shell commands."
)


def _description_for_os(os_kind: str) -> str:
    return load_desc(
        GLOB_DESC_PATH,
        {
            "MAX_MATCHES": str(MAX_MATCHES),
            "WINDOWS_PATH_HINT": WINDOWS_PATH_HINT if os_kind == "Windows" else "",
        },
    )


class Params(BaseModel):
    pattern: str = Field(description="Glob pattern. Never start with `**`.")
    directory: str | None = Field(
        description="Absolute search path. Defaults to working directory.",
        default=None,
    )
    include_dirs: bool = Field(
        description="Include directories in results.",
        default=True,
    )


class Glob(CallableTool2[Params]):
    name: str = "Glob"
    description: str = _description_for_os("")
    params: type[Params] = Params
    def __init__(self, runtime: Runtime, vfs: VFS | None = None) -> None:
        super().__init__(description=_description_for_os(runtime.environment.os_kind))
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._skills_dirs = runtime.skills_dirs
        self._vfs = vfs

    # async def _validate_directory(self, directory: KaosPath) -> ToolError | None:
    #     """Validate that the directory is safe to search."""
    #     resolved_dir = directory.canonical()

    #     # Allow directories within the workspace (work_dir or additional dirs)
    #     if is_within_workspace(resolved_dir, self._work_dir, self._additional_dirs):
    #         return None

    #     # Allow directories within any discovered skills root
    #     if any(is_within_directory(resolved_dir, d) for d in self._skills_dirs):
    #         return None

    #     return ToolError(
    #         message=(
    #             f"`{directory}` is outside the workspace. "
    #             "You can only search within the working directory, "
    #             "additional directories, and skills directories."
    #         ),
    #         brief="Directory outside workspace",
    #     )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            # Detect unsafe patterns and compute fallback
            norm = params.pattern.replace("\\", "/")
            is_unsafe = norm.startswith("**")
            if is_unsafe:
                if norm.startswith("**/"):
                    pattern = norm[3:] if norm[3:] else "*"
                else:
                    pattern = "*"
            else:
                pattern = params.pattern

            dir_path = KaosPath(str(kaos_path_from_user_input(params.directory)) if params.directory else str(self._work_dir))
            dir_path = await resolve_vfs(str(dir_path), self._vfs, for_write=False)
            if not await dir_path.exists():
                return ToolError(
                    message=f"`{params.directory}` does not exist.",
                    brief=f"Directory not found: {params.directory}",
                )
            if not await dir_path.is_dir():
                return ToolError(
                    message=f"`{params.directory}` is not a directory.",
                    brief=f"Invalid directory: {params.directory}",
                )

            # Perform the glob search - bounded streaming with inline filtering
            matches: list[KaosPath] = []
            truncated = False
            try:
                async with asyncio.timeout(10):
                    async for match in dir_path.glob(pattern):
                        if not params.include_dirs and not await match.is_file():
                            continue
                        matches.append(match)
                        if len(matches) > MAX_MATCHES:
                            truncated = True
                            matches.pop()
                            break
            except asyncio.TimeoutError:
                truncated = True

            # Sort for consistent output
            matches.sort()

            # Build output with byte limit
            output_lines: list[str] = []
            n_bytes = 0
            truncated_by_bytes = False
            for p in matches:
                line = str(p.relative_to(dir_path))
                line_bytes = len(line.encode("utf-8"))
                separator_bytes = 1 if output_lines else 0
                output_lines.append(line)
                n_bytes += separator_bytes + line_bytes
                if n_bytes >= MAX_BYTES:
                    truncated_by_bytes = True
                    break

            output = "\n".join(output_lines)

            if is_unsafe:
                return ToolError(
                    output=output,
                    message=(
                        f"Pattern `{params.pattern}` starts with `**`, which is disallowed. "
                        f"Fallback result for `{pattern}`:"
                    ),
                    brief=f"Unsafe pattern: {params.pattern}",
                )

            # Build message
            shown_count = len(output_lines)
            if shown_count > 0:
                message = f"Found {shown_count} matches for pattern `{pattern}`."
            else:
                message = f"No matches found for pattern `{pattern}`."

            if truncated:
                message += (
                    f" Showing first {MAX_MATCHES} matches. "
                    "Use a more specific pattern."
                )

            if truncated_by_bytes:
                message += f" Output truncated to {MAX_BYTES} bytes."

            return ToolOk(
                output=output,
                message=message,
                brief=f"Glob {dir_path}",
            )

        except Exception as e:
            logger.warning(
                "Glob failed: pattern={pattern}: {error}", pattern=params.pattern, error=e
            )
            return ToolError(
                message=f"Glob failed for `{params.pattern}`: {e}",
                brief=f"Glob failed: {params.pattern}",
            )
