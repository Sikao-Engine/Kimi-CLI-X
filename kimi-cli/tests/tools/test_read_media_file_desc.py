from __future__ import annotations

from typing import cast

# ruff: noqa

import pytest
from inline_snapshot import snapshot

from kimi_cli.llm import ModelCapability
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.file.read_media import ReadMediaFile


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (
            {"image_in", "video_in"},
            snapshot(
                "Read image/video up to 100MB. Supports images and videos."
            ),
        ),
        (
            {"image_in"},
            snapshot(
                "Read image/video up to 100MB. Images only."
            ),
        ),
        (
            {"video_in"},
            snapshot(
                "Read image/video up to 100MB. Videos only."
            ),
        ),
    ],
)
def test_read_media_file_description_by_capabilities(
    runtime: Runtime, capabilities: set[str], expected: str
) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], capabilities)
    assert ReadMediaFile(runtime).base.description == expected


def test_read_media_file_description_without_capabilities(runtime: Runtime) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], set())
    with pytest.raises(SkipThisTool):
        ReadMediaFile(runtime)
