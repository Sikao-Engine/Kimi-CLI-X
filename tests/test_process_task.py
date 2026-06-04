"""Comprehensive tests for ProcessTask."""

import asyncio
import queue
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimix.tools.common import ProcessTask
from kimix.tools.background.utils import _pop_task_data


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


# ---------------------------------------------------------------------------
# Construction / __init__
# ---------------------------------------------------------------------------
def test_init_stores_attributes() -> None:
    with patch("shutil.which", return_value=None):
        task = ProcessTask("my_tool", ["-c", "print(1)"], cwd="/tmp")
        assert task.path == "my_tool"
        assert task.args == ["-c", "print(1)"]
        assert task.cwd == "/tmp"
        assert task.task_id is None
        assert task.stream is None


def test_init_defaults() -> None:
    task = ProcessTask("python")
    assert task.args == []
    assert task.cwd is None


def test_init_keeps_existing_path() -> None:
    task = ProcessTask(sys.executable)
    assert task.path == sys.executable


def test_init_no_resolution_for_missing() -> None:
    task = ProcessTask("totally_fake_cmd_12345")
    assert task.path == "totally_fake_cmd_12345"


# ---------------------------------------------------------------------------
# _run_process_bg
# ---------------------------------------------------------------------------
async def test_run_process_bg_success() -> None:
    task = ProcessTask(sys.executable, ["-c", "print('hello_world')"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is True

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "hello_world" in output


async def test_run_process_bg_stderr() -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; sys.stderr.write('err_msg\\n')"],
    )
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is True

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "[stderr] err_msg" in output


async def test_run_process_bg_nonzero_exit() -> None:
    task = ProcessTask(sys.executable, ["-c", "import sys; sys.exit(42)"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is False

    messages = []
    while True:
        try:
            messages.append(q.get_nowait())
        except queue.Empty:
            break
    assert any("exited with code 42" in m for m in messages)


async def test_run_process_bg_stop_event_before_start() -> None:
    task = ProcessTask(sys.executable, ["-c", "print(1)"])
    task._stop_event.set()
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is False


async def test_run_process_bg_stop_event_during_run() -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.2)
    await task._stop_function()
    result = await asyncio.wait_for(bg, timeout=5)

    assert result is False
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "stopped by user" in output


async def test_run_process_bg_exception_on_popen() -> None:
    task = ProcessTask("this_should_not_exist_command_12345")
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is False
    msg = q.get_nowait()
    assert msg.startswith("\n[Error:")


async def test_run_process_bg_popen_raises_oserror() -> None:
    task = ProcessTask(sys.executable)
    with patch("asyncio.create_subprocess_exec", side_effect=OSError("boom")):
        q: queue.Queue[str] = queue.Queue()
        result = await task._run_process_bg(q)
        assert result is False
        msg = q.get_nowait()
        assert "boom" in msg


# ---------------------------------------------------------------------------
# _stop_function
# ---------------------------------------------------------------------------
async def test_stop_function_terminates_running_process() -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.2)
    assert task._process_ref is not None
    assert task._process_ref.returncode is None

    await task._stop_function()
    result = await asyncio.wait_for(bg, timeout=5)
    assert result is False


# ---------------------------------------------------------------------------
# _input_function
# ---------------------------------------------------------------------------
async def test_input_function_writes_to_stdin() -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; line=sys.stdin.readline(); print('echo', line.strip())"],
    )
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.1)
    success = await task._input_function("hello\n")
    assert success is True

    result = await asyncio.wait_for(bg, timeout=5)
    assert result is True

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "echo hello" in output


async def test_input_function_returns_false_when_stopped() -> None:
    task = ProcessTask(sys.executable, ["-c", "print(1)"])
    task._stop_event.set()
    success = await task._input_function("data")
    assert success is False


async def test_input_function_returns_false_when_no_process() -> None:
    task = ProcessTask(sys.executable)
    task._stop_event.set()
    success = await task._input_function("data")
    assert success is False


# ---------------------------------------------------------------------------
# start / public API integration
# ---------------------------------------------------------------------------
async def test_start_returns_task_id(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('ok')"])
    tid = await task.start(mock_session, kind="run", name="test")
    assert tid is not None
    assert tid.startswith("run_test")
    assert task.task_id == tid


async def test_start_default_name(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable)
    tid = await task.start(mock_session, kind="cmd")
    assert tid == "cmd"
    assert task.task_id == tid


async def test_start_creates_stream(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('ok')"])
    await task.start(mock_session, kind="run")
    assert task.stream is not None
    assert await task.stream.is_started() is True


async def test_wait_completes(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('done')"])
    await task.start(mock_session, kind="run")
    await task.wait(timeout=5)
    assert await task.stream.thread_is_alive() is False
    output = await task.stream.get_output()
    assert "done" in output


async def test_thread_is_alive_while_running(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(0.5)"])
    await task.start(mock_session, kind="run")
    assert await task.thread_is_alive() is True
    await task.wait(timeout=5)
    assert await task.thread_is_alive() is False


async def test_stop_via_public_api(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    await task.start(mock_session, kind="run")
    await asyncio.sleep(0.1)
    assert await task.thread_is_alive() is True
    await task.stop()
    await task.wait(timeout=2)
    assert await task.stream.is_stopped() is True


async def test_input_via_public_api(mock_session: MagicMock) -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; line=sys.stdin.readline(); print('got', line.strip())"],
    )
    await task.start(mock_session, kind="run")
    await asyncio.sleep(0.1)
    result = await task.input("hello\n")
    assert result is True
    await task.wait(timeout=5)
    output = await task.stream.get_output()
    assert "got hello" in output


async def test_input_returns_false_when_not_started() -> None:
    task = ProcessTask(sys.executable)
    result = await task.input("data")
    assert result is False


async def test_task_id_none_before_start() -> None:
    task = ProcessTask(sys.executable)
    assert task.task_id is None


async def test_stream_none_before_start() -> None:
    task = ProcessTask(sys.executable)
    assert task.stream is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
async def test_run_process_bg_empty_args() -> None:
    task = ProcessTask(sys.executable, ["-c", "print('empty_args_work')"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is True
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "empty_args_work" in output


async def test_run_process_bg_with_cwd(tmp_path: Path) -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import pathlib, sys; print(pathlib.Path.cwd())"],
        cwd=str(tmp_path),
    )
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result is True
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert str(tmp_path) in output
