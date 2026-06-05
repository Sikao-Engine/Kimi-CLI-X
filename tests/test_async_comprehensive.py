"""Comprehensive async/await correctness tests for BackgroundStream, ProcessTask, and tool integrations.

This test suite verifies:
- All async methods are properly awaited
- asyncio usage patterns are correct
- BackgroundStream and ProcessTask async behavior
- Tool integrations (Run, Python, Input, TaskOutput, TaskList, Agent) handle async correctly
- Edge cases and concurrency
"""

import asyncio
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimi_agent_sdk import ToolOk, ToolError
from kimix.tools.background.utils import (
    BackgroundStream,
    add_task,
    discard_all_tasks,
    generate_task_id,
    get_all_tasks,
    remove_task_id,
)
from kimix.tools.background import TaskOutput, TaskOutputParams
from kimix.tools.file.run import Run, RunParams
from kimix.tools.py import Python, Params as PyParams
from kimix.tools.file.input import Input, InputParams
from kimix.tools.common import ProcessTask


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    return session


@pytest.fixture(autouse=True)
def patch_find_bash() -> Any:
    with patch("kimix.tools.file.run.find_bash", return_value=None):
        with patch("kimix.tools.file.run.find_pwsh", return_value=None):
            yield


@pytest.fixture(autouse=True)
async def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    await discard_all_tasks(mock_session)


# ---------------------------------------------------------------------------
# TaskOutput block behavior fix verification
# ---------------------------------------------------------------------------
class TestTaskOutputBlockBehavior:
    """Test that TaskOutput block parameter works correctly after the fix."""

    async def test_block_false_does_not_wait(self, mock_session: MagicMock) -> None:
        """If block=False, TaskOutput should return immediately without waiting."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "long_task", stream)

        start = time.monotonic()
        result = await tool(TaskOutputParams(task_id="long_task", block=False, timeout=3))
        elapsed = time.monotonic() - start

        # Should return immediately (well under 1 second)
        assert elapsed < 1.0
        assert isinstance(result, ToolOk)
        # Task should still be registered since we didn't wait for it to finish
        assert "long_task" in get_all_tasks(mock_session)

        # Cleanup
        await stream.stop()
        await stream.wait(timeout=1)
        remove_task_id(mock_session, "long_task")

    async def test_block_true_waits_for_completion(self, mock_session: MagicMock) -> None:
        """If block=True, TaskOutput should wait for the thread to finish."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("done")
            time.sleep(0.1)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "short_task", stream)

        result = await tool(TaskOutputParams(task_id="short_task", block=True, timeout=5))

        # Task should be removed after completion because thread is dead
        assert "short_task" not in get_all_tasks(mock_session)
        assert "done" in str(result.output)

    async def test_block_false_with_kill(self, mock_session: MagicMock) -> None:
        """block=False + kill=True should kill without blocking wait."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "kill_task", stream)

        start = time.monotonic()
        result = await tool(
            TaskOutputParams(task_id="kill_task", block=False, kill=True, timeout=3)
        )
        elapsed = time.monotonic() - start

        # Should return quickly even though kill triggers stop (which is fast)
        assert elapsed < 1.0
        assert "kill_task" not in get_all_tasks(mock_session)


# ---------------------------------------------------------------------------
# BackgroundStream async method correctness
# ---------------------------------------------------------------------------
class TestBackgroundStreamAsyncMethods:
    """Verify all async methods behave correctly when awaited."""

    async def test_success_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("x")

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()

        # success() is async and must be awaited
        coro = stream.success()
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert result is True

    async def test_get_output_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("output")

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()

        coro = stream.get_output()
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert result == "output"

    async def test_pop_output_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("pop_me")

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()

        result = await stream.pop_output()
        assert result == "pop_me"
        assert await stream.get_output() == ""

    async def test_thread_is_alive_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(0.1)

        await stream.start(worker, stop_function=lambda: None)

        alive = await stream.thread_is_alive()
        assert alive is True

        await stream.wait()

        alive = await stream.thread_is_alive()
        assert alive is False

    async def test_wait_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(0.05)

        await stream.start(worker, stop_function=lambda: None)
        coro = stream.wait()
        assert asyncio.iscoroutine(coro)
        await coro
        assert await stream.thread_is_alive() is False

    async def test_stop_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        coro = stream.stop()
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert result is True
        assert await stream.is_stopped() is True

    async def test_input_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()
        received: list[str] = []

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(0.1)

        def input_handler(data: str) -> bool:
            received.append(data)
            return True

        await stream.start(worker, stop_function=lambda: None, input_function=input_handler)
        coro = stream.input("test")
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert result is True
        assert received == ["test"]
        await stream.wait()

    async def test_is_started_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()
        assert await stream.is_started() is False

        def worker(q: queue.Queue[str]) -> None:
            pass

        await stream.start(worker, stop_function=lambda: None)
        assert await stream.is_started() is True
        await stream.wait()

    async def test_is_stopped_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()
        assert await stream.is_stopped() is False

        def worker(q: queue.Queue[str]) -> None:
            pass

        await stream.start(worker, stop_function=lambda: None)
        await stream.stop()
        assert await stream.is_stopped() is True

    async def test_get_queue_is_async_and_awaitable(self) -> None:
        stream = BackgroundStream()
        coro = stream.get_queue()
        assert asyncio.iscoroutine(coro)
        assert await coro is None

        def worker(q: queue.Queue[str]) -> None:
            pass

        await stream.start(worker, stop_function=lambda: None)
        q = await stream.get_queue()
        assert isinstance(q, queue.Queue)


# ---------------------------------------------------------------------------
# BackgroundStream async function wrapping (start with coroutine function)
# ---------------------------------------------------------------------------
class TestBackgroundStreamAsyncFunctionWrapping:
    """Verify BackgroundStream correctly wraps async functions with asyncio.run."""

    async def test_start_with_async_worker(self) -> None:
        stream = BackgroundStream()

        async def worker(q: queue.Queue[str]) -> None:
            await asyncio.sleep(0.01)
            q.put("async_worker")

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()
        assert await stream.get_output() == "async_worker"
        assert await stream.success() is True

    async def test_start_with_async_worker_returning_false(self) -> None:
        stream = BackgroundStream()

        async def worker(q: queue.Queue[str]) -> bool:
            q.put("fail")
            return False

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()
        assert await stream.success() is False

    async def test_start_with_async_worker_raising_exception(self) -> None:
        stream = BackgroundStream()

        async def worker(q: queue.Queue[str]) -> None:
            raise RuntimeError("async boom")

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()
        assert await stream.success() is False

    async def test_stop_with_async_stop_function(self) -> None:
        stream = BackgroundStream()
        stopped = asyncio.Event()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(10)

        async def stopper() -> None:
            stopped.set()

        await stream.start(worker, stop_function=stopper)
        result = await stream.stop()
        assert result is True
        assert stopped.is_set()

    async def test_input_with_async_input_function(self) -> None:
        stream = BackgroundStream()
        received: list[str] = []

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(0.1)

        async def input_handler(data: str) -> bool:
            received.append(data)
            return True

        await stream.start(worker, stop_function=lambda: None, input_function=input_handler)
        result = await stream.input("async_input")
        assert result is True
        assert received == ["async_input"]
        await stream.wait()


# ---------------------------------------------------------------------------
# ProcessTask async method correctness
# ---------------------------------------------------------------------------
class TestProcessTaskAsyncMethods:
    """Verify ProcessTask async methods are correctly used with await."""

    async def test_start_is_async_and_returns_task_id(self, mock_session: MagicMock) -> None:
        task = ProcessTask(sys.executable, ["-c", "print('hello')"])
        coro = task.start(mock_session, kind="run", name="async_test")
        assert asyncio.iscoroutine(coro)
        tid = await coro
        assert tid.startswith("run_async_test")
        assert await task.thread_is_alive() is True
        await task.wait(timeout=5)
        remove_task_id(mock_session, tid)

    async def test_wait_is_async(self, mock_session: MagicMock) -> None:
        task = ProcessTask(sys.executable, ["-c", "print('wait_test')"])
        await task.start(mock_session, kind="run")
        coro = task.wait(timeout=5)
        assert asyncio.iscoroutine(coro)
        await coro
        assert await task.thread_is_alive() is False

    async def test_thread_is_alive_is_async(self, mock_session: MagicMock) -> None:
        task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(0.5)"])
        await task.start(mock_session, kind="run")
        coro = task.thread_is_alive()
        assert asyncio.iscoroutine(coro)
        alive = await coro
        assert alive is True
        await task.wait(timeout=5)
        alive = await task.thread_is_alive()
        assert alive is False

    async def test_stop_is_async(self, mock_session: MagicMock) -> None:
        task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
        await task.start(mock_session, kind="run")
        coro = task.stop()
        assert asyncio.iscoroutine(coro)
        await coro
        await task.wait(timeout=2)
        assert await task.stream.is_stopped() is True

    async def test_input_is_async(self, mock_session: MagicMock) -> None:
        task = ProcessTask(
            sys.executable,
            ["-c", "import sys; line=sys.stdin.readline(); print('echo', line.strip())"],
        )
        await task.start(mock_session, kind="run")
        await asyncio.sleep(0.1)
        coro = task.input("hello\n")
        assert asyncio.iscoroutine(coro)
        result = await coro
        assert result is True
        await task.wait(timeout=5)
        output = await task.stream.get_output()
        assert "echo hello" in output

    async def test_stream_methods_require_await(self, mock_session: MagicMock) -> None:
        task = ProcessTask(sys.executable, ["-c", "print('stream_test')"])
        await task.start(mock_session, kind="run")
        await task.wait(timeout=5)

        success_coro = task.stream.success()
        assert asyncio.iscoroutine(success_coro)
        assert await success_coro is True

        output_coro = task.stream.get_output()
        assert asyncio.iscoroutine(output_coro)
        assert "stream_test" in await output_coro

        queue_coro = task.stream.get_queue()
        assert asyncio.iscoroutine(queue_coro)
        assert await queue_coro is not None

        started_coro = task.stream.is_started()
        assert asyncio.iscoroutine(started_coro)
        assert await started_coro is True

        stopped_coro = task.stream.is_stopped()
        assert asyncio.iscoroutine(stopped_coro)
        assert await stopped_coro is False


# ---------------------------------------------------------------------------
# Run tool async patterns
# ---------------------------------------------------------------------------
class TestRunToolAsync:
    """Verify Run tool async patterns."""

    async def test_foreground_run_awaits_all_async_calls(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(command=f'{sys.executable} -c "print(\'run_async\')"', timeout=10)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "run_async" in str(result.output)

    async def test_foreground_timeout_keeps_task_registered(self, mock_session: MagicMock) -> None:
        tool = Run(session=mock_session)
        params = RunParams(
            command=f'{sys.executable} -c "import time; time.sleep(100)"',
            timeout=3,
        )
        result = await tool(params)
        assert isinstance(result, ToolError)
        # Task should remain registered after timeout
        assert len(get_all_tasks(mock_session)) >= 1
        for tid in list(get_all_tasks(mock_session).keys()):
            await get_all_tasks(mock_session)[tid].stop()
            remove_task_id(mock_session, tid)


# ---------------------------------------------------------------------------
# Python tool async patterns
# ---------------------------------------------------------------------------
class TestPythonToolAsync:
    """Verify Python tool async patterns."""

    async def test_foreground_python_awaits_all_async_calls(self, mock_session: MagicMock) -> None:
        tool = Python(session=mock_session)
        params = PyParams(code="print('py_async')", timeout=10)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "py_async" in str(result.output)



# ---------------------------------------------------------------------------
# Input tool async patterns
# ---------------------------------------------------------------------------
class TestInputToolAsync:
    """Verify Input tool async patterns."""

    async def test_input_awaits_stream_input(self, mock_session: MagicMock) -> None:
        task = ProcessTask(
            sys.executable,
            ["-c", "import sys; line=sys.stdin.readline(); print('got', line.strip())"],
        )
        tid = await task.start(mock_session, kind="run", name="input_async")
        await asyncio.sleep(0.2)

        tool = Input(session=mock_session)
        result = await tool(InputParams(task_id=tid, text="world\n"))
        assert isinstance(result, ToolOk)
        assert "sent" in str(result.output).lower()

        await task.wait(timeout=5)
        output = await task.stream.get_output()
        assert "got world" in output
        remove_task_id(mock_session, tid)

    async def test_input_to_missing_task_returns_error(self, mock_session: MagicMock) -> None:
        tool = Input(session=mock_session)
        result = await tool(InputParams(task_id="nonexistent", text="hello"))
        assert isinstance(result, ToolError)


# ---------------------------------------------------------------------------
# Agent tool async patterns (mocked to avoid external dependencies)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TaskList async patterns (via TaskOutput with task_id=None)
# ---------------------------------------------------------------------------
class TestTaskListAsync:
    """Verify TaskList tool async patterns via TaskOutput."""

    async def test_empty_task_list(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        result = await tool(TaskOutputParams(task_id=None))
        assert isinstance(result, ToolOk)
        assert "No running task" in str(result.output)

    async def test_task_list_shows_started_tasks(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("hello")

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)
        result = await tool(TaskOutputParams(task_id=None))
        await stream.wait()
        assert "run_test" in str(result.output)


# ---------------------------------------------------------------------------
# Async concurrency and stress tests
# ---------------------------------------------------------------------------
class TestAsyncConcurrency:
    """Test concurrent async operations."""

    async def test_multiple_background_streams_concurrently(self) -> None:
        streams = [BackgroundStream() for _ in range(5)]

        def make_worker(i: int):
            def worker(q: queue.Queue[str]) -> None:
                q.put(f"stream_{i}")
                time.sleep(0.05)
            return worker

        # Start all streams concurrently
        await asyncio.gather(*[
            stream.start(make_worker(i), stop_function=lambda: None)
            for i, stream in enumerate(streams)
        ])

        # Wait for all concurrently
        await asyncio.gather(*[stream.wait() for stream in streams])

        # Check outputs concurrently
        outputs = await asyncio.gather(*[stream.get_output() for stream in streams])
        for i, output in enumerate(outputs):
            assert f"stream_{i}" in output

    async def test_multiple_process_tasks_concurrently(self, mock_session: MagicMock) -> None:
        tasks = [
            ProcessTask(sys.executable, ["-c", f"print('task_{i}')"])
            for i in range(3)
        ]

        tids = await asyncio.gather(*[
            task.start(mock_session, kind="run", name=f"conc_{i}")
            for i, task in enumerate(tasks)
        ])

        await asyncio.gather(*[task.wait(timeout=5) for task in tasks])

        for i, task in enumerate(tasks):
            output = await task.stream.get_output()
            assert f"task_{i}" in output
            remove_task_id(mock_session, tids[i])

    async def test_concurrent_get_output_and_pop_output(self) -> None:
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            for i in range(10):
                q.put(str(i))

        await stream.start(worker, stop_function=lambda: None)
        await stream.wait()

        # get_output multiple times concurrently
        outputs = await asyncio.gather(*[stream.get_output() for _ in range(3)])
        for output in outputs:
            assert "0123456789" in output

        # pop_output should clear
        popped = await stream.pop_output()
        assert "0123456789" in popped
        assert await stream.get_output() == ""


# ---------------------------------------------------------------------------
# Sync utility functions should NOT be awaited
# ---------------------------------------------------------------------------
class TestSyncUtilitiesNotAwaited:
    """Verify that sync utility functions are indeed synchronous."""

    def test_generate_task_id_is_sync(self, mock_session: MagicMock) -> None:
        result = generate_task_id(mock_session, "kind", "name")
        assert result == "kind_name"
        # Should not return a coroutine
        assert not asyncio.iscoroutine(result)

    def test_add_task_is_sync(self, mock_session: MagicMock) -> None:
        stream = BackgroundStream()
        result = add_task(mock_session, "t1", stream)
        assert result is None
        assert not asyncio.iscoroutine(result)

    def test_remove_task_id_is_sync(self, mock_session: MagicMock) -> None:
        stream = BackgroundStream()
        add_task(mock_session, "t1", stream)
        result = remove_task_id(mock_session, "t1")
        assert result is None
        assert not asyncio.iscoroutine(result)

    def test_get_all_tasks_is_sync(self, mock_session: MagicMock) -> None:
        result = get_all_tasks(mock_session)
        assert isinstance(result, dict)
        assert not asyncio.iscoroutine(result)


# ---------------------------------------------------------------------------
# Edge cases for async error handling
# ---------------------------------------------------------------------------
class TestAsyncErrorHandling:
    """Test async error handling in BackgroundStream and ProcessTask."""

    async def test_background_stream_exception_in_async_worker(self) -> None:
        stream = BackgroundStream()

        async def bad_worker(q: queue.Queue[str]) -> None:
            await asyncio.sleep(0.01)
            raise ValueError("async error")

        await stream.start(bad_worker, stop_function=lambda: None)
        await stream.wait()
        assert await stream.success() is False

    async def test_background_stream_exception_in_sync_worker(self) -> None:
        stream = BackgroundStream()

        def bad_worker(q: queue.Queue[str]) -> None:
            raise ValueError("sync error")

        await stream.start(bad_worker, stop_function=lambda: None)
        await stream.wait()
        assert await stream.success() is False

    async def test_process_task_nonexistent_command(self, mock_session: MagicMock) -> None:
        task = ProcessTask("nonexistent_command_12345")
        tid = await task.start(mock_session, kind="run")
        await task.wait(timeout=5)
        assert await task.stream.success() is False
        output = await task.stream.get_output()
        assert "Error" in output or "exited" in output or "nonexistent" in output.lower()
        remove_task_id(mock_session, tid)

    async def test_task_output_handles_missing_task_gracefully(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        result = await tool(TaskOutputParams(task_id="missing", block=True, timeout=3))
        assert isinstance(result, ToolError)

    async def test_task_output_handles_exception_gracefully(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        with patch("kimix.tools.background.get_all_tasks", side_effect=RuntimeError("boom")):
            result = await tool(TaskOutputParams(task_id="any", block=False, timeout=3))
        assert isinstance(result, ToolError)
