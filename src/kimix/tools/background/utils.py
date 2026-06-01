import asyncio
import inspect
import io
import threading
import queue
from typing import Any, Awaitable, Callable, cast

from kimi_cli.session import Session


class TaskData:
    def __init__(self) -> None:
        self.task_names: dict[str, int] = {}
        self.tasks: dict[str, BackgroundStream] = {}


def _get_or_add_task_data(session: Session) -> TaskData:
    data = session.custom_data.get('background_task_data')
    if data is None:
        data = TaskData()
        session.custom_data['background_task_data'] = data
    return cast(TaskData, data)


def _get_task_data(session: Session) -> TaskData | None:
    return cast(TaskData | None, session.custom_data.get('background_task_data'))


def _pop_task_data(session: Session) -> TaskData | None:
    return cast(TaskData | None, session.custom_data.pop('background_task_data', None))


class BackgroundStream:
    """A wrapper for background thread execution with a thread-safe queue."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[str] | None = None
        self._started: bool = False
        self._stopped: bool = False
        self._stop_function: Callable[[], Any] | None = None
        self._input_function: Callable[[str], Any] | None = None
        self._lock = threading.Lock()
        self._success = False
        self._output = io.StringIO()

    async def success(self) -> bool:
        return self._success

    async def start(self, function: Callable[[queue.Queue[str]], Any] | Callable[[queue.Queue[str]], Awaitable[Any]], stop_function: Callable[[], Any] | Callable[[], Awaitable[Any]], input_function: Callable[[str], Any] | Callable[[str], Awaitable[Any]] | None = None) -> None:
        """Start the background thread with the given function.

        Args:
            function: A callable that accepts a queue.Queue[str] as its argument.
                     The function can put strings into the queue for retrieval by other threads.
        """
        with self._lock:
            if self._started:
                return

            q: queue.Queue[str] = queue.Queue()
            self._queue = q

            def func(v: BackgroundStream, function: Callable[[queue.Queue[str]], Any] | Callable[[queue.Queue[str]], Awaitable[Any]]) -> None:
                try:
                    if inspect.iscoroutinefunction(function):
                        result = asyncio.run(function(q))
                    else:
                        result = function(q)
                except Exception:
                    result = False
                v._success = False if result == False else True # defaultly success
            self._thread = threading.Thread(
                target=func, args=(self, function), daemon=True)
            self._stop_function = stop_function
            self._input_function = input_function
            self._started = True
        self._thread.start()

    async def input(self, data: str) -> bool:
        if self._input_function:
            if inspect.iscoroutinefunction(self._input_function):
                return bool(await self._input_function(data))
            return bool(self._input_function(data))
        return False

    async def thread_is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    async def wait(self, timeout: float | None = None) -> None:
        """Wait for the background thread to complete."""
        if not await self.thread_is_alive():
            return
        thread = self._thread
        if thread is None:
            return
        await asyncio.to_thread(thread.join, timeout=timeout)
        if not thread.is_alive():
            with self._lock:
                self._thread = None

    async def get_output(self) -> str:
        if self._queue is None:
            return self._output.getvalue()

        while True:
            try:
                self._output.write(self._queue.get_nowait())
            except queue.Empty:
                break
        return self._output.getvalue()

    async def pop_output(self) -> str:
        output = await self.get_output()
        self._output.truncate(0)
        self._output.seek(0)
        return output

    async def get_queue(self) -> queue.Queue[str] | None:
        """Get the thread-safe queue for retrieving messages.

        Returns:
            The queue if started, None otherwise.
        """
        return self._queue

    async def is_started(self) -> bool:
        """Check if the stream has been started."""
        return self._started

    async def is_stopped(self) -> bool:
        """Check if the stream has been stopped."""
        return self._stopped

    async def stop(self) -> bool:
        """Stop the background thread.

        Returns:
            True if the thread was stopped, False if it was not running.
        """
        with self._lock:
            if not self._started or self._stopped:
                return False
            self._stopped = True
            thread_alive = self._thread is not None and self._thread.is_alive()
            stop_func = self._stop_function if thread_alive else None

        if stop_func is not None:
            try:
                if inspect.iscoroutinefunction(stop_func):
                    await stop_func()
                else:
                    stop_func()
            except Exception:
                pass
        return thread_alive


def generate_task_id(session: Session, kind: str, name: str | None = None) -> str:
    if name:
        base_id = f"{kind}_{name}"
    else:
        base_id = kind
    data = _get_or_add_task_data(session)
    if base_id not in data.task_names:
        data.task_names[base_id] = 0
        return base_id

    data.task_names[base_id] += 1
    return f"{base_id}_{data.task_names[base_id]}".strip()


def remove_task_id(session: Session, task_id: str) -> BackgroundStream | None:
    task_id = task_id.strip()
    """Remove a task_id from the session task registry.

    Args:
        session: The session instance.
        task_id: The task identifier to remove.
    """
    try:
        data = _get_task_data(session)
        if data is not None:
            data.tasks.pop(task_id)
    except KeyError:
        pass
    return None


def add_task(session: Session, task_id: str, stream: BackgroundStream) -> None:
    task_id = task_id.strip()
    """Add a task to the session task registry.

    Args:
        session: The session instance.
        task_id: Unique identifier for the task.
        stream: The BackgroundStream instance to manage (should already be started).
    """
    _get_or_add_task_data(session).tasks[task_id] = stream


def get_all_tasks(session: Session) -> dict[str, BackgroundStream]:
    return _get_or_add_task_data(session).tasks


async def join_task(session: Session, task_id: str) -> bool:
    task_id = task_id.strip()
    """Join a task and clean up its resources.

    Args:
        session: The session instance.
        task_id: The task identifier to join.

    Returns:
        True if the task was found and joined, False otherwise.
    """
    data = _get_task_data(session)
    if (data is None) or (task_id not in data.tasks):
        return False

    stream = data.tasks.pop(task_id)
    await stream.wait()
    return True


async def discard_all_tasks(session: Session) -> None:
    """Join all tasks and clear the session registries."""
    data = _pop_task_data(session)
    if data is None:
        return
    for stream in list(data.tasks.values()):
        await stream.stop()
    del data
