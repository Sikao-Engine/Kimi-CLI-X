import asyncio
import atexit
import codecs
import io
import os
import re
import shutil
import uuid
from pathlib import Path
import queue
import threading
from typing import TYPE_CHECKING

# Common error keywords for detecting error lines in process output
_ERROR_KEYWORDS = [
    "error", "exception", "traceback", "failed", "failure",
    "fatal", "panic", "abort", "assertion", "undefined",
    "syntaxerror", "typeerror", "valueerror", "keyerror",
    "importerror", "modulenotfounderror", "attributeerror",
    "nameerror", "runtimeerror", "oserror", "ioerror",
    "zerodivisionerror", "indexerror", "memoryerror",
    "recursionerror", "unboundlocalerror", "referenceerror",
    "permission denied", "access denied", "not found",
    "cannot find", "does not exist", "no such file",
    "connection refused", "timeout", "unhandled",
]

_DEFAULT_FORBIDDEN_COMMANDS = [
    "taskkill",
    "kill",
    "killall",
    "pkill",
    "xkill",
    "rd",
    "format",
    "fdisk",
    "dd",
    "mkfs",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "reg",
    "regedit",
    "regedt32",
]

_ERROR_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _ERROR_KEYWORDS) + r')\b',
    re.IGNORECASE
)


def _find_error_line_index(output: str) -> int | None:
    """Find the 1-based line index of the first line containing a common error keyword."""
    for idx, line in enumerate(output.splitlines(), start=1):
        if _ERROR_PATTERN.search(line):
            return idx
    return None


# ANSI escape sequences (colored text)
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Emoji base characters
_EMOJI_BASE = (
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U000024FF"  # enclosed alphanumerics
    "\U0001F100-\U0001F251"  # enclosed alphanumeric / ideographic supplements
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers
)

_EMOJI_RE = re.compile(
    f"[{_EMOJI_BASE}]"
    f"(?:"
    f"  [{_EMOJI_BASE}\U0000FE0F\U000020E3]"
    f"| \U0000200D [{_EMOJI_BASE}]"
    f")*",
    re.VERBOSE | re.UNICODE,
)

# Keycap sequences (e.g. 1️⃣, #️⃣) – preserve the base digit/symbol
_KEYCAP_RE = re.compile(r"([0-9#*])\uFE0F?\u20E3")


def filter_output(text: str) -> str:
    """Process process pipeline stdout.

    Steps:
        1. Remove ANSI escape sequences (colored text).
        2. Remove emoji characters and sequences.

    Args:
        text: Raw stdout text.

    Returns:
        Cleaned plain text.
    """
    if not isinstance(text, str):
        raise TypeError("filter_output expects a string")
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _KEYCAP_RE.sub(r"\1", text)
    return text


from kimi_cli.session import Session

if TYPE_CHECKING:
    from kimix.tools.background.utils import BackgroundStream
OUTPUT_LIMIT = 65536
_temp_folder = Path.home() / '.kimi' / 'sessions' / uuid.uuid4().hex
_temp_folder.mkdir(parents=True, exist_ok=True)
_temp_idx = 0
_temp_set: dict[Path, int] = dict()


def _cleanup_temp_folder() -> None:
    if _temp_folder.exists():
        shutil.rmtree(_temp_folder, ignore_errors=True)


atexit.register(_cleanup_temp_folder)




def _create_temp_file_name(ext: str = '.md') -> str:
    global _temp_idx
    id = _temp_idx
    _temp_idx += 1
    return str(_temp_folder / (str(id) + ext))


def _export_to_temp_file(key: Path | None, content: str, ext: str = '.txt') -> tuple[str, bool]:
    global _temp_idx
    """Export content to a temporary file and return the file path."""
    id = _temp_idx
    new_id = True
    if key:
        v = _temp_set.get(key)
        if v is not None:
            id = v
            new_id = False
        else:
            # Add key to _temp_set with the new id
            _temp_set[key] = id
    if new_id:
        _temp_idx += 1
    _temp_folder.mkdir(parents=True, exist_ok=True)
    name = str(_temp_folder / (str(id) + ext))
    # Append content if key exists, otherwise overwrite/create
    mode = 'a' if not new_id else 'w'
    with open(name, mode, encoding='utf-8') as f:
        f.write(content)
    return name, new_id


async def _export_to_temp_file_async(key: Path | None, content: str, ext: str = '.txt') -> tuple[str, bool]:
    global _temp_idx
    """Async version: Export content to a temporary file and return the file path."""
    import anyio
    id = _temp_idx
    new_id = True
    if key:
        v = _temp_set.get(key)
        if v is not None:
            id = v
            new_id = False
        else:
            # Add key to _temp_set with the new id
            _temp_set[key] = id
    if new_id:
        _temp_idx += 1
    name = _temp_folder / (str(id) + ext)
    # Append content if key exists, otherwise overwrite/create
    mode = 'a' if not new_id else 'w'
    async with await anyio.open_file(name, mode, encoding='utf-8') as f:
        await f.write(content)
    return str(name), new_id


def _maybe_export_output(output: str, key: Path | None = None) -> str:
    """Check if output is too large and export to temp file if needed.

    Args:
        output: The output string to check.
        key: Optional Path to normalize and use in the output message.

    Returns:
        The output string, or a message indicating it was exported to a temp file.
    """
    if not output:
        return ''
    if len(output) > OUTPUT_LIMIT:
        if key is not None:
            if type(key) is not Path:
                key = Path(key)
            key = key.resolve()
        temp_path, new_id = _export_to_temp_file(key, output)
        return f"Output too large, {'exported' if new_id else 'added'} to file `{temp_path}`"
    return output


async def _maybe_export_output_async(output: str, key: Path | None = None) -> str:
    """Async version: Check if output is too large and export to temp file if needed.

    Args:
        output: The output string to check.
        key: Optional Path to normalize and use in the output message.

    Returns:
        The output string, or a message indicating it was exported to a temp file.
    """
    if not output:
        return ''
    if len(output) > OUTPUT_LIMIT:
        if key is not None:
            if type(key) is not Path:
                key = Path(key)
            key = key.resolve()
        temp_path, new_id = await _export_to_temp_file_async(key, output)
        return f"[Output too large, {'exported' if new_id else 'added'} to file: {temp_path}]"
    return output


class ProcessTask:
    """Run a subprocess in the background with stream output and input support."""

    def __init__(self, path: str, args: list[str] | None = None, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        import shutil
        # On Windows, subprocess.Popen with shell=False does not resolve .cmd/.bat
        # via PATHEXT. Use shutil.which to find the real executable (e.g. pnpm.CMD).
        if not Path(path).exists():
            resolved = shutil.which(path)
            if resolved:
                path = resolved
        self.path = path
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self._stop_event = threading.Event()
        self._process_ref: asyncio.subprocess.Process | None = None
        self._stream: 'BackgroundStream' | None = None
        self._task_id: str | None = None
        self._input_queue: queue.Queue[str] = queue.Queue()

    async def _run_process_bg(self, q: queue.Queue[str]) -> bool:
        """Run the process and collect output into the queue."""
        process = None
        output_buffer = io.StringIO()
        try:
            if self._stop_event.is_set():
                return False
            # Start the process
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            process = await asyncio.create_subprocess_exec(
                self.path,
                *self.args,
                cwd=self.cwd,
                env=process_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._process_ref = process
            # Read stdout and stderr concurrently with stop checking

            if process.stdout is None:
                raise RuntimeError("Subprocess stdout is None")

            async def read_stdout() -> None:
                try:
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    while True:
                        if self._stop_event.is_set():
                            break
                        data = await process.stdout.read(4096)
                        if data:
                            text = decoder.decode(data)
                            if text:
                                text = filter_output(text)
                                q.put_nowait(text)
                                output_buffer.write(text)
                        else:
                            text = decoder.decode(b'', final=True)
                            if text:
                                text = filter_output(text)
                                q.put_nowait(text)
                                output_buffer.write(text)
                            break
                except (IOError, OSError, ValueError, asyncio.CancelledError):
                    pass

            async def read_stderr() -> None:
                if process.stderr is None:
                    return
                try:
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    while True:
                        if self._stop_event.is_set():
                            break
                        data = await process.stderr.read(4096)
                        if data:
                            text = decoder.decode(data)
                            if text:
                                text = filter_output(text)
                                msg = "[stderr] " + text
                                q.put_nowait(msg)
                                output_buffer.write(msg)
                        else:
                            text = decoder.decode(b'', final=True)
                            if text:
                                text = filter_output(text)
                                msg = "[stderr] " + text
                                q.put_nowait(msg)
                                output_buffer.write(msg)
                            break
                except (IOError, OSError, ValueError, asyncio.CancelledError):
                    pass

            async def write_stdin() -> None:
                try:
                    while True:
                        if self._stop_event.is_set() or process.returncode is not None:
                            break
                        if process.stdin is None:
                            raise RuntimeError("Subprocess stdin is None")
                        try:
                            data = self._input_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue
                        process.stdin.write(data.encode('utf-8', errors='replace'))
                        await process.stdin.drain()
                except (IOError, OSError, ValueError, asyncio.CancelledError):
                    pass

            # Start reader/writer tasks
            stdout_task = asyncio.create_task(read_stdout())
            stderr_task: asyncio.Task[None] | None = None
            if process.stderr is not None:
                stderr_task = asyncio.create_task(read_stderr())
            stdin_task = asyncio.create_task(write_stdin())

            # Wait for process completion with periodic stop checking
            while process.returncode is None:
                if self._stop_event.is_set():
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    break
                await asyncio.sleep(0.1)

            if process.returncode is not None and not self._stop_event.is_set():
                await process.wait()

            # Cancel tasks and wait for them to finish
            stdout_task.cancel()
            if stderr_task is not None:
                stderr_task.cancel()
            stdin_task.cancel()
            try:
                await stdout_task
            except asyncio.CancelledError:
                pass
            if stderr_task is not None:
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            try:
                await stdin_task
            except asyncio.CancelledError:
                pass

            # Read any remaining data from stdout and stderr
            try:
                remaining_stdout = await process.stdout.read()
                if remaining_stdout:
                    text = remaining_stdout.decode('utf-8', errors='replace')
                    text = filter_output(text)
                    q.put_nowait(text)
                    output_buffer.write(text)
            except (IOError, OSError, ValueError):
                pass
            if process.stderr is not None:
                try:
                    remaining_stderr = await process.stderr.read()
                    if remaining_stderr:
                        text = remaining_stderr.decode('utf-8', errors='replace')
                        text = filter_output(text)
                        msg = "[stderr] " + text
                        q.put_nowait(msg)
                        output_buffer.write(msg)
                except (IOError, OSError, ValueError):
                    pass
            # Report completion status
            return_code = process.returncode
            if self._stop_event.is_set():
                q.put_nowait("\n[Process stopped by user]")
                return False
            elif return_code is not None and return_code != 0:
                full_output = output_buffer.getvalue()
                error_line = _find_error_line_index(full_output)
                if error_line is not None:
                    q.put_nowait(f"\n[Process exited with code {return_code}, error at line {error_line}]")
                else:
                    q.put_nowait(f"\n[Process exited with code {return_code}]")
                return False
            return True

        except Exception as e:
            q.put_nowait(f"\n[Error: {str(e)}]")
            return False
        finally:
            self._stop_event.set()
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

    async def _stop_function(self) -> None:
        """Signal the background process to stop."""
        self._stop_event.set()
        # Also try to terminate the process directly if it's running
        proc = self._process_ref
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                pass

    async def _input_function(self, data: str) -> bool:
        """Push data to the process's stdin.

        Args:
            data: The string data to write to stdin.

        Returns:
            True if data was written successfully, False otherwise.
        """
        proc = None
        # Wait for the process to be available
        while True:
            if self._stop_event.is_set():
                return False
            proc = self._process_ref
            if proc is None:
                await asyncio.sleep(0.05)
            else:
                break

        # Write data to stdin
        try:
            if proc.stdin is not None and proc.returncode is None:
                self._input_queue.put_nowait(data)
                return True
        except (IOError, OSError, ValueError):
            # Process may have terminated or stdin is closed
            pass
        return False

    async def start(self, session: Session, kind: str = "run", name: str | None = None) -> str:
        """Start the background process and register it as a task.

        Args:
            session: The session instance.
            kind: Task kind prefix for the task ID.
            name: Optional name for the task ID (defaults to the executable stem).

        Returns:
            The generated task ID.
        """
        from kimix.tools.background.utils import BackgroundStream, generate_task_id, add_task
        self._stream = BackgroundStream()
        # Generate a task ID based on the executable name
        self._task_id = generate_task_id(session, kind, name)
        await self._stream.start(self._run_process_bg,
                           self._stop_function, self._input_function)
        # Register the task
        add_task(session, self._task_id, self._stream)
        assert self._task_id is not None
        return self._task_id

    async def wait(self, timeout: float | None = None) -> None:
        await self._stream.wait(timeout)

    async def thread_is_alive(self) -> bool:
        return await self._stream.thread_is_alive()

    async def stop(self) -> None:
        """Stop the background process."""
        if self._stream is not None:
            await self._stream.stop()

    async def input(self, data: str) -> bool:
        """Push data to the process's stdin.

        Args:
            data: The string data to write to stdin.

        Returns:
            True if data was written successfully, False otherwise.
        """
        if self._stream is not None:
            return await self._stream.input(data)
        return False

    @property
    def task_id(self) -> str | None:
        """The task ID if the process has been started."""
        return self._task_id

    @property
    def stream(self) -> 'BackgroundStream' | None:
        """The underlying BackgroundStream if the process has been started."""
        return self._stream

