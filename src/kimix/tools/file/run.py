"""run tool for executing a process from a path."""
import anyio
import asyncio
from pathlib import Path

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field
from kimi_cli.session import Session
from kimix.tools.common import _maybe_export_output_async, _export_to_temp_file_async, ProcessTask
from kimix.tools.file.bash import (
    Alias, Awk, Base64, Basename, Bc, Bunzip2, Bzip2, Cal, Cat, Chgrp, Chmod, Chown, Cksum, Cmp, Comm, Cp,
    Crontab, Csplit, Curl, Cut, Date, Dc, Df, Diff, Dirname, Du, Echo, EnvsSubst, Env, Expand, Expr,
    Export, Factor, FalseCmd, File, Find, Fold, Fmt, Free, Fuser, Grep, Groups, Gunzip, Gzip, Head,
    Hexdump, History, Host, Hostname, Hwclock, Id, Ifconfig, Install, Iostat, Ip, Kill, Killall, Ln, Ls,
    LsbRelease, Lsof, Man, Md5sum, Mkdir, Mkfifo, Mktemp, Mv, Netstat, Nl, Nslookup, Od, Ping,
    Printf, Printenv, Ps, Pwd, Readlink, Realpath, Renice, Rev, Rm, Rmdir, Scriptreplay, Sed, Seq,
    Sha256sum, Shuf, Sleep, Split, Ss, Stat, Strings, SwVers, Systeminfo, Tac, Tail, Tar, Test, Top,
    Touch, Tr, Traceroute, Trap, Tree, TrueCmd, Type, Ulimit, Umask, Uname, Unexpand, Uniq, Unxz, Unzip,
    Uptime, Vmstat, Wc, Wget, Which, Who, Whoami, Xxd, Xz, Yes, Zip,
)


_BASH_COMMANDS: dict[str, CallableTool2] = {
    "alias": Alias(),
    "awk": Awk(),
    "base64": Base64(),
    "basename": Basename(),
    "bc": Bc(),
    "bunzip2": Bunzip2(),
    "bzip2": Bzip2(),
    "cal": Cal(),
    "cat": Cat(),
    "chgrp": Chgrp(),
    "chmod": Chmod(),
    "chown": Chown(),
    "cksum": Cksum(),
    "cmp": Cmp(),
    "comm": Comm(),
    "cp": Cp(),
    "crontab": Crontab(),
    "csplit": Csplit(),
    "curl": Curl(),
    "cut": Cut(),
    "date": Date(),
    "dc": Dc(),
    "df": Df(),
    "diff": Diff(),
    "dirname": Dirname(),
    "du": Du(),
    "echo": Echo(),
    "env": Env(),
    "envsubst": EnvsSubst(),
    "expand": Expand(),
    "expr": Expr(),
    "export": Export(),
    "factor": Factor(),
    "false": FalseCmd(),
    "file": File(),
    "find": Find(),
    "fold": Fold(),
    "fmt": Fmt(),
    "free": Free(),
    "fuser": Fuser(),
    "grep": Grep(),
    "groups": Groups(),
    "gunzip": Gunzip(),
    "gzip": Gzip(),
    "head": Head(),
    "hexdump": Hexdump(),
    "history": History(),
    "host": Host(),
    "hostname": Hostname(),
    "hwclock": Hwclock(),
    "id": Id(),
    "ifconfig": Ifconfig(),
    "install": Install(),
    "iostat": Iostat(),
    "ip": Ip(),
    "kill": Kill(),
    "killall": Killall(),
    "ln": Ln(),
    "ls": Ls(),
    "lsb_release": LsbRelease(),
    "lsof": Lsof(),
    "man": Man(),
    "md5sum": Md5sum(),
    "mkdir": Mkdir(),
    "mkfifo": Mkfifo(),
    "mktemp": Mktemp(),
    "mv": Mv(),
    "netstat": Netstat(),
    "nl": Nl(),
    "nslookup": Nslookup(),
    "od": Od(),
    "ping": Ping(),
    "printf": Printf(),
    "printenv": Printenv(),
    "ps": Ps(),
    "pwd": Pwd(),
    "readlink": Readlink(),
    "realpath": Realpath(),
    "renice": Renice(),
    "rev": Rev(),
    "rm": Rm(),
    "rmdir": Rmdir(),
    "scriptreplay": Scriptreplay(),
    "sed": Sed(),
    "seq": Seq(),
    "sha256sum": Sha256sum(),
    "shuf": Shuf(),
    "sleep": Sleep(),
    "split": Split(),
    "ss": Ss(),
    "stat": Stat(),
    "strings": Strings(),
    "sw_vers": SwVers(),
    "systeminfo": Systeminfo(),
    "tac": Tac(),
    "tail": Tail(),
    "tar": Tar(),
    "test": Test(),
    "top": Top(),
    "touch": Touch(),
    "tr": Tr(),
    "traceroute": Traceroute(),
    "trap": Trap(),
    "tree": Tree(),
    "true": TrueCmd(),
    "type": Type(),
    "ulimit": Ulimit(),
    "umask": Umask(),
    "uname": Uname(),
    "unexpand": Unexpand(),
    "uniq": Uniq(),
    "unxz": Unxz(),
    "unzip": Unzip(),
    "uptime": Uptime(),
    "vmstat": Vmstat(),
    "wc": Wc(),
    "wget": Wget(),
    "which": Which(),
    "who": Who(),
    "whoami": Whoami(),
    "xxd": Xxd(),
    "xz": Xz(),
    "yes": Yes(),
    "zip": Zip(),
}

# Map Windows CMD / PowerShell command names to their bash equivalents.
_WINDOWS_ALIASES: dict[str, str] = {
    # CMD
    "dir": "ls",
    "copy": "cp",
    "move": "mv",
    "del": "rm",
    "erase": "rm",
    "ren": "mv",
    "rename": "mv",
    "type": "cat",
    "fc": "diff",
    # PowerShell
    "Get-ChildItem": "ls",
    "Copy-Item": "cp",
    "Move-Item": "mv",
    "Remove-Item": "rm",
    "Get-Content": "cat",
    "Get-Location": "pwd",
    "Get-Process": "ps",
    "Select-String": "grep",
}


class RunParams(BaseModel):
    path: str = Field(
        description="Executable path or basic linux-bash cmd."
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command arguments."
    )
    timeout: int = Field(
        default=10,
        ge=3,
        le=180,
        description="Timeout in seconds."
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory."
    )
    output_path: str | None = Field(
        default=None,
        description="Output file path."
    )

class Run(CallableTool2[RunParams]):
    name: str = "Run"
    description: str = "Run an executable."
    params: type[RunParams] = RunParams

    def __init__(self, session: Session):
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        super().__init__()
        self._session = session
        self._semaphore = asyncio.Semaphore(8)

    async def _run_bash_tool(self, params: RunParams, bash_tool: CallableTool2) -> ToolReturnValue:
        import queue
        from kimix.tools.background.utils import BackgroundStream, generate_task_id, add_task, remove_task_id

        result_holder: list[ToolReturnValue] = []

        async def wrapper(q: queue.Queue[str]) -> bool:
            try:
                result = await bash_tool(params)
                result_holder.append(result)
                output_str = result.output if isinstance(result.output, str) else str(result.output)
                q.put_nowait(output_str)
                return not result.is_error
            except Exception as e:
                q.put_nowait(f"\n[Error: {str(e)}]")
                return False

        stream = BackgroundStream()
        task_id = generate_task_id(self._session, "run", params.path)
        await stream.start(wrapper, lambda: None)
        add_task(self._session, task_id, stream)

        await stream.wait(params.timeout)

        if await stream.thread_is_alive():
            output = await stream.get_output()
            return ToolError(
                output=output or f'Running in background. task_id: `{task_id}`. use `TaskOutput` or `Input`',
                message="Process timeout",
                brief="Timeout"
            )

        remove_task_id(self._session, task_id)

        if result_holder:
            return result_holder[0]

        output = await stream.pop_output()
        success = await stream.success()
        if not success:
            return ToolError(output=output, message="Command execution failed", brief="Command execution failed")
        output = await _maybe_export_output_async(output)
        return ToolOk(output=output)

    async def __call__(self, params: RunParams) -> ToolReturnValue:
        # params.path may contain arguments, split it with space, then insert to the start of params.args
        # Try progressively longer prefixes to find an existing file, so paths with spaces are handled.
        if " " in params.path:
            parts = params.path.split(" ")
            candidate = parts[0]
            for i in range(1, len(parts)):
                candidate += " " + parts[i]
                try:
                    is_file = Path(candidate).is_file()
                except OSError:
                    is_file = False
                if is_file:
                    params.path = candidate
                    remaining = parts[i + 1 :]
                    if remaining:
                        params.args.insert(0, " ".join(remaining))
                    break
            else:
                params.path = parts[0]
                remaining = parts[1:]
                if remaining:
                    params.args.insert(0, " ".join(remaining))

        async with self._semaphore:
            import sys

            bash_name = _WINDOWS_ALIASES.get(params.path, params.path)
            bash_tool = _BASH_COMMANDS.get(bash_name)
            if bash_tool:
                return await self._run_bash_tool(params, bash_tool)

            # check if using python
            if params.path == 'python':
                params.path = sys.executable
            task = ProcessTask(params.path, params.args, params.cwd)
            task_id = await task.start(self._session, "run", Path(params.path).stem)

            # Wait for completion with timeout (allow a small buffer for cleanup)
            wait_timeout = params.timeout
            await task.wait(wait_timeout)
            
            if await task.thread_is_alive():
                output = await task.stream.get_output() if task.stream else ""
                return ToolError(
                    output=output,
                    message=f"Running in background. task_id: `{task_id}`. use `TaskOutput` or `Input`",
                    brief="Timeout"
                )
            # Clean up foreground task registration
            from kimix.tools.background.utils import remove_task_id
            remove_task_id(self._session, task_id)

            # Get output
            output = await task.stream.pop_output() if task.stream else ""

            # Handle output export if needed
            if params.output_path:
                async with await anyio.open_file(params.output_path, 'w', encoding='utf-8', errors='replace') as f:
                    await f.write(output)
                output = f'saved to file `{params.output_path}`'
            
            # Check success
            success = await task.stream.success() if task.stream else False


            if not success:
                if output and not params.output_path:
                    temp_path, _ = await _export_to_temp_file_async(key=None, content=output, ext='.txt')
                    output = f'saved to file `{temp_path}`'
                return ToolError(
                    output=output,
                    message="Command execution failed",
                    brief="Command execution failed"
                )

            output = await _maybe_export_output_async(output)
            return ToolOk(output=output)
