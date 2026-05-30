"""Bash command tools — executed via the system bash when available."""
import shlex
from pathlib import Path
from .alias import Alias
from .awk import Awk
from .basename import Basename
from .bc import Bc
from .base64 import Base64
from .bunzip2 import Bunzip2
from .bzip2 import Bzip2
from .cal import Cal
from .cat import Cat
from .chmod import Chmod
from .chgrp import Chgrp
from .chown import Chown
from .cmp import Cmp
from .comm import Comm
from .cksum import Cksum
from .cp import Cp
from .crontab import Crontab
from .csplit import Csplit
from .curl import Curl
from .cut import Cut
from .date import Date
from .dc import Dc
from .df import Df
from .diff import Diff
from .dirname import Dirname
from .du import Du
from .echo import Echo
from .env import Env
from .envsubst import EnvsSubst
from .expand import Expand
from .expr import Expr
from .export import Export
from .factor import Factor
from .false import False_ as FalseCmd
from .file import File
from .find import Find
from .fold import Fold
from .free import Free
from .fmt import Fmt
from .fuser import Fuser
from .grep import Grep
from .groups import Groups
from .gunzip import Gunzip
from .gzip import Gzip
from .head import Head
from .hexdump import Hexdump
from .history import History
from .host import Host
from .hostname import Hostname
from .hwclock import Hwclock
from .id import Id
from .ip import Ip
from .ifconfig import Ifconfig
from .install import Install
from .iostat import Iostat
from .ln import Ln
from .ls import Ls
from .lsb_release import LsbRelease
from .lsof import Lsof
from .man import Man
from .md5sum import Md5sum
from .mkdir import Mkdir
from .mkfifo import Mkfifo
from .mktemp import Mktemp
from .mv import Mv
from .netstat import Netstat
from .nl import Nl
from .nslookup import Nslookup
from .od import Od
from .ping import Ping
from .printf import Printf
from .printenv import Printenv
from .ps import Ps
from .pwd import Pwd
from .readlink import Readlink
from .realpath import Realpath
from .renice import Renice
from .rev import Rev
from .rm import Rm
from .rmdir import Rmdir
from .scriptreplay import Scriptreplay
from .sed import Sed
from .seq import Seq
from .sha256sum import Sha256sum
from .shuf import Shuf
from .sleep import Sleep
from .split import Split
from .ss import Ss
from .stat import Stat
from .strings import Strings
from .sw_vers import SwVers
from .systeminfo import Systeminfo
from .tac import Tac
from .tail import Tail
from .tar import Tar
from .test import Test
from .top import Top
from .touch import Touch
from .tr import Tr
from .traceroute import Traceroute
from .trap import Trap
from .tree import Tree
from .true import True_ as TrueCmd
from .type import Type
from .ulimit import Ulimit
from .umask import Umask
from .uname import Uname
from .unexpand import Unexpand
from .uniq import Uniq
from .unxz import Unxz
from .unzip import Unzip
from .uptime import Uptime
from .vmstat import Vmstat
from .wc import Wc
from .wget import Wget
from .which import Which
from .who import Who
from .whoami import Whoami
from .xxd import Xxd
from .xz import Xz
from .yes import Yes
from .zip import Zip
import os

# ============================================================
# Bash command dispatch tool
# ============================================================

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field
from kimi_cli.session import Session


# Build the command dispatch map
BASH_COMMANDS: dict[str, CallableTool2] = {
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
WINDOWS_ALIASES: dict[str, str] = {
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

# Backward-compatible private aliases
_BASH_COMMANDS = BASH_COMMANDS
_WINDOWS_ALIASES = WINDOWS_ALIASES

# New system-bash-based Bash tool
from .bash_tool import Bash, BashParams

__all__ = [
    "Alias", "Awk", "Base64", "Basename", "Bc", "Bunzip2", "Bzip2",
    "Cal", "Cat", "Chgrp", "Chmod", "Chown", "Cksum", "Cmp", "Comm",
    "Cp", "Crontab", "Csplit", "Curl", "Cut", "Date", "Dc", "Df",
    "Diff", "Dirname", "Du", "Echo", "EnvsSubst", "Env", "Expand",
    "Expr", "Export", "Factor", "FalseCmd", "File", "Find", "Fold",
    "Fmt", "Free", "Fuser", "Grep", "Groups", "Gunzip", "Gzip",
    "Head", "Hexdump", "History", "Host", "Hostname", "Hwclock",
    "Id", "Ifconfig", "Install", "Iostat", "Ip", "Ln", "Ls",
    "LsbRelease", "Lsof", "Man", "Md5sum", "Mkdir", "Mkfifo",
    "Mktemp", "Mv", "Netstat", "Nl", "Nslookup", "Od", "Ping",
    "Printf", "Printenv", "Ps", "Pwd", "Readlink", "Realpath",
    "Renice", "Rev", "Rm", "Rmdir", "Scriptreplay", "Sed", "Seq",
    "Sha256sum", "Shuf", "Sleep", "Split", "Ss", "Stat", "Strings",
    "SwVers", "Systeminfo", "Tac", "Tail", "Tar", "Test", "Top",
    "Touch", "Tr", "Traceroute", "Trap", "Tree", "TrueCmd", "Type",
    "Ulimit", "Umask", "Uname", "Unexpand", "Uniq", "Unxz", "Unzip",
    "Uptime", "Vmstat", "Wc", "Wget", "Which", "Who", "Whoami",
    "Xxd", "Xz", "Yes", "Zip",
    # Dispatcher & params
    "Bash", "BashParams",
    # Maps
    "BASH_COMMANDS", "WINDOWS_ALIASES",
]
