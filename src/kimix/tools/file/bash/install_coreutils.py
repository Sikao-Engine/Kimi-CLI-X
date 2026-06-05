"""Install Microsoft Coreutils for Windows silently.

Strategy (in priority order):
1. WinGet (official Microsoft recommendation)
2. Direct download from GitHub latest release
3. Chocolatey (if already available)
4. Scoop (if already available)

Usage:
    python install_coreutils.py                          # default install
    python install_coreutils.py --dir "D:\\Coreutils"     # custom dir
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from kimi_cli.share import get_share_dir

# Re-use helpers from install_git to avoid duplication.
from kimix.tools.file.bash.install_git import _download_file, _ensure_in_user_path

# ============================================================
# Global configuration
# ============================================================
INSTALL_DIR = get_share_dir() / "coreutils"
"""Default install directory for the portable extraction strategy."""

_GITHUB_API_URL = "https://api.github.com/repos/microsoft/coreutils/releases/latest"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return sys.platform == "win32"


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result (stdout/stderr captured as text)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _coreutils_found(install_dir: str | None = None) -> bool:
    """Return ``True`` if ``cat.exe`` is available.

    When *install_dir* is given, checks that directory first
    (looking under ``bin/cat.exe``).  Falls back to checking
    PATH when *install_dir* is ``None``.
    """
    if install_dir:
        base = Path(install_dir)
        if (base / "bin" / "cat.exe").exists():
            return True
    return shutil.which("cat.exe") is not None


# ---------------------------------------------------------------------------
# strategy implementations
# ---------------------------------------------------------------------------

def _try_winget() -> bool:
    """Install Coreutils via WinGet (preferred, official Microsoft channel)."""
    if not shutil.which("winget"):
        return False
    try:
        print("Installing Coreutils via WinGet ...")
        result = _run(
            [
                "winget",
                "install",
                "--id",
                "Microsoft.Coreutils",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--scope",
                "user",
            ],
            timeout=300,
        )
        # WinSet returns 0 on success but may also return non-zero when
        # the package is already installed; verify by looking for cat.exe.
        return _coreutils_found()
    except Exception as exc:
        print(f"WinGet install failed: {exc}")
        return False


def _try_github_download(
    install_dir: str | None = None,
) -> bool:
    """Download the latest Coreutils installer from GitHub and run it silently.

    This is a best-effort fallback: we try common silent-install flags
    used by NSIS, Inno Setup, and WiX.  If none succeed we give up.
    """
    try:
        print("Querying GitHub API for latest Coreutils release ...")
        req = urllib.request.Request(
            _GITHUB_API_URL,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "kimix-installer"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"GitHub API query failed: {exc}")
        return False

    assets = data.get("assets", [])
    # Prefer x64 over arm64
    asset = next(
        (a for a in assets if a.get("name", "").endswith("-x64.exe")),
        None,
    )
    if asset is None:
        asset = next(
            (a for a in assets if a.get("name", "").endswith(".exe")),
            None,
        )
    if asset is None:
        print("No .exe asset found in latest release.")
        return False

    download_url = asset["browser_download_url"]
    installer_name = asset["name"]
    installer = Path(tempfile.gettempdir()) / installer_name

    # --- download ---
    try:
        print(f"Downloading {installer_name} ...")
        _download_file(download_url, installer)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False

    # --- install ---
    # Try a handful of common silent-install flag sets.
    silent_flag_sets: list[list[str]] = [
        ["/VERYSILENT", "/NORESTART"],          # Inno Setup
        ["/SILENT", "/NORESTART"],              # Inno Setup (less silent)
        ["/S"],                                  # NSIS
        ["/quiet", "/norestart"],               # WiX / MSI
        ["--silent"],                            # Generic
    ]

    ok = False
    for flags in silent_flag_sets:
        try:
            print(f"Running installer with flags {flags} ...")
            _run([str(installer), *flags], timeout=300)
        except subprocess.TimeoutExpired:
            print("Installer timed out.")
            continue
        except Exception as exc:
            print(f"Installer error: {exc}")
            continue

        if _coreutils_found():
            ok = True
            break

    # --- clean up ---
    installer.unlink(missing_ok=True)
    return ok


def _try_choco() -> bool:
    """Install Coreutils via Chocolatey (if already on the machine)."""
    if not shutil.which("choco"):
        return False
    try:
        print("Installing Coreutils via Chocolatey ...")
        result = _run(["choco", "install", "microsoft-coreutils", "-y"], timeout=300)
        return result.returncode == 0 or _coreutils_found()
    except Exception as exc:
        print(f"Chocolatey install failed: {exc}")
        return False


def _try_scoop() -> bool:
    """Install Coreutils via Scoop (if already on the machine)."""
    if not shutil.which("scoop"):
        return False
    try:
        print("Installing Coreutils via Scoop ...")
        result = _run(["scoop", "install", "coreutils"], timeout=300)
        return result.returncode == 0 or _coreutils_found()
    except Exception as exc:
        print(f"Scoop install failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def install_coreutils(
    install_dir: str | None = None,
    *,
    add_to_path: bool = True,
    timeout: int = 300,
) -> str | None:
    """Silently install Microsoft Coreutils for Windows.

    Tries, in order:
      1. WinGet (``winget install Microsoft.Coreutils --silent``).
      2. Direct download from GitHub latest release (portable installer).
      3. Chocolatey (``choco install microsoft-coreutils``).
      4. Scoop (``scoop install coreutils``).

    Parameters
    ----------
    install_dir:
        Target directory.  Defaults to ``<share_dir>/coreutils``.
    add_to_path:
        Whether to append the ``bin`` folder to the user PATH.
    timeout:
        Seconds to wait for each install subprocess.

    Returns
    -------
    The ``bin`` directory path on success, or ``None`` on failure.
    """
    if not _is_windows():
        print("install_coreutils: this script only supports Windows.", file=sys.stderr)
        return None

    target = Path(install_dir) if install_dir else INSTALL_DIR
    bin_dir = str(target / "bin")

    # Already installed on PATH or in target dir?
    if _coreutils_found(install_dir):
        where = f"at {install_dir}" if install_dir else "on PATH"
        print(f"Coreutils is already installed {where}.")
        if add_to_path:
            _ensure_in_user_path(bin_dir)
        return bin_dir

    strategies: list[tuple[str, object]] = [
        ("winget", _try_winget),
        ("github direct download", lambda: _try_github_download(install_dir)),
        ("chocolatey", _try_choco),
        ("scoop", _try_scoop),
    ]

    for name, fn in strategies:
        print(f"Trying {name} ...")
        try:
            ok = fn()  # type: ignore[operator]
        except Exception as exc:
            print(f"  {name} raised: {exc}")
            ok = False
        if ok and _coreutils_found():
            print(f"Coreutils installed successfully via {name}.")
            if add_to_path:
                _ensure_in_user_path(bin_dir)
            return bin_dir
        print(f"  {name} did not succeed.")

    print("All installation strategies failed.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Install Microsoft Coreutils for Windows silently.",
    )
    parser.add_argument(
        "--dir",
        dest="install_dir",
        default=None,
        help="Custom install directory",
    )
    args = parser.parse_args()

    success = install_coreutils(install_dir=args.install_dir)
    sys.exit(0 if success else 1)
