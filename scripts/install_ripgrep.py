"""Install ripgrep (rg) binary to the KIMI share directory.

Strategy:
1. Check if rg is already on PATH or in the share directory.
2. Download the appropriate release archive from GitHub.
3. Fallback to the Kimi CDN backup URL.
4. Extract the binary to ``<share_dir>/bin``.

Usage:
    python install_ripgrep.py
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ============================================================
# Global configuration (must match kimi-cli/tools/file/grep_local.py)
# ============================================================
_RG_VERSION = "15.1.0"
_RG_BASE_URL = f"https://github.com/BurntSushi/ripgrep/releases/download/{_RG_VERSION}"
_BACKUP_RG_VERSION = "15.0.0"
_BACKUP_RG_BASE_URL = "http://cdn.kimi.com/binaries/kimi-cli/rg"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_share_dir() -> Path:
    """Get the share directory path."""
    if share_dir := os.getenv("KIMI_SHARE_DIR"):
        share_dir = Path(share_dir)
    else:
        share_dir = Path.home() / ".kimi"
    share_dir.mkdir(parents=True, exist_ok=True)
    return share_dir


def _rg_binary_name() -> str:
    return "rg.exe" if platform.system() == "Windows" else "rg"


def _rg_found() -> bool:
    """Return ``True`` if the ripgrep binary is available."""
    bin_name = _rg_binary_name()
    share_bin = _get_share_dir() / "bin" / bin_name
    if share_bin.is_file():
        return True
    return shutil.which("rg") is not None


def _detect_target() -> str | None:
    sys_name = platform.system()
    mach = platform.machine().lower()

    if mach in ("x86_64", "amd64"):
        arch = "x86_64"
    elif mach in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        print(f"Unsupported architecture for ripgrep: {mach}", file=sys.stderr)
        return None

    if sys_name == "Darwin":
        os_name = "apple-darwin"
    elif sys_name == "Linux":
        os_name = "unknown-linux-musl" if arch == "x86_64" else "unknown-linux-gnu"
    elif sys_name == "Windows":
        os_name = "pc-windows-msvc"
    else:
        print(f"Unsupported operating system for ripgrep: {sys_name}", file=sys.stderr)
        return None

    return f"{arch}-{os_name}"


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, with a progress indicator."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "kimix-installer")

    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(str(dest), "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(100, int(downloaded * 100 / total_size))
                    sys.stdout.write(f"\r  {pct}%")
                    sys.stdout.flush()
    print()  # newline after progress


def _ensure_in_user_path(dirpath: str) -> None:
    """Add *dirpath* to the current user's PATH environment variable (persistent)."""
    if sys.platform != "win32":
        return

    import winreg

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
    except FileNotFoundError:
        return

    try:
        path_val, _ = winreg.QueryValueEx(key, "Path")
    except FileNotFoundError:
        path_val = ""

    entries = [p.strip() for p in path_val.split(";") if p.strip()]
    if dirpath in entries:
        winreg.CloseKey(key)
        return

    entries.append(dirpath)
    new_path = ";".join(entries)
    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
    winreg.CloseKey(key)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def install_ripgrep() -> Path | None:
    """Download and install ripgrep to the KIMI share directory.

    Returns
    -------
    The path to the installed binary on success, or ``None`` on failure.
    """
    if _rg_found():
        print("ripgrep is already installed, skipping.")
        existing = shutil.which("rg")
        return Path(existing) if existing else None

    target = _detect_target()
    if not target:
        print("Could not detect platform for ripgrep.", file=sys.stderr)
        return None

    is_windows = "windows" in target
    archive_ext = "zip" if is_windows else "tar.gz"
    bin_name = _rg_binary_name()

    primary_filename = f"ripgrep-{_RG_VERSION}-{target}.{archive_ext}"
    primary_url = f"{_RG_BASE_URL}/{primary_filename}"

    backup_filename = f"ripgrep-{_BACKUP_RG_VERSION}-{target}.{archive_ext}"
    backup_url = f"{_BACKUP_RG_BASE_URL}/{backup_filename}"

    share_bin_dir = _get_share_dir() / "bin"
    share_bin_dir.mkdir(parents=True, exist_ok=True)
    destination = share_bin_dir / bin_name

    with tempfile.TemporaryDirectory(prefix="kimi-rg-") as tmpdir:
        tar_path = Path(tmpdir) / primary_filename

        # Try primary URL first
        url = primary_url
        print(f"Downloading ripgrep from {url} ...")
        try:
            _download_file(url, tar_path)
        except Exception as exc:
            print(f"  Primary download failed: {exc}")
            # Try backup URL
            url = backup_url
            tar_path = Path(tmpdir) / backup_filename
            print(f"Downloading ripgrep from {url} ...")
            try:
                _download_file(url, tar_path)
            except Exception as exc2:
                print(f"  Backup download failed: {exc2}", file=sys.stderr)
                return None

        # Extract
        try:
            if is_windows:
                with zipfile.ZipFile(tar_path, "r") as zf:
                    member_name = next(
                        (name for name in zf.namelist() if Path(name).name == bin_name),
                        None,
                    )
                    if not member_name:
                        print("Ripgrep binary not found in archive.", file=sys.stderr)
                        return None
                    with zf.open(member_name) as source, open(destination, "wb") as dest_fh:
                        shutil.copyfileobj(source, dest_fh)
            else:
                with tarfile.open(tar_path, "r:gz") as tar:
                    member = next(
                        (m for m in tar.getmembers() if Path(m.name).name == bin_name),
                        None,
                    )
                    if not member:
                        print("Ripgrep binary not found in archive.", file=sys.stderr)
                        return None
                    extracted = tar.extractfile(member)
                    if not extracted:
                        print("Failed to extract ripgrep binary.", file=sys.stderr)
                        return None
                    with open(destination, "wb") as dest_fh:
                        shutil.copyfileobj(extracted, dest_fh)
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
            print(f"Failed to extract ripgrep archive: {exc}", file=sys.stderr)
            return None

    # Make executable
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Ensure share/bin is on PATH (Windows registry)
    _ensure_in_user_path(str(share_bin_dir))
    # Also update current process PATH for immediate use
    current_path = os.environ.get("PATH", "")
    if str(share_bin_dir) not in current_path:
        os.environ["PATH"] = str(share_bin_dir) + os.pathsep + current_path

    print(f"ripgrep installed to {destination}")
    return destination


if __name__ == "__main__":
    result = install_ripgrep()
    sys.exit(0 if result else 1)
