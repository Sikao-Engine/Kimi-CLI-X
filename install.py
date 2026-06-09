#!/usr/bin/env python3
"""Install script for the project using uv."""

import shutil
import subprocess
import sys
from pathlib import Path


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_command(cmd: list[str], description: str) -> bool:
    print(f"\n▶ {description} ...")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n❌ Command failed: {' '.join(cmd)}")
            return False
        print(f"✅ {description} completed.")
        return True
    except Exception as e:
        print(f"\n❌ Error running command: {' '.join(cmd)}")
        print(f"   Details: {e}")
        return False


def main() -> int:
    # 1. Check if python or uv exists
    has_python = command_exists("python") or command_exists("python3")
    has_uv = command_exists("uv")

    if not has_python and not has_uv:
        print(
            "❌ Neither 'python' nor 'uv' was found in your environment.\n"
            "   Please install Python (https://python.org) or uv (https://docs.astral.sh/uv) manually,\n"
            "   then re-run this script."
        )
        return 1

    if not has_uv:
        print(
            "⚠️  'uv' is not installed. Attempting to proceed anyway...\n"
            "   For best results, consider installing uv: https://docs.astral.sh/uv"
        )

    # 2. Delete uv.lock file
    lock_file = Path("uv.lock")
    if lock_file.exists():
        print(f"\n🗑️  Removing {lock_file} ...")
        try:
            lock_file.unlink()
            print(f"✅ Removed {lock_file}.")
        except OSError as e:
            print(f"⚠️  Could not remove {lock_file}: {e}")

    # 3. Run uv sync
    if not run_command(["uv", "sync"], "Syncing dependencies with uv"):
        print(
            "\n💔 Oops! Something went wrong while syncing dependencies.\n"
            "   Please check the error messages above and try again.\n"
            "   If the issue persists, you may need to install dependencies manually."
        )
        return 1

    # 4. Run uv tool install -e .
    if not run_command(["uv", "tool", "install", "-e", "."], "Installing tool in editable mode"):
        print(
            "\n💔 Oops! Something went wrong while installing the tool.\n"
            "   Please check the error messages above and try again.\n"
            "   If the issue persists, you may need to install the tool manually."
        )
        return 1

    coreutils_installed = False
    git_installed = False

    # 5. Install coreutils (Windows only)
    if sys.platform == "win32":
        # Check if coreutils is already installed (via PATH)
        if command_exists("cat.exe"):
            print("✅ Coreutils is already installed, skipping.")
        else:
            coreutils_script = Path(__file__).parent / "scripts" / "install_coreutils.py"
            if coreutils_script.exists():
                try:
                    scripts_dir = str(coreutils_script.parent)
                    if scripts_dir not in sys.path:
                        sys.path.insert(0, scripts_dir)
                    import install_coreutils

                    print("\n▶ Installing coreutils ...")
                    result = install_coreutils.install_coreutils()
                    if result:
                        coreutils_installed = True
                        print(f"✅ Coreutils installed at {result}.")
                    else:
                        print("⚠️  Coreutils installation was not successful (non-fatal).")
                except Exception as e:
                    print(f"⚠️  Could not install coreutils: {e}")

    # 6. Install Git (Windows only)
    if sys.platform == "win32":
        # Check if git is already installed (via PATH)
        if command_exists("git.exe"):
            print("✅ Git is already installed, skipping.")
        else:
            git_script = Path(__file__).parent / "scripts" / "install_git.py"
            if git_script.exists():
                try:
                    scripts_dir = str(git_script.parent)
                    if scripts_dir not in sys.path:
                        sys.path.insert(0, scripts_dir)
                    import install_git

                    print("\n▶ Installing Git ...")
                    result = install_git.install_git()
                    if result:
                        git_installed = True
                        print("✅ Git installed successfully.")
                    else:
                        print("⚠️  Git installation was not successful (non-fatal).")
                except Exception as e:
                    print(f"⚠️  Could not install Git: {e}")

    if coreutils_installed or git_installed:
        print(
            "\n💡 Git or Coreutils was freshly installed, which may have modified your PATH.\n"
            "   Please **restart your current shell/CLI process** before using these tools,\n"
            "   so that the updated PATH environment variable is loaded."
        )

    print("\n🎉 All done! The project has been installed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
