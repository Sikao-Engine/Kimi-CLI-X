import subprocess
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

def get_uncommitted_diff(filepaths: list[str]) -> str:
    """Return the uncommitted diff for specific files."""
    result = subprocess.run(
        ["git", "diff", "--"] + filepaths,
        capture_output=True,
        text=True,
        check=False,
        errors='replace'
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return result.stdout


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run tools/git_diff.py <filepath> [<filepath> ...]")
        sys.exit(1)

    targets = sys.argv[1:]
    try:
        diff = get_uncommitted_diff(targets)
        print(diff, end="")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
