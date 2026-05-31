"""File modification time tracker.

Detects external file modifications between read and write operations.
"""

from pathlib import Path

from kimi_cli.utils.path import kaos_path_from_user_input


class FileMTime:
    """Track file modification times to detect external changes."""

    def __init__(self) -> None:
        self._times: dict[str, float] = {}

    def _resolve(self, path: str) -> str:
        """Resolve *path* to a canonical absolute path for use as a dict key.

        Both relative and absolute paths resolve to the same key.
        """
        try:
            p = kaos_path_from_user_input(path)
            return str(p.canonical())
        except Exception:
            return str(Path(path).resolve())

    def mark_dirty(self, path: str) -> bool:
        """Check whether *path* has been modified since last recorded.

        Returns ``True`` if the file is safe to write (no external
        modification detected), ``False`` if the file was changed
        externally and should be re-read first.
        """
        key = self._resolve(path)
        try:
            current_mtime = Path(key).stat().st_mtime
        except (FileNotFoundError, OSError):
            # File doesn't exist yet — safe to create.
            self._times[key] = 0.0
            return True

        if key in self._times:
            old_mtime = self._times[key]
            if current_mtime > old_mtime:
                # File modified externally — update record and signal dirty.
                self._times[key] = current_mtime
                return True
            # Same or older mtime — no external change, but also not dirty.
            return False
        else:
            # First time tracking this file — record and allow.
            self._times[key] = current_mtime
            return True

    def clean_file(self, path: str) -> None:
        """Remove tracking entry for *path* (called after a successful read)."""
        key = self._resolve(path)
        self._times.pop(key, None)

