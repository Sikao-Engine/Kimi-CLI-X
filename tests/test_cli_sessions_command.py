from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from kimix.cli_impl import commands, constants


def test_sessions_command_registered():
    assert "sessions" in commands._command_map
    assert "sessions" in commands._command_map_keys


def test_help_includes_sessions_command():
    assert "/sessions" in constants.HELP_STR
    assert "List resumable sessions" in constants.HELP_STR


def test_sessions_command_prints_empty_state(monkeypatch, capsys):
    current = SimpleNamespace(
        _cli=SimpleNamespace(session=SimpleNamespace(work_dir="/work", id="current"))
    )
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr("kimi_cli.session.Session.list", AsyncMock(return_value=[]))

    commands._cmd_sessions(["sessions"], [])

    assert "No sessions found." in capsys.readouterr().out


def test_sessions_command_prints_sessions_in_returned_order(monkeypatch, capsys):
    current = SimpleNamespace(
        _cli=SimpleNamespace(session=SimpleNamespace(work_dir="/work", id="s2"))
    )
    sessions = [
        SimpleNamespace(id="s2", updated_at=1_700_000_100.0, title="current title"),
        SimpleNamespace(id="s1", updated_at=1_700_000_000.0, title="older title"),
    ]
    monkeypatch.setattr(commands, "get_default_session", lambda: current)
    monkeypatch.setattr("kimi_cli.session.Session.list", AsyncMock(return_value=sessions))

    commands._cmd_sessions(["sessions"], [])

    output = capsys.readouterr().out
    assert "session id" in output
    assert "*  s2" in output
    assert "   s1" in output
    assert output.index("s2") < output.index("s1")
