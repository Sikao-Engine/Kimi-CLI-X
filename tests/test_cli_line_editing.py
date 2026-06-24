from __future__ import annotations

import builtins

from kimix.cli_impl import utils


def test_input_prints_prompt_before_reading_empty_prompt(monkeypatch, capsys):
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return "typed"

    monkeypatch.setattr(builtins, "input", fake_input)

    assert utils._input("\n>>>>>>>>> Enter your prompt or command:\n", []) == "typed"

    assert prompts == [""]
    assert capsys.readouterr().out == "\n>>>>>>>>> Enter your prompt or command:\n"
