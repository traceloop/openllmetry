from __future__ import annotations

import main


def test_missing_token_returns_actionable_error(monkeypatch, capsys):
    monkeypatch.delenv("AA_TOKEN", raising=False)

    assert main.main() == 2
    captured = capsys.readouterr()
    assert "Set AA_TOKEN before running this example." in captured.err
