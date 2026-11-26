import logging
from unittest.mock import patch

from traceloop.sdk.fetcher import Fetcher


def test_api_post_logs_error_instead_of_printing(caplog, capsys):
    fetcher = Fetcher(base_url="http://example.invalid", api_key="test-key")

    with patch(
        "traceloop.sdk.fetcher.post_url",
        side_effect=RuntimeError("boom"),
    ):
        with caplog.at_level(logging.ERROR):
            fetcher.api_post("some-endpoint", {"payload": "x"})

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "expected an ERROR log record from Fetcher.api_post"

    message = error_records[-1].getMessage()
    assert "some-endpoint" in message
    assert "boom" in message

    captured = capsys.readouterr()
    assert "boom" not in captured.out
