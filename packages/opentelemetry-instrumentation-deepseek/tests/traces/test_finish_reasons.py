"""
Unit tests for DeepSeek finish reason mapping and collection.

These tests define the expected behavior for:
- _map_deepseek_finish_reason()       — maps DeepSeek's (OpenAI-compatible) finish_reason to OTel string
- _collect_finish_reasons_from_response() — extracts finish reasons from all choices

All tests here are pure unit tests (no cassettes, no network calls).
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.deepseek.span_utils import (
    _collect_finish_reasons_from_response,
    _map_deepseek_finish_reason,
)


# ---------------------------------------------------------------------------
# _map_deepseek_finish_reason
# ---------------------------------------------------------------------------


class TestMapDeepSeekFinishReason:
    """DeepSeek's API is OpenAI-compatible and returns finish_reason as a plain string."""

    def test_tool_calls(self):
        # DeepSeek returns "tool_calls" (plural, OpenAI-compatible) → OTel expects "tool_call" (singular)
        assert _map_deepseek_finish_reason("tool_calls") == "tool_call"

    def test_none_returns_empty_string(self):
        assert _map_deepseek_finish_reason(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert _map_deepseek_finish_reason("") == ""

    def test_unknown_value_is_preserved(self):
        assert _map_deepseek_finish_reason("something_unexpected") == "something_unexpected"


# ---------------------------------------------------------------------------
# _collect_finish_reasons_from_response
# ---------------------------------------------------------------------------


def _make_response(finish_reasons: list):
    """Build a mock OpenAI-compatible ChatCompletion with given finish_reasons per choice."""
    response = MagicMock()
    choices = []
    for fr in finish_reasons:
        choice = MagicMock()
        choice.finish_reason = fr
        choices.append(choice)
    response.choices = choices
    return response


class TestCollectFinishReasonsFromResponse:
    def test_single_stop(self):
        response = _make_response(["stop"])
        assert _collect_finish_reasons_from_response(response) == ["stop"]

    def test_single_length(self):
        response = _make_response(["length"])
        assert _collect_finish_reasons_from_response(response) == ["length"]

    def test_multiple_choices(self):
        response = _make_response(["stop", "length"])
        assert _collect_finish_reasons_from_response(response) == ["stop", "length"]

    def test_none_finish_reason_maps_to_empty(self):
        response = _make_response([None])
        assert _collect_finish_reasons_from_response(response) == [""]

    def test_mixed_known_and_none(self):
        response = _make_response(["stop", None])
        assert _collect_finish_reasons_from_response(response) == ["stop", ""]

    def test_empty_choices_returns_empty_list(self):
        response = _make_response([])
        assert _collect_finish_reasons_from_response(response) == []

    def test_none_response_returns_empty_list(self):
        assert _collect_finish_reasons_from_response(None) == []
