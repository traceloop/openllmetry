"""
Unit tests for Groq finish reason mapping and collection.

These tests define the expected behavior for:
- _map_groq_finish_reason()       — maps Groq API string to OTel string
- _collect_finish_reasons_from_response() — extracts finish reasons from all choices

All tests here are pure unit tests (no cassettes, no network calls).
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.groq.span_utils import (
    _map_groq_finish_reason,
    _collect_finish_reasons_from_response,
)


# ---------------------------------------------------------------------------
# _map_groq_finish_reason
# ---------------------------------------------------------------------------


class TestMapGroqFinishReason:
    """Groq API returns finish_reason as a plain string (OpenAI-compatible)."""

    def test_tool_calls(self):
        # Groq returns "tool_calls" (plural, OpenAI-compatible) → OTel expects "tool_call" (singular)
        assert _map_groq_finish_reason("tool_calls") == "tool_call"

    def test_none_returns_empty_string(self):
        assert _map_groq_finish_reason(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert _map_groq_finish_reason("") == ""

    def test_unknown_value_is_preserved(self):
        assert _map_groq_finish_reason("something_unexpected") == "something_unexpected"


# ---------------------------------------------------------------------------
# _collect_finish_reasons_from_response
# ---------------------------------------------------------------------------


def _make_response(finish_reasons: list):
    """Build a mock Groq ChatCompletion with given finish_reasons per choice."""
    response = MagicMock()
    choices = []
    for i, fr in enumerate(finish_reasons):
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
