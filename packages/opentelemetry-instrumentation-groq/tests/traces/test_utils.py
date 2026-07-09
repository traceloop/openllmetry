"""
Unit tests for utils helpers.

Covers:
  dont_throw exception handler, error_metrics_attributes, model_as_dict edge cases.
All tests are pure unit tests — no network calls, no cassettes.
"""

from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    error_metrics_attributes,
    model_as_dict,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes


# ---------------------------------------------------------------------------
# dont_throw
# ---------------------------------------------------------------------------


class TestDontThrow:
    def test_exception_logger_called_when_set(self):
        callback = MagicMock()
        original = Config.exception_logger
        Config.exception_logger = callback

        @dont_throw
        def failing():
            raise ValueError("boom")

        try:
            failing()  # must not raise
        finally:
            Config.exception_logger = original

        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], ValueError)

    def test_exception_not_propagated_without_logger(self):
        @dont_throw
        def failing():
            raise RuntimeError("silent")

        # Should return None silently
        result = failing()
        assert result is None


# ---------------------------------------------------------------------------
# error_metrics_attributes
# ---------------------------------------------------------------------------


class TestErrorMetricsAttributes:
    def test_returns_provider_name_and_error_type(self):
        result = error_metrics_attributes(ValueError("oops"))
        assert result[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "groq"
        assert result["error.type"] == "ValueError"


# ---------------------------------------------------------------------------
# model_as_dict
# ---------------------------------------------------------------------------


class TestModelAsDict:
    def test_pydantic_v2_model_uses_model_dump(self):
        model = MagicMock()
        model.model_dump.return_value = {"key": "value"}
        # hasattr returns True for MagicMock by default
        result = model_as_dict(model)
        assert result == {"key": "value"}
        model.model_dump.assert_called_once()

    def test_pydantic_v1_model_uses_dict(self):
        model = MagicMock(spec=["dict"])  # only has .dict(), no .model_dump
        model.dict.return_value = {"key": "v1"}
        with patch("opentelemetry.instrumentation.groq.utils._PYDANTIC_VERSION", "1.9.0"):
            result = model_as_dict(model)
        assert result == {"key": "v1"}

    def test_raw_api_response_with_parse_method(self):
        # Simulate a raw API response that has .parse() but no .model_dump
        inner = {"parsed": True}

        class RawResponse:
            def parse(self):
                return inner

        result = model_as_dict(RawResponse())
        assert result == {"parsed": True}

    def test_plain_dict_returned_as_is(self):
        data = {"a": 1, "b": 2}
        assert model_as_dict(data) is data
