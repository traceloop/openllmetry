"""
Unit tests for guardrail failure handlers.

Tests all built-in failure handlers from the OnFailure class.
"""
import logging
import pytest
from traceloop.sdk.guardrail.on_failure import OnFailure
from traceloop.sdk.guardrail.model import GuardedResult, GuardValidationError


class CustomException(Exception):
    """Custom exception for testing custom exception types."""
    pass


class TestOnFailureRaiseException:
    """Tests for OnFailure.raise_exception()."""

    def test_raise_exception_with_default_message(self):
        """Test raise_exception with default message."""
        handler = OnFailure.raise_exception()
        output = GuardedResult(
            result="test result",
            guard_inputs=[{"text": "test input"}]
        )

        with pytest.raises(GuardValidationError) as exc_info:
            handler(output)

        assert "Guard validation failed" in str(exc_info.value)
        assert exc_info.value.output == output

    def test_raise_exception_with_custom_message(self):
        """Test raise_exception with custom message."""
        handler = OnFailure.raise_exception("PII detected in response")
        output = GuardedResult(
            result="email@example.com",
            guard_inputs=[{"text": "email@example.com"}]
        )

        with pytest.raises(GuardValidationError) as exc_info:
            handler(output)

        assert "PII detected in response" in str(exc_info.value)

    def test_raise_exception_with_custom_exception_type(self):
        """Test raise_exception with custom exception type."""
        handler = OnFailure.raise_exception(
            message="Custom error",
            exception_type=CustomException
        )
        output = GuardedResult(
            result="test",
            guard_inputs=[{"text": "test"}]
        )

        with pytest.raises(CustomException) as exc_info:
            handler(output)

        assert "Custom error" in str(exc_info.value)

    def test_raise_exception_includes_guard_inputs(self):
        """Test that exception includes guard_inputs in string representation."""
        handler = OnFailure.raise_exception("Validation failed")
        output = GuardedResult(
            result="result",
            guard_inputs=[{"score": 0.3, "threshold": 0.5}]
        )

        with pytest.raises(GuardValidationError) as exc_info:
            handler(output)

        error_str = str(exc_info.value)
        assert "guard_inputs" in error_str
        assert "0.3" in error_str or "score" in error_str

    def test_raise_exception_handler_callable(self):
        """Test that raise_exception returns a callable handler."""
        handler = OnFailure.raise_exception()
        assert callable(handler)


class TestOnFailureLog:
    """Tests for OnFailure.log()."""

    def test_log_with_default_level(self, caplog):
        """Test log with default WARNING level."""
        handler = OnFailure.log()
        output = GuardedResult(
            result="test result",
            guard_inputs=[{"text": "test input"}]
        )

        with caplog.at_level(logging.WARNING):
            result = handler(output)

        # Check that it logs
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "Guard validation failed" in caplog.text
        assert "guard_inputs" in caplog.text

        # Check that it returns the original result
        assert result == "test result"

    def test_log_with_custom_level_info(self, caplog):
        """Test log with custom INFO level."""
        handler = OnFailure.log(level=logging.INFO)
        output = GuardedResult(
            result="test result",
            guard_inputs=[{"text": "test input"}]
        )

        with caplog.at_level(logging.INFO):
            handler(output)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO

    def test_log_with_custom_level_error(self, caplog):
        """Test log with custom ERROR level."""
        handler = OnFailure.log(level=logging.ERROR)
        output = GuardedResult(
            result="test result",
            guard_inputs=[{"text": "test input"}]
        )

        with caplog.at_level(logging.ERROR):
            handler(output)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_log_with_custom_message(self, caplog):
        """Test log with custom message."""
        handler = OnFailure.log(message="Toxic content detected")
        output = GuardedResult(
            result="bad content",
            guard_inputs=[{"text": "bad content", "score": 0.9}]
        )

        with caplog.at_level(logging.WARNING):
            handler(output)

        assert "Toxic content detected" in caplog.text

    def test_log_returns_original_result(self, caplog):
        """Test that log returns the original result unchanged."""
        handler = OnFailure.log()
        output = GuardedResult(
            result={"data": "complex result", "nested": {"value": 42}},
            guard_inputs=[{"input": "test"}]
        )

        with caplog.at_level(logging.WARNING):
            result = handler(output)

        assert result == {"data": "complex result", "nested": {"value": 42}}

    def test_log_includes_guard_inputs_in_message(self, caplog):
        """Test that log message includes guard_inputs count."""
        handler = OnFailure.log(message="Security check failed")
        output = GuardedResult(
            result="result",
            guard_inputs=[{"ip": "192.168.1.1", "user": "admin"}]
        )

        with caplog.at_level(logging.WARNING):
            handler(output)

        assert "Security check failed" in caplog.text
        assert "guard_inputs_count" in caplog.text
        # Check that guard_inputs count is logged (not content for privacy)
        log_message = caplog.records[0].getMessage()
        assert "guard_inputs_count=1" in log_message


class TestOnFailureNoop:
    """Tests for OnFailure.noop()."""

    def test_noop_returns_original_result(self):
        """Test that noop returns the original result."""
        handler = OnFailure.noop()
        output = GuardedResult(
            result="original result",
            guard_inputs=[{"text": "input"}]
        )

        result = handler(output)
        assert result == "original result"

    def test_noop_returns_complex_result(self):
        """Test that noop returns complex results unchanged."""
        handler = OnFailure.noop()
        complex_result = {
            "data": [1, 2, 3],
            "nested": {"key": "value"},
            "list": ["a", "b", "c"]
        }
        output = GuardedResult(
            result=complex_result,
            guard_inputs=[{"input": "test"}]
        )

        result = handler(output)
        assert result == complex_result
        assert result is complex_result  # Should be same object

    def test_noop_no_side_effects(self, caplog):
        """Test that noop has no side effects (no logging, no exceptions)."""
        handler = OnFailure.noop()
        output = GuardedResult(
            result="result",
            guard_inputs=[{"text": "input"}]
        )

        with caplog.at_level(logging.DEBUG):
            result = handler(output)

        # No logs should be created
        assert len(caplog.records) == 0
        # Result should be returned
        assert result == "result"

    def test_noop_handler_callable(self):
        """Test that noop returns a callable handler."""
        handler = OnFailure.noop()
        assert callable(handler)


class TestOnFailureReturnValue:
    """Tests for OnFailure.return_value()."""

    def test_return_value_with_string(self):
        """Test return_value with string fallback."""
        handler = OnFailure.return_value("Sorry, I can't help with that.")
        output = GuardedResult(
            result="original result",
            guard_inputs=[{"text": "blocked content"}]
        )

        result = handler(output)
        assert result == "Sorry, I can't help with that."

    def test_return_value_with_dict(self):
        """Test return_value with dict fallback."""
        fallback = {"error": "blocked", "reason": "PII detected"}
        handler = OnFailure.return_value(fallback)
        output = GuardedResult(
            result="original result",
            guard_inputs=[{"text": "email@example.com"}]
        )

        result = handler(output)
        assert result == fallback

    def test_return_value_with_none(self):
        """Test return_value with None fallback."""
        handler = OnFailure.return_value(None)
        output = GuardedResult(
            result="original result",
            guard_inputs=[{"text": "input"}]
        )

        result = handler(output)
        assert result is None

    def test_return_value_with_complex_object(self):
        """Test return_value with complex object fallback."""
        fallback = {
            "status": "error",
            "code": 403,
            "message": "Content blocked by guardrail",
            "details": {
                "rule": "pii_detector",
                "severity": "high"
            }
        }
        handler = OnFailure.return_value(fallback)
        output = GuardedResult(
            result="original result",
            guard_inputs=[{"text": "sensitive data"}]
        )

        result = handler(output)
        assert result == fallback
        assert result["status"] == "error"
        assert result["details"]["severity"] == "high"

    def test_return_value_ignores_original_result(self):
        """Test that return_value ignores the original result completely."""
        fallback = "fallback value"
        handler = OnFailure.return_value(fallback)

        # Test with various original results
        output1 = GuardedResult(result="ignored", guard_inputs=[{}])
        output2 = GuardedResult(result={"complex": "object"}, guard_inputs=[{}])
        output3 = GuardedResult(result=None, guard_inputs=[{}])

        assert handler(output1) == fallback
        assert handler(output2) == fallback
        assert handler(output3) == fallback

    def test_return_value_with_list(self):
        """Test return_value with list fallback."""
        fallback = ["error", "blocked", "retry"]
        handler = OnFailure.return_value(fallback)
        output = GuardedResult(
            result="original",
            guard_inputs=[{"text": "input"}]
        )

        result = handler(output)
        assert result == fallback
        assert len(result) == 3

    def test_return_value_with_number(self):
        """Test return_value with numeric fallback."""
        handler = OnFailure.return_value(0)
        output = GuardedResult(
            result=100,
            guard_inputs=[{"score": 0.3}]
        )

        result = handler(output)
        assert result == 0
        assert isinstance(result, int)
