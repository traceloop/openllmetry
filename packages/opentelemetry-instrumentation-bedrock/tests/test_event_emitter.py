"""Unit tests for event_emitter.py finish_reason mapping.

These validate that event emitter functions:
1. Map provider finish reasons to OTel enum values (#17)
2. Do NOT use "unknown" as a fallback (#16)
3. Omit finish_reason when None (#16)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.bedrock.event_emitter import (
    emit_choice_events,
    emit_response_event_converse,
    emit_streaming_converse_response_event,
    emit_streaming_response_event,
)
from opentelemetry.instrumentation.bedrock.event_models import ChoiceEvent


class TestChoiceEventModel:
    """#16: ChoiceEvent.finish_reason should default to None, not 'unknown'."""

    def test_default_finish_reason_is_none(self):
        event = ChoiceEvent(index=0, message={"content": "hi", "role": "assistant"})
        assert event.finish_reason is None

    def test_explicit_finish_reason(self):
        event = ChoiceEvent(
            index=0,
            message={"content": "hi", "role": "assistant"},
            finish_reason="stop",
        )
        assert event.finish_reason == "stop"


class TestEmitChoiceEvents:
    """#17: emit_choice_events must map provider finish reasons to OTel enum."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_anthropic_completion_end_turn_mapped_to_stop(self, mock_prompts, mock_events):
        """Anthropic 'end_turn' must be mapped to 'stop' in events."""
        response_body = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
        }
        response = {"body": MagicMock()}
        response["body"].read.return_value = json.dumps(response_body).encode()

        logger = MagicMock()
        emit_choice_events(logger, response)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_anthropic_completion_stop_sequence_mapped_to_stop(self, mock_prompts, mock_events):
        """Anthropic 'stop_sequence' must be mapped to 'stop' in events."""
        response_body = {
            "completion": "Hello!",
            "stop_reason": "stop_sequence",
        }
        response = {"body": MagicMock()}
        response["body"].read.return_value = json.dumps(response_body).encode()

        logger = MagicMock()
        emit_choice_events(logger, response)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_titan_completionreason_finish_mapped_to_stop(self, mock_prompts, mock_events):
        """Titan 'FINISH' must be mapped to 'stop' in events."""
        response_body = {
            "results": [
                {"outputText": "Hello!", "completionReason": "FINISH"}
            ]
        }
        response = {"body": MagicMock()}
        response["body"].read.return_value = json.dumps(response_body).encode()

        logger = MagicMock()
        emit_choice_events(logger, response)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_converse_guardrail_intervened_mapped_to_content_filter(self, mock_prompts, mock_events):
        """Converse 'guardrail_intervened' must be mapped to 'content_filter' in events."""
        response_body = {
            "output": {"message": {"content": [{"text": "Blocked"}], "role": "assistant"}},
            "stopReason": "guardrail_intervened",
        }
        response = {"body": MagicMock()}
        response["body"].read.return_value = json.dumps(response_body).encode()

        logger = MagicMock()
        emit_choice_events(logger, response)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "content_filter"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_missing_finish_reason_not_unknown(self, mock_prompts, mock_events):
        """#16: Missing finish_reason should NOT be 'unknown'."""
        response_body = {
            "completion": "Hello!",
            # no stop_reason key
        }
        response = {"body": MagicMock()}
        response["body"].read.return_value = json.dumps(response_body).encode()

        logger = MagicMock()
        emit_choice_events(logger, response)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        # Should NOT be "unknown"
        assert log_record.body.get("finish_reason") != "unknown"


class TestEmitResponseEventConverse:
    """#17: emit_response_event_converse must map finish reasons."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_end_turn_mapped_to_stop(self, mock_prompts, mock_events):
        response = {
            "output": {"message": {"content": [{"text": "Hi"}], "role": "assistant"}},
            "stopReason": "end_turn",
        }
        logger = MagicMock()
        emit_response_event_converse(response, logger)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_guardrail_intervened_mapped(self, mock_prompts, mock_events):
        response = {
            "output": {"message": {"content": [{"text": "Blocked"}], "role": "assistant"}},
            "stopReason": "guardrail_intervened",
        }
        logger = MagicMock()
        emit_response_event_converse(response, logger)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "content_filter"


class TestEmitStreamingResponseEvent:
    """#17: emit_streaming_response_event must map finish reasons."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_end_turn_mapped_to_stop(self, mock_prompts, mock_events):
        response_body = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
        }
        logger = MagicMock()
        emit_streaming_response_event(response_body, logger)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_missing_stop_reason_not_unknown(self, mock_prompts, mock_events):
        """#16: Missing stop_reason should not produce 'unknown'."""
        response_body = {"outputText": "Hello"}
        logger = MagicMock()
        emit_streaming_response_event(response_body, logger)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body.get("finish_reason") != "unknown"


class TestEmitStreamingConverseResponseEvent:
    """#17: emit_streaming_converse_response_event must map finish reasons."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_end_turn_mapped_to_stop(self, mock_prompts, mock_events):
        logger = MagicMock()
        emit_streaming_converse_response_event(logger, ["Hello"], "assistant", "end_turn")

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body["finish_reason"] == "stop"

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_none_finish_reason_not_unknown(self, mock_prompts, mock_events):
        """#16: None finish_reason should not produce 'unknown'."""
        logger = MagicMock()
        emit_streaming_converse_response_event(logger, ["Hello"], "assistant", None)

        logger.emit.assert_called_once()
        log_record = logger.emit.call_args[0][0]
        assert log_record.body.get("finish_reason") != "unknown"
