"""unit tests."""

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiSystemValues,
)

VALID_OTEL_FINISH_REASONS = {"stop", "tool_call", "length", "content_filter", "error"}
VALID_OTEL_PART_TYPES = {"text", "tool_call", "tool_call_response", "blob", "uri", "reasoning"}


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    """Shared helper: validate an emitted OTel GenAI event log record."""
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == GenAiSystemValues.AWS_BEDROCK.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


def assert_valid_parts(parts, context=""):
    """Assert every part has a valid OTel type and required fields."""
    assert isinstance(parts, list), f"{context}: parts must be a list, got {type(parts)}"
    for i, part in enumerate(parts):
        assert isinstance(part, dict), f"{context}: part[{i}] must be a dict"
        assert "type" in part, f"{context}: part[{i}] missing 'type' key: {part}"
        ptype = part["type"]
        assert ptype in VALID_OTEL_PART_TYPES, (
            f"{context}: part[{i}] has invalid type '{ptype}'. "
            f"Must be one of {VALID_OTEL_PART_TYPES}. Got: {part}"
        )


def assert_valid_output_message(msg, context=""):
    """Assert an output message has role, valid parts, and valid finish_reason."""
    assert isinstance(msg, dict), f"{context}: message must be a dict"
    assert "role" in msg, f"{context}: message missing 'role'"
    if "parts" in msg:
        assert_valid_parts(msg["parts"], context)
    if "finish_reason" in msg and msg["finish_reason"] is not None:
        assert msg["finish_reason"] in VALID_OTEL_FINISH_REASONS, (
            f"{context}: invalid finish_reason '{msg['finish_reason']}'. "
            f"Must be one of {VALID_OTEL_FINISH_REASONS}"
        )
