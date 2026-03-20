from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import safety
from opentelemetry.instrumentation.fortifyroot import (
    SAFETY_EVENT_NAME,
    SafetyDecision,
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    get_completion_safety_handler,
    get_object_value,
    get_prompt_safety_handler,
    register_completion_safety_handler,
    run_completion_safety,
    set_object_value,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.fr


def teardown_function():
    clear_safety_handlers()


def _span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_get_and_set_object_value_support_mapping_and_object():
    mapping = {"value": 1}
    obj = SimpleNamespace(value=2)

    assert get_object_value(mapping, "value") == 1
    assert get_object_value(obj, "value") == 2
    assert get_object_value(obj, "missing", "fallback") == "fallback"

    assert set_object_value(mapping, "value", 3) is True
    assert set_object_value(obj, "value", 4) is True
    assert mapping["value"] == 3
    assert obj.value == 4


def test_set_object_value_returns_false_when_attribute_is_not_writable():
    class ReadOnly:
        @property
        def value(self):
            return "x"

    assert set_object_value(ReadOnly(), "value", "y") is False


def test_run_completion_safety_normalizes_and_emits_event_without_overall_action():
    exporter, tracer = _span()
    register_completion_safety_handler(
        lambda _: SafetyResult(
            text=None,
            overall_action="mask",
            findings=[
                SafetyFinding(
                    category="pii",
                    severity="low",
                    action="mask",
                    rule_name="PII.email",
                    start="1",
                    end="4",
                )
            ],
        )
    )

    with tracer.start_as_current_span("test-span") as span:
        result = run_completion_safety(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            text="raw",
            location=SafetyLocation.COMPLETION,
            request_type="chat",
            segment_index=2,
            segment_role="assistant",
            metadata={"ignored": True},
        )

    assert result is not None
    assert result.text == "raw"
    assert result.overall_action == SafetyDecision.MASK.value
    assert result.findings[0].category == "PII"
    assert result.findings[0].severity == "LOW"
    assert result.findings[0].start == 1
    assert result.findings[0].end == 4

    span = exporter.get_finished_spans()[0]
    assert span.events[0].name == SAFETY_EVENT_NAME
    assert "fortifyroot.safety.overall_action" not in span.events[0].attributes
    assert "fortifyroot.safety.provider" not in span.events[0].attributes
    assert "fortifyroot.safety.request_type" not in span.events[0].attributes


def test_run_completion_safety_returns_none_for_empty_text_or_missing_handler():
    assert get_completion_safety_handler() is None
    assert run_completion_safety(
        span=None,
        provider="OpenAI",
        span_name="openai.chat",
        text="",
        location=SafetyLocation.COMPLETION,
    ) is None


def test_run_completion_safety_returns_none_when_handler_returns_none():
    register_completion_safety_handler(lambda _: None)
    assert run_completion_safety(
        span=None,
        provider="OpenAI",
        span_name="openai.chat",
        text="hello",
        location=SafetyLocation.COMPLETION,
    ) is None


def test_handler_getters_follow_registration():
    register_completion_safety_handler(lambda _: None)
    assert get_completion_safety_handler() is not None
    assert get_prompt_safety_handler() is None
    clear_safety_handlers()
    assert get_completion_safety_handler() is None


def test_emit_findings_noops_for_non_recording_span_and_none_span():
    context = safety.SafetyContext(
        provider="OpenAI",
        text="raw",
        location=SafetyLocation.PROMPT,
        span_name="openai.chat",
    )
    result = SafetyResult(
        text="raw",
        findings=[
            SafetyFinding(
                category="PII",
                severity="HIGH",
                action="MASK",
                rule_name="PII.email",
                start=0,
                end=3,
            )
        ],
        overall_action="MASK",
    )

    safety._emit_findings(None, context, result)

    class NotRecordingSpan:
        def is_recording(self):
            return False

    safety._emit_findings(NotRecordingSpan(), context, result)


def test_emit_findings_omits_optional_attributes_when_not_present():
    exporter, tracer = _span()
    context = safety.SafetyContext(
        provider="OpenAI",
        text="raw",
        location=SafetyLocation.PROMPT,
        span_name="openai.chat",
    )
    result = SafetyResult(
        text="raw",
        findings=[
            SafetyFinding(
                category="PII",
                severity="HIGH",
                action="ALLOW",
                rule_name="PII.email",
                start=0,
                end=3,
            )
        ],
        overall_action="ALLOW",
    )

    with tracer.start_as_current_span("test-span") as span:
        safety._emit_findings(span, context, result)

    attributes = exporter.get_finished_spans()[0].events[0].attributes
    assert "fortifyroot.safety.request_type" not in attributes
    assert "fortifyroot.safety.segment_index" not in attributes
    assert "fortifyroot.safety.segment_role" not in attributes
    assert "fortifyroot.safety.provider" not in attributes


def test_normalize_decision_defaults_unknown_to_allow():
    assert safety._normalize_decision("mask") == SafetyDecision.MASK.value
    assert safety._normalize_decision("something-else") == SafetyDecision.ALLOW.value
