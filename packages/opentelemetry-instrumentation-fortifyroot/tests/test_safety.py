import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SAFETY_EVENT_NAME,
    SafetyDecision,
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    clone_value,
    register_prompt_safety_handler,
    run_prompt_safety,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.safety


def test_run_prompt_safety_emits_one_event_per_finding():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    register_prompt_safety_handler(
        lambda _: SafetyResult(
            text="masked value",
            overall_action="mask",
            findings=[
                SafetyFinding(
                    category="pii",
                    severity="high",
                    action="mask",
                    rule_name="PII.email",
                    start=2,
                    end=7,
                ),
                SafetyFinding(
                    category="secret",
                    severity="medium",
                    action="allow",
                    rule_name="SECRET.api_key",
                    start=9,
                    end=13,
                ),
            ],
        )
    )

    with tracer.start_as_current_span("test-span") as span:
        result = run_prompt_safety(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            text="raw value",
            location=SafetyLocation.PROMPT,
            request_type="chat",
            segment_index=1,
            segment_role="user",
        )

    spans = exporter.get_finished_spans()
    assert result is not None
    assert result.text == "masked value"
    assert result.overall_action == SafetyDecision.MASK.value
    assert len(spans) == 1
    assert len(spans[0].events) == 2
    assert spans[0].events[0].name == SAFETY_EVENT_NAME
    assert spans[0].events[0].attributes["fortifyroot.safety.category"] == "PII"
    assert spans[0].events[0].attributes["fortifyroot.safety.segment_index"] == 1
    assert spans[0].events[0].attributes["fortifyroot.safety.segment_role"] == "user"
    assert "fortifyroot.safety.request_type" not in spans[0].events[0].attributes
    assert "fortifyroot.safety.provider" not in spans[0].events[0].attributes
    assert spans[0].events[1].attributes["fortifyroot.safety.action"] == SafetyDecision.ALLOW.value

    clear_safety_handlers()


def test_run_prompt_safety_returns_none_without_handler():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    clear_safety_handlers()

    with tracer.start_as_current_span("test-span") as span:
        result = run_prompt_safety(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            text="raw value",
            location=SafetyLocation.PROMPT,
        )

    assert result is None
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].events == ()


def test_clone_value_falls_back_cleanly():
    class NotCloneable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no clone")

    value = NotCloneable()
    assert clone_value(value) is value


def test_run_prompt_safety_fail_opens_when_handler_raises():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    register_prompt_safety_handler(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    with tracer.start_as_current_span("test-span") as span:
        result = run_prompt_safety(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            text="raw value",
            location=SafetyLocation.PROMPT,
        )

    assert result is None
