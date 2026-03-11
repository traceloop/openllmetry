from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.anthropic import (
    _handle_input,
    _apply_completion_safety,
    _apply_prompt_safety,
)
from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.safety


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_prompt_safety_masks_system_message():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.system]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.system",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {
        "system": "secret",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with tracer.start_as_current_span("anthropic.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs, "anthropic.chat")

    assert kwargs["system"] == "secret"
    assert updated_kwargs["system"] == "[PII.system]"


def test_prompt_safety_masks_span_prompt_attributes():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.system]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.system",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {
        "system": "secret",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with tracer.start_as_current_span("anthropic.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs, "anthropic.chat")
        _handle_input(span, None, updated_kwargs)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == "[PII.system]"
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"


def test_completion_safety_masks_text_blocks():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.token]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.token",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="secret")])
    with tracer.start_as_current_span("anthropic.chat") as span:
        _apply_completion_safety(span, response, "anthropic.chat")

    assert response.content[0].text == "[SECRET.token]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1
