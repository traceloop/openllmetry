from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.cohere import (
    _handle_input_content,
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
from opentelemetry.semconv_ai import LLMRequestTypeValues
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


def test_prompt_safety_masks_chat_preamble():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.preamble]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="LOW",
                    action="MASK",
                    rule_name="PII.preamble",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"preamble": "secret", "message": "hello"}
    with tracer.start_as_current_span("cohere.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span,
            kwargs,
            LLMRequestTypeValues.CHAT,
            "cohere.chat",
        )

    assert kwargs["preamble"] == "secret"
    assert updated_kwargs["preamble"] == "[PII.preamble]"


def test_prompt_safety_masks_span_prompt_attributes():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.preamble]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="LOW",
                    action="MASK",
                    rule_name="PII.preamble",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"preamble": "secret", "message": "hello"}
    with tracer.start_as_current_span("cohere.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span,
            kwargs,
            LLMRequestTypeValues.CHAT,
            "cohere.chat",
        )
        _handle_input_content(span, None, LLMRequestTypeValues.CHAT, updated_kwargs)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == "[PII.preamble]"
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"


def test_completion_safety_masks_chat_message_content():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.output]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.output",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(message=SimpleNamespace(content="secret"))
    with tracer.start_as_current_span("cohere.chat") as span:
        _apply_completion_safety(
            span,
            response,
            LLMRequestTypeValues.CHAT,
            "cohere.chat",
        )

    assert response.message.content == "[SECRET.output]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1
