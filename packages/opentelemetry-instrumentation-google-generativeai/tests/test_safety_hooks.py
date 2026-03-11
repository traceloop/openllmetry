from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.google_generativeai import (
    _handle_request,
    _apply_completion_safety,
    _apply_prompt_safety,
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


def test_prompt_safety_masks_positional_prompt_args():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.prompt",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        updated_args, updated_kwargs = _apply_prompt_safety(
            span,
            ("secret",),
            {},
            "gemini.generate_content",
        )

    assert updated_args == ("[PII.prompt]",)
    assert updated_kwargs == {}


def test_prompt_safety_masks_span_prompt_attributes():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.prompt",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        updated_args, updated_kwargs = _apply_prompt_safety(
            span,
            ("secret",),
            {},
            "gemini.generate_content",
        )
        _handle_request(span, updated_args, updated_kwargs, "gemini-2.0", None)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == '[{"type": "text", "text": "[PII.prompt]"}]'
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"


def test_completion_safety_masks_candidate_parts():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.gemini]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.gemini",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="secret")])
            )
        ]
    )
    with tracer.start_as_current_span("gemini.generate_content") as span:
        _apply_completion_safety(span, response, "gemini.generate_content")

    assert response.candidates[0].content.parts[0].text == "[SECRET.gemini]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1
