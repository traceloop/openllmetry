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
from opentelemetry.instrumentation.vertexai.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_prompt_value,
    _resolve_masked_text,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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


def test_prompt_safety_masks_positional_text_arg():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        updated_args, _ = _apply_prompt_safety(
            span, ("secret",), {}, "vertexai.generate_content"
        )

    assert updated_args[0] == "[PII.prompt]"


def test_completion_safety_masks_response_text():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.output]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(text="secret", candidates=[SimpleNamespace(text="secret")])
    with tracer.start_as_current_span("vertexai.generate_content") as span:
        _apply_completion_safety(span, response, "vertexai.generate_content")

    assert response.text == "[SECRET.output]"
    assert response.candidates[0].text == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 2


def test_prompt_and_completion_cover_contents_parts_and_candidate_parts():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    contents = [SimpleNamespace(parts=[SimpleNamespace(text="prompt-part")])]
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                text=None,
                content=SimpleNamespace(parts=[SimpleNamespace(text="completion-part")]),
            )
        ],
    )
    with tracer.start_as_current_span("vertexai.generate_content") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span, (), {"contents": contents}, "vertexai.generate_content"
        )
        _apply_completion_safety(span, response, "vertexai.generate_content")
        assert _mask_prompt_value(
            span, None, span_name="vertexai.generate_content", segment_index=0, segment_role="user"
        ) == (None, False)

    assert updated_kwargs["contents"][0].parts[0].text == "[MASKED:prompt-part]"
    assert response.candidates[0].content.parts[0].text == "[MASKED:completion-part]"
    assert _resolve_masked_text("same", None) == ("same", False)
