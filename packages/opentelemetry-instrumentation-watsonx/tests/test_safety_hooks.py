import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.watsonx.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
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


def test_prompt_safety_masks_prompt_arg():
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

    with tracer.start_as_current_span("watsonx.generate") as span:
        updated_args, _ = _apply_prompt_safety(span, ("secret",), {}, "watsonx.generate")

    assert updated_args[0] == "[PII.prompt]"


def test_completion_safety_masks_generated_text():
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

    response = {"results": [{"generated_text": "secret"}]}
    with tracer.start_as_current_span("watsonx.generate") as span:
        _apply_completion_safety(span, response, "watsonx.generate")

    assert response["results"][0]["generated_text"] == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_list_paths():
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

    with tracer.start_as_current_span("watsonx.generate") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span, (), {"prompt": ["a", "b", 1]}, "watsonx.generate"
        )
        updated_args, _ = _apply_prompt_safety(
            span, (["x", "y"],), {}, "watsonx.generate"
        )
        responses = [
            {"results": [{"generated_text": "one"}, {"generated_text": "one-b"}]},
            {"results": [{"generated_text": "two"}]},
        ]
        _apply_completion_safety(span, responses, "watsonx.generate")

    assert updated_kwargs["prompt"][:2] == ["[MASKED:a]", "[MASKED:b]"]
    assert updated_args[0][:2] == ["[MASKED:x]", "[MASKED:y]"]
    assert responses[0]["results"][0]["generated_text"] == "[MASKED:one]"
    assert responses[0]["results"][1]["generated_text"] == "[MASKED:one-b]"
    assert responses[1]["results"][0]["generated_text"] == "[MASKED:two]"
    assert _resolve_masked_text("same", None) == ("same", False)
