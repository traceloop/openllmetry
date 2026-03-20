import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.transformers.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _resolve_masked_text,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.fr


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


def test_prompt_safety_masks_text_generation_input():
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

    with tracer.start_as_current_span("transformers_text_generation_pipeline.call") as span:
        updated_args, _ = _apply_prompt_safety(
            span, ("secret",), {}, "transformers_text_generation_pipeline.call"
        )

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

    response = [{"generated_text": "secret"}]
    with tracer.start_as_current_span("transformers_text_generation_pipeline.call") as span:
        _apply_completion_safety(span, response, "transformers_text_generation_pipeline.call")

    assert response[0]["generated_text"] == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_keyword_and_nested_batch_paths():
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

    with tracer.start_as_current_span("transformers_text_generation_pipeline.call") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span,
            (),
            {"text_inputs": ["prompt-a", "prompt-b", 123]},
            "transformers_text_generation_pipeline.call",
        )
        unchanged_args, unchanged_kwargs = _apply_prompt_safety(
            span, (), {"args": {"not": "supported"}}, "transformers_text_generation_pipeline.call"
        )
        _, updated_chat_kwargs = _apply_prompt_safety(
            span,
            (),
            {"text_inputs": [{"role": "user", "content": "chat-secret"}]},
            "transformers_text_generation_pipeline.call",
        )
        response = [
            [{"generated_text": "completion-a"}],
            [{"generated_text": [{"role": "assistant", "content": "chat-completion"}]}],
            {"generated_text": None},
            "ignored",
        ]
        _apply_completion_safety(span, response, "transformers_text_generation_pipeline.call")
        _apply_completion_safety(span, {"generated_text": "ignored"}, "transformers_text_generation_pipeline.call")

    assert updated_kwargs["text_inputs"][:2] == ["[MASKED:prompt-a]", "[MASKED:prompt-b]"]
    assert updated_chat_kwargs["text_inputs"][0]["content"] == "[MASKED:chat-secret]"
    assert unchanged_args == ()
    assert unchanged_kwargs == {"args": {"not": "supported"}}
    assert response[0][0]["generated_text"] == "[MASKED:completion-a]"
    assert response[1][0]["generated_text"][0]["content"] == "[MASKED:chat-completion]"
    assert _resolve_masked_text("same", None) == ("same", False)
