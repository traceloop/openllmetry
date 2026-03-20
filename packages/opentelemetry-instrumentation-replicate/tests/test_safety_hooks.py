from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.replicate.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.replicate.streaming_safety import (
    ReplicateStreamingSafety,
)
from opentelemetry.instrumentation.replicate import _wrap
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


def test_prompt_safety_masks_input_prompt():
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

    with tracer.start_as_current_span("replicate.run") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span, (), {"input": {"prompt": "secret"}}, "replicate.run"
        )

    assert updated_kwargs["input"]["prompt"] == "[PII.prompt]"


def test_completion_safety_masks_prediction_output():
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

    response = SimpleNamespace(output="secret")
    with tracer.start_as_current_span("replicate.predictions.create") as span:
        _apply_completion_safety(span, response, "replicate.predictions.create")

    assert response.output == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_safety_supports_args_path_and_passthrough():
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

    with tracer.start_as_current_span("replicate.run") as span:
        updated_args, unchanged_kwargs = _apply_prompt_safety(
            span,
            ("model", {"prompt": "secret"}),
            {},
            "replicate.run",
        )
        _, nested_kwargs = _apply_prompt_safety(
            span,
            (),
            {
                "input": {
                    "query": "secret",
                    "image": "https://example.invalid/cat.png",
                    "nested": {"negative_prompt": "secret"},
                }
            },
            "replicate.run",
        )
        passthrough_args, passthrough_kwargs = _apply_prompt_safety(
            span,
            (),
            {"input": {"prompt": None}},
            "replicate.run",
        )

    assert updated_args[1]["prompt"] == "[PII.prompt]"
    assert unchanged_kwargs == {}
    assert nested_kwargs["input"]["query"] == "[PII.prompt]"
    assert nested_kwargs["input"]["nested"]["negative_prompt"] == "[PII.prompt]"
    assert nested_kwargs["input"]["image"] == "https://example.invalid/cat.png"
    assert passthrough_args == ()
    assert passthrough_kwargs == {"input": {"prompt": None}}


def test_completion_safety_covers_list_and_string_branches():
    _, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    list_response = ["secret", 123]
    object_response = SimpleNamespace(output=["secret", "second"])
    with tracer.start_as_current_span("replicate.predictions.create") as span:
        updated_string = _apply_completion_safety(
            span,
            "plain-string",
            "replicate.predictions.create",
        )
        _apply_completion_safety(span, list_response, "replicate.predictions.create")
        _apply_completion_safety(span, object_response, "replicate.predictions.create")

    assert list_response[0] == "[MASKED:secret]"
    assert object_response.output == ["[MASKED:secret]", "[MASKED:second]"]
    assert updated_string == "[MASKED:plain-string]"
    assert _resolve_masked_text("same", None) == ("same", False)
    unchanged = SafetyResult(text="same", overall_action="MASK", findings=[])
    assert _resolve_masked_text("same", unchanged) == ("same", False)


def test_wrapper_masks_plain_string_response_before_attributes_are_written():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[MASKED:plain-string]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "plain-string"
        else None
    )

    wrapper = _wrap(tracer, None, {"span_name": "replicate.run"})
    response = wrapper(
        lambda *args, **kwargs: "plain-string",
        None,
        (),
        {"input": {"prompt": "secret"}},
    )

    assert response == "[MASKED:plain-string]"
    span = exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.prompt.0.user"] == "[PII.prompt]"
    assert span.attributes["gen_ai.completion.0.content"] == "[MASKED:plain-string]"


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_streaming_helper_masks_text_and_flushes_tail():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())

    with tracer.start_as_current_span("replicate.stream") as span:
        helper = ReplicateStreamingSafety(span, "replicate.stream")
        assert helper.process_text("secret") == "masked"
        assert helper.flush() == "-tail"


def test_stream_wrapper_preserves_provider_chunk_count_when_flushing_tail():
    exporter, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    wrapper = _wrap(tracer, None, {"span_name": "replicate.stream"})

    def _stream(*args, **kwargs):
        yield "secret"
        yield "chunk"

    response = list(
        wrapper(
            _stream,
            None,
            (),
            {"input": {"prompt": "plain"}},
        )
    )

    assert response == ["masked", "masked-tail"]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
