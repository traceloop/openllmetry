import json
from io import BytesIO

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_completion_safety_stream_factory,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.sagemaker.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.sagemaker.streaming_safety import (
    create_streaming_wrapper,
)
from opentelemetry.instrumentation.sagemaker import (
    _instrumented_endpoint_invoke_with_response_stream,
)
from opentelemetry.instrumentation.sagemaker.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _decode_payload,
    _encode_payload,
    _mask_completion_payload,
    _mask_prompt_payload,
    _resolve_masked_text,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def teardown_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


class _FakeStreamSession:
    def __init__(self, results=None, flush_result=""):
        self._results = list(results or [])
        self._flush_result = flush_result

    def process_chunk(self, text):
        if self._results:
            return SafetyResult(text=self._results.pop(0), overall_action="MASK", findings=[])
        return SafetyResult(text=text, overall_action="ALLOW", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="ALLOW", findings=[])


def test_prompt_safety_masks_request_body():
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

    kwargs = {"Body": json.dumps({"inputs": "secret"})}
    with tracer.start_as_current_span("sagemaker.completion") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs, "sagemaker.completion")

    assert json.loads(updated_kwargs["Body"])["inputs"] == "[PII.prompt]"


def test_completion_safety_masks_response_body():
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

    raw_response = json.dumps([{"generated_text": "secret"}])
    with tracer.start_as_current_span("sagemaker.completion") as span:
        updated_response, changed = _apply_completion_safety(
            span, raw_response, "sagemaker.completion"
        )

    assert changed is True
    assert json.loads(updated_response)[0]["generated_text"] == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_payload_helpers_cover_nested_and_invalid_inputs():
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

    payload = {"items": [{"inputs": "prompt-a"}]}
    completion_payload = {"items": [{"generated_text": "completion-a"}]}
    with tracer.start_as_current_span("sagemaker.completion") as span:
        masked_prompt, prompt_changed = _mask_prompt_payload(
            span, payload, span_name="sagemaker.completion", segment_index=0
        )
        masked_completion, completion_changed = _mask_completion_payload(
            span, completion_payload, span_name="sagemaker.completion", segment_index=0
        )

    assert prompt_changed is True
    assert masked_prompt["items"][0]["inputs"] == "[MASKED:prompt-a]"
    assert completion_changed is True
    assert masked_completion["items"][0]["generated_text"] == "[MASKED:completion-a]"
    assert _decode_payload("not-json") == (None, False)
    assert _decode_payload(123) == (None, False)
    assert _encode_payload({"text": "x"}, False) == '{"text": "x"}'
    assert _encode_payload({"text": "x"}, True) == b'{"text": "x"}'
    assert _resolve_masked_text("same", None) == ("same", False)


def test_streaming_invoke_applies_prompt_safety_before_provider_call():
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

    wrapped_kwargs = {}

    class _Body:
        def __iter__(self):
            return iter(())

    def fn(*args, **kwargs):
        wrapped_kwargs.update(kwargs)
        return {"Body": _Body()}

    instrumented = _instrumented_endpoint_invoke_with_response_stream(
        fn,
        tracer,
        event_logger=None,
    )
    instrumented(EndpointName="demo", Body=json.dumps({"inputs": "secret"}))

    assert json.loads(wrapped_kwargs["Body"])["inputs"] == "[PII.prompt]"


def test_streaming_wrapper_masks_payload_parts_and_flushes_tail():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    completed = []
    with tracer.start_as_current_span("sagemaker.completion") as span:
        wrapper = create_streaming_wrapper(
            [{"PayloadPart": {"Bytes": b"a"}}],
            span=span,
            stream_done_callback=lambda body: completed.append(body),
        )
        events = list(wrapper)

    assert len(events) == 1
    assert events[0]["PayloadPart"]["Bytes"] == b"masked-atail"
    assert completed == ["masked-atail"]


def test_streaming_wrapper_handles_invalid_events_and_factory_helper():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="")
    )

    completed = []
    invalid_utf8 = bytes([0xFF])
    with tracer.start_as_current_span("sagemaker.completion") as span:
        wrapper = create_streaming_wrapper(
            [
                {"Unexpected": True},
                {"PayloadPart": {"Bytes": invalid_utf8}},
                {"PayloadPart": {"Bytes": b"a"}},
            ],
            span=span,
            stream_done_callback=lambda body: completed.append(body),
        )
        events = list(wrapper)

    assert events[0] == {"Unexpected": True}
    assert events[1]["PayloadPart"]["Bytes"] == invalid_utf8
    assert events[2]["PayloadPart"]["Bytes"] == b"masked-a"
    assert completed == ["masked-a"]
