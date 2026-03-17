import json
from io import BytesIO

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.sagemaker.reusable_streaming_body import (
    ReusableStreamingBody,
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
