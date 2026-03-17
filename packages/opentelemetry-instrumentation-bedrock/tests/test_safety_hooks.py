import json
from io import BytesIO

import pytest

from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.bedrock.safety import (
    _apply_converse_completion_safety,
    _apply_converse_prompt_safety,
    _decode_payload,
    _encode_payload,
    _apply_invoke_completion_safety,
    _apply_invoke_prompt_safety,
    _mask_completion_payload,
    _mask_prompt_payload,
    _prepare_invoke_response,
    _request_type,
    _resolve_masked_text,
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


def test_invoke_prompt_safety_masks_json_body():
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

    kwargs = {"body": json.dumps({"prompt": "secret"}), "modelId": "anthropic.claude"}
    with tracer.start_as_current_span("bedrock.completion") as span:
        updated_kwargs = _apply_invoke_prompt_safety(span, kwargs, "bedrock.completion")

    assert json.loads(updated_kwargs["body"])["prompt"] == "[PII.prompt]"


def test_converse_prompt_safety_masks_message_content():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.chat]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"messages": [{"role": "user", "content": [{"text": "secret"}]}]}
    with tracer.start_as_current_span("bedrock.converse") as span:
        updated_kwargs = _apply_converse_prompt_safety(span, kwargs, "bedrock.converse")

    assert updated_kwargs["messages"][0]["content"][0]["text"] == "[PII.chat]"


def test_invoke_completion_safety_masks_json_response():
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

    raw_response = json.dumps({"completion": "secret"})
    with tracer.start_as_current_span("bedrock.completion") as span:
        updated_response, changed = _apply_invoke_completion_safety(
            span, raw_response, "bedrock.completion"
        )

    assert changed is True
    assert json.loads(updated_response)["completion"] == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prepare_invoke_response_masks_and_rebuilds_streaming_body():
    _, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.output]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    raw_response = json.dumps({"completion": "secret"}).encode("utf-8")
    response = {
        "body": ReusableStreamingBody(BytesIO(raw_response), len(raw_response)),
    }

    with tracer.start_as_current_span("bedrock.completion") as span:
        parsed_response = _prepare_invoke_response(span, response, "bedrock.completion")

    assert parsed_response["completion"] == "[SECRET.output]"
    assert json.loads(response["body"].read().decode("utf-8"))["completion"] == "[SECRET.output]"


def test_converse_helpers_cover_system_messages_and_completion_output():
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

    kwargs = {
        "system": [{"text": "system-secret"}, {"ignore": "x"}],
        "messages": [{"role": "user", "content": [{"text": "message-secret"}, {"image": "ignored"}]}],
    }
    response = {"output": {"message": {"content": [{"text": "completion-secret"}]}}}
    with tracer.start_as_current_span("bedrock.converse") as span:
        updated_kwargs = _apply_converse_prompt_safety(span, kwargs, "bedrock.converse")
        _apply_converse_completion_safety(span, response, "bedrock.converse")

    assert updated_kwargs["system"][0]["text"] == "[MASKED:system-secret]"
    assert updated_kwargs["messages"][0]["content"][0]["text"] == "[MASKED:message-secret]"
    assert response["output"]["message"]["content"][0]["text"] == "[MASKED:completion-secret]"


def test_payload_helpers_cover_nested_structures_and_invalid_inputs():
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

    prompt_payload = {"items": [{"prompt": "prompt-secret"}], "messages": [{"content": [{"text": "message-secret"}]}]}
    completion_payload = {"items": [{"completion": "completion-secret"}], "generations": [[{"text": "gen-secret"}]]}
    with tracer.start_as_current_span("bedrock.completion") as span:
        masked_prompt, prompt_changed = _mask_prompt_payload(
            span,
            prompt_payload,
            span_name="bedrock.completion",
            request_type="completion",
            segment_index=0,
        )
        masked_completion, completion_changed = _mask_completion_payload(
            span,
            completion_payload,
            span_name="bedrock.completion",
            request_type="completion",
            segment_index=0,
        )

    assert prompt_changed is True
    assert masked_prompt["items"][0]["prompt"] == "[MASKED:prompt-secret]"
    assert masked_prompt["messages"][0]["content"][0]["text"] == "[MASKED:message-secret]"
    assert completion_changed is True
    assert masked_completion["items"][0]["completion"] == "[MASKED:completion-secret]"
    assert masked_completion["generations"][0][0]["text"] == "[MASKED:gen-secret]"
    assert _decode_payload("not-json") == (None, False)
    assert _decode_payload(123) == (None, False)
    assert _encode_payload({"x": 1}, False) == '{"x": 1}'
    assert _encode_payload({"x": 1}, True) == b'{"x": 1}'
    assert _request_type("bedrock.converse") == "chat"
    assert _request_type("bedrock.completion") == "completion"
    assert _resolve_masked_text("same", None) == ("same", False)
