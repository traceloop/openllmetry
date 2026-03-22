import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.bedrock import (
    _instrumented_converse_stream,
    _instrumented_model_invoke_with_response_stream,
)
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.bedrock.streaming_safety import (
    BedrockConverseSafetyStream,
    _BedrockChunkStreamingSafety,
    _BedrockConverseStreamingSafety,
    _decode_chunk_event,
    _encode_chunk_event,
    _set_payload_text,
    create_converse_stream_wrapper,
    create_invoke_stream_wrapper,
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
    clear_completion_safety_stream_factory,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
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

    def fn(*args, **kwargs):
        wrapped_kwargs.update(kwargs)
        return {
            "body": iter(
                [
                    {
                        "chunk": {
                            "bytes": json.dumps({"completion": "secret"}).encode("utf-8")
                        }
                    }
                ]
            )
        }

    instrumented = _instrumented_model_invoke_with_response_stream(
        fn,
        tracer,
        metric_params=SimpleNamespace(),
        event_logger=None,
    )
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-secret"], flush_result="tail")
    )
    response = instrumented(
        body=json.dumps({"prompt": "secret"}),
        modelId="anthropic.claude",
    )

    assert json.loads(wrapped_kwargs["body"])["prompt"] == "[PII.prompt]"
    yielded = list(response["body"])
    assert json.loads(yielded[0]["chunk"]["bytes"].decode("utf-8"))["completion"] == "masked-secrettail"


def test_streaming_converse_applies_prompt_safety_before_provider_call():
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

    wrapped_kwargs = {}

    def fn(*args, **kwargs):
        wrapped_kwargs.update(kwargs)
        return {"stream": None}

    instrumented = _instrumented_converse_stream(
        fn,
        tracer,
        metric_params=SimpleNamespace(),
        event_logger=None,
    )
    instrumented(messages=[{"role": "user", "content": [{"text": "secret"}]}], modelId="anthropic.claude")

    assert wrapped_kwargs["messages"][0]["content"][0]["text"] == "[PII.chat]"


def test_invoke_streaming_wrapper_masks_chunk_bytes_and_accumulates_masked_body():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    completed = []
    event = {
        "chunk": {
            "bytes": json.dumps(
                {"contentBlockDelta": {"delta": {"text": "a"}}}
            ).encode("utf-8")
        }
    }
    with tracer.start_as_current_span("bedrock.completion") as span:
        wrapper = create_invoke_stream_wrapper(
            [event],
            span=span,
            stream_done_callback=lambda body: completed.append(body),
        )
        events = list(wrapper)

    assert len(events) == 1
    decoded = json.loads(events[0]["chunk"]["bytes"].decode("utf-8"))
    assert decoded["contentBlockDelta"]["delta"]["text"] == "masked-atail"
    assert completed == [{"outputText": "masked-atail"}]


def test_converse_streaming_wrapper_masks_deltas_before_span_attributes():
    exporter, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "a"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
    ]
    span = tracer.start_span("bedrock.converse")
    wrapper = create_converse_stream_wrapper(
        events,
        span=span,
        provider="AWS",
        model="demo",
        metric_params=SimpleNamespace(
            guardrail_activation=SimpleNamespace(add=lambda *args, **kwargs: None),
            token_histogram=None,
            duration_histogram=None,
            vendor="AWS",
            model="demo",
            is_stream=True,
            start_time=0.0,
        ),
        event_logger=None,
    )
    yielded = list(wrapper)

    assert yielded[1]["contentBlockDelta"]["delta"]["text"] == "masked-atail"
    finished_span = exporter.get_finished_spans()[0]
    assert (
        finished_span.attributes["gen_ai.completion.0.content"]
        == "masked-atail"
    )


def test_bedrock_chunk_streaming_safety_covers_payload_variants_and_helpers():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(
            [
                "masked-delta",
                "masked-start",
                "masked-nested-delta",
                "masked-nested-start",
                "masked-choice",
            ],
            flush_result="tail",
        )
    )

    with tracer.start_as_current_span("bedrock.completion") as span:
        helper = _BedrockChunkStreamingSafety(
            span=span,
            span_name="bedrock.completion",
            request_type="completion",
        )
        delta_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "index": 1, "delta": {"text": "a"}}
                ).encode("utf-8")
            }
        }
        start_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_start", "index": 2, "content_block": {"text": "b"}}
                ).encode("utf-8")
            }
        }
        nested_delta_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"contentBlockDelta": {"contentBlockIndex": 3, "delta": {"text": "c"}}}
                ).encode("utf-8")
            }
        }
        nested_start_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"contentBlockStart": {"contentBlockIndex": 4, "start": {"text": "d"}}}
                ).encode("utf-8")
            }
        }
        choice_event = {
            "chunk": {
                "bytes": json.dumps({"outputText": "e"}).encode("utf-8")
            }
        }

        helper.process_event(delta_event)
        helper.flush_transition(delta_event, {"chunk": {"bytes": b"{\"messageStop\": {\"stopReason\": \"stop\"}}"}})
        helper.process_event(start_event)
        helper.process_event(nested_delta_event)
        helper.process_event(nested_start_event)
        helper.process_event(choice_event)
        helper.flush_pending_event(choice_event)

        payload = _decode_chunk_event(choice_event)
        _set_payload_text(payload, "replaced")
        _encode_chunk_event(choice_event, payload)

    assert _decode_chunk_event(delta_event)["delta"]["text"] == "masked-deltatail"
    assert _decode_chunk_event(start_event)["content_block"]["text"] == "masked-delta"
    assert _decode_chunk_event(nested_delta_event)["contentBlockDelta"]["delta"]["text"] == "masked-delta"
    assert _decode_chunk_event(nested_start_event)["contentBlockStart"]["start"]["text"] == "masked-delta"
    assert _decode_chunk_event(choice_event)["outputText"] == "replaced"
    assert _decode_chunk_event({"chunk": {}}) is None
    assert _decode_chunk_event({"chunk": {"bytes": b"not-json"}}) is None
    _encode_chunk_event({}, {"ignored": True})


def test_bedrock_invoke_wrapper_accumulates_message_and_provider_payload_shapes():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a", "masked-b"], flush_result="tail")
    )

    events = [
        {"chunk": {"bytes": json.dumps({"type": "message_start", "message": {"role": "assistant"}}).encode("utf-8")}},
        {"chunk": {"bytes": json.dumps({"type": "content_block_start", "content_block": {"text": "a"}}).encode("utf-8")}},
        {"chunk": {"bytes": json.dumps({"type": "content_block_delta", "delta": {"text": "b"}}).encode("utf-8")}},
        {"chunk": {"bytes": json.dumps({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 1}}).encode("utf-8")}},
        {"chunk": {"bytes": json.dumps({"contentBlockDelta": {"delta": {"text": "c"}}, "messageStop": {"stopReason": "end_turn"}, "completion": "d"}).encode("utf-8")}},
    ]
    completed = []

    with tracer.start_as_current_span("bedrock.completion") as span:
        wrapper = create_invoke_stream_wrapper(
            events,
            span=span,
            stream_done_callback=lambda body: completed.append(body),
        )
        yielded = list(wrapper)
        wrapper._accumulate_event("not-a-dict")

    assert len(yielded) == 5
    assert completed
    assert completed[0]["content"][0]["text"].startswith("masked-a")
    assert completed[0]["outputText"] == "ctail"
    assert completed[0]["stop_reason"] == "end_turn"


def test_bedrock_converse_streaming_safety_and_wrapper_cover_observation_paths(monkeypatch):
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a", "masked-b"], flush_result="tail")
    )
    recorded = {"span": [], "guardrail": [], "usage": [], "events": []}
    monkeypatch.setattr(
        "opentelemetry.instrumentation.bedrock.streaming_safety.should_emit_events",
        lambda: False,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.bedrock.streaming_safety.set_converse_streaming_response_span_attributes",
        lambda response_msg, role, span: recorded["span"].append((list(response_msg), role)),
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.bedrock.streaming_safety.guardrail_converse",
        lambda *args: recorded["guardrail"].append(args),
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.bedrock.streaming_safety.converse_usage_record",
        lambda *args: recorded["usage"].append(args),
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.bedrock.streaming_safety.emit_streaming_converse_response_event",
        lambda *args: recorded["events"].append(args),
    )

    with tracer.start_as_current_span("bedrock.converse") as span:
        helper = _BedrockConverseStreamingSafety(span=span, span_name="bedrock.converse")
        start_event = {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": "a"}}}
        delta_event = {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "b"}}}
        helper.process_event(start_event)
        helper.process_event(delta_event)
        helper.flush_transition(start_event, {"messageStop": {"stopReason": "end_turn"}})
        helper.flush_pending_event(delta_event)
        helper.process_event("not-a-dict")

    assert start_event["contentBlockStart"]["start"]["text"] == "masked-atail"
    assert delta_event["contentBlockDelta"]["delta"]["text"] == "masked-atail"

    span = tracer.start_span("bedrock.converse")
    wrapper = BedrockConverseSafetyStream(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": "a"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
        ],
        span=span,
        provider="AWS",
        model="demo",
        metric_params=SimpleNamespace(),
        event_logger=None,
    )
    yielded = list(wrapper)
    wrapper._observe_event("not-a-dict")

    assert yielded[1]["contentBlockStart"]["start"]["text"] == "masked-atail"
    assert recorded["span"] == [(["masked-atail"], "assistant")]
    assert recorded["guardrail"]
    assert recorded["usage"]


def _raise_boom(context):
    raise RuntimeError("boom")


def test_apply_invoke_prompt_safety_returns_kwargs_on_handler_exception():
    """MF-5: _apply_invoke_prompt_safety must return original kwargs when safety handler raises."""
    _, tracer = _test_span()
    register_prompt_safety_handler(_raise_boom)

    kwargs = {"body": json.dumps({"prompt": "secret"}), "modelId": "anthropic.claude"}
    with tracer.start_as_current_span("bedrock.completion") as span:
        result = _apply_invoke_prompt_safety(span, kwargs, "bedrock.completion")

    assert result is kwargs


def test_apply_invoke_completion_safety_returns_response_on_handler_exception():
    """MF-5: _apply_invoke_completion_safety must return original response when safety handler raises."""
    _, tracer = _test_span()
    register_completion_safety_handler(_raise_boom)

    raw_response = json.dumps({"completion": "secret"})
    with tracer.start_as_current_span("bedrock.completion") as span:
        result, changed = _apply_invoke_completion_safety(
            span, raw_response, "bedrock.completion"
        )

    assert result is raw_response
    assert changed is False


def test_streaming_with_safety_span_attributes_contain_masked_data():
    """MF-6: When streaming with safety enabled, span attributes must contain MASKED data,
    not unmasked data. The stream_done callback should only be called once with masked body."""
    exporter, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["[MASKED]"], flush_result="")
    )

    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {"contentBlockDelta": {"delta": {"text": "secret-data"}}}
                ).encode("utf-8")
            }
        },
    ]

    span = tracer.start_span("bedrock.completion")

    callback_bodies = []

    def stream_done(response_body):
        callback_bodies.append(dict(response_body))
        if span.is_recording():
            span.set_attribute("gen_ai.completion.0.content", response_body.get("outputText", ""))
        span.end()

    from opentelemetry.instrumentation.bedrock.streaming_wrapper import StreamingWrapper
    wrapper = create_invoke_stream_wrapper(
        StreamingWrapper(events),
        span=span,
        stream_done_callback=stream_done,
    )
    yielded = list(wrapper)

    assert len(callback_bodies) == 1
    assert "secret-data" not in callback_bodies[0].get("outputText", "")
    assert "[MASKED]" in callback_bodies[0].get("outputText", "")

    finished_spans = exporter.get_finished_spans()
    assert len(finished_spans) == 1
    completion_attr = finished_spans[0].attributes.get("gen_ai.completion.0.content", "")
    assert "secret-data" not in completion_attr
    assert "[MASKED]" in completion_attr
