from types import SimpleNamespace

import pytest
from openai._legacy_response import LegacyAPIResponse
from opentelemetry import context as context_api

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
from opentelemetry.instrumentation.openai.v1.assistant_safety import (
    AssistantStreamingSafety,
    _assistant_block_text,
    _mask_assistant_content,
    _set_assistant_block_text,
    apply_assistant_instruction_prompt_safety,
    apply_assistant_message_prompt_safety,
    apply_assistant_messages_list_safety,
)
from opentelemetry.instrumentation.openai.v1.event_handler_wrapper import (
    EventHandleWrapper,
)
from opentelemetry.instrumentation.openai.v1.realtime_safety import (
    RealtimeStreamingSafety,
    apply_realtime_event_prompt_safety,
    apply_realtime_item_prompt_safety,
    apply_realtime_session_prompt_safety,
)
from opentelemetry.instrumentation.openai.v1.responses_safety import (
    _mask_prompt_content_blocks,
    _mask_response_input,
    _mask_response_input_item,
    _mask_response_output_content,
    _response_output_key,
    ResponsesStreamingSafety,
    apply_response_completion_safety,
    apply_response_prompt_safety,
)
from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
    RealtimeConnectionWrapper,
    RealtimeEventProcessor,
    RealtimeSessionState,
)
from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    ResponseStream,
    TracedData,
    _cache_legacy_parsed_response,
    responses,
    responses_get_or_create_wrapper,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.fr


class _FakeStreamSession:
    def __init__(self, process_results, flush_result=""):
        self._process_results = list(process_results)
        self._flush_result = flush_result

    def process_chunk(self, text):
        value = self._process_results.pop(0)
        return SafetyResult(text=value, overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="allow", findings=[])


class _EmptyStream:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self):
        return None


class _UncachedLegacyResponse(LegacyAPIResponse):
    def __init__(self, factory):
        self._factory = factory
        self._parsed_by_type = {}
        self._cast_to = "response"

    def parse(self, *, to=None):
        cache_key = to if to is not None else self._cast_to
        cached = self._parsed_by_type.get(cache_key)
        if cached is not None:
            return cached
        return self._factory()


class _OriginalAssistantHandler:
    def __init__(self):
        self.delta_values = []
        self.done_value = None
        self.message = None

    def on_end(self):
        pass

    def on_event(self, event):
        pass

    def on_run_step_created(self, run_step):
        pass

    def on_run_step_delta(self, delta, snapshot):
        pass

    def on_run_step_done(self, run_step):
        pass

    def on_tool_call_created(self, tool_call):
        pass

    def on_tool_call_delta(self, delta, snapshot):
        pass

    def on_tool_call_done(self, tool_call):
        pass

    def on_exception(self, exception: Exception):
        pass

    def on_timeout(self):
        pass

    def on_message_created(self, message):
        pass

    def on_message_delta(self, delta, snapshot):
        pass

    def on_message_done(self, message):
        self.message = message

    def on_text_created(self, text):
        pass

    def on_text_delta(self, delta, snapshot):
        self.delta_values.append(delta.value)

    def on_text_done(self, text):
        self.done_value = text.value

    def on_image_file_done(self, image_file):
        pass


class _FakeRealtimeConnection:
    def __init__(self):
        self.sent_event = None

    async def send(self, event):
        self.sent_event = event
        return "sent"


def setup_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()
    responses.clear()


def teardown_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()
    responses.clear()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def _mask_result(text, category, rule_name, location):
    return SafetyResult(
        text=text,
        overall_action="MASK",
        findings=[
            SafetyFinding(
                category=category,
                severity="HIGH",
                action="MASK",
                rule_name=rule_name,
                start=0,
                end=6,
            )
        ],
    )


def test_response_prompt_safety_masks_instructions_and_input_without_mutating_source():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {
        "instructions": "secret",
        "input": [{"role": "user", "content": "secret"}],
    }

    with tracer.start_as_current_span("openai.response") as span:
        updated_kwargs = apply_response_prompt_safety(span, kwargs)

    assert kwargs["instructions"] == "secret"
    assert kwargs["input"][0]["content"] == "secret"
    assert updated_kwargs["instructions"] == "[PII.email]"
    assert updated_kwargs["input"][0]["content"] == "[PII.email]"
    span = exporter.get_finished_spans()[0]
    assert len(span.events) == 2


def test_response_completion_safety_masks_message_output_blocks():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: _mask_result(
            "[SECRET.api_key]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="secret")],
            )
        ],
        output_text="secret",
    )

    with tracer.start_as_current_span("openai.response") as span:
        masked = apply_response_completion_safety(span, response)

    assert masked == "[SECRET.api_key]"
    assert response.output[0].content[0].text == "[SECRET.api_key]"
    assert response.output_text == "[SECRET.api_key]"
    span = exporter.get_finished_spans()[0]
    assert len(span.events) == 1


def test_response_streaming_safety_uses_done_event_for_tail_delivery():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    delta_chunk = SimpleNamespace(
        type="response.output_text.delta",
        output_index=0,
        content_index=0,
        delta="secret",
    )
    done_chunk = SimpleNamespace(
        type="response.output_text.done",
        output_index=0,
        content_index=0,
        text="secret",
    )

    with tracer.start_as_current_span("openai.response") as span:
        helper = ResponsesStreamingSafety(span)
        helper.process_chunk(delta_chunk)
        helper.process_chunk(done_chunk)

    assert delta_chunk.delta == "masked"
    assert done_chunk.delta.text == "tail"
    assert done_chunk.text == "maskedtail"


def test_response_wrapper_defers_span_end_for_non_completed_response():
    """Non-completed responses (in_progress, queued) should NOT end the span
    immediately -- the span stays open for continuation polling."""
    exporter, tracer = _test_span()

    wrapper = responses_get_or_create_wrapper(tracer)
    response = wrapper(
        lambda **_: SimpleNamespace(
            id="resp_1",
            status="in_progress",
            output=[],
            usage=None,
            model="gpt-test",
            service_tier=None,
        ),
        SimpleNamespace(_client=None),
        (),
        {"model": "gpt-test", "input": "hello"},
    )

    assert response.status == "in_progress"
    # Span should NOT be ended for non-completed responses
    spans = exporter.get_finished_spans()
    assert len(spans) == 0


def test_response_wrapper_ends_span_for_completed_response():
    """Completed responses should have their span ended immediately."""
    exporter, tracer = _test_span()

    wrapper = responses_get_or_create_wrapper(tracer)
    response = wrapper(
        lambda **_: SimpleNamespace(
            id="resp_2",
            status="completed",
            output=[],
            usage=None,
            model="gpt-test",
            service_tier=None,
        ),
        SimpleNamespace(_client=None),
        (),
        {"model": "gpt-test", "input": "hello"},
    )

    assert response.status == "completed"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.response"


def test_response_stream_does_not_duplicate_done_delta_text():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    delta_chunk = SimpleNamespace(
        type="response.output_text.delta",
        output_index=0,
        content_index=0,
        delta="secret",
    )
    done_chunk = SimpleNamespace(
        type="response.output_text.done",
        output_index=0,
        content_index=0,
        text="secret",
    )

    with tracer.start_as_current_span("openai.response") as span:
        stream = ResponseStream(
            span=span,
            response=_EmptyStream(),
            start_time=1,
            request_kwargs={"model": "gpt-test", "input": "hello"},
            tracer=tracer,
        )
        stream._process_chunk(delta_chunk)
        stream._process_chunk(done_chunk)

        assert stream._output_text == "maskedtail"


def test_response_stream_creates_span_for_continuation_only_requests():
    exporter, tracer = _test_span()
    responses["resp_parent"] = TracedData(
        start_time=111,
        response_id="resp_parent",
        input=[],
        instructions=None,
        tools=[],
        output_blocks={},
        usage=None,
        output_text="masked",
        request_model="gpt-test",
        response_model="gpt-test",
        request_reasoning_summary=None,
        request_reasoning_effort=None,
        response_reasoning_effort=None,
        request_service_tier=None,
        response_service_tier=None,
        trace_context=context_api.get_current(),
    )

    stream = ResponseStream(
        span=None,
        response=_EmptyStream(),
        start_time=222,
        request_kwargs={"model": "gpt-test", "previous_response_id": "resp_parent"},
        tracer=tracer,
    )

    assert stream._span is not None
    stream.close()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.response"


def test_response_legacy_cache_helper_reuses_masked_parsed_object():
    register_completion_safety_handler(
        lambda context: _mask_result(
            "[SECRET.api_key]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    parsed_response = SimpleNamespace(
        id="resp_legacy",
        status="completed",
        output=[
            SimpleNamespace(
                id="msg_1",
                type="message",
                content=[SimpleNamespace(type="output_text", text="secret")],
            )
        ],
        output_text="secret",
        usage=None,
        model="gpt-test",
        service_tier=None,
    )

    legacy_response = _UncachedLegacyResponse(
        lambda: SimpleNamespace(
            output_text="secret",
            output=[
                SimpleNamespace(
                    content=[SimpleNamespace(text="secret")]
                )
            ],
        )
    )

    exporter, tracer = _test_span()
    with tracer.start_as_current_span("openai.response") as span:
        apply_response_completion_safety(span, parsed_response)
    _cache_legacy_parsed_response(legacy_response, parsed_response)

    reparsed = legacy_response.parse()

    assert reparsed.output[0].content[0].text == "[SECRET.api_key]"
    assert reparsed.output_text == "[SECRET.api_key]"
    assert len(exporter.get_finished_spans()) == 1


def test_assistant_message_prompt_safety_masks_content_argument():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    args, kwargs = apply_assistant_message_prompt_safety(
        None,
        (),
        {"role": "user", "content": "secret"},
    )

    assert args == ()
    assert kwargs["content"] == "[PII.email]"


def test_assistant_messages_list_safety_masks_prompt_and_completion_messages():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _mask_result(
            "[SECRET.api_key]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                role="user",
                content=[SimpleNamespace(text=SimpleNamespace(value="secret"))],
            ),
            SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(text=SimpleNamespace(value="secret"))],
            ),
        ]
    )

    with tracer.start_as_current_span("openai.assistant.run") as span:
        apply_assistant_messages_list_safety(span, response)

    assert response.data[0].content[0].text.value == "[PII.email]"
    assert response.data[1].content[0].text.value == "[SECRET.api_key]"
    span = exporter.get_finished_spans()[0]
    assert len(span.events) == 2


def test_event_handler_wrapper_masks_streamed_assistant_text_and_flushes_tail():
    exporter, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    original_handler = _OriginalAssistantHandler()
    message = SimpleNamespace(
        id="msg_1",
        role="assistant",
        content=[
            SimpleNamespace(
                model_dump=lambda: {"type": "output_text", "text": {"value": "secret"}},
                text=SimpleNamespace(value="secret"),
            )
        ],
    )

    with tracer.start_as_current_span("openai.assistant.run_stream") as span:
        handler = EventHandleWrapper(original_handler, span)
        handler.on_text_delta(SimpleNamespace(value="secret"), SimpleNamespace(value="secret"))
        handler.on_text_done(SimpleNamespace(value="secret"))
        handler.on_message_done(message)

    assert original_handler.delta_values == ["masked"]
    assert original_handler.done_value == "maskedtail"
    assert original_handler.message.content[0].text.value == "maskedtail"
    span = exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.completion.0.content"] == "maskedtail"


def test_realtime_prompt_helpers_mask_session_and_item_payloads():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    event = {
        "type": "conversation.item.create",
        "item": {"content": [{"type": "input_text", "text": "secret"}]},
    }
    kwargs = {"item": {"content": [{"type": "input_text", "text": "secret"}]}}
    session_kwargs = {"session": {"instructions": "secret"}}

    masked_event = apply_realtime_event_prompt_safety(event)
    masked_item = apply_realtime_item_prompt_safety(kwargs)
    masked_session = apply_realtime_session_prompt_safety(session_kwargs)

    assert masked_event["item"]["content"][0]["text"] == "[PII.email]"
    assert masked_item["item"]["content"][0]["text"] == "[PII.email]"
    assert masked_session["session"]["instructions"] == "[PII.email]"


def test_realtime_prompt_helper_masks_typed_raw_send_events():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    typed_event = SimpleNamespace(
        type="conversation.item.create",
        item=SimpleNamespace(
            content=[SimpleNamespace(type="input_text", text="secret")]
        ),
    )

    masked_event = apply_realtime_event_prompt_safety(typed_event)

    assert masked_event is not typed_event
    assert masked_event.item.content[0].text == "[PII.email]"
    assert typed_event.item.content[0].text == "secret"


@pytest.mark.asyncio
async def test_realtime_connection_send_masks_typed_event_and_tracks_masked_input():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: _mask_result(
            "[PII.email]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    state = RealtimeSessionState(tracer, "gpt-test")
    fake_connection = _FakeRealtimeConnection()
    wrapper = RealtimeConnectionWrapper(fake_connection, state)

    event = SimpleNamespace(
        type="conversation.item.create",
        item=SimpleNamespace(
            content=[SimpleNamespace(type="input_text", text="secret")]
        ),
    )

    result = await wrapper.send(event)

    assert result == "sent"
    assert fake_connection.sent_event.item.content[0].text == "[PII.email]"
    assert state.input_text == "[PII.email]"
    assert event.item.content[0].text == "secret"
    assert exporter.get_finished_spans() == ()


def test_realtime_streaming_safety_masks_delta_without_synthetic_done_events():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    delta_event = SimpleNamespace(type="response.text.delta", delta="secret")
    done_event = SimpleNamespace(type="response.done")

    with tracer.start_as_current_span("openai.realtime") as span:
        helper = RealtimeStreamingSafety(span)
        streamed_events = helper.process_event("resp_1", delta_event)
        done_events = helper.process_event("resp_1", done_event)
        text_tail, transcript_tail = helper.consume_done_tails("resp_1")

    assert streamed_events == [delta_event]
    assert delta_event.delta == "masked"
    assert done_events == [done_event]
    assert text_tail == "tail"
    assert transcript_tail == ""


def test_realtime_processor_appends_done_tail_without_extra_public_events():
    exporter, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    state = RealtimeSessionState(tracer, "gpt-test")
    processor = RealtimeEventProcessor(state)
    processor.start_response_span()
    processor.expand_and_process_event(
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        start_span_if_none=False,
    )
    delta_event = SimpleNamespace(type="response.text.delta", delta="secret")
    done_event = SimpleNamespace(
        type="response.done",
        response=SimpleNamespace(id="resp_1", usage=None),
    )

    assert processor.expand_and_process_event(delta_event, start_span_if_none=False) == [
        delta_event
    ]
    assert processor.expand_and_process_event(done_event, start_span_if_none=False) == [
        done_event
    ]

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.completion.0.content"] == "maskedtail"


def test_assistant_instruction_and_message_prompt_safety_cover_noop_kwargs_and_args():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            f"[PII:{context.text}]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )

    kwargs = {"instructions": "secret"}
    updated_kwargs = apply_assistant_instruction_prompt_safety(None, kwargs)
    untouched_kwargs = apply_assistant_instruction_prompt_safety(None, {"instructions": 123})

    args_result, kwargs_result = apply_assistant_message_prompt_safety(
        None,
        (),
        {"role": "user", "content": "secret"},
    )
    args_from_positional, kwargs_from_positional = apply_assistant_message_prompt_safety(
        None,
        ("secret",),
        {"role": "user"},
    )
    untouched_args, untouched_message_kwargs = apply_assistant_message_prompt_safety(
        None,
        (),
        {},
    )

    assert updated_kwargs["instructions"] == "[PII:secret]"
    assert untouched_kwargs == {"instructions": 123}
    assert args_result == ()
    assert kwargs_result["content"] == "[PII:secret]"
    assert args_from_positional == ("[PII:secret]",)
    assert kwargs_from_positional == {"role": "user"}
    assert untouched_args == ()
    assert untouched_message_kwargs == {}


def test_assistant_content_helpers_cover_list_blocks_and_fallback_setters():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            f"[PII:{context.text}]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: _mask_result(
            f"[SECRET:{context.text}]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    prompt_content, prompt_changed = _mask_assistant_content(
        None,
        ["secret", {"text": {"value": "secret-2"}}, {"ignore": "x"}],
        location="prompt",
        segment_index=0,
        segment_role="user",
    )
    completion_content, completion_changed = _mask_assistant_content(
        None,
        [{"text": "secret-3"}],
        location="completion",
        segment_index=1,
        segment_role="assistant",
    )
    unchanged_content, unchanged = _mask_assistant_content(
        None,
        {"not": "a-list"},
        location="prompt",
        segment_index=0,
        segment_role="user",
    )

    block = {}
    _set_assistant_block_text(block, "masked")
    string_block = {"text": "raw"}
    _set_assistant_block_text(string_block, "masked-2")

    assert prompt_changed is True
    assert prompt_content[0] == "[PII:secret]"
    assert prompt_content[1]["text"]["value"] == "[PII:secret-2]"
    assert completion_changed is True
    assert completion_content[0]["text"] == "[SECRET:secret-3]"
    assert unchanged is False
    assert unchanged_content == {"not": "a-list"}
    assert _assistant_block_text({"text": {"value": "x"}}) == "x"
    assert _assistant_block_text({"text": "y"}) == "y"
    assert _assistant_block_text({}) is None
    assert block["text"] == "masked"
    assert string_block["text"] == "masked-2"


def test_assistant_messages_list_and_streaming_safety_cover_accumulated_and_fallback_paths():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            f"[PII:{context.text}]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: _mask_result(
            f"[SECRET:{context.text}]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )

    response = SimpleNamespace(
        data=[
            SimpleNamespace(role="user", content=["secret"]),
            SimpleNamespace(role="assistant", content=[{"text": "secret"}]),
        ]
    )
    apply_assistant_messages_list_safety(None, response)

    with _test_span()[1].start_as_current_span("openai.chat") as span:
        helper = AssistantStreamingSafety(span)
        assert helper.process_text_delta(SimpleNamespace(value=None), None, text_index=0) == ""
        delta = SimpleNamespace(value="secret")
        snapshot = SimpleNamespace(value="")
        assert helper.process_text_delta(delta, snapshot, text_index=0) == "masked"
        text = SimpleNamespace(value=None)
        assert helper.flush_text(text, text_index=0) == "tail"
        message = SimpleNamespace(content=[SimpleNamespace(text=SimpleNamespace(value="secret"))])
        helper.apply_message_safety(message, text_index=0)
        fallback_message = SimpleNamespace(
            content=[SimpleNamespace(text=SimpleNamespace(value="secret-2"))]
        )
        helper.apply_message_safety(fallback_message, text_index=1)
        helper.apply_message_safety(SimpleNamespace(content=None), text_index=1)

    assert response.data[0].content[0] == "[PII:secret]"
    assert response.data[1].content[0]["text"] == "[SECRET:secret]"
    assert delta.value == "masked"
    assert snapshot.value == "masked"
    assert text.value == "maskedtail"
    assert message.content[0].text.value == "maskedtail"
    assert fallback_message.content[0].text.value == "[SECRET:secret-2]"


def test_response_helper_units_cover_prompt_and_completion_branches():
    register_prompt_safety_handler(
        lambda context: _mask_result(
            f"[PII:{context.text}]",
            "PII",
            "PII.email",
            SafetyLocation.PROMPT,
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: _mask_result(
            f"[SECRET:{context.text}]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    masked_string_input, masked_string_changed = _mask_response_input(
        None, "secret", start_index=2
    )
    unchanged_input, unchanged_input_changed = _mask_response_input(
        None, {"not": "supported"}, start_index=0
    )
    masked_item, item_changed = _mask_response_input_item(
        None,
        {
            "role": "assistant",
            "content": ["secret", {"text": "secret-2"}, {"ignore": "x"}],
            "text": "secret-3",
            "output": "secret-4",
        },
        segment_index=1,
    )
    masked_blocks, blocks_changed = _mask_prompt_content_blocks(
        None,
        ["secret", {"text": "secret-2"}, {"ignore": "x"}],
        segment_index=0,
        segment_role="user",
    )
    unchanged_blocks, unchanged_blocks_changed = _mask_prompt_content_blocks(
        None,
        {"not": "list"},
        segment_index=0,
        segment_role="user",
    )
    masked_output_content, aggregated_output, output_changed = _mask_response_output_content(
        None,
        [{"text": "secret"}, {"text": "secret-2"}, {"ignore": "x"}],
        output_index=0,
    )
    passthrough_output, empty_text, passthrough_changed = _mask_response_output_content(
        None,
        "not-a-list",
        output_index=0,
    )
    output_text_only = SimpleNamespace(output=None, output_text="secret")
    no_output_text = SimpleNamespace(output=[SimpleNamespace(type="tool_call")], output_text=None)

    assert masked_string_changed is True
    assert masked_string_input == "[PII:secret]"
    assert unchanged_input == {"not": "supported"}
    assert unchanged_input_changed is False
    assert item_changed is True
    assert masked_item["content"][0] == "[PII:secret]"
    assert masked_item["content"][1]["text"] == "[PII:secret-2]"
    assert masked_item["text"] == "[PII:secret-3]"
    assert masked_item["output"] == "[PII:secret-4]"
    assert blocks_changed is True
    assert masked_blocks[0] == "[PII:secret]"
    assert masked_blocks[1]["text"] == "[PII:secret-2]"
    assert unchanged_blocks == {"not": "list"}
    assert unchanged_blocks_changed is False
    assert output_changed is True
    assert aggregated_output == "[SECRET:secret][SECRET:secret-2]"
    assert masked_output_content[0]["text"] == "[SECRET:secret]"
    assert passthrough_output == "not-a-list"
    assert empty_text == ""
    assert passthrough_changed is False
    assert apply_response_completion_safety(None, output_text_only) == "[SECRET:secret]"
    assert output_text_only.output_text == "[SECRET:secret]"
    assert apply_response_completion_safety(None, no_output_text) is None


def test_response_streaming_safety_helper_units_cover_object_delta_and_completed_sync():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a", "masked-b"], flush_result="tail")
    )
    register_completion_safety_handler(
        lambda context: _mask_result(
            f"[SECRET:{context.text}]",
            "SECRET",
            "SECRET.api_key",
            SafetyLocation.COMPLETION,
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    with _test_span()[1].start_as_current_span("openai.response") as span:
        helper = ResponsesStreamingSafety(span)
        delta_object_chunk = SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=1,
            delta=SimpleNamespace(text="secret"),
        )
        helper.process_chunk(delta_object_chunk)
        done_chunk = SimpleNamespace(
            type="response.output_text.done",
            output_index=0,
            content_index=1,
            text="secret",
        )
        helper.process_chunk(done_chunk)
        helper.process_chunk(SimpleNamespace(type="response.completed", response=None))
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(text="masked-a"), SimpleNamespace(text="masked-btail")],
                )
            ],
            output_text="masked-a[stale]",
        )
        helper._sync_aggregated_text_from_response(response, "")

    assert delta_object_chunk.delta.text == "masked-a"
    assert done_chunk.delta.text == "tail"
    assert done_chunk.text == "masked-atail"
    assert helper.aggregated_text() == "masked-amasked-btail"
    assert _response_output_key(SimpleNamespace(output_index=None, content_index=None)) == (0, 0)


def test_response_streaming_safety_flush_text_collects_pending_tails_once():
    class _OneShotFlushSession:
        def __init__(self):
            self._flushed = False

        def process_chunk(self, text):
            return SafetyResult(text="masked", overall_action="allow", findings=[])

        def flush(self):
            if self._flushed:
                return SafetyResult(text="", overall_action="allow", findings=[])
            self._flushed = True
            return SafetyResult(text="tail", overall_action="allow", findings=[])

    clear_completion_safety_stream_factory()
    register_completion_safety_stream_factory(lambda _: _OneShotFlushSession())

    with _test_span()[1].start_as_current_span("openai.response") as span:
        helper = ResponsesStreamingSafety(span)
        helper.process_chunk(
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="secret",
            )
        )
        assert helper.flush_text() == "maskedtail"


def test_response_stream_process_chunk_does_not_double_count_object_delta():
    """Verify that _process_chunk does not double-count text when chunk.delta has a .text attribute."""
    _, tracer = _test_span()

    stream = ResponseStream(
        span=tracer.start_span("openai.response"),
        response=_EmptyStream(),
        start_time=1,
        request_kwargs={"model": "gpt-test", "input": "hello"},
        tracer=tracer,
    )

    # Simulate a chunk with an object delta that has a .text attribute.
    # This exercises the path where chunk.type != "response.output_text.delta"
    # but chunk.delta.text is present. Only one path should fire.
    chunk = SimpleNamespace(
        type="response.content_part.delta",
        delta=SimpleNamespace(text="hello"),
        response=None,
    )
    stream._process_chunk(chunk)

    # Text should appear exactly once, not double-counted
    assert stream._output_text == "hello"
