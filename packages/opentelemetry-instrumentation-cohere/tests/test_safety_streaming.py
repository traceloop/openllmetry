from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.cohere.streaming import (
    DEFAULT_MESSAGE,
    _accumulate_stream_item,
)
from opentelemetry.instrumentation.cohere.streaming_safety import CohereStreamingSafety
from opentelemetry.instrumentation.fortifyroot import (
    clear_safety_handlers,
    register_completion_safety_stream_factory,
    SafetyResult,
)
from opentelemetry.sdk.trace import TracerProvider

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_span():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    return tracer.start_span("cohere.chat")


class _FakeStreamSession:
    def __init__(self, results=None, flush_result=""):
        self._results = list(results or [])
        self._flush_result = flush_result

    def process_chunk(self, text):
        if self._results:
            return SafetyResult(text=self._results.pop(0), overall_action="mask", findings=[])
        return SafetyResult(text=text, overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="allow", findings=[])


def test_cohere_v1_streaming_masks_text_and_flushes_tail():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    try:
        helper = CohereStreamingSafety(span, "cohere.chat", "chat")
        item = SimpleNamespace(event_type="text-generation", text="a")
        item = helper.process_v1_item(item)
        helper.flush_transition(item, SimpleNamespace(event_type="stream-end"))

        assert item.text == "masked-atail"
    finally:
        span.end()


def test_cohere_v2_streaming_masks_text_and_thinking_paths():
    register_completion_safety_stream_factory(
        lambda context: _FakeStreamSession(
            ["masked-thinking"] if context.segment_role == "thinking" else ["masked-text"],
            flush_result="tail",
        )
    )
    span = _test_span()
    try:
        helper = CohereStreamingSafety(span, "cohere.chat", "chat")
        text_item = SimpleNamespace(
            type="content-delta",
            index=1,
            delta=SimpleNamespace(
                message=SimpleNamespace(content=SimpleNamespace(text="a", thinking=None))
            ),
        )
        thinking_item = SimpleNamespace(
            type="content-delta",
            index=2,
            delta=SimpleNamespace(
                message=SimpleNamespace(content=SimpleNamespace(text=None, thinking="b"))
            ),
        )

        text_item = helper.process_v2_item(text_item)
        thinking_item = helper.process_v2_item(thinking_item)
        helper.flush_transition(thinking_item, SimpleNamespace(type="message-end"))

        assert text_item.delta.message.content.text == "masked-text"
        assert thinking_item.delta.message.content.thinking == "masked-thinkingtail"
    finally:
        span.end()


def test_cohere_streaming_safety_covers_noop_and_flush_pending_paths():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    try:
        helper = CohereStreamingSafety(span, "cohere.chat", "chat")
        noop_v1 = helper.process_v1_item(SimpleNamespace(event_type="stream-start", text="a"))
        noop_v2 = helper.process_v2_item(SimpleNamespace(type="message-start"))
        pending_item = SimpleNamespace(event_type="text-generation", text="a")
        helper.process_v1_item(pending_item)
        helper.flush_pending_item(pending_item)

        assert noop_v1.event_type == "stream-start"
        assert noop_v2.type == "message-start"
        assert pending_item.text == "masked-atail"
    finally:
        span.end()


def test_cohere_streaming_safety_covers_missing_content_and_non_terminal_transitions():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    try:
        helper = CohereStreamingSafety(span, "cohere.chat", "chat")
        missing_content = SimpleNamespace(
            type="content-delta",
            index=0,
            delta=SimpleNamespace(message=SimpleNamespace(content=None)),
        )
        no_text = SimpleNamespace(
            type="content-delta",
            index=1,
            delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace())),
        )
        pending_item = SimpleNamespace(
            type="content-delta",
            index=2,
            delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="a"))),
        )

        assert helper.process_v2_item(missing_content) is missing_content
        assert helper.process_v2_item(no_text) is no_text
        helper.process_v2_item(pending_item)
        helper.flush_transition(pending_item, SimpleNamespace(type="content-start"))
        helper.flush_pending_item(SimpleNamespace(type="message-start"))

        assert pending_item.delta.message.content.text == "masked-a"
    finally:
        span.end()


def test_default_message_not_corrupted_by_missing_message_start():
    """Verify DEFAULT_MESSAGE is not mutated when a stream skips 'message-start'."""
    import copy

    final_response = {
        "finish_reason": None,
        "message": copy.deepcopy(DEFAULT_MESSAGE),
        "usage": {},
        "id": "",
        "error": None,
    }
    current_content_item = {"type": "text", "thinking": None, "text": "hello"}
    current_tool_call_item = {
        "id": "tc_1",
        "type": "function",
        "function": {"name": "fn", "arguments": "", "description": ""},
    }

    # _accumulate_stream_item calls to_dict() which expects dict items
    # Simulate a content-end event WITHOUT a preceding message-start,
    # which appends to final_response["message"]["content"].
    content_end_item = {"type": "content-end"}
    _accumulate_stream_item(
        content_end_item, current_content_item, current_tool_call_item, final_response
    )

    # Simulate a tool-call-end event WITHOUT a preceding message-start.
    tool_call_end_item = {"type": "tool-call-end"}
    _accumulate_stream_item(
        tool_call_end_item, current_content_item, current_tool_call_item, final_response
    )

    # The deep-copied message should have accumulated data
    assert len(final_response["message"]["content"]) == 1
    assert len(final_response["message"]["tool_calls"]) == 1

    # The module-level DEFAULT_MESSAGE must remain pristine
    assert DEFAULT_MESSAGE["content"] == []
    assert DEFAULT_MESSAGE["tool_calls"] == []
