from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.anthropic import safety
from opentelemetry.instrumentation.anthropic.streaming_safety import (
    AnthropicStreamingSafety,
)
from opentelemetry.instrumentation.anthropic.streaming import (
    AnthropicAsyncStream,
    AnthropicStream,
)
from opentelemetry.instrumentation.fortifyroot import SafetyDecision, SafetyResult
from opentelemetry.instrumentation.fortifyroot import (
    clear_safety_handlers,
    register_completion_safety_stream_factory,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.fr


def test_apply_prompt_safety_masks_prompt_system_and_messages(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))

    kwargs = {
        "prompt": "secret",
        "system": [{"type": "text", "text": "sys-secret"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "msg-secret"}]}],
    }
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.chat")

    assert kwargs["prompt"] == "secret"
    assert updated["prompt"] == "masked:secret"
    assert updated["system"][0]["text"] == "masked:sys-secret"
    assert updated["messages"][0]["content"][0]["text"] == "masked:msg-secret"


def test_apply_prompt_safety_returns_partial_update_when_messages_missing():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    kwargs = {"prompt": "secret"}
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.completion")
    assert updated is not kwargs
    assert updated["prompt"] == "masked:secret"
    monkeypatch.undo()


def test_apply_completion_safety_masks_completion_and_content(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    response = SimpleNamespace(
        completion="secret",
        content=[
            {"type": "text", "text": "text-secret"},
            {"type": "thinking", "thinking": "thought-secret"},
            {"type": "tool_use", "name": "ignored"},
        ],
    )

    safety._apply_completion_safety(None, response, "anthropic.chat")

    assert response.completion == "masked:secret"
    assert response.content[0]["text"] == "masked:text-secret"
    assert response.content[1]["thinking"] == "masked:thought-secret"


def test_anthropic_prompt_and_completion_helpers_cover_noop_branches(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    updated, changed = safety._mask_prompt_content(
        None,
        [{"type": "tool_use", "text": "ignored"}, {"type": "text", "text": 1}],
        span_name="anthropic.chat",
        request_type="chat",
        segment_index=0,
        segment_role="user",
    )
    assert changed is False
    assert updated[0]["text"] == "ignored"

    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    response = SimpleNamespace(completion="keep", content=[{"type": "text", "text": 1}, {"type": "tool_use", "name": "ignored"}])
    safety._apply_completion_safety(None, response, "anthropic.chat")
    assert response.completion == "keep"

    response = SimpleNamespace(completion="keep", content="not-a-list")
    assert safety._apply_completion_safety(None, response, "anthropic.chat") is None


def test_anthropic_message_only_change_path(monkeypatch):
    def _prompt(**kwargs):
        return SafetyResult(text=kwargs["text"], overall_action="MASK") if kwargs["text"] == "keep" else SafetyResult(text="masked:secret", overall_action="MASK")

    monkeypatch.setattr(safety, "run_prompt_safety", _prompt)
    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.chat")
    assert updated["messages"][0]["content"] == "masked:secret"


def test_anthropic_request_type_and_resolve_masked_text():
    assert safety._request_type("anthropic.completion") == "completion"
    assert safety._request_type("anthropic.chat") == "chat"
    assert safety._resolve_masked_text("x", None) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="x", overall_action=SafetyDecision.MASK.value),
    ) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="y", overall_action=SafetyDecision.ALLOW.value),
    ) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="y", overall_action=SafetyDecision.MASK.value),
    ) == ("y", True)


def test_anthropic_fail_opens_on_internal_error(monkeypatch):
    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    monkeypatch.setattr(safety, "_request_type", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_prompt_safety(None, kwargs, "anthropic.chat") is kwargs

    response = SimpleNamespace(completion="secret")
    monkeypatch.setattr(safety, "_request_type", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_completion_safety(None, response, "anthropic.chat") is None


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


class _Iterator:
    def __init__(self, items):
        self._items = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._items)


class _AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _test_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_anthropic_stream_masks_text_delta_and_flushes_on_block_stop():
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    exporter, tracer = _test_tracer()

    items = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="secret"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
    ]

    with tracer.start_as_current_span("anthropic.chat") as span:
        stream = AnthropicStream(
            span,
            _Iterator(items),
            SimpleNamespace(count_tokens=lambda text: len(text)),
            0.0,
            kwargs={},
        )
        first = next(stream)
        second = next(stream)
        with pytest.raises(StopIteration):
            next(stream)

    assert first.delta.text == "masked-tail"
    assert second.type == "content_block_stop"
    assert exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_anthropic_async_stream_masks_text_delta_and_flushes_on_block_stop():
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    exporter, tracer = _test_tracer()

    items = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="secret"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
    ]

    with tracer.start_as_current_span("anthropic.chat") as span:
        stream = AnthropicAsyncStream(
            span,
            _AsyncIterator(items),
            SimpleNamespace(count_tokens=lambda text: len(text)),
            0.0,
            kwargs={},
        )
        first = await stream.__anext__()
        second = await stream.__anext__()
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    assert first.delta.text == "masked-tail"
    assert second.type == "content_block_stop"
    assert exporter.get_finished_spans()


def test_anthropic_streaming_safety_handles_thinking_and_noop_paths():
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    _, tracer = _test_tracer()

    with tracer.start_as_current_span("anthropic.chat") as span:
        helper = AnthropicStreamingSafety(span, "anthropic.chat")

        thinking_item = SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="thinking_delta", thinking="secret"),
        )
        helper.process_item(thinking_item)
        helper.flush_transition(thinking_item, SimpleNamespace(type="message_stop"))
        helper.flush_pending_item(None)
        noop = helper.process_item(SimpleNamespace(type="message_start"))

    assert thinking_item.delta.thinking == "masked-tail"
    assert noop.type == "message_start"


def test_anthropic_streaming_safety_covers_unknown_and_mismatch_branches():
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    _, tracer = _test_tracer()

    with tracer.start_as_current_span("anthropic.chat") as span:
        helper = AnthropicStreamingSafety(span, "anthropic.chat")
        unknown_delta = SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="tool_delta"),
        )
        no_text = SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text=None),
        )
        pending = SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="secret"),
        )
        helper.process_item(pending)
        helper.flush_transition(pending, SimpleNamespace(type="message_start"))
        helper.flush_pending_item(SimpleNamespace(type="message_start"))
        helper._append_tail(
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text="raw"),
            ),
            (0, "assistant"),
            "tail",
        )

    assert helper.process_item(unknown_delta) is unknown_delta
    assert helper.process_item(no_text) is no_text
    assert pending.delta.text == "masked"


class _ErrorIterator:
    """Iterator that raises a RuntimeError on the first call to __next__."""
    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("stream error")


def test_anthropic_sync_stream_ends_span_on_error():
    """Verify that AnthropicStream.__next__ sets error status and ends the span on exception."""
    clear_safety_handlers()
    exporter, tracer = _test_tracer()

    with tracer.start_as_current_span("anthropic.chat") as span:
        stream = AnthropicStream(
            span,
            _ErrorIterator(),
            SimpleNamespace(count_tokens=lambda text: len(text)),
            0.0,
            kwargs={},
        )
        with pytest.raises(RuntimeError, match="stream error"):
            next(stream)

    finished_spans = exporter.get_finished_spans()
    assert len(finished_spans) == 1
    assert finished_spans[0].status.status_code.name == "ERROR"
    assert "stream error" in finished_spans[0].status.description
    assert stream._instrumentation_completed is True


def test_anthropic_sync_stream_span_not_ended_before_last_item_consumed():
    """D-13: Span must not be ended until the consumer has processed the last pending item.
    When StopIteration is caught with a pending item, the item is returned WITHOUT
    calling _complete_instrumentation(). On the NEXT __next__ call, instrumentation completes."""
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    exporter, tracer = _test_tracer()

    items = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="hello"),
        ),
    ]

    span = tracer.start_span("anthropic.chat")
    stream = AnthropicStream(
        span,
        _Iterator(items),
        SimpleNamespace(count_tokens=lambda text: len(text)),
        0.0,
        kwargs={},
    )

    # First call returns the pending item (flushed from the stream end)
    last_item = next(stream)
    assert last_item.type == "content_block_delta"
    # At this point the span must NOT be ended yet -- the consumer hasn't processed the item
    assert not stream._instrumentation_completed
    assert len(exporter.get_finished_spans()) == 0

    # Second call triggers StopIteration and completes instrumentation
    with pytest.raises(StopIteration):
        next(stream)

    assert stream._instrumentation_completed
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_anthropic_async_stream_span_not_ended_before_last_item_consumed():
    """D-13: Async variant - span must not be ended until the consumer has processed the last pending item."""
    clear_safety_handlers()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    exporter, tracer = _test_tracer()

    items = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="hello"),
        ),
    ]

    span = tracer.start_span("anthropic.chat")
    stream = AnthropicAsyncStream(
        span,
        _AsyncIterator(items),
        SimpleNamespace(count_tokens=lambda text: len(text)),
        0.0,
        kwargs={},
    )

    last_item = await stream.__anext__()
    assert last_item.type == "content_block_delta"
    assert not stream._instrumentation_completed
    assert len(exporter.get_finished_spans()) == 0

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()

    assert stream._instrumentation_completed
    assert len(exporter.get_finished_spans()) == 1
