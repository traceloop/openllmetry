from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_stream_factory,
)
from opentelemetry.instrumentation.writer.streaming_safety import (
    WriterStreamingSafety,
    create_async_stream_processor,
    create_stream_processor,
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
    return tracer.start_span("writerai.chat")


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


def test_writer_streaming_helper_masks_chat_and_completion_chunks():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-chat", "masked-completion"], flush_result="tail")
    )
    span = _test_span()
    try:
        helper = WriterStreamingSafety(span, "chat", "writerai.chat")
        chat_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    message=SimpleNamespace(content="a"),
                    delta=SimpleNamespace(content="a"),
                )
            ]
        )
        completion_chunk = SimpleNamespace(value="b")

        chat_chunk = helper.process_chunk(chat_chunk)
        completion_chunk = helper.process_chunk(completion_chunk)
        helper.flush_pending_chunk(completion_chunk)

        assert chat_chunk.choices[0].delta.content == "masked-chat"
        assert chat_chunk.choices[0].message.content == "masked-chat"
        assert completion_chunk.value == "masked-completiontail"
    finally:
        span.end()


def test_writer_streaming_helper_flushes_tail_into_message_content():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-chat"], flush_result="tail")
    )
    span = _test_span()
    try:
        helper = WriterStreamingSafety(span, "chat", "writerai.chat")
        helper.process_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason=None,
                        message=SimpleNamespace(content="a"),
                        delta=SimpleNamespace(content="a"),
                    )
                ]
            )
        )
        final_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content=None),
                    delta=SimpleNamespace(content=None),
                )
            ]
        )

        helper.flush_pending_chunk(final_chunk)

        assert final_chunk.choices[0].delta.content == "tail"
        assert final_chunk.choices[0].message.content == "tail"
    finally:
        span.end()


def test_writer_stream_processor_flushes_final_tail_before_completion():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    updates = []
    handled = []

    class _FakeResponse:
        def __init__(self, chunks):
            self._chunks = iter(chunks)
            self.response = SimpleNamespace(
                request=SimpleNamespace(url=SimpleNamespace(path="/v1/completions"))
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._chunks)

    response = _FakeResponse([SimpleNamespace(value="a")])
    processor = create_stream_processor(
        response,
        span=span,
        event_logger=None,
        start_time=0.0,
        duration_histogram=None,
        streaming_time_to_first_token=None,
        streaming_time_to_generate=None,
        token_histogram=None,
        method="create",
        span_name="writerai.completions",
        update_accumulated_response=lambda acc, chunk: updates.append(chunk.value),
        handle_response=lambda *args: handled.append(True),
    )

    assert [chunk.value for chunk in processor] == ["masked-atail"]
    assert updates == ["masked-atail"]
    assert handled == [True]


def test_writer_streaming_helper_skips_missing_chat_content_and_empty_tail():
    class _NoTailSession:
        def process_chunk(self, text):
            return SafetyResult(text="masked", overall_action="allow", findings=[])

        def flush(self):
            return SafetyResult(text="", overall_action="allow", findings=[])

    register_completion_safety_stream_factory(lambda _: _NoTailSession())
    span = _test_span()
    try:
        helper = WriterStreamingSafety(span, "chat", "writerai.chat")
        missing_delta_chunk = SimpleNamespace(
            choices=[SimpleNamespace(index=0, finish_reason=None, delta=None)]
        )
        non_string_delta_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content="raw"),
                    delta=SimpleNamespace(content=None),
                )
            ]
        )
        non_string_completion = SimpleNamespace(value=None)

        assert helper.process_chunk(missing_delta_chunk) is missing_delta_chunk
        assert helper.process_chunk(non_string_delta_chunk) is non_string_delta_chunk
        assert helper.process_chunk(non_string_completion) is non_string_completion
        helper.flush_pending_chunk(non_string_delta_chunk)

        assert non_string_delta_chunk.choices[0].message.content == "raw"
    finally:
        span.end()


def test_writer_stream_processor_records_metrics_and_error_path():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    ttft = []
    ttg = []
    duration = []
    handled = []

    class _Histogram:
        def __init__(self, values):
            self._values = values

        def record(self, value, attributes=None):
            self._values.append((value, dict(attributes or {})))

    class _ExplodingResponse:
        def __init__(self):
            self._chunks = iter([SimpleNamespace(value="a"), RuntimeError("boom")])
            self.response = SimpleNamespace(
                request=SimpleNamespace(url=SimpleNamespace(path="/v1/completions"))
            )

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self._chunks)
            if isinstance(item, Exception):
                raise item
            return item

    with pytest.raises(RuntimeError, match="boom"):
        list(
            create_stream_processor(
                _ExplodingResponse(),
                span=span,
                event_logger=None,
                start_time=0.0,
                duration_histogram=_Histogram(duration),
                streaming_time_to_first_token=_Histogram(ttft),
                streaming_time_to_generate=_Histogram(ttg),
                token_histogram=None,
                method="create",
                span_name="writerai.completions",
                update_accumulated_response=lambda acc, chunk: None,
                handle_response=lambda *args: handled.append(True),
            )
        )

    assert handled == [True]
    assert ttft and ttg and duration


@pytest.mark.asyncio
async def test_writer_async_stream_processor_flushes_tail_and_calls_handler():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    updates = []
    handled = []

    class _AsyncResponse:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.response = SimpleNamespace(
                request=SimpleNamespace(url=SimpleNamespace(path="/v1/completions"))
            )

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    processor = create_async_stream_processor(
        _AsyncResponse([SimpleNamespace(value="a")]),
        span=span,
        event_logger=None,
        start_time=0.0,
        duration_histogram=None,
        streaming_time_to_first_token=None,
        streaming_time_to_generate=None,
        token_histogram=None,
        method="create",
        span_name="writerai.completions",
        update_accumulated_response=lambda acc, chunk: updates.append(chunk.value),
        handle_response=lambda *args: handled.append(True),
    )

    assert [chunk.value async for chunk in processor] == ["masked-atail"]
    assert updates == ["masked-atail"]
    assert handled == [True]
