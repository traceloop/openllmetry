from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.together.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.trace.status import Status, StatusCode


class TogetherStreamingSafety:
    def __init__(self, *, span, llm_request_type, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=llm_request_type.value,
        )

    def process_chunk(self, chunk):
        for choice in get_object_value(chunk, "choices") or []:
            index = get_object_value(choice, "index", 0) or 0
            delta = get_object_value(choice, "delta")
            if delta is None:
                continue
            text = get_object_value(delta, "content")
            if not isinstance(text, str):
                continue
            masked = self._streams.process(
                ("choice", index),
                text,
                segment_index=index,
                segment_role="assistant",
            )
            set_object_value(delta, "content", masked)
            if get_object_value(choice, "finish_reason"):
                tail = self._streams.flush(("choice", index))
                if tail:
                    set_object_value(delta, "content", f"{masked}{tail}")
        return chunk

    def flush_pending_chunk(self, chunk):
        for choice in get_object_value(chunk, "choices") or []:
            index = get_object_value(choice, "index", 0) or 0
            tail = self._streams.flush(("choice", index))
            if not tail:
                continue
            delta = get_object_value(choice, "delta")
            if delta is None:
                continue
            current_text = get_object_value(delta, "content") or ""
            set_object_value(delta, "content", f"{current_text}{tail}")
        return chunk


def build_streaming_response(
    response,
    *,
    span,
    event_logger,
    llm_request_type,
    span_name,
    handle_response,
) -> Iterator:
    processor = _TogetherStreamProcessor(
        span=span,
        llm_request_type=llm_request_type,
        span_name=span_name,
        event_logger=event_logger,
        handle_response=handle_response,
    )

    try:
        for chunk in response:
            yielded = processor.process_chunk(chunk)
            if yielded is not None:
                yield yielded

        final_chunk = processor.finish()
        if final_chunk is not None:
            yield final_chunk
        processor.finalize()
    except GeneratorExit:
        if span.is_recording():
            span.end()
        raise
    except BaseException:
        if span.is_recording():
            span.end()
        raise


async def build_async_streaming_response(
    response,
    *,
    span,
    event_logger,
    llm_request_type,
    span_name,
    handle_response,
) -> AsyncIterator:
    processor = _TogetherStreamProcessor(
        span=span,
        llm_request_type=llm_request_type,
        span_name=span_name,
        event_logger=event_logger,
        handle_response=handle_response,
    )

    try:
        async for chunk in response:
            yielded = processor.process_chunk(chunk)
            if yielded is not None:
                yield yielded

        final_chunk = processor.finish()
        if final_chunk is not None:
            yield final_chunk
        processor.finalize()
    except GeneratorExit:
        if span.is_recording():
            span.end()
        raise
    except BaseException:
        if span.is_recording():
            span.end()
        raise


class _TogetherStreamingState:
    def __init__(self, llm_request_type):
        self._llm_request_type = llm_request_type
        self._response_id = None
        self._response_model = None
        self._usage = None
        self._choices: dict[int, dict[str, str | int | None]] = {}

    def add_chunk(self, chunk):
        self._response_id = get_object_value(chunk, "id") or self._response_id
        self._response_model = get_object_value(chunk, "model") or self._response_model
        usage = get_object_value(chunk, "usage")
        if usage is not None:
            self._usage = usage

        for choice in get_object_value(chunk, "choices") or []:
            index = get_object_value(choice, "index", 0) or 0
            entry = self._choices.setdefault(
                index,
                {"index": index, "text": "", "finish_reason": None},
            )
            delta = get_object_value(choice, "delta")
            delta_text = get_object_value(delta, "content") if delta is not None else None
            if isinstance(delta_text, str):
                entry["text"] = f"{entry['text']}{delta_text}"
            finish_reason = get_object_value(choice, "finish_reason")
            if finish_reason:
                entry["finish_reason"] = finish_reason

    def build_response(self):
        choices = []
        for index in sorted(self._choices):
            entry = self._choices[index]
            if self._llm_request_type == LLMRequestTypeValues.CHAT:
                choices.append(
                    SimpleNamespace(
                        index=index,
                        finish_reason=entry["finish_reason"],
                        message=SimpleNamespace(
                            content=entry["text"],
                            role="assistant",
                        ),
                    )
                )
            else:
                choices.append(
                    SimpleNamespace(
                        index=index,
                        finish_reason=entry["finish_reason"],
                        text=entry["text"],
                    )
                )

        return SimpleNamespace(
            id=self._response_id,
            model=self._response_model,
            usage=self._usage,
            choices=choices,
        )


class _TogetherStreamProcessor:
    def __init__(self, *, span, llm_request_type, span_name, event_logger, handle_response):
        self._span = span
        self._event_logger = event_logger
        self._llm_request_type = llm_request_type
        self._handle_response = handle_response
        self._tracker = _TogetherStreamingState(llm_request_type)
        self._streaming_safety = TogetherStreamingSafety(
            span=span,
            llm_request_type=llm_request_type,
            span_name=span_name,
        )
        self._pending_chunk = None

    def process_chunk(self, chunk):
        chunk = self._streaming_safety.process_chunk(chunk)
        if self._pending_chunk is None:
            self._pending_chunk = chunk
            return None

        yielded = self._pending_chunk
        self._tracker.add_chunk(yielded)
        self._pending_chunk = chunk
        return yielded

    def finish(self):
        if self._pending_chunk is None:
            return None
        self._streaming_safety.flush_pending_chunk(self._pending_chunk)
        self._tracker.add_chunk(self._pending_chunk)
        chunk = self._pending_chunk
        self._pending_chunk = None
        return chunk

    def finalize(self):
        _finalize_stream(
            self._span,
            self._event_logger,
            self._llm_request_type,
            self._tracker,
            self._handle_response,
        )


def _finalize_stream(span, event_logger, llm_request_type, tracker, handle_response):
    handle_response(span, event_logger, llm_request_type, tracker.build_response())
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()
