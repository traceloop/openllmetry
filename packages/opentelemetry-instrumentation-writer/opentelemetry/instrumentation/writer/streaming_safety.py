from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator

from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.writer.safety import PROVIDER
from opentelemetry.instrumentation.writer.utils import (
    initialize_accumulated_response,
    request_type_by_method,
    response_attributes,
)


class WriterStreamingSafety:
    def __init__(self, span, method: str, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=request_type_by_method(method).value,
        )

    def process_chunk(self, chunk):
        if get_object_value(chunk, "choices"):
            return self._process_chat_chunk(chunk)
        return self._process_completion_chunk(chunk)

    def flush_pending_chunk(self, chunk):
        if get_object_value(chunk, "choices"):
            for choice in get_object_value(chunk, "choices") or []:
                index = get_object_value(choice, "index", 0) or 0
                tail = self._streams.flush(("choice", index))
                if not tail:
                    continue
                self._append_chat_choice_text(choice, tail)
            return chunk

        tail = self._streams.flush(("choice", 0))
        if tail:
            current_text = get_object_value(chunk, "value") or ""
            set_object_value(chunk, "value", f"{current_text}{tail}")
        return chunk

    def _process_chat_chunk(self, chunk):
        for choice in get_object_value(chunk, "choices") or []:
            index = get_object_value(choice, "index", 0) or 0
            delta = get_object_value(choice, "delta")
            if delta is None:
                continue
            content = get_object_value(delta, "content")
            if not isinstance(content, str):
                continue
            masked = self._streams.process(
                ("choice", index),
                content,
                segment_index=index,
                segment_role="assistant",
            )
            self._set_chat_choice_text(choice, masked)
            if get_object_value(choice, "finish_reason"):
                tail = self._streams.flush(("choice", index))
                if tail:
                    self._append_chat_choice_text(choice, tail)
        return chunk

    def _process_completion_chunk(self, chunk):
        text = get_object_value(chunk, "value")
        if not isinstance(text, str):
            return chunk
        masked = self._streams.process(
            ("choice", 0),
            text,
            segment_index=0,
            segment_role="assistant",
        )
        set_object_value(chunk, "value", masked)
        return chunk

    def _set_chat_choice_text(self, choice, text: str) -> None:
        delta = get_object_value(choice, "delta")
        if delta is not None:
            set_object_value(delta, "content", text)
        message = get_object_value(choice, "message")
        if message is not None:
            set_object_value(message, "content", text)

    def _append_chat_choice_text(self, choice, tail: str) -> None:
        delta = get_object_value(choice, "delta")
        if delta is not None:
            current_text = get_object_value(delta, "content") or ""
            set_object_value(delta, "content", f"{current_text}{tail}")
        message = get_object_value(choice, "message")
        if message is not None:
            current_text = get_object_value(message, "content") or ""
            set_object_value(message, "content", f"{current_text}{tail}")


def create_stream_processor(
    response,
    *,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
    span_name,
    update_accumulated_response,
    handle_response,
) -> Iterator:
    accumulated_response = initialize_accumulated_response(response)
    first_token_time = None
    last_token_time = start_time
    error: Exception | None = None
    pending_chunk = None
    streaming_safety = WriterStreamingSafety(span, method, span_name)

    try:
        for chunk in response:
            if first_token_time is None:
                first_token_time = time.time()

            chunk = streaming_safety.process_chunk(chunk)
            if pending_chunk is None:
                pending_chunk = chunk
                continue

            update_accumulated_response(accumulated_response, pending_chunk)
            yield pending_chunk
            pending_chunk = chunk

        last_token_time = time.time()

        if pending_chunk is not None:
            streaming_safety.flush_pending_chunk(pending_chunk)
            update_accumulated_response(accumulated_response, pending_chunk)
            yield pending_chunk
    except Exception as ex:
        error = ex
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        metrics_attributes = response_attributes(accumulated_response, method) or {}
        metrics_attributes.update({"stream": True})

        if streaming_time_to_first_token:
            ttft = (first_token_time or last_token_time) - start_time
            streaming_time_to_first_token.record(ttft, attributes=metrics_attributes)

        if streaming_time_to_generate:
            streaming_time_to_generate.record(
                last_token_time - (first_token_time or last_token_time),
                attributes=metrics_attributes,
            )

        if duration_histogram:
            duration_histogram.record(
                last_token_time - start_time, attributes=metrics_attributes
            )

        handle_response(
            span, accumulated_response, token_histogram, event_logger, method
        )

        if span.is_recording() and error is None:
            span.set_status(Status(StatusCode.OK))

        span.end()


async def create_async_stream_processor(
    response,
    *,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
    span_name,
    update_accumulated_response,
    handle_response,
) -> AsyncIterator:
    accumulated_response = initialize_accumulated_response(response)
    first_token_time = None
    last_token_time = start_time
    error: Exception | None = None
    pending_chunk = None
    streaming_safety = WriterStreamingSafety(span, method, span_name)

    try:
        async for chunk in response:
            if first_token_time is None:
                first_token_time = time.time()

            chunk = streaming_safety.process_chunk(chunk)
            if pending_chunk is None:
                pending_chunk = chunk
                continue

            update_accumulated_response(accumulated_response, pending_chunk)
            yield pending_chunk
            pending_chunk = chunk

        last_token_time = time.time()

        if pending_chunk is not None:
            streaming_safety.flush_pending_chunk(pending_chunk)
            update_accumulated_response(accumulated_response, pending_chunk)
            yield pending_chunk
    except Exception as ex:
        error = ex
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        metrics_attributes = response_attributes(accumulated_response, method) or {}
        metrics_attributes.update({"stream": True})

        if streaming_time_to_first_token:
            ttft = (first_token_time or last_token_time) - start_time
            streaming_time_to_first_token.record(ttft, attributes=metrics_attributes)

        if streaming_time_to_generate:
            streaming_time_to_generate.record(
                last_token_time - (first_token_time or last_token_time),
                attributes=metrics_attributes,
            )

        if duration_histogram:
            duration_histogram.record(
                last_token_time - start_time, attributes=metrics_attributes
            )

        handle_response(
            span, accumulated_response, token_histogram, event_logger, method
        )

        if span.is_recording() and error is None:
            span.set_status(Status(StatusCode.OK))

        span.end()
