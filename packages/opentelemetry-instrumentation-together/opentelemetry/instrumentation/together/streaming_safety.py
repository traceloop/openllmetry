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
    tracker = _TogetherStreamingState(llm_request_type)
    streaming_safety = TogetherStreamingSafety(
        span=span,
        llm_request_type=llm_request_type,
        span_name=span_name,
    )
    pending_chunk = None

    for chunk in response:
        chunk = streaming_safety.process_chunk(chunk)
        if pending_chunk is not None:
            tracker.add_chunk(pending_chunk)
            yield pending_chunk
        pending_chunk = chunk

    if pending_chunk is not None:
        streaming_safety.flush_pending_chunk(pending_chunk)
        tracker.add_chunk(pending_chunk)
        yield pending_chunk

    _finalize_stream(span, event_logger, llm_request_type, tracker, handle_response)


async def build_async_streaming_response(
    response,
    *,
    span,
    event_logger,
    llm_request_type,
    span_name,
    handle_response,
) -> AsyncIterator:
    tracker = _TogetherStreamingState(llm_request_type)
    streaming_safety = TogetherStreamingSafety(
        span=span,
        llm_request_type=llm_request_type,
        span_name=span_name,
    )
    pending_chunk = None

    async for chunk in response:
        chunk = streaming_safety.process_chunk(chunk)
        if pending_chunk is not None:
            tracker.add_chunk(pending_chunk)
            yield pending_chunk
        pending_chunk = chunk

    if pending_chunk is not None:
        streaming_safety.flush_pending_chunk(pending_chunk)
        tracker.add_chunk(pending_chunk)
        yield pending_chunk

    _finalize_stream(span, event_logger, llm_request_type, tracker, handle_response)


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


def _finalize_stream(span, event_logger, llm_request_type, tracker, handle_response):
    handle_response(span, event_logger, llm_request_type, tracker.build_response())
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()
