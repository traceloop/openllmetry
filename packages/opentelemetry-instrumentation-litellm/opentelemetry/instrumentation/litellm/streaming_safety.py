from __future__ import annotations

import inspect
from types import SimpleNamespace

from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.litellm.safety import PROVIDER, extract_text_content
from opentelemetry.semconv_ai import SpanAttributes


def is_sync_streaming_response(kwargs, response) -> bool:
    """Check if this is a streaming response."""
    if kwargs.get("stream"):
        return not inspect.iscoroutine(response) and not inspect.isasyncgen(response)
    return False


def is_async_streaming_response(kwargs, response) -> bool:
    """Check if this is an async streaming response."""
    if kwargs.get("stream"):
        return inspect.iscoroutine(response) or inspect.isasyncgen(response)
    return False


def wrap_sync_streaming_response(span, response, request_type, span_name, set_response_attributes):
    streams = CompletionTextStreamGroup(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        request_type=request_type,
    )
    complete_response = {"choices": [], "model": None, "usage": None}

    try:
        for chunk in response:
            _mask_streaming_chunk(streams, chunk)
            _accumulate_streaming_chunk(complete_response, chunk)
            yield chunk
        _finalize_streaming_span(span, complete_response, set_response_attributes)
    except Exception as exc:
        _close_streaming_response(response)
        _record_span_error(span, exc)
        raise
    finally:
        if span.is_recording():
            span.end()


async def wrap_async_streaming_response(
    span,
    response,
    request_type,
    span_name,
    set_response_attributes,
):
    streams = CompletionTextStreamGroup(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        request_type=request_type,
    )
    complete_response = {"choices": [], "model": None, "usage": None}

    try:
        async for chunk in response:
            _mask_streaming_chunk(streams, chunk)
            _accumulate_streaming_chunk(complete_response, chunk)
            yield chunk
        _finalize_streaming_span(span, complete_response, set_response_attributes)
    except Exception as exc:
        await _aclose_streaming_response(response)
        _record_span_error(span, exc)
        raise
    finally:
        if span.is_recording():
            span.end()


def _mask_streaming_chunk(streams, chunk):
    for index, choice in enumerate(get_object_value(chunk, "choices") or []):
        finish_reason = get_object_value(choice, "finish_reason")
        message = get_object_value(choice, "message")
        content = extract_text_content(get_object_value(message, "content")) if message is not None else None
        if isinstance(content, str):
            masked = streams.process(
                ("choice", index),
                content,
                segment_index=index,
                segment_role="assistant",
            )
            set_object_value(message, "content", masked)
            if get_object_value(choice, "text") is not None:
                set_object_value(choice, "text", masked)
        else:
            text = get_object_value(choice, "text")
            if isinstance(text, str):
                masked = streams.process(
                    ("choice", index),
                    text,
                    segment_index=index,
                    segment_role="assistant",
                )
                set_object_value(choice, "text", masked)
        if finish_reason:
            tail = streams.flush(("choice", index))
            if tail:
                if message is not None:
                    current_content = extract_text_content(get_object_value(message, "content"))
                    if current_content is None:
                        current_content = ""
                    set_object_value(message, "content", f"{current_content}{tail}")
                if message is None or get_object_value(choice, "text") is not None:
                    current_text = get_object_value(choice, "text") or ""
                    set_object_value(choice, "text", f"{current_text}{tail}")


def _accumulate_streaming_chunk(complete_response, chunk):
    model = get_object_value(chunk, "model")
    if model is not None:
        complete_response["model"] = model
    usage = get_object_value(chunk, "usage")
    if usage is not None:
        complete_response["usage"] = usage

    for index, choice in enumerate(get_object_value(chunk, "choices") or []):
        while len(complete_response["choices"]) <= index:
            complete_response["choices"].append(
                {"message": {"role": "assistant", "content": ""}, "text": ""}
            )
        aggregate = complete_response["choices"][index]
        finish_reason = get_object_value(choice, "finish_reason")
        if finish_reason is not None:
            aggregate["finish_reason"] = finish_reason

        message = get_object_value(choice, "message")
        if message is not None:
            role = get_object_value(message, "role")
            if role is not None:
                aggregate["message"]["role"] = role
            content = extract_text_content(get_object_value(message, "content"))
            if isinstance(content, str):
                aggregate["message"]["content"] += content
                aggregate["text"] += content
                continue

        text = get_object_value(choice, "text")
        if isinstance(text, str):
            aggregate["text"] += text


def _finalize_streaming_span(span, complete_response, set_response_attributes):
    response = SimpleNamespace(
        model=complete_response["model"],
        usage=complete_response["usage"],
        choices=[
            SimpleNamespace(
                finish_reason=choice.get("finish_reason"),
                message=SimpleNamespace(
                    role=get_object_value(choice.get("message"), "role") or "assistant",
                    content=get_object_value(choice.get("message"), "content"),
                ),
                text=choice.get("text"),
            )
            for choice in complete_response["choices"]
        ],
    )
    set_response_attributes(span, response)
    span.set_status(Status(StatusCode.OK))


def _record_span_error(span, exc):
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _close_streaming_response(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


async def _aclose_streaming_response(response) -> None:
    aclose = getattr(response, "aclose", None)
    if callable(aclose):
        await aclose()
