from __future__ import annotations

from opentelemetry.instrumentation.writer.streaming_safety import (
    create_async_stream_processor,
    create_stream_processor,
)


def create_stream_processor_delegate(
    response,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
):
    from opentelemetry.instrumentation.writer import (
        _handle_response,
        _update_accumulated_response,
    )

    return create_stream_processor(
        response,
        span=span,
        event_logger=event_logger,
        start_time=start_time,
        duration_histogram=duration_histogram,
        streaming_time_to_first_token=streaming_time_to_first_token,
        streaming_time_to_generate=streaming_time_to_generate,
        token_histogram=token_histogram,
        method=method,
        span_name=getattr(span, "name", "writerai.chat"),
        update_accumulated_response=_update_accumulated_response,
        handle_response=_handle_response,
    )


async def create_async_stream_processor_delegate(
    response,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
):
    from opentelemetry.instrumentation.writer import (
        _handle_response,
        _update_accumulated_response,
    )

    return await create_async_stream_processor(
        response,
        span=span,
        event_logger=event_logger,
        start_time=start_time,
        duration_histogram=duration_histogram,
        streaming_time_to_first_token=streaming_time_to_first_token,
        streaming_time_to_generate=streaming_time_to_generate,
        token_histogram=token_histogram,
        method=method,
        span_name=getattr(span, "name", "writerai.chat"),
        update_accumulated_response=_update_accumulated_response,
        handle_response=_handle_response,
    )
