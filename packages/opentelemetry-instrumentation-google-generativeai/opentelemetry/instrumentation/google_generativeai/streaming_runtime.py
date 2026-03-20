from __future__ import annotations

from opentelemetry.instrumentation.google_generativeai.streaming_safety import (
    build_async_streaming_response,
    build_streaming_response,
)


def build_streaming_response_delegate(span, response, llm_model, event_logger, token_histogram):
    from opentelemetry.instrumentation.google_generativeai import (
        emit_choice_events,
        set_model_response_attributes,
        set_response_attributes,
        should_emit_events,
    )

    finalize_response = lambda complete_response, final_chunk, llm_model: (
        emit_choice_events(response, event_logger)
        if should_emit_events() and event_logger
        else set_response_attributes(span, complete_response, llm_model),
        set_model_response_attributes(
            span,
            final_chunk,
            llm_model,
            token_histogram,
        ),
        span.end(),
    )
    yield from build_streaming_response(
        response,
        span=span,
        llm_model=llm_model,
        finalize_response=finalize_response,
    )


async def build_async_streaming_response_delegate(
    span, response, llm_model, event_logger, token_histogram
):
    from opentelemetry.instrumentation.google_generativeai import (
        emit_choice_events,
        set_model_response_attributes,
        set_response_attributes,
        should_emit_events,
    )

    finalize_response = lambda complete_response, final_chunk, llm_model: (
        emit_choice_events(response, event_logger)
        if should_emit_events() and event_logger
        else set_response_attributes(span, complete_response, llm_model),
        set_model_response_attributes(
            span,
            final_chunk,
            llm_model,
            token_histogram,
        ),
        span.end(),
    )
    async for item in build_async_streaming_response(
        response,
        span=span,
        llm_model=llm_model,
        finalize_response=finalize_response,
    ):
        yield item
