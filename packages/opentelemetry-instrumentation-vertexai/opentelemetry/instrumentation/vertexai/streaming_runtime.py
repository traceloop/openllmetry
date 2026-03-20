from __future__ import annotations

from opentelemetry.instrumentation.vertexai.streaming_safety import (
    build_async_streaming_response,
    build_streaming_response,
)
from opentelemetry.trace.status import Status, StatusCode


def build_streaming_response_delegate(span, event_logger, response, llm_model):
    from opentelemetry.instrumentation.vertexai import handle_streaming_response

    finalize_response = lambda complete_response, token_usage, llm_model: (
        handle_streaming_response(
            span,
            event_logger,
            llm_model,
            complete_response,
            token_usage,
        ),
        span.set_status(Status(StatusCode.OK)),
        span.end(),
    )
    yield from build_streaming_response(
        response,
        span=span,
        llm_model=llm_model,
        finalize_response=finalize_response,
    )


async def build_async_streaming_response_delegate(span, event_logger, response, llm_model):
    from opentelemetry.instrumentation.vertexai import handle_streaming_response

    finalize_response = lambda complete_response, token_usage, llm_model: (
        handle_streaming_response(
            span,
            event_logger,
            llm_model,
            complete_response,
            token_usage,
        ),
        span.set_status(Status(StatusCode.OK)),
        span.end(),
    )
    async for item in build_async_streaming_response(
        response,
        span=span,
        llm_model=llm_model,
        finalize_response=finalize_response,
    ):
        yield item
