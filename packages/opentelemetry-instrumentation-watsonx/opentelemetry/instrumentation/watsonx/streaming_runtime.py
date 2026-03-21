from __future__ import annotations

import time

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.watsonx.streaming_safety import (
    build_streaming_response,
)


def build_and_set_stream_response_delegate(
    span,
    event_logger,
    response,
    raw_flag,
    token_histogram,
    response_counter,
    duration_histogram,
    start_time,
):
    from opentelemetry.instrumentation.watsonx import (
        _handle_stream_response,
        _metric_shared_attributes,
    )

    def finalize_response(stream_state):
        stream_model_id = stream_state["model_id"]
        stream_generated_text = stream_state["generated_text"]
        stream_stop_reason = stream_state["stop_reason"]
        stream_generated_token_count = stream_state["generated_token_count"]
        stream_input_token_count = stream_state["input_token_count"]

        shared_attributes = _metric_shared_attributes(
            response_model=stream_model_id,
            is_streaming=True,
        )
        stream_response = {
            "model_id": stream_model_id,
            "generated_text": stream_generated_text,
            "generated_token_count": stream_generated_token_count,
            "input_token_count": stream_input_token_count,
        }
        _handle_stream_response(
            span, event_logger, stream_response, stream_generated_text, stream_stop_reason
        )
        if response_counter:
            response_counter.add(
                1,
                attributes={
                    **shared_attributes,
                    SpanAttributes.LLM_RESPONSE_STOP_REASON: stream_stop_reason,
                },
            )
        if token_histogram:
            token_histogram.record(
                stream_generated_token_count,
                attributes={
                    **shared_attributes,
                    GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                },
            )
            token_histogram.record(
                stream_input_token_count,
                attributes={
                    **shared_attributes,
                    GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                },
            )
        duration = (
            time.time() - start_time
            if start_time and isinstance(start_time, (float, int))
            else None
        )
        if duration and isinstance(duration, (float, int)) and duration_histogram:
            duration_histogram.record(duration, attributes=shared_attributes)
        span.set_status(Status(StatusCode.OK))
        span.end()

    yield from build_streaming_response(
        response,
        span=span,
        raw_flag=raw_flag,
        finalize_response=finalize_response,
        span_name=getattr(span, "name", "watsonx.generate_text_stream"),
    )
