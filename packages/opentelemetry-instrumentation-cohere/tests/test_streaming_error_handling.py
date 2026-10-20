import pytest
from opentelemetry.instrumentation.cohere.streaming import aprocess_chat_v2_streaming_response
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.trace import StatusCode


@pytest.mark.asyncio
async def test_async_v2_streaming_exception_ends_span(span_exporter, tracer_provider):
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("cohere.chat")

    async def failing_stream():
        yield {"type": "message-start", "delta": {"message": {"role": "assistant"}}, "id": "response-id"}
        raise RuntimeError("cohere stream failure")

    with pytest.raises(RuntimeError, match="cohere stream failure"):
        async for _ in aprocess_chat_v2_streaming_response(
            span, None, LLMRequestTypeValues.CHAT, failing_stream()
        ):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert "cohere stream failure" in spans[0].status.description
    assert any(
        event.name == "exception"
        and "cohere stream failure" in event.attributes.get("exception.message", "")
        for event in spans[0].events
    )
