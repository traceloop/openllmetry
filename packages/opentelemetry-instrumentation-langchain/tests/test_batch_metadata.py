import pytest
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
@pytest.mark.skip(reason="VCR is not working for this test in CI - need to fix")
def test_batch_metadata_in_span_attributes(instrument_legacy, span_exporter):
    """Test that metadata from batch calls are populated as span attributes."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Test batch with metadata
    test_metadata = {
        "user_id": "12345",
        "session_id": "abc-123",
        "priority": "high"
    }
    messages_list = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "How are you?"}]
    ]

    # Call batch with metadata
    llm.batch(messages_list, config={"metadata": test_metadata})

    spans = span_exporter.get_finished_spans()

    # Find the LLM spans
    llm_spans = [span for span in spans if span.name.endswith(".chat")]

    # There should be 2 LLM spans (one for each message in the batch)
    assert len(llm_spans) == 2

    # Each span should contain the metadata as attributes
    for span in llm_spans:
        # Check if metadata is present as span attributes
        user_id_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        session_id_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        priority_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.priority"

        assert ("user_id" in span.attributes or user_id_key in span.attributes)
        assert ("session_id" in span.attributes or session_id_key in span.attributes)
        assert ("priority" in span.attributes or priority_key in span.attributes)

        # Check the values
        user_id_attr = span.attributes.get("user_id") or span.attributes.get(user_id_key)
        session_id_attr = span.attributes.get("session_id") or span.attributes.get(session_id_key)
        priority_attr = span.attributes.get("priority") or span.attributes.get(priority_key)

        assert user_id_attr == "12345"
        assert session_id_attr == "abc-123"
        assert priority_attr == "high"


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.skip(reason="VCR is not working for this test in CI - need to fix")
async def test_async_batch_metadata_in_span_attributes(instrument_legacy, span_exporter):
    """Test that metadata from abatch calls are populated as span attributes."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Test abatch with metadata
    test_metadata = {
        "user_id": "67890",
        "session_id": "def-456",
        "environment": "production"
    }
    messages_list = [
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "Explain machine learning"}]
    ]

    # Call abatch with metadata
    await llm.abatch(messages_list, config={"metadata": test_metadata})

    spans = span_exporter.get_finished_spans()

    # Find the LLM spans
    llm_spans = [span for span in spans if span.name.endswith(".chat")]

    # There should be 2 LLM spans (one for each message in the batch)
    assert len(llm_spans) == 2

    # Each span should contain the metadata as attributes
    for span in llm_spans:
        # Check if metadata is present as span attributes
        user_id_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        session_id_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        environment_key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.environment"

        assert ("user_id" in span.attributes or user_id_key in span.attributes)
        assert ("session_id" in span.attributes or session_id_key in span.attributes)
        assert ("environment" in span.attributes or environment_key in span.attributes)

        # Check the values
        user_id_attr = span.attributes.get("user_id") or span.attributes.get(user_id_key)
        session_id_attr = span.attributes.get("session_id") or span.attributes.get(session_id_key)
        environment_attr = span.attributes.get("environment") or span.attributes.get(environment_key)

        assert user_id_attr == "67890"
        assert session_id_attr == "def-456"
        assert environment_attr == "production"
