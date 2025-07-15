import pytest
from opentelemetry.semconv_ai import SpanAttributes
from pinecone import Pinecone

@pytest.mark.vcr
def test_pinecone_assistant_chat(traces_exporter):
    pc = Pinecone(api_key="test")
    
    #create an assistant
    assistant = pc.assistant.create_assistant(
        assistant_name="test-assistant",
        instructions="You are a helpful assistant"
    )
    
    #chat with the assistant
    response = assistant.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o"
    )
    
    spans = traces_exporter.get_finished_spans()
    chat_span = next(span for span in spans if span.name == "pinecone.assistant.chat")
    
    assert chat_span.attributes.get(SpanAttributes.GEN_AI_SYSTEM) == "pinecone"
    assert chat_span.attributes.get(SpanAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o"
    assert chat_span.attributes.get("gen_ai.prompt.count") == 1
    assert chat_span.attributes.get("gen_ai.prompt.0.role") == "user"
    assert chat_span.attributes.get("gen_ai.prompt.0.content") == "Hello"
