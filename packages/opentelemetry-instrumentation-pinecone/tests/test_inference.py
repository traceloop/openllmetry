import pytest
from opentelemetry.semconv_ai import SpanAttributes
from pinecone import Pinecone

@pytest.mark.vcr
def test_pinecone_inference_embed(traces_exporter):
    pc = Pinecone(api_key="test")
    
    response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=["Hello world", "Test embedding"],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    spans = traces_exporter.get_finished_spans()
    embed_span = next(span for span in spans if span.name == "pinecone.inference.embed")
    
    assert embed_span.attributes.get(SpanAttributes.GEN_AI_SYSTEM) == "pinecone"
    assert embed_span.attributes.get(SpanAttributes.GEN_AI_REQUEST_MODEL) == "multilingual-e5-large"
    assert embed_span.attributes.get("gen_ai.prompt.count") == 2
    assert embed_span.attributes.get("pinecone.inference.input_type") == "passage"
