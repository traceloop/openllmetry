import json

import pytest
from opentelemetry.instrumentation.bedrock.prompt_caching import CacheSpanAttrs
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_invoke_model_cache_tokens(instrument_legacy, brt, span_exporter, log_exporter):
    """invoke_model emits both legacy cache marker and numeric cache token attrs."""
    def call():
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": "very very long system prompt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How do I write js?",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.1,
            "stop_sequences": ["stop"],
            "top_k": 250,
        }
        return brt.invoke_model(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            body=json.dumps(body),
        )

    call()
    call()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    write_span = next(s for s in spans if s.attributes.get(CacheSpanAttrs.CACHED) == "write")
    read_span = next(s for s in spans if s.attributes.get(CacheSpanAttrs.CACHED) == "read")

    # legacy marker
    assert write_span.attributes.get(CacheSpanAttrs.CACHED) == "write"
    assert read_span.attributes.get(CacheSpanAttrs.CACHED) == "read"

    # numeric token counts
    assert write_span.attributes.get(SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS) == 18131
    assert write_span.attributes.get(SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS) == 0
    assert read_span.attributes.get(SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS) == 18131
    assert read_span.attributes.get(SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS) == 0
