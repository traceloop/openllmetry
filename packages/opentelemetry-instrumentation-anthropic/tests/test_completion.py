import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from opentelemetry.semconv_ai import SpanAttributes

from .utils import verify_metrics


@pytest.mark.vcr
def test_anthropic_completion(exporter, reader):
    client = Anthropic()
    client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert anthropic_span.attributes.get("gen_ai.response.id") == "compl_01EjfrPvPEsRDRUKD6VoBxtK"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(resource_metrics, "claude-instant-1.2", ignore_zero_input_tokens=True)
