import json

import pytest
from oci.generative_ai_inference import models
from opentelemetry.instrumentation.oci_genai.span_utils import (
    GEN_AI_OCI_EMBEDDINGS_INPUT_COUNT,
    GEN_AI_OCI_EMBEDDINGS_INPUT_TYPE,
    GEN_AI_OCI_SERVING_MODE,
    OCI_GENAI_PROVIDER,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from tests import assert_message_in_logs

EMBED_MODEL = "cohere.embed-v4.0"
INPUTS = ["OpenTelemetry is an observability framework", "OCI Generative AI serves Cohere embeddings"]


def _embed(client, compartment_id):
    return client.embed_text(
        models.EmbedTextDetails(
            compartment_id=compartment_id,
            serving_mode=models.OnDemandServingMode(model_id=EMBED_MODEL),
            inputs=INPUTS,
            input_type=models.EmbedTextDetails.INPUT_TYPE_SEARCH_DOCUMENT,
            truncate=models.EmbedTextDetails.TRUNCATE_END,
        )
    )


@pytest.mark.vcr
def test_embed_text_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _embed(oci_client, compartment_id)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"embeddings {EMBED_MODEL}"]
    span = spans[0]

    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == OCI_GENAI_PROVIDER
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "embeddings"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == EMBED_MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == EMBED_MODEL
    assert span.attributes[GEN_AI_OCI_SERVING_MODE] == "ON_DEMAND"
    assert span.attributes[GEN_AI_OCI_EMBEDDINGS_INPUT_COUNT] == 2
    assert span.attributes[GEN_AI_OCI_EMBEDDINGS_INPUT_TYPE] == "SEARCH_DOCUMENT"
    assert span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == len(response.data.embeddings[0])
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID]

    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == response.data.usage.prompt_tokens
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == response.data.usage.total_tokens
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [{"role": "user", "parts": [{"type": "text", "content": text}]} for text in INPUTS]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes

    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.vcr
def test_embed_text_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    _embed(oci_client, compartment_id)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"embeddings {EMBED_MODEL}"]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": INPUTS[0]})
    assert_message_in_logs(logs[1], "gen_ai.user.message", {"content": INPUTS[1]})
