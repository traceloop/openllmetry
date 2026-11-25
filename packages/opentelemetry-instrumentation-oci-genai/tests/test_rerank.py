import json

import pytest
from oci.generative_ai_inference import models
from opentelemetry.instrumentation.oci_genai.span_utils import (
    GEN_AI_OCI_RERANK_DOCUMENT_COUNT,
    GEN_AI_OCI_RERANK_TOP_N,
    GEN_AI_OCI_SERVING_MODE,
    OCI_GENAI_PROVIDER,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from tests import assert_message_in_logs, assert_valid_output_message

RERANK_MODEL = "cohere.rerank-v4.0-fast"
QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Paris.",
]


def _rerank(client, compartment_id):
    return client.rerank_text(
        models.RerankTextDetails(
            compartment_id=compartment_id,
            serving_mode=models.OnDemandServingMode(model_id=RERANK_MODEL),
            input=QUERY,
            documents=DOCUMENTS,
            top_n=2,
        )
    )


@pytest.mark.vcr
def test_rerank_text_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _rerank(oci_client, compartment_id)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"rerank {RERANK_MODEL}"]
    span = spans[0]

    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == OCI_GENAI_PROVIDER
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "rerank"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == RERANK_MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == RERANK_MODEL
    assert span.attributes[GEN_AI_OCI_SERVING_MODE] == "ON_DEMAND"
    assert span.attributes[GEN_AI_OCI_RERANK_TOP_N] == 2
    assert span.attributes[GEN_AI_OCI_RERANK_DOCUMENT_COUNT] == 3
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID]

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert [message["role"] for message in input_messages] == ["system", "system", "system", "user"]
    assert [message["parts"][0]["content"] for message in input_messages] == DOCUMENTS + [QUERY]

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 2
    for message, rank in zip(output_messages, response.data.document_ranks):
        assert_valid_output_message(message)
        assert message["role"] == "assistant"
        assert message["parts"][0]["content"] == f"Doc {rank.index}, Score: {rank.relevance_score}"
    assert response.data.document_ranks[0].index == 0

    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.vcr
def test_rerank_text_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    response = _rerank(oci_client, compartment_id)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"rerank {RERANK_MODEL}"]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6
    for log, document in zip(logs[:3], DOCUMENTS):
        assert_message_in_logs(log, "gen_ai.system.message", {"content": document})
    assert_message_in_logs(logs[3], "gen_ai.user.message", {"content": QUERY})
    for index, (log, rank) in enumerate(zip(logs[4:], response.data.document_ranks)):
        assert_message_in_logs(
            log,
            "gen_ai.choice",
            {
                "index": index,
                "finish_reason": "stop",
                "message": {"content": f"Doc {rank.index}, Score: {rank.relevance_score}"},
            },
        )
