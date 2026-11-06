"""``generate_text`` (legacy completion API) tests.

The legacy text generation endpoint is no longer served in the commercial realm (the service answers 404 "The
requested API is not available"), so these tests stub ``BaseClient.call_api`` with the documented response shapes
instead of recording cassettes.
"""

import json
from unittest.mock import patch

import oci
from oci.generative_ai_inference import models
from opentelemetry.instrumentation.oci_genai.span_utils import GEN_AI_OCI_API_FORMAT, OCI_GENAI_PROVIDER
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from tests import assert_message_in_logs, assert_valid_output_message

PROMPT = "Write a haiku about opentelemetry"


def _response(result):
    return oci.response.Response(200, {"opc-request-id": "REDACTED"}, result, None)


def _cohere_result():
    return models.GenerateTextResult(
        model_id="cohere.command",
        model_version="15.6",
        inference_response=models.CohereLlmInferenceResponse(
            generated_texts=[
                models.GeneratedText(id="gen-1", text="Traces flow like streams", finish_reason="COMPLETE"),
                models.GeneratedText(id="gen-2", text="Spans light the darkest path", finish_reason="MAX_TOKENS"),
            ]
        ),
    )


def _llama_result():
    return models.GenerateTextResult(
        model_id="meta.llama-2-70b-chat",
        model_version="1.0",
        inference_response=models.LlamaLlmInferenceResponse(
            created="2024-01-01T00:00:00Z",
            choices=[models.Choice(index=0, text="Signals converge", finish_reason="stop")],
        ),
    )


def _generate(client, compartment_id, model, inference_request):
    return client.generate_text(
        models.GenerateTextDetails(
            compartment_id=compartment_id,
            serving_mode=models.OnDemandServingMode(model_id=model),
            inference_request=inference_request,
        )
    )


def test_generate_text_cohere_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    request = models.CohereLlmInferenceRequest(
        prompt=PROMPT, max_tokens=50, temperature=0.7, top_p=0.9, num_generations=2, stop_sequences=["\n\n"]
    )
    with patch.object(oci_client.base_client, "call_api", return_value=_response(_cohere_result())):
        response = _generate(oci_client, compartment_id, "cohere.command", request)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["text_completion cohere.command"]
    span = spans[0]

    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == OCI_GENAI_PROVIDER
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "text_completion"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "cohere.command"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "cohere.command"
    assert span.attributes[GEN_AI_OCI_API_FORMAT] == "COHERE"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] == 2
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == ("\n\n",)
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is False

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [{"role": "user", "parts": [{"type": "text", "content": PROMPT}]}]

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 2
    for message in output_messages:
        assert_valid_output_message(message)
    assert output_messages[0]["parts"] == [{"type": "text", "content": "Traces flow like streams"}]
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[1]["finish_reason"] == "length"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop", "length")

    assert response.data.inference_response.generated_texts[0].text == "Traces flow like streams"
    assert len(log_exporter.get_finished_logs()) == 0


def test_generate_text_llama_legacy(instrument_legacy, oci_client, compartment_id, span_exporter):
    request = models.LlamaLlmInferenceRequest(prompt=PROMPT, max_tokens=50, top_k=10)
    with patch.object(oci_client.base_client, "call_api", return_value=_response(_llama_result())):
        _generate(oci_client, compartment_id, "meta.llama-2-70b-chat", request)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["text_completion meta.llama-2-70b-chat"]
    span = spans[0]

    assert span.attributes[GEN_AI_OCI_API_FORMAT] == "LLAMA"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 10
    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Signals converge"}], "finish_reason": "stop"}
    ]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


def test_generate_text_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    request = models.CohereLlmInferenceRequest(prompt=PROMPT, max_tokens=50)
    with patch.object(oci_client.base_client, "call_api", return_value=_response(_cohere_result())):
        _generate(oci_client, compartment_id, "cohere.command", request)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["text_completion cohere.command"]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": PROMPT})
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "stop", "message": {"content": "Traces flow like streams"}},
    )
    assert_message_in_logs(
        logs[2],
        "gen_ai.choice",
        {"index": 1, "finish_reason": "length", "message": {"content": "Spans light the darkest path"}},
    )
