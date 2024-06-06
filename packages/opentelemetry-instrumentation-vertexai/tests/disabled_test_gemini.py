import pytest
import vertexai
from opentelemetry.semconv.ai import SpanAttributes
from vertexai.preview.generative_models import GenerativeModel, Part

vertexai.init()


@pytest.mark.vcr
def test_vertexai_generate_content(exporter):
    multimodal_model = GenerativeModel("gemini-pro-vision")
    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg",
                mime_type="image/jpeg",
            ),
            "what is shown in this image?",
        ]
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.generate_content",
    ]

    vertexai_span = spans[0]
    assert (
        "what is shown in this image?" in vertexai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
    )
    assert vertexai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-pro-vision"
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response._raw_response.usage_metadata.total_token_count
    )
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == response._raw_response.usage_metadata.prompt_token_count
    )
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        == response._raw_response.usage_metadata.candidates_token_count
    )
    assert vertexai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == response.text
