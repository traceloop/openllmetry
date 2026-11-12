import asyncio
from unittest.mock import patch

import pytest
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from vertexai.language_models import (
    ChatModel,
    ChatSession,
    InputOutputTextPair,
    TextGenerationModel,
)

from .mocks import (
    init_vertexai,
    offline_from_pretrained,
    offline_predict,
    offline_predict_async,
    offline_predict_streaming,
    offline_predict_streaming_async,
    offline_send_message,
    offline_send_message_streaming,
)

init_vertexai()


@pytest.fixture(name="mock_language_models")
def fixture_mock_language_models():
    """Patch the SDK methods that would make real gRPC calls to Vertex AI."""
    with (
        patch.object(
            TextGenerationModel, "from_pretrained", classmethod(offline_from_pretrained)
        ),
        patch.object(TextGenerationModel, "predict", offline_predict),
        patch.object(TextGenerationModel, "predict_async", offline_predict_async),
        patch.object(
            TextGenerationModel, "predict_streaming", offline_predict_streaming
        ),
        patch.object(
            TextGenerationModel,
            "predict_streaming_async",
            offline_predict_streaming_async,
        ),
        patch.object(
            ChatModel, "from_pretrained", classmethod(offline_from_pretrained)
        ),
        patch.object(ChatSession, "send_message", offline_send_message),
        patch.object(
            ChatSession, "send_message_streaming", offline_send_message_streaming
        ),
    ):
        yield


def test_vertexai_predict(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        "Give me ten interview questions for the role of program manager.",
        **parameters,
    )

    response = response.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-bison@001"
    )
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_predict_async(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    async def async_predict_text() -> str:
        """Ideation example with a Large Language Model"""

        parameters = {
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40,
        }

        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = await model.predict_async(
            "Give me ten interview questions for the role of program manager.",
            **parameters,
        )

        return response.text

    response = asyncio.run(async_predict_text())

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_async",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-bison@001"
    )
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_stream(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    text_generation_model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }
    responses = text_generation_model.predict_streaming(
        prompt="Give me ten interview questions for the role of program manager.",
        **parameters,
    )

    result = [response.text for response in responses]
    response = result

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_streaming",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-bison"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(response)


def test_vertexai_stream_async(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    async def async_streaming_prediction() -> list:
        """Streaming Text Example with a Large Language Model"""

        text_generation_model = TextGenerationModel.from_pretrained("text-bison")
        parameters = {
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40,
        }

        responses = text_generation_model.predict_streaming_async(
            prompt="Give me ten interview questions for the role of program manager.",
            **parameters,
        )
        result = [response.text async for response in responses]
        return result

    response = asyncio.run(async_streaming_prediction())

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_streaming_async",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-bison"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(response)


def test_vertexai_chat(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }

    chat = chat_model.start_chat(
        context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )

    response = chat.send_message(
        "How many planets are there in the solar system?", **parameters
    )

    response = response.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "chat-bison@001"
    )
    assert (
        "How many planets are there in the solar system?"
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.95
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_chat_stream(
    mock_language_models, instrument_legacy, span_exporter, log_exporter
):
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    parameters = {
        "temperature": 0.8,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }

    chat = chat_model.start_chat(
        context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )

    responses = chat.send_message_streaming(
        message="How many planets are there in the solar system?", **parameters
    )

    result = [response.text for response in responses]
    response = result

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message_streaming",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "chat-bison@001"
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.95
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(response)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    # In OpenTelemetry 1.37.0+, event_name is a field on LogRecord, not in attributes
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
