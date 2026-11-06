import asyncio

import vertexai
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from vertexai.language_models import ChatModel, InputOutputTextPair, TextGenerationModel

from tests.mocks import (
    BISON_STREAM_CHUNKS,
    CHAT_STREAM_CHUNKS,
    configure_chat_predict,
    configure_predict,
    configure_predict_async,
    patch_bison_streaming,
    patch_bison_streaming_async,
    patch_chat_from_pretrained,
    patch_chat_streaming,
    patch_text_generation_from_pretrained,
)

vertexai.init()

PROMPT = "Give me ten interview questions for the role of program manager."
CHAT_MESSAGE = "How many planets are there in the solar system?"


def test_vertexai_predict(instrument_legacy, span_exporter, log_exporter):
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    with patch_text_generation_from_pretrained():
        model = TextGenerationModel.from_pretrained("text-bison@001")
        configure_predict(model)
        response = model.predict(PROMPT, **parameters)

    response = response.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-bison@001"
    )
    assert PROMPT in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_predict_async(instrument_legacy, span_exporter, log_exporter):
    async def async_predict_text() -> str:
        parameters = {
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40,
        }

        with patch_text_generation_from_pretrained():
            model = TextGenerationModel.from_pretrained("text-bison@001")
            configure_predict_async(model)
            response = await model.predict_async(PROMPT, **parameters)

        return response.text

    response = asyncio.run(async_predict_text())

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_async",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-bison@001"
    )
    assert PROMPT in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_stream(instrument_legacy, span_exporter, log_exporter):
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    with (
        patch_text_generation_from_pretrained(),
        patch_bison_streaming(BISON_STREAM_CHUNKS),
    ):
        text_generation_model = TextGenerationModel.from_pretrained("text-bison")
        responses = text_generation_model.predict_streaming(prompt=PROMPT, **parameters)
        result = [response.text for response in responses]

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_streaming",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-bison"
    assert PROMPT in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(result)


def test_vertexai_stream_async(instrument_legacy, span_exporter, log_exporter):
    async def async_streaming_prediction() -> list:
        parameters = {
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40,
        }

        with (
            patch_text_generation_from_pretrained(),
            patch_bison_streaming_async(BISON_STREAM_CHUNKS),
        ):
            text_generation_model = TextGenerationModel.from_pretrained("text-bison")
            responses = await text_generation_model.predict_streaming_async(
                prompt=PROMPT, **parameters
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
    assert PROMPT in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(response)


def test_vertexai_chat(instrument_legacy, span_exporter, log_exporter):
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }

    with patch_chat_from_pretrained():
        chat_model = ChatModel.from_pretrained("chat-bison@001")
        configure_chat_predict(chat_model)
        chat = chat_model.start_chat(
            context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
            examples=[
                InputOutputTextPair(
                    input_text="How many moons does Mars have?",
                    output_text="The planet Mars has two moons, Phobos and Deimos.",
                ),
            ],
        )
        response = chat.send_message(CHAT_MESSAGE, **parameters)

    response = response.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "chat-bison@001"
    )
    assert (
        CHAT_MESSAGE
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.95
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )


def test_vertexai_chat_stream(instrument_legacy, span_exporter, log_exporter):
    parameters = {
        "temperature": 0.8,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }

    with (
        patch_chat_from_pretrained(),
        patch_chat_streaming(CHAT_STREAM_CHUNKS),
    ):
        chat_model = ChatModel.from_pretrained("chat-bison@001")
        chat = chat_model.start_chat(
            context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
            examples=[
                InputOutputTextPair(
                    input_text="How many moons does Mars have?",
                    output_text="The planet Mars has two moons, Phobos and Deimos.",
                ),
            ],
        )
        responses = chat.send_message_streaming(message=CHAT_MESSAGE, **parameters)
        result = [response.text for response in responses]

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message_streaming",
    ]

    vertexai_span = spans[0]
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "chat-bison@001"
    )
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.95
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.8
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert vertexai_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ] == "".join(result)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
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
