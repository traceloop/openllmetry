import asyncio
import pytest
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel, InputOutputTextPair

vertexai.init()


@pytest.mark.vcr
def test_vertexai_predict(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "text-bison@001"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes["gen_ai.prompt.0.user"]
    )
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.8
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == response


@pytest.mark.vcr
def test_vertexai_predict_async(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "text-bison@001"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes["gen_ai.prompt.0.user"]
    )
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.8
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == response


@pytest.mark.vcr
def test_vertexai_stream(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "text-bison"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes["gen_ai.prompt.0.user"]
    )
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.8
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == "".join(response)


@pytest.mark.vcr
def test_vertexai_stream_async(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "text-bison"
    assert (
        "Give me ten interview questions for the role of program manager."
        in vertexai_span.attributes["gen_ai.prompt.0.user"]
    )
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.8
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == "".join(response)


@pytest.mark.vcr
def test_vertexai_chat(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "chat-bison@001"
    assert (
        "How many planets are there in the solar system?"
        in vertexai_span.attributes["gen_ai.prompt.0.user"]
    )
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.95
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == response


@pytest.mark.vcr
def test_vertexai_chat_stream(exporter):
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

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.send_message",
    ]

    vertexai_span = spans[0]
    assert vertexai_span.attributes["gen_ai.request.model"] == "chat-bison@001"
    assert vertexai_span.attributes["gen_ai.request.top_p"] == 0.95
    assert vertexai_span.attributes["gen_ai.request.temperature"] == 0.8
    assert vertexai_span.attributes["gen_ai.request.max_tokens"] == 256
    assert vertexai_span.attributes["llm.top_k"] == 40
    assert vertexai_span.attributes["gen_ai.completion.0.content"] == "".join(response)
