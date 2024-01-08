import os
import asyncio
import vertexai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, aworkflow
from vertexai.language_models import TextGenerationModel, ChatModel, InputOutputTextPair

Traceloop.init(app_name="stream_prediction_service")

project_id = os.getenv('VERTEXAI_PROJECT_ID')
location = os.getenv('VERTEXAI_LOCATION')

vertexai.init(project=project_id, location=location)


@workflow("stream_prediction")
def streaming_prediction() -> str:
    """Streaming Text Example with a Large Language Model"""

    text_generation_model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }
    responses = text_generation_model.predict_streaming(
        prompt="Give me ten interview questions for the role of program manager.",
        **parameters)
    result = [response for response in responses]

    return result


@aworkflow("stream_prediction_async")
async def async_streaming_prediction() -> str:
    """Async Streaming Text Example with a Large Language Model"""

    text_generation_model = TextGenerationModel.from_pretrained(
        "text-bison")
    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    responses = text_generation_model.predict_streaming_async(
        prompt="Give me ten interview questions for the role of program manager.",
        **parameters)

    result = [response async for response in responses]
    return result


@workflow("send_message_streaming")
def chat_streaming() -> str:
    """Streaming Chat Example with a Large Language Model"""

    chat_model = ChatModel.from_pretrained("chat-bison")

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

    result = [response for response in responses]
    return result


if __name__ == "__main__":
    print(chat_streaming())