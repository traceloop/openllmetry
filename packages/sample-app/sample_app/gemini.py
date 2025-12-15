import os
import asyncio
import google.genai as genai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(app_name="gemini_example")

client = genai.Client(api_key=os.environ.get("GENAI_API_KEY"))


@workflow("predict")
def predict_text() -> str:
    """Ideation example with a Large Language Model"""

    response = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents="Give me ten interview questions for the role of program manager.",
    )

    return response.text


@workflow("predict_async")
async def async_predict_text() -> str:
    """Async Ideation example with a Large Language Model"""

    response = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents="Give me ten interview questions for the role of program manager.",
    )

    return response.text


@workflow("chat")
def chat() -> str:
    """Real chat example with conversation context"""

    # First message
    response1 = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents="Hello, how are you?",
    )

    # Second message with conversation history
    conversation = [
        {"role": "user", "parts": [{"text": "Hello, how are you?"}]},
        {"role": "model", "parts": [{"text": response1.text}]},
        {"role": "user", "parts": [{"text": "What is the capital of France?"}]},
    ]

    response2 = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents=conversation,
    )

    return response2.text


if __name__ == "__main__":
    print(chat())
    print(predict_text())
    print(asyncio.run(async_predict_text()))
