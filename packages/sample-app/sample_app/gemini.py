import os
import asyncio
import google.generativeai as genai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(app_name="gemini_example")

genai.configure(api_key=os.environ.get("GENAI_API_KEY"))


@workflow("predict")
def predict_text() -> str:
    """Ideation example with a Large Language Model"""

    model = genai.GenerativeModel("gemini-1.5-pro-002")
    response = model.generate_content(
        "Give me ten interview questions for the role of program manager.",
    )

    return response.text


@workflow("predict_async")
async def async_predict_text() -> str:
    """Async Ideation example with a Large Language Model"""

    model = genai.GenerativeModel("gemini-1.5-pro-002")
    response = await model.generate_content_async(
        "Give me ten interview questions for the role of program manager.",
    )

    return response.text


@workflow("chat")
def chat() -> str:
    """Chat example with a Large Language Model"""

    model = genai.GenerativeModel("gemini-1.5-pro-002")
    chat = model.start_chat()
    response = chat.send_message("Hello, how are you?")
    response = chat.send_message("What is the capital of France?")

    return response.text


if __name__ == "__main__":
    print(chat())
    print(predict_text())
    print(asyncio.run(async_predict_text()))
