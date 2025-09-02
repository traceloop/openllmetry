import os
import asyncio
import google.genai as genai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# Initialize with console exporter for debugging
Traceloop.init(
    app_name="gemini_example",
)

# Configure client with API key
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
    """Chat example with a Large Language Model"""

    # For chat, we'll simulate with multiple generate_content calls
    response1 = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents="Hello, how are you?",
    )
    
    response2 = client.models.generate_content(
        model="gemini-1.5-pro-002",
        contents="What is the capital of France?",
    )

    return response2.text


if __name__ == "__main__":
    print(chat())
    print(predict_text())
    print(asyncio.run(async_predict_text()))
