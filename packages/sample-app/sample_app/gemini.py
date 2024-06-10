import os
import asyncio
import google.generativeai as genai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, aworkflow

Traceloop.init(app_name="gemini_example")

genai.configure(api_key=os.environ.get("GENAI_API_KEY"))


@workflow("predict")
def predict_text() -> str:
    """Ideation example with a Large Language Model"""

    model = genai.GenerativeModel("gemini-1.0-pro-latest")
    response = model.generate_content(
        "Give me ten interview questions for the role of program manager.",
    )

    return response.text


@aworkflow("predict_async")
async def async_predict_text() -> str:
    """Async Ideation example with a Large Language Model"""

    model = genai.GenerativeModel("gemini-1.0-pro-latest")
    response = await model.generate_content_async(
        "Give me ten interview questions for the role of program manager.",
    )

    return response.text


if __name__ == "__main__":
    print(predict_text())
    print(asyncio.run(async_predict_text()))
