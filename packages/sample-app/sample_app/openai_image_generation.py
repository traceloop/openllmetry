import os
import asyncio
from openai import AsyncOpenAI
import base64

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="story_service")


@workflow(name="image_generation")
async def joke_workflow():
    prompt = """
    A children's book drawing of an OpenTelemetry instrumented application.
    """

    result = await client.images.generate(model="gpt-image-1", prompt=prompt)

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open("otel_app.png", "wb") as f:
        f.write(image_bytes)


asyncio.run(joke_workflow())
