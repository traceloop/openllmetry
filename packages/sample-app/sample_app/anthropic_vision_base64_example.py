import asyncio
import base64
import os
import anthropic
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(
    app_name="sample-app",
    api_key=os.environ.get("TRACELOOP_API_KEY"),
)

api_key = os.environ.get("ANTHROPIC_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@task(name="generate_description")
async def task():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "..", "data", "vision", "elephant.jpeg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file does not exist at: {image_path}")

    base64_image = encode_image(image_path)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "You are a sub-system ..."},
            {"role": "assistant", "content": "I understand the context!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            },
        ],
    )

    print(message)


@workflow(name="image_description_generation")
def workflow():
    asyncio.run(task())


if __name__ == "__main__":
    workflow()
