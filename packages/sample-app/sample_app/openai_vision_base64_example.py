import base64
import os
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(
    app_name="sample-app",
    api_key=os.environ.get("TRACELOOP_API_KEY"),
)

api_key = os.environ.get("OPENAI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@task(name="generate_description")
def task():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = ["elephant.jpeg", "zebra-by-sutirta-budiman.jpg"]

    images_path = [
        os.path.join(current_dir, "..", "data", "vision", filename)
        for filename in filenames
    ]
    images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
        }
        for image_path in images_path
    ]

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    *images,
                ],
            }
        ],
    )

    print(completion)


@workflow(name="image_description_generation")
def workflow():
    task()


if __name__ == "__main__":
    workflow()
