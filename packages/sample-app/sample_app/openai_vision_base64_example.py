import base64
import os
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task


api_key = os.environ.get("OPENAI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "..", "data", "vision", "elephant.jpeg")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image file does not exist at: {image_path}")

base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

Traceloop.init()


@task(name="generate_description")
def task():
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What’s in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
    )

    print(completion)


@workflow(name="image_description_generation")
def workflow():
    task()


if __name__ == "__main__":
    workflow()
