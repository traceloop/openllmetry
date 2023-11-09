import os
import openai

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

openai.api_key = os.getenv("OPENAI_API_KEY")
Traceloop.init(app_name="story_service")


@workflow(name="streaming_story")
def joke_workflow():
    stream = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a story about opentelemetry"}],
        stream=True,
    )

    for part in stream:
        print(part.choices[0].delta.get("content") or "", end="")
    print()


joke_workflow()
