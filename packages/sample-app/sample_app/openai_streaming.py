import os
from openai import OpenAI


from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="story_service")


@workflow(name="streaming_story")
def joke_workflow():
    stream = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": "Tell me a story about opentelemetry"}],
        stream=True,
    )

    for part in stream:
        print(part.choices[0].delta.content or "", end="")
    print()


joke_workflow()
