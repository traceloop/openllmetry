from openai import OpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

client = OpenAI()
Traceloop.init(app_name="joke_generation_service")


@workflow(name="streaming_joke_creation")
def joke_workflow():
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    for part in stream:
        print(part.choices[0].delta.content or "", end="")
    print()


joke_workflow()
