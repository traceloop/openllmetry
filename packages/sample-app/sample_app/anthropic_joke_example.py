from anthropic import Anthropic

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(
    app_name="sample-app",
    disable_batch=True,
)


@workflow(name="pirate_joke_generator")
def joke_workflow():
    anthropic = Anthropic()
    response = anthropic.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-haiku-4-5-20251001",
    )
    print(response.content)
    return response


joke_workflow()
