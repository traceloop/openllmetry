from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init()


@workflow(name="pirate_joke_generator")
def joke_workflow():
    anthropic = Anthropic()
    response = anthropic.completions.create(
        prompt=f"{HUMAN_PROMPT}\nTell me a joke about OpenTelemetry in a pirate style\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    print(response.completion)
    return response


joke_workflow()
