import os

from aleph_alpha_client import Client, CompletionRequest, Prompt
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow


Traceloop.init(app_name="alephalpha_example")

client = Client(token=os.environ.get("ALEPH_ALPHA_API_KEY") or os.environ.get("AA_TOKEN"))


@task(name="complete_prompt")
def complete_prompt():
    request = CompletionRequest(
        prompt=Prompt.from_text("Tell me one sentence about OpenTelemetry."),
        maximum_tokens=64,
    )

    response = client.complete(request, model="luminous-base")
    return response.completions[0].completion


@workflow(name="alephalpha_completion_demo")
def alephalpha_completion_demo():
    completion = complete_prompt()
    print(completion)


if __name__ == "__main__":
    alephalpha_completion_demo()
