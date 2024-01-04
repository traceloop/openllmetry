import os
import cohere
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init()


@workflow(name="pirate_joke_generator")
def joke_workflow():
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    return co.chat(model="command", message="Tell me a joke, pirate style")


res = joke_workflow()
