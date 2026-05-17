import replicate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="chat_stream_generation_service")


@task(name="chat_stream_generation")
def generate_chat_stream():
    model_version = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
    chat_stream = replicate.stream(model_version, input={"prompt": "tiny robot"})
    for event in chat_stream:
        print(str(event), end="")


@workflow(name="chat_stream_generator")
def chat_stream_workflow():
    generate_chat_stream()


chat_stream_workflow()
