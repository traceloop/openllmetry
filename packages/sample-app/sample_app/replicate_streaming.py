import replicate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="chat_stream_generation_service")

@task(name="chat_stream_generation")
def generate_chat_stream():
    chat_stream = replicate.run(
      "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
      input={"prompt": "tiny robot"}
    )
    return chat_stream

@workflow(name="chat_stream_generator")
def chat_stream_workflow():
    for event in generate_chat_stream():
        print(str(event), end="")

chat_stream_workflow()

