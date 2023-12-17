import replicate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="image_generation_service")

@task(name="image_generation")
def generate_image():
    images = replicate.run(
      "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
      input={"prompt": "tiny robot"}
    )
    return images

@workflow(name="robot_image_generator")
def image_workflow():
    print(generate_image())

image_workflow()

#model_version = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
#for event in replicate.stream(
#    model_version,
#    input={
#        "prompt": "Please write a haiku about llamas.",
#    },
#):
#    print(str(event), end="")
#
#model = replicate.models.get("kvfrans/clipdraw")
#version = model.versions.get("5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b")
#prediction = replicate.predictions.create(
#    version=version,
#    input={"prompt":"Watercolor painting of an underwater submarine"})
#print(prediction)

