import replicate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="image_generation_service")


@task(name="image_generation")
def generate_image():
    images = replicate.run(
        "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        input={"prompt": "tiny robot"},
    )
    return images


@workflow(name="robot_image_generator")
def image_workflow():
    print(generate_image())


image_workflow()
