import replicate

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="image_generation_service")


@task(name="image_generation")
def generate_image():
    images = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": "tiny robot"},
    )
    return images


@workflow(name="robot_image_generator")
def image_workflow():
    print(generate_image())


image_workflow()
