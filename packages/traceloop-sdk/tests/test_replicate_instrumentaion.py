import replicate
from traceloop.sdk.decorators import workflow, task


def test_replicate_completion(exporter):
    @task(name="image_generation")
    def generate_image():
        image = replicate.run(
            "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
            input={"prompt": "a 19th century portrait of a wombat gentleman"}
        )
        print(image)


    @workflow(name="robot_image_generator")
    def image_workflow():
        generate_image()

    image_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.run",
        "image_generation.task",
        "robot_image_generator.workflow",
    ]
