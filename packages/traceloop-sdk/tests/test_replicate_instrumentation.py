import sys
import pytest
import replicate
from traceloop.sdk.decorators import workflow, task


def test_replicate_image_generation(exporter):
    @task(name="image_generation")
    def generate_image():
        replicate.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={"prompt": "robots"},
        )

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


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9")
def test_replicate_image_generation_stream(exporter):
    @task(name="image_generation_stream")
    def generate_image_stream():
        model_version = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        for event in replicate.stream(
            model_version,
            input={
                "prompt": "tell me a joke about opentelemetry",
            },
        ):
            continue

    @workflow(name="robot_image_generator_stream")
    def image_workflow():
        generate_image_stream()

    image_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.stream",
        "image_generation_stream.task",
        "robot_image_generator_stream.workflow",
    ]


def test_replicate_image_generation_predictions(exporter):
    @task(name="image_generation_predictions")
    def generate_image_predictions():
        model = replicate.models.get("kvfrans/clipdraw")
        version = model.versions.get(
            "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b"
        )
        replicate.predictions.create(version, input={"prompt": "robots"})

    @workflow(name="robot_image_generator_predictions")
    def image_workflow():
        generate_image_predictions()

    image_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.predictions.create",
        "image_generation_predictions.task",
        "robot_image_generator_predictions.workflow",
    ]
