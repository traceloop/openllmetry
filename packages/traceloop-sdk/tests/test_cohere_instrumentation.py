import os
import cohere
from traceloop.sdk.decorators import workflow, task


def test_cohere_completion(exporter):
    @task(name="joke_creation")
    def create_joke():
        co = cohere.Client(os.environ.get("COHERE_API_KEY"))
        prediction = co.generate(
            model="command", prompt="Tell me a joke, pirate style", max_tokens=10
        )
        print(prediction)
        return prediction

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "cohere.completion",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
