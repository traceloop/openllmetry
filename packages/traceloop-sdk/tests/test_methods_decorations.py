"""Methods decorations test module."""
import openai
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task


def test_simple_workflow():
    """Test the hello function."""
    exporter = InMemorySpanExporter()
    Traceloop.init(app_name="joke_generation_service", exporter=exporter)

    @task(name="joke_creation")
    def create_joke():
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
