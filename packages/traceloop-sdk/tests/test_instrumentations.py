import os
import pytest
import openai
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task


exporter = InMemorySpanExporter()
Traceloop.init(app_name="test", disable_batch=True, exporter=exporter)


@pytest.fixture(autouse=True)
def clear_exporter():
    exporter.clear()


def test_simple_workflow():
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


def test_haystack():
    prompt = PromptTemplate(
        prompt="Tell me a joke about {query}\n",
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(
        model_name_or_path="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        default_prompt_template=prompt,
    )

    pipeline = Pipeline()
    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Query"])
    pipeline.run("OpenTelemetry")

    spans = exporter.get_finished_spans()
    assert set(
        [
            "openai.chat",
            "PromptNode.task",
            "haystack_pipeline.workflow",
        ]
    ).issubset([span.name for span in spans])
