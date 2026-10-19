import os

import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from traceloop.sdk.tracing.tracing import TracerWrapper


@pytest.fixture(autouse=True)
def disable_trace_content():
    os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
    yield
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_simple_workflow(exporter, openai_client):
    @task(name="joke_creation")
    def create_joke():
        completion = openai_client.chat.completions.create(
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
    open_ai_span = spans[0]
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert not open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert not open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.completions"
    )


@pytest.mark.vcr
def test_trace_content_param_disables_content(exporter, openai_client, monkeypatch):
    """This file's autouse disable_trace_content fixture already sets
    TRACELOOP_TRACE_CONTENT=false for every test, so we force it back to
    "true" here. With the env var forced to "true", the only thing that
    can still suppress content is trace_content=False itself.
    (exporter=exporter is passed so init() takes the real init path instead
    of returning early at the missing-API-key check.)"""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    monkeypatch.setattr(
        TracerWrapper, "enable_content_tracing", TracerWrapper.enable_content_tracing
    )

    Traceloop.init(exporter=exporter, disable_batch=True, trace_content=False)

    @task(name="joke_creation")
    def create_joke():
        completion = openai_client.chat.completions.create(
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
    openai_span = next(s for s in spans if s.name == "openai.chat")

    # Metadata must still be present.
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert not openai_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert not openai_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
