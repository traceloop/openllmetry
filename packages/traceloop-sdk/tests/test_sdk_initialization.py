import pytest
from openai import OpenAI
from traceloop.sdk.decorators import workflow


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_resource_attributes(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert open_ai_span.resource.attributes["something"] == "yes"
    assert open_ai_span.resource.attributes["service.name"] == "test"


def test_custom_span_processor(exporter_with_custom_span_processor, openai_client):
    @workflow()
    def run_workflow():
        pass

    run_workflow()

    spans = exporter_with_custom_span_processor.get_finished_spans()
    workflow_span = spans[0]
    assert workflow_span.attributes["custom_span"] == "yes"
