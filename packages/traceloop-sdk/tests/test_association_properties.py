import json

import pytest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.tracing.tracing import metrics_common_attributes


def test_association_properties(exporter):
    @workflow(name="test_workflow")
    def test_workflow():
        return test_task()

    @task(name="test_task")
    def test_task():
        return

    Traceloop.set_association_properties({"user_id": 1, "user_name": "John Doe"})
    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_task.task",
        "test_workflow.workflow",
    ]

    some_task_span = spans[0]
    some_workflow_span = spans[1]
    assert (
        some_workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == 1
    )
    assert (
        some_workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_name"
        ]
        == "John Doe"
    )
    assert (
        some_task_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == 1
    )
    assert (
        some_task_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_name"
        ]
        == "John Doe"
    )


def test_association_properties_within_workflow(exporter):
    @workflow(name="test_workflow_within")
    def test_workflow():
        Traceloop.set_association_properties({"session_id": 15})
        return

    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_workflow_within.workflow",
    ]

    some_workflow_span = spans[0]
    assert (
        some_workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 15
    )


def test_list_valued_association_properties_in_metrics():
    Traceloop.set_association_properties(
        {"tags": ["a", "b"], "metadata": {"z": 1, "a": 2}, "user_id": 1}
    )

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    meter = meter_provider.get_meter("test-association-properties")
    counter = meter.create_counter("test.counter")
    histogram = meter.create_histogram("test.histogram")

    attributes = metrics_common_attributes()
    counter.add(1, attributes=attributes)
    histogram.record(0.5, attributes=attributes)

    metrics_data = reader.get_metrics_data()
    data_points = {
        metric.name: metric.data.data_points
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }

    assert set(data_points) == {"test.counter", "test.histogram"}
    for points in data_points.values():
        assert len(points) == 1
        attributes = points[0].attributes
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.tags"
        ] == json.dumps(["a", "b"])
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.metadata"
        ] == json.dumps({"z": 1, "a": 2}, sort_keys=True)
        assert (
            attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
            ]
            == 1
        )


@pytest.mark.vcr
def test_langchain_association_properties(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")

    chain = prompt | model
    chain.invoke(
        {"input": "tell me a short joke"},
        {"metadata": {"user_id": "1234", "session_id": 456}},
    )

    spans = exporter.get_finished_spans()

    assert [
        "execute_task ChatPromptTemplate",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    prompt_span = next(span for span in spans if span.name == "execute_task ChatPromptTemplate")
    chat_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert (
        workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )
    assert (
        chat_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        chat_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )
    assert (
        prompt_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        prompt_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )


@pytest.mark.vcr
def test_langchain_and_external_association_properties(exporter):
    @workflow(name="test_workflow_external")
    def test_workflow_external():
        Traceloop.set_association_properties({"workspace_id": "789"})

        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are helpful assistant"), ("user", "{input}")]
        )
        model = ChatOpenAI(model="gpt-3.5-turbo")

        chain = prompt | model
        chain.invoke(
            {"input": "tell me a short joke"},
            {"metadata": {"user_id": "1234", "session_id": 456}},
        )

    test_workflow_external()

    spans = exporter.get_finished_spans()

    assert [
        "execute_task ChatPromptTemplate",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
        "test_workflow_external.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    prompt_span = next(span for span in spans if span.name == "execute_task ChatPromptTemplate")
    chat_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert (
        workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )
    assert (
        workflow_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.workspace_id"
        ]
        == "789"
    )
    assert (
        chat_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        chat_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )
    assert (
        chat_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.workspace_id"
        ]
        == "789"
    )
    assert (
        prompt_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        == "1234"
    )
    assert (
        prompt_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
        ]
        == 456
    )
    assert (
        prompt_span.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.workspace_id"
        ]
        == "789"
    )
