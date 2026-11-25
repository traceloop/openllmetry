import json

import pytest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.tracing.tracing import metrics_common_attributes


class _NonSerializable:
    """A value that cannot be serialized to JSON."""

    def __str__(self):
        return "non-serializable-object"


class _Unrepresentable:
    """A value that cannot even be turned into a string."""

    def __str__(self):
        raise ValueError("this value cannot be represented")


def _record_metrics(attributes):
    """Record a counter and a histogram with the given attributes and collect them."""
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    meter = meter_provider.get_meter("test-association-properties")
    counter = meter.create_counter("test.counter")
    histogram = meter.create_histogram("test.histogram")

    counter.add(1, attributes=attributes)
    histogram.record(0.5, attributes=attributes)

    metrics_data = reader.get_metrics_data()
    return {
        metric.name: metric.data.data_points
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


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

    attributes = metrics_common_attributes()
    data_points = _record_metrics(attributes)

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


def test_set_valued_association_properties_in_metrics():
    """Sets are not hashable, so they must be serialized before they reach the metrics."""
    Traceloop.set_association_properties({"tags": {"b", "a"}, "user_id": 1})

    attributes = metrics_common_attributes()
    data_points = _record_metrics(attributes)

    assert set(data_points) == {"test.counter", "test.histogram"}
    for points in data_points.values():
        assert len(points) == 1
        attributes = points[0].attributes
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.tags"
        ] == json.dumps(["a", "b"])
        assert (
            attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
            ]
            == 1
        )


def test_tuple_valued_association_properties_in_metrics():
    """Tuples are not hashable, so they must be serialized before they reach the metrics."""
    Traceloop.set_association_properties({"tags": ("b", "a"), "user_id": 1})

    attributes = metrics_common_attributes()
    data_points = _record_metrics(attributes)

    assert set(data_points) == {"test.counter", "test.histogram"}
    for points in data_points.values():
        assert len(points) == 1
        attributes = points[0].attributes
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.tags"
        ] == json.dumps(["b", "a"])
        assert (
            attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
            ]
            == 1
        )


def test_non_serializable_association_properties_in_metrics():
    """Values that cannot be serialized are converted into a safe attribute value."""
    Traceloop.set_association_properties(
        {
            "obj": _NonSerializable(),
            "unrepresentable": _Unrepresentable(),
            "user_id": 1,
        }
    )

    attributes = metrics_common_attributes()
    data_points = _record_metrics(attributes)

    assert set(data_points) == {"test.counter", "test.histogram"}
    for points in data_points.values():
        assert len(points) == 1
        attributes = points[0].attributes
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.obj"
        ] == "non-serializable-object"
        assert attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.unrepresentable"
        ] == "<unserializable>"
        assert (
            attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
            ]
            == 1
        )


def test_response_span_attributes_recorded_with_problematic_metric_attributes(
    exporter,
):
    """A problematic association property must not abort the response telemetry.

    The OpenAI instrumentation records the chat metrics before it sets the response
    attributes on the span, so an exception raised while recording the metrics used
    to drop `gen_ai.output.messages`, `gen_ai.response.model` and the `gen_ai.usage.*`
    attributes as well.
    See https://github.com/traceloop/openllmetry/issues/4436.
    """
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _handle_response

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    meter = meter_provider.get_meter("test-association-properties")
    token_counter = meter.create_histogram("gen_ai.client.token.usage")
    choice_counter = meter.create_counter("gen_ai.client.choice.count")
    duration_histogram = meter.create_histogram("gen_ai.client.operation.duration")

    Traceloop.set_association_properties({"tags": {"a", "b"}})

    response = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    tracer = trace.get_tracer("test-association-properties")
    with tracer.start_as_current_span("ChatOpenAI.chat") as span:
        _handle_response(
            response,
            span,
            instance=None,
            token_counter=token_counter,
            choice_counter=choice_counter,
            duration_histogram=duration_histogram,
            duration=0.1,
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == 15
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert "Hello!" in span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]

    metrics_data = reader.get_metrics_data()
    data_points = [
        data_point
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
        for data_point in metric.data.data_points
    ]
    assert data_points
    for data_point in data_points:
        assert data_point.attributes[
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.tags"
        ] == json.dumps(["a", "b"])
