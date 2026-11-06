"""Regression test: openai-agents + openai instrumentation must not double-emit LLM spans.

When both OpenAIAgentsInstrumentor and OpenAIV1Instrumentor are active, each
model call must produce exactly ONE openai.response span. Before the fix the
responses_get_or_create_wrapper did not check SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
so the openai instrumentation fired a second span for the same call even when the
agents processor had set that suppression key.
"""
from unittest.mock import MagicMock


from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry import context as context_api


def _make_provider_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestAgentsProcessorSuppressToken:
    """The agents processor must set SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    while a generation/response span is active."""

    def test_suppress_token_set_during_generation_span(self):
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = _make_provider_and_exporter()
        set_tracer_provider(provider)
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        fake_span_data = MagicMock()
        fake_span_data.__class__.__name__ = "ResponseSpanData"
        fake_span_data.response = None
        fake_span_data.input = None
        fake_span_data.model = "gpt-4o"

        fake_sdk_span = MagicMock()
        fake_sdk_span.span_data = fake_span_data
        fake_sdk_span.trace_id = "trace-suppress-test"
        fake_sdk_span.error = None

        workflow_span = tracer.start_span("Agent Workflow")
        processor._root_spans["trace-suppress-test"] = workflow_span

        suppressed_during_span = []

        original_start = processor._start_generation_span

        def recording_start(agent_ctx, span_data):
            suppressed_during_span.append(
                context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
            )
            return original_start(agent_ctx, span_data)

        processor._start_generation_span = recording_start

        processor.on_span_start(fake_sdk_span)

        assert suppressed_during_span, "_start_generation_span was never called"
        assert suppressed_during_span[0] is True, (
            "Expected SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY=True during "
            "generation span start, got {!r}".format(suppressed_during_span[0])
        )

        processor.on_span_end(fake_sdk_span)

        after_value = context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        assert not after_value, (
            "SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY leaked after span end: {!r}".format(after_value)
        )

        workflow_span.end()

    def test_non_generation_spans_do_not_suppress(self):
        """Agent spans must NOT set the suppression key."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import AgentSpanData

        provider, exporter = _make_provider_and_exporter()
        set_tracer_provider(provider)
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        fake_span_data = MagicMock(spec=AgentSpanData)
        fake_span_data.name = "helper.agent"
        fake_span_data.handoffs = []
        fake_span_data.tools = []
        fake_span_data.output_type = None

        fake_sdk_span = MagicMock()
        fake_sdk_span.span_data = fake_span_data
        fake_sdk_span.trace_id = "trace-agent-test"
        fake_sdk_span.error = None

        workflow_span = tracer.start_span("Agent Workflow")
        processor._root_spans["trace-agent-test"] = workflow_span

        suppressed_values = []
        original_start = processor._start_agent_span

        def recording_start(span_data, parent_context, trace_id):
            suppressed_values.append(
                context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
            )
            return original_start(span_data, parent_context, trace_id)

        processor._start_agent_span = recording_start

        processor.on_span_start(fake_sdk_span)

        assert suppressed_values, "_start_agent_span was never called"
        assert not suppressed_values[0], (
            "Agent span must NOT suppress LLM instrumentation, got {!r}".format(suppressed_values[0])
        )

        processor.on_span_end(fake_sdk_span)
        workflow_span.end()


RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4.1-nano",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Paris.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def test_agents_and_openai_instrumentors_record_one_response_span(exporter):
    """An Agents run with both instrumentors on must record the model call once."""
    import httpx
    from openai import AsyncOpenAI
    from agents import Agent, OpenAIResponsesModel, Runner
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    client = AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=RESPONSE))
        ),
    )
    agent = Agent(
        name="Geo",
        instructions="Answer briefly.",
        model=OpenAIResponsesModel(model="gpt-4.1-nano", openai_client=client),
    )

    openai_instrumentor = OpenAIInstrumentor()
    openai_instrumentor.instrument()
    try:
        result = Runner.run_sync(agent, "Capital of France?")
    finally:
        openai_instrumentor.uninstrument()

    assert result.final_output == "Paris."

    response_spans = [s for s in exporter.get_finished_spans() if s.name == "openai.response"]
    assert len(response_spans) == 1, [s.instrumentation_scope.name for s in response_spans]

    attrs = response_spans[0].attributes
    assert "Capital of France?" in attrs["gen_ai.input.messages"]
    assert "Paris." in attrs["gen_ai.output.messages"]
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.usage.total_tokens"] == 18
