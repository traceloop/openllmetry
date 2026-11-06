"""
Tests for CrewAI native-async instrumentation.

CrewAI 1.7.0+ exposes a native async/await execution chain that is entirely
separate from the synchronous one:

    Crew.akickoff -> Task.aexecute_sync -> Agent.aexecute_task -> LLM.acall

(`kickoff_async`, by contrast, just runs the sync `kickoff` in a thread and is
already covered by the sync wrapper, so it is intentionally NOT instrumented.)

These tests drive the async wrappers directly with a real TracerProvider +
InMemorySpanExporter, so they run on any crewai version.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiOperationNameValues
from opentelemetry.trace import StatusCode

from opentelemetry.instrumentation.crewai.instrumentation import (
    wrap_akickoff,
    wrap_aexecute_task,
    wrap_aexecute_sync,
    wrap_acall,
)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


@pytest.fixture(autouse=True)
def _patch_span_attrs():
    """CrewAISpanAttributes walks a real instance's __dict__; stub it out here."""
    with patch("opentelemetry.instrumentation.crewai.instrumentation.CrewAISpanAttributes"):
        yield


def spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _make_crew(stream=False):
    Crew = type("Crew", (), {})
    instance = Crew()
    instance.__dict__.clear()
    instance.stream = stream
    return instance


def _make_agent(role="Researcher", model="gpt-4"):
    Agent = type("Agent", (), {})
    instance = Agent()
    instance.role = role
    instance.__dict__ = {"role": role}
    llm = MagicMock()
    llm.model = model
    instance.llm = llm
    instance._token_process = MagicMock()
    instance._token_process.get_summary.return_value = MagicMock(prompt_tokens=10, completion_tokens=20)
    return instance


def _make_task(description="Do the thing"):
    Task = type("Task", (), {})
    instance = Task()
    instance.description = description
    return instance


def _make_llm(model="gpt-4"):
    LLM = type("LLM", (), {})
    instance = LLM()
    instance.model = model
    instance.last_token_usage = None
    return instance


# ---------------------------------------------------------------------------
# Crew.akickoff -> crewai.workflow span
# ---------------------------------------------------------------------------


class TestWrapAkickoff:
    def _run(self, tracer, instance, result="crew result"):
        async def real_akickoff(*a, **k):
            return result

        wrapper = wrap_akickoff(tracer, None, None)
        return asyncio.run(wrapper(real_akickoff, instance, (), {}))

    def test_creates_workflow_span(self, tracer, exporter):
        self._run(tracer, _make_crew())
        spans = spans_named(exporter, "crewai.workflow")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("gen_ai.provider.name") == "crewai"
        assert attrs.get("gen_ai.operation.name") == GenAiOperationNameValues.INVOKE_AGENT.value

    def test_deprecated_gen_ai_system_not_emitted(self, tracer, exporter):
        self._run(tracer, _make_crew())
        attrs = dict(spans_named(exporter, "crewai.workflow")[0].attributes or {})
        assert GenAIAttributes.GEN_AI_SYSTEM not in attrs

    def test_streaming_defers_to_nested_call(self, tracer, exporter):
        result = self._run(tracer, _make_crew(stream=True), result="streaming-output")
        assert result == "streaming-output"
        assert spans_named(exporter, "crewai.workflow") == []

    def test_error_sets_error_status(self, tracer, exporter):
        async def boom(*a, **k):
            raise RuntimeError("kaboom")

        wrapper = wrap_akickoff(tracer, None, None)
        with pytest.raises(RuntimeError):
            asyncio.run(wrapper(boom, _make_crew(), (), {}))
        assert spans_named(exporter, "crewai.workflow")[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Agent.aexecute_task -> <role>.agent span
# ---------------------------------------------------------------------------


class TestWrapAexecuteTask:
    def _run(self, tracer, role="Analyst"):
        instance = _make_agent(role=role)

        async def real_aexecute_task(*a, **k):
            return "done"

        wrapper = wrap_aexecute_task(tracer, None, None)
        asyncio.run(wrapper(real_aexecute_task, instance, (), {}))

    def test_span_name_and_attrs(self, tracer, exporter):
        self._run(tracer, role="Analyst")
        span = spans_named(exporter, "Analyst.agent")[0]
        attrs = dict(span.attributes or {})
        assert attrs.get("gen_ai.provider.name") == "crewai"
        assert attrs.get("gen_ai.operation.name") == GenAiOperationNameValues.INVOKE_AGENT.value
        assert attrs.get("gen_ai.agent.name") == "Analyst"

    def test_usage_tokens_recorded(self, tracer, exporter):
        self._run(tracer)
        attrs = dict(spans_named(exporter, "Analyst.agent")[0].attributes or {})
        assert attrs.get("gen_ai.usage.input_tokens") == 10
        assert attrs.get("gen_ai.usage.output_tokens") == 20

    def test_error_sets_error_status(self, tracer, exporter):
        instance = _make_agent()

        async def boom(*a, **k):
            raise ValueError("fail")

        wrapper = wrap_aexecute_task(tracer, None, None)
        with pytest.raises(ValueError):
            asyncio.run(wrapper(boom, instance, (), {}))
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Task.aexecute_sync -> <description>.task span
# ---------------------------------------------------------------------------


class TestWrapAexecuteSync:
    def test_task_span_with_output(self, tracer, exporter):
        instance = _make_task("async task")

        async def real_aexecute_sync(*a, **k):
            return "task output"

        wrapper = wrap_aexecute_sync(tracer, None, None)
        asyncio.run(wrapper(real_aexecute_sync, instance, (), {}))

        span = spans_named(exporter, "async task.task")[0]
        attrs = dict(span.attributes or {})
        assert attrs.get("gen_ai.provider.name") == "crewai"
        assert attrs.get("gen_ai.operation.name") == GenAiOperationNameValues.INVOKE_AGENT.value
        assert attrs.get("traceloop.entity.output") == "task output"

    def test_error_sets_error_status(self, tracer, exporter):
        instance = _make_task()

        async def boom(*a, **k):
            raise RuntimeError("boom")

        wrapper = wrap_aexecute_sync(tracer, None, None)
        with pytest.raises(RuntimeError):
            asyncio.run(wrapper(boom, instance, (), {}))
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# LLM.acall -> <model>.llm span
# ---------------------------------------------------------------------------


class TestWrapAcall:
    def _run(self, tracer, model="gpt-4", messages=None):
        instance = _make_llm(model=model)

        async def real_acall(*a, **k):
            return "response text"

        wrapper = wrap_acall(tracer, None, None)
        args = (messages,) if messages is not None else ()
        asyncio.run(wrapper(real_acall, instance, args, {}))

    def test_span_name_includes_model(self, tracer, exporter):
        self._run(tracer, model="claude-3-5")
        assert spans_named(exporter, "claude-3-5.llm")

    def test_operation_is_chat(self, tracer, exporter):
        self._run(tracer, model="gpt-4o")
        attrs = dict(spans_named(exporter, "gpt-4o.llm")[0].attributes or {})
        assert attrs.get("gen_ai.operation.name") == GenAiOperationNameValues.CHAT.value

    def test_provider_inferred(self, tracer, exporter):
        self._run(tracer, model="claude-3-5-sonnet")
        attrs = dict(spans_named(exporter, "claude-3-5-sonnet.llm")[0].attributes or {})
        assert attrs.get("gen_ai.provider.name") == "anthropic"

    def test_input_output_messages_captured(self, tracer, exporter):
        self._run(tracer, model="gpt-4", messages="What is 2+2?")
        attrs = dict(spans_named(exporter, "gpt-4.llm")[0].attributes or {})
        assert "gen_ai.input.messages" in attrs
        assert "gen_ai.output.messages" in attrs

    def test_error_sets_error_status(self, tracer, exporter):
        instance = _make_llm(model="gpt-4")

        async def boom(*a, **k):
            raise ConnectionError("no api key")

        wrapper = wrap_acall(tracer, None, None)
        with pytest.raises(ConnectionError):
            asyncio.run(wrapper(boom, instance, (), {}))
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# End-to-end: real Crew.akickoff drives the whole native-async chain
# ---------------------------------------------------------------------------


class TestAkickoffEndToEnd:
    def test_native_async_chain_nests_spans(self, exporter):
        crewai = pytest.importorskip("crewai")
        if not hasattr(crewai.Crew, "akickoff"):
            pytest.skip("crewai < 1.7.0 has no native async akickoff")

        from crewai import Agent, Crew, Task
        from crewai.llms.base_llm import BaseLLM
        from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

        class StubLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None, available_functions=None,
                     from_task=None, from_agent=None, response_model=None):
                return "Final Answer: mocked"

            async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                            from_task=None, from_agent=None, response_model=None):
                return "Final Answer: mocked"

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        instrumentor = CrewAIInstrumentor()
        instrumentor.instrument(tracer_provider=provider)
        try:
            llm = StubLLM(model="gpt-4o")
            agent = Agent(role="Researcher", goal="g", backstory="b", llm=llm, max_iter=1)
            task = Task(description="Say hi", expected_output="a greeting", agent=agent)
            crew = Crew(agents=[agent], tasks=[task], tracing=False)

            asyncio.run(crew.akickoff(inputs={}))
        finally:
            instrumentor.uninstrument()

        spans = exporter.get_finished_spans()
        by_id = {s.context.span_id: s for s in spans}

        workflow = next(s for s in spans if s.name == "crewai.workflow")
        task_span = next(s for s in spans if s.name == "Say hi.task")
        agent_span = next(s for s in spans if s.name == "Researcher.agent")

        # crewai.workflow is a root created by akickoff; task nests under it,
        # agent nests under the task — proving the native async chain is traced.
        assert task_span.parent is not None
        assert by_id[task_span.parent.span_id].name == "crewai.workflow"
        assert agent_span.parent is not None
        assert by_id[agent_span.parent.span_id].name == "Say hi.task"
        assert workflow.attributes.get("gen_ai.provider.name") == "crewai"
