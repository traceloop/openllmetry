"""
Tests for CrewAI instrumentation — OTel GenAI semantic convention attribute names.

All attribute assertions use a real TracerProvider + InMemorySpanExporter so we
verify the names that actually land on emitted spans, not mock call arguments.

Coverage:
  - CrewAISpanAttributes._process_llm: all field → attribute name mappings
  - wrap_* spans: gen_ai.provider.name + gen_ai.operation.name (not deprecated gen_ai.system)
  - wrap_kickoff / agent / task: operation invoke_agent; wrap_llm_call: operation chat
  - Absence of every legacy llm.* attribute name
"""

import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiOperationNameValues

from opentelemetry.instrumentation.crewai.crewai_span_attributes import CrewAISpanAttributes
from opentelemetry.instrumentation.crewai.instrumentation import (
    wrap_kickoff,
    wrap_agent_execute_task,
    wrap_llm_call,
    wrap_task_execute,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def span_attrs(exporter, span_name=None):
    """Return attribute dict for the first (or named) finished span."""
    spans = exporter.get_finished_spans()
    if span_name:
        spans = [s for s in spans if s.name == span_name]
    assert spans, f"No finished spans{' named ' + repr(span_name) if span_name else ''}"
    return dict(spans[0].attributes or {})


def make_llm_instance(**overrides):
    """
    Return a plain object whose __class__.__name__ == "LLM" so that
    CrewAISpanAttributes dispatches to _process_llm correctly.
    Using type() avoids the MagicMock __class__ mutation issues.
    """
    LLM = type("LLM", (), {})
    instance = LLM()
    defaults = dict(
        model="gpt-4",
        temperature=None,
        top_p=None,
        n=None,
        stop=None,
        max_completion_tokens=None,
        max_tokens=None,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(instance, k, v)
    return instance


def make_crew_instance():
    """Plain object with __class__.__name__ == 'Crew', minimal __dict__."""
    Crew = type("Crew", (), {})
    instance = Crew()
    instance.__dict__.clear()
    return instance


def make_agent_instance(role="Researcher", model="gpt-4"):
    """Plain object with __class__.__name__ == 'Agent'."""
    Agent = type("Agent", (), {})
    instance = Agent()
    instance.role = role
    instance.__dict__ = {"role": role}
    llm = MagicMock()
    llm.model = model
    instance.llm = llm
    instance._token_process = MagicMock()
    instance._token_process.get_summary.return_value = MagicMock(
        prompt_tokens=10, completion_tokens=20
    )
    return instance


# ---------------------------------------------------------------------------
# _process_llm — frequency_penalty and presence_penalty
# ---------------------------------------------------------------------------


class TestProcessLlmPenalties:
    """frequency_penalty and presence_penalty must use gen_ai.request.* namespace."""

    def test_frequency_penalty_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(frequency_penalty=0.3))
        assert span_attrs(exporter).get("gen_ai.request.frequency_penalty") == 0.3

    def test_frequency_penalty_old_name_absent(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(frequency_penalty=0.3))
        assert "llm.frequency_penalty" not in span_attrs(exporter)

    def test_presence_penalty_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(presence_penalty=0.5))
        assert span_attrs(exporter).get("gen_ai.request.presence_penalty") == 0.5

    def test_presence_penalty_old_name_absent(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(presence_penalty=0.5))
        assert "llm.presence_penalty" not in span_attrs(exporter)

    def test_penalties_not_set_when_none(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance())
        attrs = span_attrs(exporter)
        assert "gen_ai.request.frequency_penalty" not in attrs
        assert "gen_ai.request.presence_penalty" not in attrs

    def test_semconv_constants_are_migrated(self):
        """Guard: the upstream constants must have the expected gen_ai.* strings."""
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
            GEN_AI_REQUEST_FREQUENCY_PENALTY,
            GEN_AI_REQUEST_PRESENCE_PENALTY,
        )
        assert GEN_AI_REQUEST_FREQUENCY_PENALTY == "gen_ai.request.frequency_penalty"
        assert GEN_AI_REQUEST_PRESENCE_PENALTY == "gen_ai.request.presence_penalty"


# ---------------------------------------------------------------------------
# _process_llm — fallback fields (n, stop, max_completion_tokens, seed)
# ---------------------------------------------------------------------------


class TestProcessLlmFallbackFields:
    """Fields that previously fell through to f'llm.{field}' must now use gen_ai.*."""

    def test_seed_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(seed=42))
        attrs = span_attrs(exporter)
        assert "gen_ai.request.seed" in attrs
        assert attrs["gen_ai.request.seed"] == 42

    def test_seed_old_name_absent(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(seed=42))
        assert "llm.seed" not in span_attrs(exporter)

    def test_stop_sequences_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(stop=["END", "STOP"]))
        attrs = span_attrs(exporter)
        assert "gen_ai.request.stop_sequences" in attrs
        assert attrs["gen_ai.request.stop_sequences"] == ("END", "STOP")
        assert "llm.stop" not in attrs

    def test_n_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(n=3))
        attrs = span_attrs(exporter)
        assert "gen_ai.request.n" in attrs
        assert "llm.n" not in attrs

    def test_max_completion_tokens_attribute_name(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(max_completion_tokens=512))
        attrs = span_attrs(exporter)
        assert "gen_ai.request.max_completion_tokens" in attrs
        assert "llm.max_completion_tokens" not in attrs

    def test_fallback_fields_absent_when_none(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance())
        attrs = span_attrs(exporter)
        for name in ("gen_ai.request.seed", "gen_ai.request.stop_sequences",
                     "gen_ai.request.n", "gen_ai.request.max_completion_tokens"):
            assert name not in attrs


# ---------------------------------------------------------------------------
# _process_llm — standard OTel fields unchanged
# ---------------------------------------------------------------------------


class TestProcessLlmStandardFields:
    """Fields already mapped via SpanAttributes constants must still be correct."""

    def test_model(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(model="gpt-4o"))
        assert span_attrs(exporter).get("gen_ai.request.model") == "gpt-4o"

    def test_temperature(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(temperature=0.7))
        assert span_attrs(exporter).get("gen_ai.request.temperature") == 0.7

    def test_top_p(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(top_p=0.95))
        assert span_attrs(exporter).get("gen_ai.request.top_p") == 0.95

    def test_max_tokens(self, tracer, exporter):
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=make_llm_instance(max_tokens=2048))
        assert span_attrs(exporter).get("gen_ai.request.max_tokens") == 2048


# ---------------------------------------------------------------------------
# _process_llm — comprehensive sweep: zero legacy llm.* attribute names
# ---------------------------------------------------------------------------


class TestProcessLlmNoLegacyNames:
    def test_no_llm_prefix_attributes(self, tracer, exporter):
        instance = make_llm_instance(
            model="gpt-4",
            temperature=0.7,
            top_p=0.9,
            n=2,
            stop=["END"],
            max_completion_tokens=512,
            max_tokens=1024,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            seed=1,
        )
        with tracer.start_as_current_span("test") as span:
            CrewAISpanAttributes(span=span, instance=instance)
        legacy = [k for k in span_attrs(exporter) if k.startswith("llm.")]
        assert legacy == [], f"Found legacy llm.* attributes: {legacy}"


# ---------------------------------------------------------------------------
# wrap_kickoff — gen_ai.provider.name on span (not deprecated gen_ai.system)
# (CrewAISpanAttributes patched out — we're testing what the wrapper itself sets)
# ---------------------------------------------------------------------------


class TestWrapKickoff:
    @pytest.fixture(autouse=True)
    def _patch_span_attrs(self):
        with patch("opentelemetry.instrumentation.crewai.instrumentation.CrewAISpanAttributes"):
            yield

    def _run(self, tracer, result=None):
        instance = make_crew_instance()
        mock_wrapped = MagicMock(return_value=result)
        wrap_kickoff(tracer, None, None)(mock_wrapped, instance, [], {})

    def test_deprecated_gen_ai_system_not_emitted(self, tracer, exporter):
        self._run(tracer)
        assert GenAIAttributes.GEN_AI_SYSTEM not in span_attrs(exporter, "crewai.workflow")

    def test_gen_ai_provider_name_is_crewai(self, tracer, exporter):
        self._run(tracer)
        assert span_attrs(exporter, "crewai.workflow").get("gen_ai.provider.name") == "crewai"

    def test_gen_ai_operation_name_is_invoke_agent(self, tracer, exporter):
        self._run(tracer)
        assert (
            span_attrs(exporter, "crewai.workflow").get("gen_ai.operation.name")
            == GenAiOperationNameValues.INVOKE_AGENT.value
        )

    def test_span_name(self, tracer, exporter):
        self._run(tracer)
        assert any(s.name == "crewai.workflow" for s in exporter.get_finished_spans())

    def test_error_sets_error_status(self, tracer, exporter):
        from opentelemetry.trace import StatusCode
        instance = make_crew_instance()
        mock_wrapped = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            wrap_kickoff(tracer, None, None)(mock_wrapped, instance, [], {})
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# wrap_agent_execute_task — gen_ai.provider.name on span
# ---------------------------------------------------------------------------


class TestWrapAgentExecuteTask:
    @pytest.fixture(autouse=True)
    def _patch_span_attrs(self):
        with patch("opentelemetry.instrumentation.crewai.instrumentation.CrewAISpanAttributes"):
            yield

    def _run(self, tracer, role="Researcher"):
        instance = make_agent_instance(role=role)
        mock_wrapped = MagicMock(return_value="done")
        wrap_agent_execute_task(tracer, None, None)(mock_wrapped, instance, [], {})

    def test_deprecated_gen_ai_system_not_emitted(self, tracer, exporter):
        self._run(tracer)
        assert GenAIAttributes.GEN_AI_SYSTEM not in span_attrs(exporter)

    def test_gen_ai_provider_name_is_crewai(self, tracer, exporter):
        self._run(tracer)
        assert span_attrs(exporter).get("gen_ai.provider.name") == "crewai"

    def test_gen_ai_operation_name_is_invoke_agent(self, tracer, exporter):
        self._run(tracer)
        assert (
            span_attrs(exporter).get("gen_ai.operation.name")
            == GenAiOperationNameValues.INVOKE_AGENT.value
        )

    def test_span_name_includes_role(self, tracer, exporter):
        self._run(tracer, role="Analyst")
        assert exporter.get_finished_spans()[0].name == "Analyst.agent"

    def test_error_sets_error_status(self, tracer, exporter):
        from opentelemetry.trace import StatusCode
        instance = make_agent_instance()
        mock_wrapped = MagicMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError):
            wrap_agent_execute_task(tracer, None, None)(mock_wrapped, instance, [], {})
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# wrap_task_execute — gen_ai span attributes on task span
# ---------------------------------------------------------------------------


class TestWrapTaskExecute:
    @pytest.fixture(autouse=True)
    def _patch_span_attrs(self):
        with patch("opentelemetry.instrumentation.crewai.instrumentation.CrewAISpanAttributes"):
            yield

    def _run(self, tracer, description="Do the thing"):
        Task = type("Task", (), {})
        instance = Task()
        instance.description = description
        mock_wrapped = MagicMock(return_value="task output")
        wrap_task_execute(tracer, None, None)(mock_wrapped, instance, [], {})

    def test_gen_ai_provider_and_operation(self, tracer, exporter):
        self._run(tracer)
        attrs = span_attrs(exporter)
        assert GenAIAttributes.GEN_AI_SYSTEM not in attrs
        assert attrs.get("gen_ai.provider.name") == "crewai"
        assert attrs.get("gen_ai.operation.name") == GenAiOperationNameValues.INVOKE_AGENT.value


# ---------------------------------------------------------------------------
# wrap_llm_call — inferred gen_ai.provider.name for underlying LLM (not "crewai")
# ---------------------------------------------------------------------------


class TestWrapLlmCall:
    @pytest.fixture(autouse=True)
    def _patch_span_attrs(self):
        with patch("opentelemetry.instrumentation.crewai.instrumentation.CrewAISpanAttributes"):
            yield

    def _run(self, tracer, model="gpt-4"):
        instance = make_llm_instance(model=model)
        mock_wrapped = MagicMock(return_value="response text")
        wrap_llm_call(tracer, None, None)(mock_wrapped, instance, [], {})

    def test_deprecated_gen_ai_system_not_emitted(self, tracer, exporter):
        self._run(tracer)
        assert GenAIAttributes.GEN_AI_SYSTEM not in span_attrs(exporter)

    def test_gen_ai_provider_name_inferred_openai_for_gpt(self, tracer, exporter):
        self._run(tracer, model="gpt-4")
        assert span_attrs(exporter).get("gen_ai.provider.name") == "openai"

    def test_gen_ai_provider_name_inferred_anthropic_for_claude(self, tracer, exporter):
        self._run(tracer, model="claude-3-5-sonnet")
        assert span_attrs(exporter).get("gen_ai.provider.name") == "anthropic"

    def test_gen_ai_provider_name_omitted_when_unknown_model(self, tracer, exporter):
        self._run(tracer, model="custom-local-model")
        assert "gen_ai.provider.name" not in span_attrs(exporter)

    def test_gen_ai_operation_name_is_chat(self, tracer, exporter):
        self._run(tracer)
        assert (
            span_attrs(exporter).get("gen_ai.operation.name")
            == GenAiOperationNameValues.CHAT.value
        )

    def test_span_name_includes_model(self, tracer, exporter):
        self._run(tracer, model="claude-3-5")
        assert exporter.get_finished_spans()[0].name == "claude-3-5.llm"

    def test_error_sets_error_status(self, tracer, exporter):
        from opentelemetry.trace import StatusCode
        instance = make_llm_instance(model="gpt-4")
        mock_wrapped = MagicMock(side_effect=ConnectionError("no api key"))
        with pytest.raises(ConnectionError):
            wrap_llm_call(tracer, None, None)(mock_wrapped, instance, [], {})
        assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR
