"""Tests for the prompt handling of the text generation pipeline wrapper.

The wrapped pipeline call is replaced by a canned pipeline response, so these
tests need neither a model download nor a network connection. The wrapper, the
input attributes and the event emitter are the real instrumented code paths.
"""

from types import SimpleNamespace

import transformers

from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.transformers.config import Config
from opentelemetry.instrumentation.transformers.text_generation_pipeline_wrapper import (  # noqa: E501
    text_generation_pipeline_wrapper,
)
from opentelemetry.instrumentation.transformers.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

PROMPT = "Tell me a joke about OpenTelemetry."
COMPLETION = "Because it has too many spans to keep track of."
SPAN_NAME = "transformers_text_generation_pipeline.call"
TO_WRAP = {"span_name": SPAN_NAME, "method": "__call__"}


def _pipeline_stand_in():
    """A TextGenerationPipeline that never runs __init__, so no model is loaded."""
    pipeline = object.__new__(transformers.TextGenerationPipeline)
    pipeline._forward_params = {"temperature": 0.7, "max_length": 16}
    pipeline.model = SimpleNamespace(
        config=SimpleNamespace(name_or_path="gpt2", model_type="gpt2")
    )
    return pipeline


def _call_pipeline(tracer, event_logger, args, kwargs):
    def wrapped(*args, **kwargs):
        return [{"generated_text": COMPLETION}]

    instrumented = text_generation_pipeline_wrapper(tracer, event_logger, TO_WRAP)
    return instrumented(wrapped, _pipeline_stand_in(), args, kwargs)


def test_prompt_passed_positionally_is_recorded(
    span_exporter, tracer_provider, monkeypatch
):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")
    monkeypatch.setattr(Config, "use_legacy_attributes", True)

    _call_pipeline(tracer_provider.get_tracer(__name__), None, (PROMPT,), {})

    span = span_exporter.get_finished_spans()[0]
    assert span.name == SPAN_NAME
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == PROMPT
    assert (
        span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == COMPLETION
    )


def test_prompt_passed_by_keyword_is_recorded(
    span_exporter, tracer_provider, monkeypatch
):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")
    monkeypatch.setattr(Config, "use_legacy_attributes", True)

    # ``text_inputs`` is the keyword the pipeline itself declares for its input.
    _call_pipeline(
        tracer_provider.get_tracer(__name__), None, (), {"text_inputs": PROMPT}
    )

    span = span_exporter.get_finished_spans()[0]
    assert span.name == SPAN_NAME
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == PROMPT
    assert (
        span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == COMPLETION
    )


def test_prompt_passed_by_keyword_emits_prompt_event(
    span_exporter, log_exporter, tracer_provider, logger_provider, monkeypatch
):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")
    monkeypatch.setattr(Config, "use_legacy_attributes", False)
    event_logger = get_logger(__name__, logger_provider=logger_provider)

    _call_pipeline(
        tracer_provider.get_tracer(__name__), event_logger, (), {"text_inputs": PROMPT}
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert logs[0].log_record.event_name == "gen_ai.user.message"
    assert dict(logs[0].log_record.body) == {"content": PROMPT}
    assert (
        logs[0].log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == "transformers"
    )

    assert logs[1].log_record.event_name == "gen_ai.choice"
    assert dict(logs[1].log_record.body)["message"] == {"content": COMPLETION}
