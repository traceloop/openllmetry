"""Unit tests for request model extraction in ``set_request_params`` (issue #3098)."""

from opentelemetry.instrumentation.langchain.span_utils import (
    SpanHolder,
    set_request_params,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _make_holder(span):
    return SpanHolder(span, None, None, [], "LangGraph", "", "")


def _request_model(span_exporter):
    span = span_exporter.get_finished_spans()[-1]
    return span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)


def test_request_model_falls_back_to_ls_model_name(tracer_provider, span_exporter):
    """Providers such as ChatBedrockConverse invoked through LangGraph don't expose
    the model via ``kwargs``/``invocation_params``; it is only available in
    LangChain's standard run metadata (``ls_model_name``)."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("chat")
    holder = _make_holder(span)

    set_request_params(
        span,
        {"invocation_params": {"temperature": 0.7}},
        holder,
        {"ls_model_name": "us.anthropic.claude-3-5-haiku-20241022-v1:0"},
    )
    span.end()

    assert holder.request_model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    assert _request_model(span_exporter) == "us.anthropic.claude-3-5-haiku-20241022-v1:0"


def test_request_model_prefers_invocation_params_over_metadata(
    tracer_provider, span_exporter
):
    """When the model is available in ``invocation_params`` it takes precedence over
    the ``ls_model_name`` metadata fallback."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("chat")
    holder = _make_holder(span)

    set_request_params(
        span,
        {"invocation_params": {"model_name": "gpt-4o"}},
        holder,
        {"ls_model_name": "should-not-be-used"},
    )
    span.end()

    assert holder.request_model == "gpt-4o"
    assert _request_model(span_exporter) == "gpt-4o"


def test_request_model_defaults_to_unknown(tracer_provider, span_exporter):
    """With no model in kwargs, invocation params, or metadata the attribute falls
    back to ``unknown`` and the holder is left unset."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("chat")
    holder = _make_holder(span)

    set_request_params(span, {"invocation_params": {}}, holder, None)
    span.end()

    assert holder.request_model is None
    assert _request_model(span_exporter) == "unknown"
