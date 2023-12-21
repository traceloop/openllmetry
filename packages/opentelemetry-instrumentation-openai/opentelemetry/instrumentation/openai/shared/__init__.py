import os
import openai
import json
import types
import logging

from importlib.metadata import version

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.instrumentation.openai.utils import is_openai_v1


OPENAI_API_VERSION = "openai.api_version"
OPENAI_API_BASE = "openai.api_base"
OPENAI_API_TYPE = "openai.api_type"

logger = logging.getLogger(__name__)


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_api_attributes(span):
    if not span.is_recording():
        return

    try:
        base_url = openai.base_url if hasattr(openai, "base_url") else openai.api_base

        _set_span_attribute(span, OPENAI_API_BASE, base_url)
        _set_span_attribute(span, OPENAI_API_TYPE, openai.api_type)
        _set_span_attribute(span, OPENAI_API_VERSION, openai.api_version)
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set api attributes for openai span, error: %s", str(ex)
        )

    return


def _set_functions_attributes(span, functions):
    if not functions:
        return

    for i, function in enumerate(functions):
        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


def _set_request_attributes(span, llm_request_type, kwargs):
    if not span.is_recording():
        return

    try:
        _set_api_attributes(span)
        _set_span_attribute(span, SpanAttributes.LLM_VENDOR, "OpenAI")
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_TYPE, llm_request_type.value
        )
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature")
        )
        _set_span_attribute(span, SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))
        _set_span_attribute(
            span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
        )
        _set_span_attribute(span, SpanAttributes.LLM_USER, kwargs.get("user"))
        _set_span_attribute(
            span, SpanAttributes.LLM_HEADERS, str(kwargs.get("headers"))
        )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for openai span, error: %s", str(ex)
        )


def _set_response_attributes(span, response):
    if not span.is_recording():
        return

    try:
        _set_span_attribute(
            span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model")
        )

        usage = response.get("usage")
        if not usage:
            return

        if is_openai_v1() and not isinstance(usage, dict):
            usage = usage.__dict__

        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )

        return
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set response attributes for openai span, error: %s", str(ex)
        )


def is_streaming_response(response):
    if is_openai_v1():
        return isinstance(response, openai.Stream)

    return isinstance(response, types.GeneratorType) or isinstance(response, types.AsyncGeneratorType)


def model_as_dict(model):
    if version("pydantic") < "2.0.0":
        return model.dict()

    return model.model_dump()
