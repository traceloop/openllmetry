import os
import openai
import json
import types
import logging

from importlib.metadata import version

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.instrumentation.openai.utils import (
    dont_throw,
    is_openai_v1,
    should_record_stream_token_usage,
)

OPENAI_API_VERSION = "openai.api_version"
OPENAI_API_BASE = "openai.api_base"
OPENAI_API_TYPE = "openai.api_type"

OPENAI_LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens"]

# tiktoken encodings map for different model, key is model_name, value is tiktoken encoding
tiktoken_encodings = {}

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


def _set_client_attributes(span, instance):
    if not span.is_recording():
        return

    if not is_openai_v1():
        return

    client = instance._client  # pylint: disable=protected-access
    if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
        _set_span_attribute(span, OPENAI_API_BASE, str(client.base_url))
    if isinstance(client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)):
        _set_span_attribute(
            span, OPENAI_API_VERSION, client._api_version
        )  # pylint: disable=protected-access


def _set_api_attributes(span):
    if not span.is_recording():
        return

    if is_openai_v1():
        return

    base_url = openai.base_url if hasattr(openai, "base_url") else openai.api_base

    _set_span_attribute(span, OPENAI_API_BASE, base_url)
    _set_span_attribute(span, OPENAI_API_TYPE, openai.api_type)
    _set_span_attribute(span, OPENAI_API_VERSION, openai.api_version)

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


def set_tools_attributes(span, tools):
    if not tools:
        return

    for i, tool in enumerate(tools):
        function = tool.get("function")
        if not function:
            continue

        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


def _set_request_attributes(span, kwargs):
    if not span.is_recording():
        return

    _set_api_attributes(span)
    _set_span_attribute(span, SpanAttributes.LLM_VENDOR, "OpenAI")
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(span, SpanAttributes.LLM_USER, kwargs.get("user"))
    _set_span_attribute(span, SpanAttributes.LLM_HEADERS, str(kwargs.get("headers")))
    # The new OpenAI SDK removed the `headers` and create new field called `extra_headers`
    if kwargs.get("extra_headers") is not None:
        _set_span_attribute(
            span, SpanAttributes.LLM_HEADERS, str(kwargs.get("extra_headers"))
        )
    _set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )


@dont_throw
def _set_response_attributes(span, response):
    if not span.is_recording():
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

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


@dont_throw
def _set_span_stream_usage(span, prompt_tokens, completion_tokens):
    if not span.is_recording():
        return

    if type(completion_tokens) is int and completion_tokens >= 0:
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )

    if type(prompt_tokens) is int and prompt_tokens >= 0:
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)

    if (
        type(prompt_tokens) is int
        and type(completion_tokens) is int
        and completion_tokens + prompt_tokens >= 0
    ):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            completion_tokens + prompt_tokens,
        )


def _get_openai_base_url(instance):
    if hasattr(instance, "_client"):
        client = instance._client  # pylint: disable=protected-access
        if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
            return str(client.base_url)

    return ""


def is_streaming_response(response):
    if is_openai_v1():
        return isinstance(response, openai.Stream) or isinstance(
            response, openai.AsyncStream
        )

    return isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    )


def model_as_dict(model):
    if version("pydantic") < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):  # Raw API response
        return model_as_dict(model.parse())
    else:
        return model


def get_token_count_from_string(string: str, model_name: str):
    if not should_record_stream_token_usage():
        return None

    import tiktoken

    if tiktoken_encodings.get(model_name) is None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError as ex:
            # no such model_name in tiktoken
            logger.warning(
                f"Failed to get tiktoken encoding for model_name {model_name}, error: {str(ex)}"
            )
            return None

        tiktoken_encodings[model_name] = encoding
    else:
        encoding = tiktoken_encodings.get(model_name)

    token_count = len(encoding.encode(string))
    return token_count
