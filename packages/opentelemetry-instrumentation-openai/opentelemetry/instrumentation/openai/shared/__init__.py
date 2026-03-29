import json
import logging
import types
import openai
import pydantic
from importlib.metadata import version

from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import (
    dont_throw,
    is_openai_v1,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

OPENAI_LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens"]
PROMPT_FILTER_KEY = "prompt_filter_results"
OPENAI_FINISH_REASON_MAP = {
    "tool_calls": "tool_call",
    "function_call": "tool_call",
}

_PYDANTIC_VERSION = version("pydantic")


logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is None or value == "":
        return

    if hasattr(openai, "NOT_GIVEN") and value == openai.NOT_GIVEN:
        return

    span.set_attribute(name, value)


def _set_client_attributes(span, instance):
    if not span.is_recording():
        return

    if not is_openai_v1():
        return

    client = instance._client  # pylint: disable=protected-access
    if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_OPENAI_API_BASE, str(client.base_url)
        )
    if isinstance(client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)):
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_OPENAI_API_VERSION, client._api_version
        )  # pylint: disable=protected-access


def _set_api_attributes(span):
    if not span.is_recording():
        return

    if is_openai_v1():
        return

    base_url = openai.base_url if hasattr(openai, "base_url") else openai.api_base

    _set_span_attribute(span, SpanAttributes.GEN_AI_OPENAI_API_BASE, base_url)
    _set_span_attribute(span, SpanAttributes.GEN_AI_OPENAI_API_TYPE, openai.api_type)
    _set_span_attribute(span, SpanAttributes.GEN_AI_OPENAI_API_VERSION, openai.api_version)

    return


def _parse_arguments(raw_args):
    """Best-effort parse of a JSON argument string to dict. Falls back to raw string."""
    if raw_args is None:
        return None
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return raw_args


def _build_tool_def_dict(function_dict, tool_type=None):
    """Build a tool definition dict matching OTel source system format."""
    tool_def = {}
    t = tool_type or function_dict.get("type")
    if t:
        tool_def["type"] = t
    if function_dict.get("name"):
        tool_def["name"] = function_dict["name"]
    if function_dict.get("description"):
        tool_def["description"] = function_dict["description"]
    if function_dict.get("parameters"):
        tool_def["parameters"] = function_dict["parameters"]
    return tool_def


def _set_tool_definitions_json(span, tool_defs):
    """Set gen_ai.tool.definitions as a single JSON string attribute."""
    if tool_defs:
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs)
        )


def _set_functions_attributes(span, functions):
    if not functions:
        return

    tool_defs = [
        d for f in functions
        if (d := _build_tool_def_dict(f, tool_type="function"))
    ]
    _set_tool_definitions_json(span, tool_defs)


def set_tools_attributes(span, tools):
    if not tools:
        return

    tool_defs = [
        d for tool in tools
        if tool.get("function")
        and (d := _build_tool_def_dict(tool["function"], tool_type=tool.get("type")))
    ]
    _set_tool_definitions_json(span, tool_defs)


def _set_request_attributes(span, kwargs, instance=None):
    if not span.is_recording():
        return

    _set_api_attributes(span)

    base_url = _get_openai_base_url(instance) if instance else ""
    vendor = _get_vendor_from_url(base_url)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_PROVIDER_NAME, vendor)

    model = kwargs.get("model")
    if vendor == "aws.bedrock" and model and "." in model:
        model = _cross_region_check(model)
    elif vendor == "openrouter":
        model = _extract_model_name_from_provider_format(model)

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(span, SpanAttributes.GEN_AI_USER, kwargs.get("user"))
    headers = kwargs.get("extra_headers") or kwargs.get("headers")
    if headers is not None:
        _set_span_attribute(span, SpanAttributes.GEN_AI_HEADERS, str(headers))
    _set_span_attribute(
        span, SpanAttributes.GEN_AI_IS_STREAMING, kwargs.get("stream") or False
    )
    _set_span_attribute(
        span, OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER, kwargs.get("service_tier")
    )
    if response_format := kwargs.get("response_format"):
        # backward-compatible check for
        # openai.types.shared_params.response_format_json_schema.ResponseFormatJSONSchema
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and response_format.get("json_schema")
        ):
            schema = dict(response_format.get("json_schema")).get("schema")
            if schema:
                _set_span_attribute(
                    span,
                    SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                    json.dumps(schema),
                )
        elif (
            isinstance(response_format, pydantic.BaseModel)
            or (
                hasattr(response_format, "model_json_schema")
                and callable(response_format.model_json_schema)
            )
        ):
            _set_span_attribute(
                span,
                SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(response_format.model_json_schema()),
            )
        else:
            schema = None
            try:
                schema = json.dumps(pydantic.TypeAdapter(response_format).json_schema())
            except Exception:
                try:
                    schema = json.dumps(response_format)
                except Exception:
                    pass

            if schema:
                _set_span_attribute(
                    span,
                    SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                    schema,
                )


@dont_throw
def _set_response_attributes(span, response):
    if not span.is_recording():
        return

    if "error" in response:
        error_data = response.get("error")
        if isinstance(error_data, dict):
            error_type = error_data.get("type") or error_data.get("code") or "api_error"
        else:
            error_type = "api_error"
        _set_span_attribute(span, "error.type", error_type)
        span.set_status(Status(StatusCode.ERROR, str(error_data)))
        return

    response_model = response.get("model")
    if response_model:
        response_model = _extract_model_name_from_provider_format(response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    # Set gen_ai.response.finish_reasons (top-level, not gated by content opt-in)
    choices = response.get("choices")
    if choices:
        finish_reasons = tuple(
            OPENAI_FINISH_REASON_MAP.get(c.get("finish_reason"), c.get("finish_reason"))
            for c in choices
            if c.get("finish_reason")
        )
        if finish_reasons:
            _set_span_attribute(
                span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
            )

    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
        response.get("system_fingerprint"),
    )
    _set_span_attribute(
        span,
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER,
        response.get("service_tier"),
    )
    _log_prompt_filter(span, response)
    usage = response.get("usage")
    if not usage:
        return

    if is_openai_v1() and not isinstance(usage, dict):
        usage = usage.__dict__

    _set_span_attribute(
        span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        usage.get("completion_tokens"),
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens")
    )
    prompt_tokens_details = dict(usage.get("prompt_tokens_details", {}))
    _set_span_attribute(
        span,
        SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        prompt_tokens_details.get("cached_tokens", 0),
    )
    return


def _log_prompt_filter(span, response_dict):
    if response_dict.get("prompt_filter_results"):
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}",
            json.dumps(response_dict.get("prompt_filter_results")),
        )


@dont_throw
def _set_span_stream_usage(span, prompt_tokens, completion_tokens):
    if not span.is_recording():
        return

    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )

    if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)

    if (
        isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
        and completion_tokens + prompt_tokens >= 0
    ):
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
            completion_tokens + prompt_tokens,
        )


def _get_openai_base_url(instance):
    if hasattr(instance, "_client"):
        client = instance._client  # pylint: disable=protected-access
        if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
            return str(client.base_url)

    return ""


def _get_vendor_from_url(base_url):
    if not base_url:
        return "openai"

    if "openai.azure.com" in base_url:
        return "azure.ai.openai"
    elif "amazonaws.com" in base_url or "bedrock" in base_url:
        return "aws.bedrock"
    elif "googleapis.com" in base_url or "vertex" in base_url:
        return "gcp.vertex_ai"
    elif "openrouter.ai" in base_url:
        return "openrouter"

    return "openai"


def _cross_region_check(value):
    if not value or "." not in value:
        return value

    prefixes = ["us", "us-gov", "eu", "apac"]
    if any(value.startswith(prefix + ".") for prefix in prefixes):
        parts = value.split(".")
        if len(parts) > 2:
            return parts[2]
        else:
            return value
    else:
        vendor, model = value.split(".", 1)
        return model


def _extract_model_name_from_provider_format(model_name):
    """
    Extract model name from provider/model format.
    E.g., 'openai/gpt-4o' -> 'gpt-4o', 'anthropic/claude-3-sonnet' -> 'claude-3-sonnet'
    """
    if not model_name:
        return model_name

    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]  # Return the last part (actual model name)

    return model_name


def is_streaming_response(response):
    if is_openai_v1():
        return isinstance(response, openai.Stream) or isinstance(
            response, openai.AsyncStream
        )

    return isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    )


def model_as_dict(model):
    if isinstance(model, dict):
        return model
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):  # Raw API response
        return model_as_dict(model.parse())
    else:
        return model


def _token_type(token_type: str):
    if token_type == "prompt_tokens":
        return "input"
    elif token_type == "completion_tokens":
        return "output"

    return None


def metric_shared_attributes(
    response_model: str, operation: str, server_address: str, is_streaming: bool = False
):
    attributes = Config.get_common_metrics_attributes()
    vendor = _get_vendor_from_url(server_address)

    return {
        **attributes,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: vendor,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
        "server.address": server_address,
        "stream": is_streaming,
    }


def propagate_trace_context(span, kwargs):
    if is_openai_v1():
        extra_headers = kwargs.get("extra_headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
        kwargs["extra_headers"] = extra_headers
    else:
        headers = kwargs.get("headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(headers, context=ctx)
        kwargs["headers"] = headers
