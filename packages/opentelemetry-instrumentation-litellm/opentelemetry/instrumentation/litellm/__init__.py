"""OpenTelemetry LiteLLM instrumentation.

Instruments the LiteLLM *library* (in-process) by wrapping its public module-level
functions ``completion`` / ``acompletion`` / ``embedding`` / ``aembedding``. Because
LiteLLM normalizes every provider's reply into an OpenAI-style ``ModelResponse`` — including
custom providers registered via ``litellm.CustomLLM`` — a single set of wrappers produces
consistent ``gen_ai.*`` spans and metrics across all back-ends.
"""

import json
import logging
import time
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.litellm.config import Config
from opentelemetry.instrumentation.litellm.event_emitter import emit_event
from opentelemetry.instrumentation.litellm.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.litellm.utils import (
    dont_throw,
    is_metrics_enabled,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.litellm.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy, wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("litellm >= 1.0.0",)

# Fallback ``gen_ai.system`` value when the provider cannot be resolved from the
# response or the model prefix. Kept as a literal (mirroring other instrumentors) to
# avoid coupling to a specific opentelemetry-semantic-conventions-ai release that adds
# ``GenAISystem.LITELLM``.
LITELLM_SYSTEM = "litellm"

WRAPPED_METHODS = [
    {
        "method": "completion",
        "span_name": "litellm.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": False,
    },
    {
        "method": "acompletion",
        "span_name": "litellm.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": True,
    },
    {
        "method": "embedding",
        "span_name": "litellm.embeddings",
        "request_type": LLMRequestTypeValues.EMBEDDING,
        "is_async": False,
    },
    {
        "method": "aembedding",
        "span_name": "litellm.embeddings",
        "request_type": LLMRequestTypeValues.EMBEDDING,
        "is_async": True,
    },
]


def _set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def _model_as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return dict(getattr(obj, "__dict__", {}) or {})


# -- provider resolution -----------------------------------------------------


def _resolve_system_from_response(response):
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        return hidden.get("custom_llm_provider")
    return None


def _resolve_system_from_kwargs(kwargs):
    provider = kwargs.get("custom_llm_provider")
    if provider:
        return provider
    model = kwargs.get("model") or ""
    if "/" in model:
        return model.split("/", 1)[0]
    return LITELLM_SYSTEM


def _get_model(args, kwargs):
    return kwargs.get("model") or (args[0] if args else None)


def _get_messages(args, kwargs):
    return kwargs.get("messages") or (args[1] if len(args) > 1 else None)


def _get_embedding_input(args, kwargs):
    return kwargs.get("input") or (args[1] if len(args) > 1 else None)


# -- request attributes ------------------------------------------------------


@dont_throw
def _set_request_attributes(span, kwargs):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, bool(kwargs.get("stream")))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )


@dont_throw
def _set_prompts(span, request_type, args, kwargs):
    if not span.is_recording() or not should_send_prompts():
        return
    if request_type == LLMRequestTypeValues.CHAT:
        for index, message in enumerate(_get_messages(args, kwargs) or []):
            if not isinstance(message, dict):
                message = _model_as_dict(message)
            prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{index}"
            _set_span_attribute(span, f"{prefix}.role", message.get("role"))
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                content = json.dumps(content)
            _set_span_attribute(span, f"{prefix}.content", content)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                _set_tool_calls(span, prefix, tool_calls)
            _set_span_attribute(
                span, f"{prefix}.tool_call_id", message.get("tool_call_id")
            )
    else:
        embedding_input = _get_embedding_input(args, kwargs)
        if isinstance(embedding_input, str):
            embedding_input = [embedding_input]
        for index, prompt in enumerate(embedding_input or []):
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.role", "user"
            )
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.content", prompt
            )


# -- response attributes -----------------------------------------------------


def _set_tool_calls(span, prefix, tool_calls):
    for index, tool_call in enumerate(tool_calls):
        tool_call = _model_as_dict(tool_call)
        function = tool_call.get("function") or {}
        _set_span_attribute(span, f"{prefix}.tool_calls.{index}.id", tool_call.get("id"))
        _set_span_attribute(
            span, f"{prefix}.tool_calls.{index}.name", function.get("name")
        )
        _set_span_attribute(
            span, f"{prefix}.tool_calls.{index}.arguments", function.get("arguments")
        )


@dont_throw
def _set_response_attributes(span, request_type, response_dict):
    if not span.is_recording() or not response_dict:
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_dict.get("id"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_dict.get("model")
    )

    usage = response_dict.get("usage")
    if usage:
        usage = _model_as_dict(usage)
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens")
        )
        prompt_tokens_details = _model_as_dict(usage.get("prompt_tokens_details"))
        cached_tokens = prompt_tokens_details.get("cached_tokens")
        if cached_tokens is not None:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                cached_tokens,
            )

    if request_type == LLMRequestTypeValues.EMBEDDING:
        return

    if should_send_prompts():
        for choice in response_dict.get("choices", []):
            index = choice.get("index", 0)
            prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
            _set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
            message = choice.get("message") or {}
            _set_span_attribute(span, f"{prefix}.role", message.get("role"))
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                content = json.dumps(content)
            _set_span_attribute(span, f"{prefix}.content", content)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                _set_tool_calls(span, prefix, tool_calls)


# -- events ------------------------------------------------------------------


@dont_throw
def _emit_input_events(request_type, args, kwargs, event_logger):
    if request_type == LLMRequestTypeValues.CHAT:
        for message in _get_messages(args, kwargs) or []:
            if not isinstance(message, dict):
                message = _model_as_dict(message)
            emit_event(
                MessageEvent(
                    content=message.get("content"),
                    role=message.get("role", "user"),
                    tool_calls=message.get("tool_calls"),
                ),
                event_logger,
            )
    else:
        embedding_input = _get_embedding_input(args, kwargs)
        if isinstance(embedding_input, str):
            embedding_input = [embedding_input]
        for prompt in embedding_input or []:
            emit_event(MessageEvent(content=prompt, role="user"), event_logger)


@dont_throw
def _emit_choice_events(request_type, response_dict, event_logger):
    if request_type == LLMRequestTypeValues.EMBEDDING or not response_dict:
        return
    for choice in response_dict.get("choices", []):
        message = choice.get("message") or {}
        emit_event(
            ChoiceEvent(
                index=choice.get("index", 0),
                message={
                    "content": message.get("content"),
                    "role": message.get("role") or "assistant",
                },
                finish_reason=choice.get("finish_reason") or "unknown",
            ),
            event_logger,
        )


def _handle_input(span, request_type, args, kwargs, event_logger):
    _set_request_attributes(span, kwargs)
    if should_emit_events() and event_logger:
        _emit_input_events(request_type, args, kwargs, event_logger)
    else:
        _set_prompts(span, request_type, args, kwargs)


def _handle_response(span, request_type, response_dict, event_logger):
    _set_response_attributes(span, request_type, response_dict)
    if should_emit_events() and event_logger:
        _emit_choice_events(request_type, response_dict, event_logger)


# -- metrics -----------------------------------------------------------------


@dont_throw
def _record_metrics(metrics, system, model, duration, response_dict, request_type):
    if not metrics:
        return
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: system,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }
    common = Config.get_common_metrics_attributes()
    if common:
        attributes.update(common)

    if metrics.get("duration_histogram"):
        metrics["duration_histogram"].record(duration, attributes=attributes)

    usage = response_dict.get("usage")
    if usage and metrics.get("tokens_histogram"):
        usage = _model_as_dict(usage)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            metrics["tokens_histogram"].record(
                prompt_tokens,
                attributes={**attributes, SpanAttributes.GEN_AI_USAGE_TOKEN_TYPE: "input"},
            )
        if completion_tokens is not None:
            metrics["tokens_histogram"].record(
                completion_tokens,
                attributes={**attributes, SpanAttributes.GEN_AI_USAGE_TOKEN_TYPE: "output"},
            )

    choices = response_dict.get("choices")
    if (
        choices
        and request_type == LLMRequestTypeValues.CHAT
        and metrics.get("chat_choice_counter")
    ):
        metrics["chat_choice_counter"].add(len(choices), attributes=attributes)


@dont_throw
def _record_exception_metric(metrics, system, model, error):
    if not metrics or not metrics.get("exception_counter"):
        return
    metrics["exception_counter"].add(
        1,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: system,
            GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
            ERROR_TYPE: error.__class__.__name__,
        },
    )


# -- streaming accumulation --------------------------------------------------


@dont_throw
def _accumulate_chunk(accumulated, chunk):
    chunk_dict = _model_as_dict(chunk)
    if chunk_dict.get("id"):
        accumulated["id"] = chunk_dict["id"]
    if chunk_dict.get("model"):
        accumulated["model"] = chunk_dict["model"]
    if chunk_dict.get("usage"):
        accumulated["usage"] = chunk_dict["usage"]
    if accumulated.get("system") is None:
        system = _resolve_system_from_response(chunk)
        if system:
            accumulated["system"] = system

    for choice in chunk_dict.get("choices", []):
        index = choice.get("index", 0)
        slot = accumulated["choices"].setdefault(
            index,
            {"index": index, "content": "", "role": "assistant", "finish_reason": None, "tool_calls": []},
        )
        delta = choice.get("delta") or {}
        if delta.get("role"):
            slot["role"] = delta["role"]
        if delta.get("content"):
            slot["content"] += delta["content"]
        if choice.get("finish_reason"):
            slot["finish_reason"] = choice["finish_reason"]
        for tool_call in delta.get("tool_calls") or []:
            tool_call = _model_as_dict(tool_call)
            tindex = tool_call.get("index", 0)
            while len(slot["tool_calls"]) <= tindex:
                slot["tool_calls"].append({"id": None, "function": {"name": "", "arguments": ""}})
            tc_slot = slot["tool_calls"][tindex]
            if tool_call.get("id"):
                tc_slot["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                tc_slot["function"]["name"] += function["name"]
            if function.get("arguments"):
                tc_slot["function"]["arguments"] += function["arguments"]


def _build_accumulated_response(accumulated):
    choices = []
    for index in sorted(accumulated["choices"].keys()):
        choice = accumulated["choices"][index]
        message = {"role": choice["role"], "content": choice["content"]}
        if choice["tool_calls"]:
            message["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": tc["function"]}
                for tc in choice["tool_calls"]
            ]
        choices.append(
            {"index": index, "finish_reason": choice["finish_reason"], "message": message}
        )
    return {
        "id": accumulated["id"],
        "model": accumulated["model"],
        "choices": choices,
        "usage": accumulated["usage"],
    }


def _finalize(span, metrics, event_logger, request_type, response, model, kwargs, duration, accumulated_system=None):
    system = (
        accumulated_system
        or _resolve_system_from_response(response)
        or _resolve_system_from_kwargs(kwargs)
    )
    # Serialize the response once and reuse it for attributes, events, and metrics —
    # model_dump() on a ModelResponse is the most expensive step in the wrapper.
    response_dict = _model_as_dict(response)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_SYSTEM, system)
    _handle_response(span, request_type, response_dict, event_logger)
    _record_metrics(metrics, system, model, duration, response_dict, request_type)
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()


def _handle_exception(span, metrics, error, model, kwargs):
    try:
        span.set_attribute(ERROR_TYPE, error.__class__.__name__)
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        _record_exception_metric(metrics, _resolve_system_from_kwargs(kwargs), model, error)
    finally:
        span.end()


def _new_accumulator():
    return {"id": None, "model": None, "choices": {}, "usage": None, "system": None}


# -- wrappers ----------------------------------------------------------------


def _start_span(tracer, to_wrap, kwargs):
    return tracer.start_span(
        to_wrap["span_name"],
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: _resolve_system_from_kwargs(kwargs),
            SpanAttributes.LLM_REQUEST_TYPE: to_wrap["request_type"].value,
        },
    )


def _is_suppressed():
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    )


def _wrap_factory(tracer, metrics, event_logger, to_wrap):
    request_type = to_wrap["request_type"]

    def wrapper(wrapped, instance, args, kwargs):
        if _is_suppressed():
            return wrapped(*args, **kwargs)

        span = _start_span(tracer, to_wrap, kwargs)
        _handle_input(span, request_type, args, kwargs, event_logger)
        model = _get_model(args, kwargs)

        start = time.time()
        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )
        try:
            response = wrapped(*args, **kwargs)
        except Exception as error:
            _handle_exception(span, metrics, error, model, kwargs)
            raise
        finally:
            context_api.detach(token)

        duration = time.time() - start

        if kwargs.get("stream") and response is not None:
            return _LiteLLMStream(
                response, span, metrics, event_logger, to_wrap, model, kwargs, start
            )

        _finalize(span, metrics, event_logger, request_type, response, model, kwargs, duration)
        return response

    return wrapper


def _awrap_factory(tracer, metrics, event_logger, to_wrap):
    request_type = to_wrap["request_type"]

    async def wrapper(wrapped, instance, args, kwargs):
        if _is_suppressed():
            return await wrapped(*args, **kwargs)

        span = _start_span(tracer, to_wrap, kwargs)
        _handle_input(span, request_type, args, kwargs, event_logger)
        model = _get_model(args, kwargs)

        start = time.time()
        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as error:
            _handle_exception(span, metrics, error, model, kwargs)
            raise
        finally:
            context_api.detach(token)

        duration = time.time() - start

        if kwargs.get("stream") and response is not None:
            return _LiteLLMAsyncStream(
                response, span, metrics, event_logger, to_wrap, model, kwargs, start
            )

        _finalize(span, metrics, event_logger, request_type, response, model, kwargs, duration)
        return response

    return wrapper


def _finalize_stream(span, metrics, event_logger, to_wrap, accumulated, model, kwargs, start):
    final = _build_accumulated_response(accumulated)
    _finalize(
        span, metrics, event_logger, to_wrap["request_type"], final, model, kwargs,
        time.time() - start, accumulated_system=accumulated.get("system"),
    )


class _StreamWrapper(ObjectProxy):
    """Proxy over litellm's ``CustomStreamWrapper`` that accumulates chunks and ends
    the span when the stream is exhausted, errors, or is abandoned early — while
    preserving the wrapped object's interface (helper methods, ``isinstance`` checks).

    A bare generator would tie the span's lifecycle to full consumption: an early
    ``break``, a timeout, or an exception in the consumer would leave the span open
    forever. Ending the span from ``__next__``/``__anext__`` (on stop or error) and
    from ``close``/``aclose``/``__del__`` guarantees it always closes exactly once.
    """

    def __init__(self, response, span, metrics, event_logger, to_wrap, model, kwargs, start):
        super().__init__(response)
        self._self_span = span
        self._self_metrics = metrics
        self._self_event_logger = event_logger
        self._self_to_wrap = to_wrap
        self._self_model = model
        self._self_kwargs = kwargs
        self._self_start = start
        self._self_accumulated = _new_accumulator()
        self._self_done = False

    @dont_throw
    def _finish(self):
        if self._self_done:
            return
        self._self_done = True
        _finalize_stream(
            self._self_span, self._self_metrics, self._self_event_logger,
            self._self_to_wrap, self._self_accumulated, self._self_model,
            self._self_kwargs, self._self_start,
        )

    def _finish_error(self, error):
        if self._self_done:
            return
        self._self_done = True
        _handle_exception(
            self._self_span, self._self_metrics, error, self._self_model, self._self_kwargs
        )

    def __del__(self):
        self._finish()


class _LiteLLMStream(_StreamWrapper):
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish_error(error)
            raise
        _accumulate_chunk(self._self_accumulated, chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finish()

    def close(self):
        self._finish()
        wrapped_close = getattr(self.__wrapped__, "close", None)
        if callable(wrapped_close):
            wrapped_close()


class _LiteLLMAsyncStream(_StreamWrapper):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish_error(error)
            raise
        _accumulate_chunk(self._self_accumulated, chunk)
        return chunk

    async def aclose(self):
        self._finish()
        wrapped_aclose = getattr(self.__wrapped__, "aclose", None)
        if callable(wrapped_aclose):
            await wrapped_aclose()


def _build_metrics(meter):
    if not is_metrics_enabled():
        return {}
    return {
        "tokens_histogram": meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        ),
        "duration_histogram": meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        ),
        "chat_choice_counter": meter.create_counter(
            name=Meters.LLM_GENERATION_CHOICES,
            unit="choice",
            description="Number of choices returned by chat completions call",
        ),
        "exception_counter": meter.create_counter(
            name=Meters.LLM_COMPLETIONS_EXCEPTIONS,
            unit="time",
            description="Number of exceptions occurred during chat completions",
        ),
    }


class LiteLLMInstrumentor(BaseInstrumentor):
    """An instrumentor for the LiteLLM library."""

    def __init__(
        self,
        exception_logger=None,
        use_legacy_attributes=True,
        get_common_metrics_attributes=lambda: {},
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        Config.get_common_metrics_attributes = get_common_metrics_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)
        metrics = _build_metrics(meter)

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        for to_wrap in WRAPPED_METHODS:
            factory = _awrap_factory if to_wrap["is_async"] else _wrap_factory
            try:
                wrap_function_wrapper(
                    "litellm",
                    to_wrap["method"],
                    factory(tracer, metrics, event_logger, to_wrap),
                )
            except (AttributeError, ModuleNotFoundError):
                logger.debug("litellm.%s not found, skipping", to_wrap["method"])

    def _uninstrument(self, **kwargs):
        import litellm

        for to_wrap in WRAPPED_METHODS:
            try:
                unwrap(litellm, to_wrap["method"])
            except (AttributeError, ModuleNotFoundError):
                pass
