"""OpenTelemetry IBM Watsonx AI instrumentation"""

import logging
import os
import types
import time
from typing import Collection, Dict, Any, Optional, Union
from opentelemetry.instrumentation.watsonx.config import Config
from opentelemetry.instrumentation.watsonx.utils import dont_throw
from opentelemetry.instrumentation.watsonx.events import (
    prompt_to_event,
    completion_to_event,
)
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import get_meter
from opentelemetry.metrics import Counter, Histogram

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    Meters,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.watsonx.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("ibm-watson-machine-learning >= 1.0.333",)

WRAPPED_METHODS_WATSON_ML_VERSION_1 = [
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "__init__",
        "span_name": "watsonx.model_init",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "generate",
        "span_name": "watsonx.generate",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "generate_text_stream",
        "span_name": "watsonx.generate_text_stream",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "get_details",
        "span_name": "watsonx.get_details",
    },
]

WRAPPED_METHODS_WATSON_AI_VERSION_1 = [
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "__init__",
        "span_name": "watsonx.model_init",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "generate",
        "span_name": "watsonx.generate",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "generate_text_stream",
        "span_name": "watsonx.generate_text_stream",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "get_details",
        "span_name": "watsonx.get_details",
    },
]

WATSON_MODULES = [
    WRAPPED_METHODS_WATSON_ML_VERSION_1,
    WRAPPED_METHODS_WATSON_AI_VERSION_1,
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_api_attributes(span):
    _set_span_attribute(
        span,
        WatsonxSpanAttributes.WATSONX_API_BASE,
        "https://us-south.ml.cloud.ibm.com",
    )
    _set_span_attribute(span, WatsonxSpanAttributes.WATSONX_API_TYPE, "watsonx.ai")
    _set_span_attribute(span, WatsonxSpanAttributes.WATSONX_API_VERSION, "1.0")

    return


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _set_input_attributes(span, instance, kwargs):
    if should_send_prompts() and kwargs is not None and len(kwargs) > 0:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, list):
            for index, input in enumerate(prompt):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.user",
                    input,
                )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.user",
                prompt,
            )

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.model_id)
    # Set other attributes
    modelParameters = instance.params
    if modelParameters is not None:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_DECODING_METHOD,
            modelParameters.get("decoding_method", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_RANDOM_SEED,
            modelParameters.get("random_seed", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_MAX_NEW_TOKENS,
            modelParameters.get("max_new_tokens", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_MIN_NEW_TOKENS,
            modelParameters.get("min_new_tokens", None),
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_TOP_K, modelParameters.get("top_k", None)
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_REPETITION_PENALTY,
            modelParameters.get("repetition_penalty", None),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_REQUEST_TEMPERATURE,
            modelParameters.get("temperature", None),
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_TOP_P, modelParameters.get("top_p", None)
        )

    return


def _set_stream_response_attributes(span, stream_response):
    _set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, stream_response.get("model_id")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        stream_response.get("input_token_count"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        stream_response.get("generated_token_count"),
    )
    total_token = stream_response.get("input_token_count") + stream_response.get(
        "generated_token_count"
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        total_token,
    )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
        stream_response.get("generated_text"),
    )


def _set_completion_content_attributes(
    span, response, index, response_counter
) -> Optional[str]:
    if not isinstance(response, dict):
        return None

    if results := response.get("results"):
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
            results[0]["generated_text"],
        )
        model_id = response.get("model_id")

        if response_counter:
            attributes_with_reason = {
                SpanAttributes.LLM_RESPONSE_MODEL: model_id,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: results[0]["stop_reason"],
            }
            response_counter.add(1, attributes=attributes_with_reason)

        return model_id

    return None


def _token_usage_count(responses):
    prompt_token = 0
    completion_token = 0
    if isinstance(responses, list):
        for response in responses:
            prompt_token += response["results"][0]["input_token_count"]
            completion_token += response["results"][0]["generated_token_count"]
    elif isinstance(responses, dict):
        response = responses
        prompt_token = response["results"][0]["input_token_count"]
        completion_token = response["results"][0]["generated_token_count"]

    return prompt_token, completion_token


@dont_throw
def _set_response_attributes(
    span, responses, token_histogram, response_counter, duration_histogram, duration
):
    if not isinstance(responses, (list, dict)):
        return

    if isinstance(responses, list):
        if len(responses) == 0:
            return
        for index, response in enumerate(responses):
            model_id = _set_completion_content_attributes(
                span, response, index, response_counter
            )
    elif isinstance(responses, dict):
        response = responses
        model_id = _set_completion_content_attributes(
            span, response, 0, response_counter
        )

    if model_id is None:
        return
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model_id)

    prompt_token, completion_token = _token_usage_count(responses)
    if (prompt_token + completion_token) != 0:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            completion_token,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            prompt_token,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_token + completion_token,
        )

        shared_attributes = _metric_shared_attributes(response_model=model_id)

        if token_histogram:
            attributes_with_token_type = {
                **shared_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            }
            token_histogram.record(
                completion_token, attributes=attributes_with_token_type
            )
            attributes_with_token_type = {
                **shared_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            }
            token_histogram.record(prompt_token, attributes=attributes_with_token_type)

    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)


def _extract_completion_data(response: Any) -> Dict[str, Any]:
    """Extract completion data from various response types."""
    if isinstance(response, dict):
        if "results" in response:
            result = response["results"][0]
            return {
                "generated_text": result.get("generated_text", ""),
                "token_usage": {
                    "prompt_tokens": result.get("input_token_count", 0),
                    "generated_tokens": result.get("generated_token_count", 0),
                },
            }
        return response
    elif hasattr(response, "generated_text"):
        return {
            "generated_text": response.generated_text,
            "token_usage": response.token_usage if hasattr(response, "token_usage") else None,
        }
    else:
        return {"generated_text": str(response)}


def _extract_prompt_data(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract prompt data from args or kwargs."""
    if args and len(args) > 0:
        prompt = args[0]
        if isinstance(prompt, list):
            return "\n".join(str(p) for p in prompt)
        return str(prompt)
    elif "prompt" in kwargs:
        prompt = kwargs["prompt"]
        if isinstance(prompt, list):
            return "\n".join(str(p) for p in prompt)
        return str(prompt)
    return None


@dont_throw
def _handle_request(span, args, kwargs, llm_model, event_logger=None, capture_content=True):
    """Handle request by setting attributes and emitting prompt event."""
    if span.is_recording():
        _set_api_attributes(span)
        
        # Add event-based tracking for prompts
        if event_logger is not None:
            prompt = _extract_prompt_data(args, kwargs)
            if prompt is not None:
                event_logger.emit(
                    prompt_to_event(
                        prompt=prompt,
                        model_name=llm_model,
                        capture_content=capture_content,
                        trace_id=span.get_span_context().trace_id,
                        span_id=span.get_span_context().span_id,
                        trace_flags=span.get_span_context().trace_flags,
                    )
                )


@dont_throw
def _handle_response(span, response, llm_model, event_logger=None, capture_content=True):
    """Handle response by setting attributes and emitting completion event."""
    if not span.is_recording():
        return

    try:
        completion_data = _extract_completion_data(response)
        
        # Add event-based tracking for completions
        if event_logger is not None:
            event_logger.emit(
                completion_to_event(
                    completion=completion_data,
                    model_name=llm_model,
                    capture_content=capture_content,
                    trace_id=span.get_span_context().trace_id,
                    span_id=span.get_span_context().span_id,
                    trace_flags=span.get_span_context().trace_flags,
                )
            )

        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        if Config.exception_logger:
            Config.exception_logger(e)


def _build_and_set_stream_response(
    span,
    response,
    raw_flag,
    token_histogram,
    response_counter,
    duration_histogram,
    start_time,
    event_logger=None,
    capture_content=True,
):
    """Build and set stream response while handling events."""
    stream_generated_text = ""
    stream_generated_token_count = 0
    stream_input_token_count = 0
    stream_model_id = None
    
    try:
        for item in response:
            if not isinstance(item, dict) or "results" not in item:
                if raw_flag:
                    yield item
                else:
                    yield str(item)
                continue

            stream_model_id = item.get("model_id")
            result = item["results"][0]
            stream_generated_text += result.get("generated_text", "")
            stream_input_token_count += result.get("input_token_count", 0)
            stream_generated_token_count = result.get("generated_token_count", 0)
            stream_stop_reason = result.get("stop_reason")

            if raw_flag:
                yield item
            else:
                yield result.get("generated_text", "")

        # Emit completion event after collecting all streaming data
        if event_logger is not None and stream_model_id:
            completion_data = {
                "generated_text": stream_generated_text,
                "token_usage": {
                    "prompt_tokens": stream_input_token_count,
                    "generated_tokens": stream_generated_token_count,
                },
            }
            event_logger.emit(
                completion_to_event(
                    completion=completion_data,
                    model_name=stream_model_id,
                    capture_content=capture_content,
                    trace_id=span.get_span_context().trace_id,
                    span_id=span.get_span_context().span_id,
                    trace_flags=span.get_span_context().trace_flags,
                )
            )

        # Update metrics
        shared_attributes = _metric_shared_attributes(
            response_model=stream_model_id, is_streaming=True
        )
        
        if response_counter and stream_model_id:
            attributes_with_reason = {
                **shared_attributes,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: stream_stop_reason,
            }
            response_counter.add(1, attributes=attributes_with_reason)

        if token_histogram and stream_model_id:
            token_histogram.record(
                stream_generated_token_count,
                attributes={**shared_attributes, SpanAttributes.LLM_TOKEN_TYPE: "output"},
            )
            token_histogram.record(
                stream_input_token_count,
                attributes={**shared_attributes, SpanAttributes.LLM_TOKEN_TYPE: "input"},
            )

        if duration_histogram and start_time and stream_model_id:
            duration = time.time() - start_time
            duration_histogram.record(duration, attributes=shared_attributes)

        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        if Config.exception_logger:
            Config.exception_logger(e)
    finally:
        span.end()


def _metric_shared_attributes(response_model: str, is_streaming: bool = False):
    return {
        SpanAttributes.LLM_RESPONSE_MODEL: response_model,
        SpanAttributes.LLM_SYSTEM: "watsonx",
        "stream": is_streaming,
    }


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(
        tracer,
        to_wrap,
        token_histogram,
        response_counter,
        duration_histogram,
        exception_counter,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                to_wrap,
                token_histogram,
                response_counter,
                duration_histogram,
                exception_counter,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(
    tracer,
    to_wrap,
    token_histogram: Histogram,
    response_counter: Counter,
    duration_histogram: Histogram,
    exception_counter: Counter,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Watsonx",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    event_logger = getattr(Config, "event_logger", None)
    capture_content = getattr(Config, "capture_content", True)
    _handle_request(span, args, kwargs, instance.model_id, event_logger, capture_content)

    try:
        start_time = time.time()
        response = wrapped(*args, **kwargs)
        end_time = time.time()
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0

        attributes = {
            "error.type": e.__class__.__name__,
        }

        if duration > 0 and duration_histogram:
            duration_histogram.record(duration, attributes=attributes)
        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        raise e

    if "generate" in name:
        if isinstance(response, types.GeneratorType):
            return _build_and_set_stream_response(
                span,
                response,
                raw_flag,
                token_histogram,
                response_counter,
                duration_histogram,
                start_time,
                event_logger,
                capture_content,
            )
        else:
            duration = end_time - start_time
            _handle_response(span, response, instance.model_id, event_logger, capture_content)

    _handle_response(span, response, instance.model_id, event_logger, capture_content)

    span.end()
    return response


class WatsonxSpanAttributes:
    WATSONX_API_VERSION = "watsonx.api_version"
    WATSONX_API_BASE = "watsonx.api_base"
    WATSONX_API_TYPE = "watsonx.api_type"


class WatsonxInstrumentor(BaseInstrumentor):
    """An instrumentor for Watsonx's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        # Store event logger and capture_content in Config
        Config.event_logger = kwargs.get("event_logger")
        Config.capture_content = kwargs.get("capture_content", True)

        if is_metrics_enabled():
            token_histogram = meter.create_histogram(
                name=Meters.LLM_TOKEN_USAGE,
                unit="token",
                description="Measures number of input and output tokens used",
            )

            response_counter = meter.create_counter(
                name=Meters.LLM_WATSONX_COMPLETIONS_RESPONSES,
                unit="response",
                description="Number of response returned by completions call",
            )

            duration_histogram = meter.create_histogram(
                name=Meters.LLM_OPERATION_DURATION,
                unit="s",
                description="GenAI operation duration",
            )

            exception_counter = meter.create_counter(
                name=Meters.LLM_WATSONX_COMPLETIONS_EXCEPTIONS,
                unit="time",
                description="Number of exceptions occurred during completions",
            )
        else:
            (
                token_histogram,
                response_counter,
                duration_histogram,
                exception_counter,
            ) = (
                None,
                None,
                None,
                None,
            )

        for wrapped_methods in WATSON_MODULES:
            for wrapped_method in wrapped_methods:
                wrap_module = wrapped_method.get("module")
                wrap_object = wrapped_method.get("object")
                wrap_method = wrapped_method.get("method")
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        wrapped_method,
                        token_histogram,
                        response_counter,
                        duration_histogram,
                        exception_counter,
                    ),
                )

    def _uninstrument(self, **kwargs):
        # Clean up Config
        Config.event_logger = None
        Config.capture_content = True
        
        for wrapped_methods in WATSON_MODULES:
            for wrapped_method in wrapped_methods:
                wrap_module = wrapped_method.get("module")
                wrap_object = wrapped_method.get("object")
                unwrap(
                    f"{wrap_module}.{wrap_object}",
                    wrapped_method.get("method", ""),
                )
