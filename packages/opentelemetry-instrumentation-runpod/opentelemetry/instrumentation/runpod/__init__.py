"""OpenTelemetry RunPod instrumentation"""

import inspect
import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.runpod.config import Config
from opentelemetry.instrumentation.runpod.span_utils import (
    OPERATION_RUN,
    OPERATION_RUN_SYNC,
    RUNPOD_ENDPOINT_ID,
    set_input_content_attributes,
    set_span_async_response_attributes,
    set_span_request_attributes,
    set_span_status_error,
    set_span_status_ok,
    set_span_sync_response_attributes,
)
from opentelemetry.instrumentation.runpod.utils import dont_throw
from opentelemetry.instrumentation.runpod.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("runpod >= 1.0.0",)

# `runpod.Endpoint` is the synchronous Serverless client of the RunPod Python SDK:
# `run` submits a job and returns a `Job` handle, `run_sync` submits a job and
# waits for its output. `runpod.AsyncioEndpoint.run` is the asyncio equivalent of
# `run`. The remaining SDK entry points that reach the Serverless API are handled
# by other code paths and are intentionally out of scope here - see README.
WRAPPED_METHODS = [
    {
        "module": "runpod",
        "object": "Endpoint",
        "method": "run",
        "span_name": "runpod.run",
        "operation_name": OPERATION_RUN,
        "response_handler": None,
    },
    {
        "module": "runpod",
        "object": "Endpoint",
        "method": "run_sync",
        "span_name": "runpod.run_sync",
        "operation_name": OPERATION_RUN_SYNC,
        "response_handler": set_span_sync_response_attributes,
    },
    {
        "module": "runpod",
        "object": "AsyncioEndpoint",
        "method": "run",
        "span_name": "runpod.run",
        "operation_name": OPERATION_RUN,
        "response_handler": set_span_async_response_attributes,
    },
]


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@dont_throw
def _start_span(tracer, to_wrap, instance):
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: "runpod",
        RUNPOD_ENDPOINT_ID: getattr(instance, "endpoint_id", None),
    }
    return tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={key: value for key, value in attributes.items() if value is not None},
    )


@_with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = _start_span(tracer, to_wrap, instance)

    try:
        set_span_request_attributes(span, to_wrap, instance)
        set_input_content_attributes(span, args, kwargs)

        response = wrapped(*args, **kwargs)

        response_handler = to_wrap.get("response_handler")
        if response_handler:
            response_handler(span, response)
        set_span_status_ok(span)
        span.end()
        return response
    except Exception as e:
        set_span_status_error(span, e)
        span.end()
        raise


@_with_tracer_wrapper
async def _awrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and awaits async RunPod serverless calls."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = _start_span(tracer, to_wrap, instance)

    try:
        set_span_request_attributes(span, to_wrap, instance)
        set_input_content_attributes(span, args, kwargs)

        response = await wrapped(*args, **kwargs)

        response_handler = to_wrap.get("response_handler")
        if response_handler:
            response_handler(span, response)
        set_span_status_ok(span)
        span.end()
        return response
    except Exception as e:
        set_span_status_error(span, e)
        span.end()
        raise


def _resolve(wrap_module, wrap_object, wrap_method):
    """Returns the target method, or None when the installed SDK does not expose it."""
    try:
        module = __import__(wrap_module, fromlist=[wrap_object])
    except (ImportError, ModuleNotFoundError):
        return None

    target_object = getattr(module, wrap_object, None)
    if target_object is None:
        return None

    return getattr(target_object, wrap_method, None)


class RunpodInstrumentor(BaseInstrumentor):
    """An instrumentor for RunPod's Serverless client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            target = _resolve(wrap_module, wrap_object, wrap_method)
            if target is None:
                logger.debug(f"Failed to instrument {wrap_module}.{wrap_object}.{wrap_method}")
                continue

            wrapper = _awrap if inspect.iscoroutinefunction(target) else _wrap
            try:
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    wrapper(tracer, wrapped_method),
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to instrument {wrap_module}.{wrap_object}.{wrap_method}")

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                module = __import__(wrap_module, fromlist=[wrap_object])
                unwrap(getattr(module, wrap_object), wrap_method)
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to uninstrument {wrap_module}.{wrap_object}.{wrap_method}")
