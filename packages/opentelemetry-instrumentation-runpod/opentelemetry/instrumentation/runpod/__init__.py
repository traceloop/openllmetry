"""OpenTelemetry RunPod instrumentation"""

import inspect
import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.runpod.config import Config
from opentelemetry.instrumentation.runpod.event_emitter import (
    emit_request_event,
    emit_response_event,
)
from opentelemetry.instrumentation.runpod.span_utils import (
    OPERATION_RUN,
    OPERATION_RUN_SYNC,
    RUNPOD_ENDPOINT_ID,
    content_is_recorded,
    set_input_content_attributes,
    set_response_content_attributes,
    set_span_job_response_attributes,
    set_span_request_attributes,
    set_span_status_error,
    set_span_status_ok,
    set_span_sync_response_attributes,
)
from opentelemetry.instrumentation.runpod.utils import (
    dont_throw,
    get_request_content,
    normalize_request_input,
    wrap_request_input,
)
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
#
# Each entry names the normalizer that mirrors how that method wraps the payload
# before POSTing it, so that the recorded content is the request body the SDK sends.
WRAPPED_METHODS = [
    {
        "module": "runpod",
        "object": "Endpoint",
        "method": "run",
        "span_name": "runpod.run",
        "operation_name": OPERATION_RUN,
        "request_normalizer": normalize_request_input,
        "response_handler": set_span_job_response_attributes,
    },
    {
        "module": "runpod",
        "object": "Endpoint",
        "method": "run_sync",
        "span_name": "runpod.run_sync",
        "operation_name": OPERATION_RUN_SYNC,
        "request_normalizer": normalize_request_input,
        "response_handler": set_span_sync_response_attributes,
    },
    {
        "module": "runpod",
        "object": "AsyncioEndpoint",
        "method": "run",
        "span_name": "runpod.run",
        "operation_name": OPERATION_RUN,
        "request_normalizer": wrap_request_input,
        "response_handler": set_span_job_response_attributes,
    },
]


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

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


@dont_throw
def _handle_request(span, event_logger, to_wrap, args, kwargs):
    """
    Records the submitted payload, as a legacy attribute or - when the instrumentor
    runs with ``use_legacy_attributes=False`` - as a ``gen_ai.user.message`` event.
    """
    request_content = None
    if content_is_recorded(span):
        request_content = get_request_content(
            args, kwargs, to_wrap.get("request_normalizer", normalize_request_input)
        )

    set_input_content_attributes(span, request_content)
    emit_request_event(event_logger, request_content)


@dont_throw
def _handle_response(span, event_logger, to_wrap, response):
    """
    Records the response, as a legacy attribute or - when the instrumentor runs with
    ``use_legacy_attributes=False`` - as a ``gen_ai.choice`` event.
    """
    response_handler = to_wrap.get("response_handler")
    response_content = response_handler(span, response) if response_handler else None

    set_response_content_attributes(span, response_content)
    emit_response_event(event_logger, response_content)


@_with_tracer_wrapper
def _wrap(tracer: Tracer, event_logger, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = _start_span(tracer, to_wrap, instance)

    try:
        set_span_request_attributes(span, to_wrap, instance)
        _handle_request(span, event_logger, to_wrap, args, kwargs)

        response = wrapped(*args, **kwargs)

        _handle_response(span, event_logger, to_wrap, response)
        set_span_status_ok(span)
        return response
    except Exception as e:
        set_span_status_error(span, e)
        raise
    finally:
        # `finally` rather than a call on each path, so that the span is also ended
        # when a BaseException - a KeyboardInterrupt, say - unwinds the call.
        span.end()


@_with_tracer_wrapper
async def _awrap(tracer: Tracer, event_logger, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and awaits async RunPod serverless calls."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = _start_span(tracer, to_wrap, instance)

    try:
        set_span_request_attributes(span, to_wrap, instance)
        _handle_request(span, event_logger, to_wrap, args, kwargs)

        response = await wrapped(*args, **kwargs)

        _handle_response(span, event_logger, to_wrap, response)
        set_span_status_ok(span)
        return response
    except Exception as e:
        set_span_status_error(span, e)
        raise
    finally:
        # `asyncio.CancelledError` derives from BaseException, so it bypasses the
        # handler above; ending the span here keeps cancelled calls from leaking an
        # unfinished span.
        span.end()


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

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(__name__, __version__, logger_provider=logger_provider)

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
                    wrapper(tracer, event_logger, wrapped_method),
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
