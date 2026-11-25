"""OpenTelemetry Together AI instrumentation"""

import logging
import warnings
from typing import Collection, Optional

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.event_emitter import (
    emit_completion_event,
    emit_prompt_events,
)
from opentelemetry.instrumentation.together.span_utils import (
    set_completion_attributes,
    set_model_completion_attributes,
    set_model_prompt_attributes,
    set_prompt_attributes,
)
from opentelemetry.instrumentation.together.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.together.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("together >= 1.2.0, <4",)

_CHAT_MODULE = "together.resources.chat.completions"
_COMPLETIONS_MODULE = "together.resources.completions"

# together 1.x and together 2.x/3.x use the same modules but different class
# names. Every entry is tried; the ones missing from the installed version are
# skipped in `_instrument`.
WRAPPED_METHODS = [
    # together 1.x
    {
        "module": _CHAT_MODULE,
        "object": "ChatCompletions",
        "span_name": "together.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": False,
    },
    {
        "module": _CHAT_MODULE,
        "object": "AsyncChatCompletions",
        "span_name": "together.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": True,
    },
    {
        "module": _COMPLETIONS_MODULE,
        "object": "Completions",
        "span_name": "together.completion",
        "request_type": LLMRequestTypeValues.COMPLETION,
        "is_async": False,
    },
    {
        "module": _COMPLETIONS_MODULE,
        "object": "AsyncCompletions",
        "span_name": "together.completion",
        "request_type": LLMRequestTypeValues.COMPLETION,
        "is_async": True,
    },
    # together 2.x / 3.x
    {
        "module": _CHAT_MODULE,
        "object": "CompletionsResource",
        "span_name": "together.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": False,
    },
    {
        "module": _CHAT_MODULE,
        "object": "AsyncCompletionsResource",
        "span_name": "together.chat",
        "request_type": LLMRequestTypeValues.CHAT,
        "is_async": True,
    },
    {
        "module": _COMPLETIONS_MODULE,
        "object": "CompletionsResource",
        "span_name": "together.completion",
        "request_type": LLMRequestTypeValues.COMPLETION,
        "is_async": False,
    },
    {
        "module": _COMPLETIONS_MODULE,
        "object": "AsyncCompletionsResource",
        "span_name": "together.completion",
        "request_type": LLMRequestTypeValues.COMPLETION,
        "is_async": True,
    },
]


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _is_suppressed():
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    )


@dont_throw
def _handle_input(span, event_logger, llm_request_type, kwargs):
    set_model_prompt_attributes(span, kwargs)

    if should_emit_events() and event_logger:
        emit_prompt_events(event_logger, llm_request_type, kwargs)
    else:
        set_prompt_attributes(span, llm_request_type, kwargs)


@dont_throw
def _handle_response(span, event_logger, llm_request_type, response):
    if should_emit_events() and event_logger:
        emit_completion_event(event_logger, llm_request_type, response)
    else:
        set_completion_attributes(span, llm_request_type, response)

    set_model_completion_attributes(span, response)


def _start_span(tracer, event_logger, to_wrap, kwargs):
    llm_request_type = to_wrap.get("request_type")
    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    _handle_input(span, event_logger, llm_request_type, kwargs)
    return span


def _end_span_with_error(span, error):
    span.set_attribute(ERROR_TYPE, error.__class__.__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.end()


def _end_span_with_response(span, event_logger, to_wrap, response):
    if response:
        _handle_response(span, event_logger, to_wrap.get("request_type"), response)
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()


@_with_tracer_wrapper
def _wrap(
    tracer,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every sync function defined in WRAPPED_METHODS."""
    if _is_suppressed():
        return wrapped(*args, **kwargs)

    span = _start_span(tracer, event_logger, to_wrap, kwargs)
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _end_span_with_error(span, e)
        raise

    _end_span_with_response(span, event_logger, to_wrap, response)
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every async function defined in WRAPPED_METHODS."""
    if _is_suppressed():
        return await wrapped(*args, **kwargs)

    span = _start_span(tracer, event_logger, to_wrap, kwargs)
    try:
        response = await wrapped(*args, **kwargs)
    except BaseException as e:
        # BaseException so the span also ends on asyncio cancellation.
        _end_span_with_error(span, e)
        raise

    _end_span_with_response(span, event_logger, to_wrap, response)
    return response


class TogetherAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Together AI's client library."""

    def __init__(
        self,
        exception_logger=None,
        use_attributes: Optional[bool] = None,
        use_legacy_attributes: Optional[bool] = None,
    ):
        super().__init__()
        if use_attributes is not None and use_legacy_attributes is not None:
            raise TypeError(
                "Cannot pass both `use_attributes` and `use_legacy_attributes`; "
                "`use_legacy_attributes` is deprecated, use `use_attributes` instead."
            )
        if use_legacy_attributes is not None:
            warnings.warn(
                "`use_legacy_attributes` is deprecated and will be removed in a "
                "future release; use `use_attributes` instead. The current OTel "
                "GenAI spec emits prompts/completions as span attributes "
                "(`gen_ai.input.messages` / `gen_ai.output.messages`), which is "
                "what `use_attributes=True` (the default) does. "
                "`use_attributes=False` opts into the events path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_attributes = use_legacy_attributes
        if use_attributes is None:
            use_attributes = True
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            obj = wrapped_method.get("object")
            wrapper = _awrap if wrapped_method.get("is_async") else _wrap
            try:
                wrap_function_wrapper(
                    module,
                    f"{obj}.create",
                    wrapper(tracer, event_logger, wrapped_method),
                )
            except (ImportError, AttributeError):
                # Class isn't in the installed together version.
                logger.debug("Skipping %s.%s.create", module, obj)

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            obj = wrapped_method.get("object")
            try:
                unwrap(f"{module}.{obj}", "create")
            except ImportError:
                # Class isn't in the installed together version.
                pass
