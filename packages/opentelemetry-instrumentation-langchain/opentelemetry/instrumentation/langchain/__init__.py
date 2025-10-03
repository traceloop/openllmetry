"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api


from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.utils import is_package_available
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv_ai import Meters, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import get_tracer
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("langchain-core > 0.1.0", )


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def __init__(
        self,
        exception_logger=None,
        disable_trace_context_propagation=False,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        self.disable_trace_context_propagation = disable_trace_context_propagation

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Add meter creation
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        # Create duration histogram
        duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )

        # Create token histogram
        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )

        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            Config.event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        traceloopCallbackHandler = TraceloopCallbackHandler(
            tracer, duration_histogram, token_histogram
        )
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(traceloopCallbackHandler),
        )

        if not self.disable_trace_context_propagation:
            self._wrap_openai_functions_for_tracing(traceloopCallbackHandler)

    def _wrap_openai_functions_for_tracing(self, traceloopCallbackHandler):
        openai_tracing_wrapper = _OpenAITracingWrapper(traceloopCallbackHandler)

        if is_package_available("langchain_community"):
            # Wrap langchain_community.llms.openai.BaseOpenAI
            wrap_function_wrapper(
                module="langchain_community.llms.openai",
                name="BaseOpenAI._generate",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_community.llms.openai",
                name="BaseOpenAI._agenerate",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_community.llms.openai",
                name="BaseOpenAI._stream",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_community.llms.openai",
                name="BaseOpenAI._astream",
                wrapper=openai_tracing_wrapper,
            )

        if is_package_available("langchain_openai"):
            # Wrap langchain_openai.llms.base.BaseOpenAI
            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._generate",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._agenerate",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._stream",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._astream",
                wrapper=openai_tracing_wrapper,
            )

            # langchain_openai.chat_models.base.BaseOpenAI
            wrap_function_wrapper(
                module="langchain_openai.chat_models.base",
                name="BaseChatOpenAI._generate",
                wrapper=openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                module="langchain_openai.chat_models.base",
                name="BaseChatOpenAI._agenerate",
                wrapper=openai_tracing_wrapper,
            )

            # Doesn't work :(
            # wrap_function_wrapper(
            #     module="langchain_openai.chat_models.base",
            #     name="BaseChatOpenAI._stream",
            #     wrapper=openai_tracing_wrapper,
            # )
            # wrap_function_wrapper(
            #     module="langchain_openai.chat_models.base",
            #     name="BaseChatOpenAI._astream",
            #     wrapper=openai_tracing_wrapper,
            # )

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")
        if not self.disable_trace_context_propagation:
            if is_package_available("langchain_community"):
                unwrap("langchain_community.llms.openai", "BaseOpenAI._generate")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._agenerate")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._stream")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._astream")
            if is_package_available("langchain_openai"):
                unwrap("langchain_openai.llms.base", "BaseOpenAI._generate")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._agenerate")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._stream")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._astream")
                unwrap("langchain_openai.chat_models.base", "BaseOpenAI._generate")
                unwrap("langchain_openai.chat_models.base", "BaseOpenAI._agenerate")
                # unwrap("langchain_openai.chat_models.base", "BaseOpenAI._stream")
                # unwrap("langchain_openai.chat_models.base", "BaseOpenAI._astream")


class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_handler: "TraceloopCallbackHandler"):
        self._callback_handler = callback_handler

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._callback_handler)):
                break
        else:
            # Add a property to the handler which indicates the CallbackManager instance.
            # Since the CallbackHandler only propagates context for sync callbacks,
            # we need a way to determine the type of CallbackManager being wrapped.
            self._callback_handler._callback_manager = instance
            instance.add_handler(self._callback_handler, True)


# This class wraps a function call to inject tracing information (trace headers) into
# OpenAI client requests. It assumes the following:
# 1. The wrapped function includes a `run_manager` keyword argument that contains a `run_id`.
#    The `run_id` is used to look up a corresponding tracing span from the callback manager.
# 2. The `kwargs` passed to the wrapped function are forwarded to the OpenAI client. This
#    allows us to add extra headers (including tracing headers) to the OpenAI request by
#    modifying the `extra_headers` argument in `kwargs`.
class _OpenAITracingWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        run_manager = kwargs.get("run_manager")
        if run_manager:
            run_id = run_manager.run_id
            span_holder = self._callback_manager.spans[run_id]

            extra_headers = kwargs.get("extra_headers", {})

            # Inject tracing context into the extra headers
            ctx = set_span_in_context(span_holder.span)
            TraceContextTextMapPropagator().inject(extra_headers, context=ctx)

            # Update kwargs to include the modified headers
            kwargs["extra_headers"] = extra_headers

        # In legacy chains like LLMChain, suppressing model instrumentations
        # within create_llm_span doesn't work, so this should helps as a fallback
        try:
            context_api.attach(
                context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
            )
        except Exception:
            # If context setting fails, continue without suppression
            # This is not critical for core functionality
            pass

        return wrapped(*args, **kwargs)
