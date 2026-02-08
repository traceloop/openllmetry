"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api


from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.patch import (
    create_graph_invocation_wrapper,
    create_command_init_wrapper,
    create_middleware_hook_wrapper,
    create_async_middleware_hook_wrapper,
    create_agent_wrapper,
)
from opentelemetry.instrumentation.langchain.utils import is_package_available
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv_ai import Meters, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, SpanAttributes
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
        metadata_key_prefix: str = SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES
    ):
        """Create a Langchain instrumentor instance.

        Args:
            exception_logger: A callable that takes an Exception as input. This will be
                used to log exceptions that occur during instrumentation. If None, exceptions will not be logged.
            disable_trace_context_propagation: If True, disables trace context propagation to LLM providers.
            use_legacy_attributes: If True, uses span attributes for Inputs/Outputs instead of events.
            metadata_key_prefix: Prefix for metadata keys added to spans. Defaults to
                `SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES`.
                Useful for using with other backends.
        """
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        Config.metadata_key_prefix = metadata_key_prefix
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
            logger_provider = kwargs.get("logger_provider")
            Config.event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        traceloopCallbackHandler = TraceloopCallbackHandler(
            tracer, duration_histogram, token_histogram
        )
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(traceloopCallbackHandler),
        )

        # Wrap LangGraph components if available
        self._wrap_langgraph_components(tracer)

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

    def _wrap_langgraph_components(self, tracer):
        """Wrap LangGraph components for instrumentation."""
        # Wrap Pregel.stream and Pregel.astream (graph invocation)
        if is_package_available("langgraph"):
            try:
                wrap_function_wrapper(
                    module="langgraph.pregel",
                    name="Pregel.stream",
                    wrapper=create_graph_invocation_wrapper(tracer, is_async=False),
                )
                wrap_function_wrapper(
                    module="langgraph.pregel",
                    name="Pregel.astream",
                    wrapper=create_graph_invocation_wrapper(tracer, is_async=True),
                )
            except Exception as e:
                logger.debug("Failed to wrap Pregel methods: %s", e)

            # Wrap Command.__init__ to capture routing commands
            try:
                wrap_function_wrapper(
                    module="langgraph.types",
                    name="Command.__init__",
                    wrapper=create_command_init_wrapper(tracer),
                )
            except Exception as e:
                logger.debug("Failed to wrap Command.__init__: %s", e)

        # Wrap AgentMiddleware hooks if available
        if is_package_available("langchain"):
            self._wrap_middleware_hooks(tracer)
            self._wrap_agent_factories(tracer)

    def _wrap_agent_factories(self, tracer):
        """Wrap agent factory functions for instrumentation."""
        # LangGraph prebuilt agents - patch both actual module and re-export location
        if is_package_available("langgraph"):
            langgraph_agent_wrapper = create_agent_wrapper(tracer, provider_name="langgraph")
            # Patch the actual module where the function is defined
            try:
                wrap_function_wrapper(
                    module="langgraph.prebuilt.chat_agent_executor",
                    name="create_react_agent",
                    wrapper=langgraph_agent_wrapper,
                )
            except Exception as e:
                logger.debug("Failed to wrap langgraph.prebuilt.chat_agent_executor.create_react_agent: %s", e)
            # Also patch the re-export location for imports from langgraph.prebuilt
            try:
                wrap_function_wrapper(
                    module="langgraph.prebuilt",
                    name="create_react_agent",
                    wrapper=langgraph_agent_wrapper,
                )
            except Exception as e:
                logger.debug("Failed to wrap langgraph.prebuilt.create_react_agent: %s", e)

        # LangChain agents - patch both actual module and re-export location
        if is_package_available("langchain"):
            agent_wrapper = create_agent_wrapper(tracer, provider_name="langchain")
            # Patch the actual module where the function is defined
            try:
                wrap_function_wrapper(
                    module="langchain.agents.factory",
                    name="create_agent",
                    wrapper=agent_wrapper,
                )
            except Exception as e:
                logger.debug("Failed to wrap langchain.agents.factory.create_agent: %s", e)
            # Also patch the re-export location for imports from langchain.agents
            try:
                wrap_function_wrapper(
                    module="langchain.agents",
                    name="create_agent",
                    wrapper=agent_wrapper,
                )
            except Exception as e:
                logger.debug("Failed to wrap langchain.agents.create_agent: %s", e)

    def _wrap_middleware_hooks(self, tracer):
        """Wrap AgentMiddleware hook methods for instrumentation."""
        # Sync hooks
        sync_hooks = ["before_model", "after_model", "before_agent", "after_agent"]
        for hook_name in sync_hooks:
            try:
                wrap_function_wrapper(
                    module="langchain.agents.middleware.types",
                    name=f"AgentMiddleware.{hook_name}",
                    wrapper=create_middleware_hook_wrapper(tracer, hook_name),
                )
            except Exception as e:
                logger.debug("Failed to wrap AgentMiddleware.%s: %s", hook_name, e)

        # Async hooks
        async_hooks = ["abefore_model", "aafter_model", "abefore_agent", "aafter_agent"]
        for hook_name in async_hooks:
            try:
                wrap_function_wrapper(
                    module="langchain.agents.middleware.types",
                    name=f"AgentMiddleware.{hook_name}",
                    wrapper=create_async_middleware_hook_wrapper(tracer, hook_name),
                )
            except Exception as e:
                logger.debug("Failed to wrap AgentMiddleware.%s: %s", hook_name, e)

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")

        # Unwrap LangGraph components
        if is_package_available("langgraph"):
            try:
                unwrap("langgraph.pregel", "Pregel.stream")
                unwrap("langgraph.pregel", "Pregel.astream")
            except Exception:
                pass
            try:
                unwrap("langgraph.types", "Command.__init__")
            except Exception:
                pass

        # Unwrap AgentMiddleware hooks
        if is_package_available("langchain"):
            sync_hooks = ["before_model", "after_model", "before_agent", "after_agent"]
            async_hooks = ["abefore_model", "aafter_model", "abefore_agent", "aafter_agent"]
            for hook_name in sync_hooks + async_hooks:
                try:
                    unwrap("langchain.agents.middleware.types", f"AgentMiddleware.{hook_name}")
                except Exception:
                    pass

        # Unwrap LangGraph agent factories (both actual module and re-export)
        if is_package_available("langgraph"):
            try:
                unwrap("langgraph.prebuilt.chat_agent_executor", "create_react_agent")
            except Exception:
                pass
            try:
                unwrap("langgraph.prebuilt", "create_react_agent")
            except Exception:
                pass

        # Unwrap LangChain agent factories (both actual module and re-export)
        if is_package_available("langchain"):
            try:
                unwrap("langchain.agents.factory", "create_agent")
            except Exception:
                pass
            try:
                unwrap("langchain.agents", "create_agent")
            except Exception:
                pass

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
            span_holder = self._callback_manager.spans.get(run_id)

            if span_holder:
                extra_headers = kwargs.get("extra_headers", {})
                ctx = set_span_in_context(span_holder.span)
                TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
                kwargs["extra_headers"] = extra_headers
            else:
                logger.debug(
                    "No span found for run_id %s, skipping header injection",
                    run_id
                )

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
