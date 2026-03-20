import os
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

_instruments = ("ag2 >= 0.11.0",)


def _should_send_prompts():
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


class AG2Instrumentor(BaseInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")

        # Use AG2's built-in OpenTelemetry instrumentation for LLM calls
        from autogen.opentelemetry import instrument_llm_wrapper

        instrument_llm_wrapper(
            tracer_provider=tracer_provider,
            capture_messages=_should_send_prompts(),
        )

        # Patch ConversableAgent.__init__ to auto-instrument every agent instance
        from autogen.opentelemetry import instrument_agent

        def _wrap_agent_init(wrapped, instance, args, kwargs):
            result = wrapped(*args, **kwargs)
            instrument_agent(instance, tracer_provider=tracer_provider)
            return result

        wrap_function_wrapper(
            "autogen.agentchat.conversable_agent",
            "ConversableAgent.__init__",
            _wrap_agent_init,
        )

        # Patch Pattern.prepare_group_chat to auto-instrument patterns
        try:
            from autogen.opentelemetry import instrument_pattern

            def _wrap_pattern_init(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                instrument_pattern(instance, tracer_provider=tracer_provider)
                return result

            wrap_function_wrapper(
                "autogen.agentchat.group.patterns.pattern",
                "Pattern.__init__",
                _wrap_pattern_init,
            )
        except (ImportError, AttributeError):
            pass

    def _uninstrument(self, **kwargs):
        from opentelemetry.instrumentation.utils import unwrap

        try:
            from autogen.agentchat.conversable_agent import ConversableAgent
            unwrap(ConversableAgent, "__init__")
        except (ImportError, AttributeError):
            pass

        try:
            from autogen.agentchat.group.patterns.pattern import Pattern
            unwrap(Pattern, "__init__")
        except (ImportError, AttributeError):
            pass

        try:
            from autogen.oai.client import OpenAIWrapper
            if hasattr(OpenAIWrapper.create, "__otel_wrapped__"):
                # instrument_llm_wrapper stores original as a closure;
                # best-effort: remove the wrapper flag so re-instrument works
                delattr(OpenAIWrapper.create, "__otel_wrapped__")
        except (ImportError, AttributeError):
            pass
