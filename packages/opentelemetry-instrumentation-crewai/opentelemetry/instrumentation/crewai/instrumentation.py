import os
import time
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)
from opentelemetry.semconv_ai import GenAISystem, SpanAttributes, TraceloopSpanKindValues, Meters
from .crewai_span_attributes import CrewAISpanAttributes, set_span_attribute
from .utils import _messages_to_otel_input, _response_to_otel_output

_instruments = ("crewai >= 1.0.0",)

# Maps LiteLLM vendor prefixes (e.g. "openai" in "openai/gpt-4") to OTel provider name values.
# Uses GenAISystem (semconv-ai) and GenAiSystemValues (OTel upstream) — no raw strings.
_LITELLM_PREFIX_TO_OTEL_PROVIDER = {
    "openai":      GenAISystem.OPENAI.value,
    "anthropic":   GenAISystem.ANTHROPIC.value,
    "gemini":      GenAiSystemValues.GCP_GEMINI.value,
    "vertex_ai":   GenAiSystemValues.GCP_VERTEX_AI.value,
    "bedrock":     GenAISystem.AWS.value,
    "azure":       GenAiSystemValues.AZURE_AI_OPENAI.value,
    "groq":        GenAISystem.GROQ.value,
    "mistral":     GenAISystem.MISTRALAI.value,
    "cohere":      GenAISystem.COHERE.value,
    "ollama":      GenAISystem.OLLAMA.value,
}

# Maps bare model name patterns to OTel provider name values.
_MODEL_PATTERN_TO_OTEL_PROVIDER = [
    ("claude",   GenAISystem.ANTHROPIC.value),
    ("gemini",   GenAiSystemValues.GCP_GEMINI.value),
    ("mistral",  GenAISystem.MISTRALAI.value),
    ("command",  GenAISystem.COHERE.value),
]


def _infer_llm_provider_from_model(model: object | None) -> str | None:
    """Resolve gen_ai.provider.name for the underlying LLM on a chat span.

    LiteLLM-prefixed strings ("openai/gpt-4") use the prefix via
    _LITELLM_PREFIX_TO_OTEL_PROVIDER. Bare model names use pattern matching
    via _MODEL_PATTERN_TO_OTEL_PROVIDER. Returns None when unknown — never guesses.
    """
    if not model:
        return None
    s = str(model).strip()
    if "/" in s:
        return _LITELLM_PREFIX_TO_OTEL_PROVIDER.get(s.split("/")[0].lower())
    lower = s.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return GenAISystem.OPENAI.value
    for pattern, provider in _MODEL_PATTERN_TO_OTEL_PROVIDER:
        if pattern in lower:
            return provider
    return None

# GenAI memory semantic convention attribute keys (fallback to string
# literals when the installed semconv package doesn't define them yet).
_GEN_AI_OPERATION_NAME = getattr(GenAIAttributes, "GEN_AI_OPERATION_NAME", "gen_ai.operation.name")
_GEN_AI_PROVIDER_NAME = getattr(GenAIAttributes, "GEN_AI_PROVIDER_NAME", "gen_ai.provider.name")
_GEN_AI_MEMORY_SCOPE = getattr(GenAIAttributes, "GEN_AI_MEMORY_SCOPE", "gen_ai.memory.scope")
_GEN_AI_MEMORY_TYPE = getattr(GenAIAttributes, "GEN_AI_MEMORY_TYPE", "gen_ai.memory.type")
_GEN_AI_MEMORY_QUERY = getattr(GenAIAttributes, "GEN_AI_MEMORY_QUERY", "gen_ai.memory.query")
_GEN_AI_MEMORY_CONTENT = getattr(GenAIAttributes, "GEN_AI_MEMORY_CONTENT", "gen_ai.memory.content")
_GEN_AI_MEMORY_NAMESPACE = getattr(GenAIAttributes, "GEN_AI_MEMORY_NAMESPACE", "gen_ai.memory.namespace")
_GEN_AI_MEMORY_SEARCH_RESULT_COUNT = getattr(
    GenAIAttributes, "GEN_AI_MEMORY_SEARCH_RESULT_COUNT", "gen_ai.memory.search.result.count"
)
_GEN_AI_MEMORY_UPDATE_STRATEGY = getattr(
    GenAIAttributes, "GEN_AI_MEMORY_UPDATE_STRATEGY", "gen_ai.memory.update.strategy"
)
_GEN_AI_MEMORY_IMPORTANCE = getattr(GenAIAttributes, "GEN_AI_MEMORY_IMPORTANCE", "gen_ai.memory.importance")
_ERROR_TYPE = getattr(ErrorAttributes, "ERROR_TYPE", "error.type")

_PROVIDER = "crewai"


def _capture_content() -> bool:
    """Check if memory content capture is enabled."""
    return os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower() in ("true", "1")


class CrewAIInstrumentor(BaseInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        token_histogram = None
        duration_histogram = None
        if is_metrics_enabled():
            token_histogram, duration_histogram = _create_metrics(meter)

        wrap_function_wrapper("crewai.crew", "Crew.kickoff",
                              wrap_kickoff(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.agent", "Agent.execute_task",
                              wrap_agent_execute_task(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.task", "Task.execute_sync",
                              wrap_task_execute(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.llm", "LLM.call",
                              wrap_llm_call(tracer, duration_histogram, token_histogram))

        # Memory operations (crewai.memory.unified_memory.Memory)
        try:
            wrap_function_wrapper(
                "crewai.memory.unified_memory", "Memory.remember",
                wrap_memory_remember(tracer, duration_histogram))
            wrap_function_wrapper(
                "crewai.memory.unified_memory", "Memory.recall",
                wrap_memory_recall(tracer, duration_histogram))
            wrap_function_wrapper(
                "crewai.memory.unified_memory", "Memory.forget",
                wrap_memory_forget(tracer, duration_histogram))
            wrap_function_wrapper(
                "crewai.memory.unified_memory", "Memory.reset",
                wrap_memory_reset(tracer, duration_histogram))
        except Exception:
            # CrewAI versions before unified_memory may not have these classes
            pass

    def _uninstrument(self, **kwargs):
        unwrap("crewai.crew.Crew", "kickoff")
        unwrap("crewai.agent.Agent", "execute_task")
        unwrap("crewai.task.Task", "execute_sync")
        unwrap("crewai.llm.LLM", "call")

        # Memory unwrap (ignore if not wrapped)
        try:
            from crewai.memory.unified_memory import Memory as UnifiedMemory
            unwrap(UnifiedMemory, "remember")
            unwrap(UnifiedMemory, "recall")
            unwrap(UnifiedMemory, "forget")
            unwrap(UnifiedMemory, "reset")
        except Exception:
            pass


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, duration_histogram, token_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@with_tracer_wrapper
def wrap_kickoff(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                 wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        "crewai.workflow",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            if result:
                class_name = instance.__class__.__name__
                span.set_attribute(f"crewai.{class_name.lower()}.result", str(result))
                span.set_status(Status(StatusCode.OK))
                if class_name == "Crew":
                    for attr in ["tasks_output", "token_usage", "usage_metrics"]:
                        if hasattr(result, attr):
                            span.set_attribute(f"crewai.crew.{attr}", str(getattr(result, attr)))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_agent_execute_task(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.role if hasattr(instance, "role") else "agent"
    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            if hasattr(instance, "role") and instance.role:
                set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_NAME, instance.role)
            if hasattr(instance, "id"):
                set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_ID, str(instance.id))
            result = wrapped(*args, **kwargs)
            if token_histogram:
                token_histogram.record(
                    instance._token_process.get_summary().prompt_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    }
                )
                token_histogram.record(
                    instance._token_process.get_summary().completion_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    },
                )

            set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, str(instance.llm.model))
            set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.llm.model))
            summary = instance._token_process.get_summary()
            if summary.prompt_tokens:
                set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, summary.prompt_tokens)
            if summary.completion_tokens:
                set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, summary.completion_tokens)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_task_execute(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    task_name = instance.description if hasattr(instance, "description") else "task"

    with tracer.start_as_current_span(
        f"{task_name}.task",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_llm_call(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    model = str(instance.model) if hasattr(instance, "model") else "llm"
    provider = _infer_llm_provider_from_model(getattr(instance, "model", None))

    span_attrs = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }
    if provider:
        span_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider

    with tracer.start_as_current_span(
        f"{model}.llm", kind=SpanKind.CLIENT, attributes=span_attrs,
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            messages_arg = args[0] if args else kwargs.get("messages")
            result = wrapped(*args, **kwargs)

            _set_messages_attributes(span, messages_arg, result)
            _set_response_attributes(span, instance)
            _record_duration(duration_histogram, start_time, model, provider)

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def _set_messages_attributes(span, messages_arg, result):
    input_json = _messages_to_otel_input(messages_arg)
    if input_json:
        set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_json)
    output_json = _response_to_otel_output(result)
    if output_json:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_json)


def _set_response_attributes(span, instance):
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.model))
    if hasattr(instance, "last_token_usage") and instance.last_token_usage:
        usage = instance.last_token_usage
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                           getattr(usage, "prompt_tokens", None))
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                           getattr(usage, "completion_tokens", None))


def _record_duration(duration_histogram, start_time, model, provider):
    if not duration_histogram:
        return
    attrs = {GenAIAttributes.GEN_AI_RESPONSE_MODEL: model}
    if provider:
        attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider
    duration_histogram.record(time.time() - start_time, attributes=attrs)


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, duration_histogram


# ---------------------------------------------------------------------------
# Memory operation wrappers — aligned with GenAI memory semantic conventions
# ---------------------------------------------------------------------------


def _infer_memory_scope(instance) -> str:
    """Infer memory scope from the Memory instance or its MemoryScope wrapper."""
    # MemoryScope has a _root attribute like "/agent/1" or "/user/123"
    root = getattr(instance, "_root", None)
    if root:
        parts = root.strip("/").split("/")
        if parts:
            first = parts[0].lower()
            if first in ("user", "agent", "session", "team", "global"):
                return first
    return "agent"


def _infer_memory_type(kwargs) -> str:
    """Infer memory type from kwargs categories hint, defaulting to long_term."""
    categories = kwargs.get("categories")
    if categories and isinstance(categories, list):
        for cat in categories:
            cl = str(cat).lower()
            if "short" in cl:
                return "short_term"
            if "entity" in cl:
                return "entity"
    return "long_term"


def _set_memory_error(span, exc):
    """Record error details on the span."""
    error_type = type(exc).__qualname__
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    set_span_attribute(span, _ERROR_TYPE, error_type)
    return error_type


def _record_memory_duration(duration_histogram, duration_s, operation, error_type=None):
    """Record memory operation duration metric."""
    if not duration_histogram:
        return
    attrs = {
        _GEN_AI_OPERATION_NAME: operation,
        GenAIAttributes.GEN_AI_SYSTEM: _PROVIDER,
    }
    if error_type:
        attrs[_ERROR_TYPE] = error_type
    duration_histogram.record(max(duration_s, 0.0), attributes=attrs)


def wrap_memory_remember(tracer: Tracer, duration_histogram: Histogram):
    """Wrap Memory.remember() → update_memory span."""
    def _wrapper(wrapped, instance, args, kwargs):
        operation = "update_memory"
        span_name = f"{operation} {_PROVIDER}"
        error_type = None
        start_time = time.time()
        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT,
            attributes={GenAIAttributes.GEN_AI_SYSTEM: _PROVIDER}
        ) as span:
            set_span_attribute(span, _GEN_AI_OPERATION_NAME, operation)
            set_span_attribute(span, _GEN_AI_PROVIDER_NAME, _PROVIDER)
            set_span_attribute(span, _GEN_AI_MEMORY_SCOPE, _infer_memory_scope(instance))
            set_span_attribute(span, _GEN_AI_MEMORY_TYPE, _infer_memory_type(kwargs))
            set_span_attribute(span, _GEN_AI_MEMORY_UPDATE_STRATEGY, "merge")

            # Namespace from source kwarg
            source = kwargs.get("source")
            if source:
                set_span_attribute(span, _GEN_AI_MEMORY_NAMESPACE, str(source))

            # Scope path
            scope = kwargs.get("scope")
            if scope:
                set_span_attribute(span, "crewai.memory.scope_path", str(scope))

            importance = kwargs.get("importance")
            if importance is not None:
                set_span_attribute(span, _GEN_AI_MEMORY_IMPORTANCE, float(importance))

            # Content (opt-in)
            if _capture_content() and args:
                content = args[0] if args else kwargs.get("content")
                if content and isinstance(content, str):
                    set_span_attribute(span, _GEN_AI_MEMORY_CONTENT, content)

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                # MemoryRecord has an id attribute
                if result and hasattr(result, "id"):
                    set_span_attribute(span, "gen_ai.memory.id", str(result.id))
                return result
            except Exception as ex:
                error_type = _set_memory_error(span, ex)
                raise
            finally:
                _record_memory_duration(
                    duration_histogram, time.time() - start_time, operation, error_type
                )
    return _wrapper


def wrap_memory_recall(tracer: Tracer, duration_histogram: Histogram):
    """Wrap Memory.recall() → search_memory span."""
    def _wrapper(wrapped, instance, args, kwargs):
        operation = "search_memory"
        span_name = f"{operation} {_PROVIDER}"
        error_type = None
        start_time = time.time()
        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT,
            attributes={GenAIAttributes.GEN_AI_SYSTEM: _PROVIDER}
        ) as span:
            set_span_attribute(span, _GEN_AI_OPERATION_NAME, operation)
            set_span_attribute(span, _GEN_AI_PROVIDER_NAME, _PROVIDER)
            set_span_attribute(span, _GEN_AI_MEMORY_SCOPE, _infer_memory_scope(instance))
            set_span_attribute(span, _GEN_AI_MEMORY_TYPE, _infer_memory_type(kwargs))

            # Query (opt-in)
            query = args[0] if args else kwargs.get("query")
            if _capture_content() and query and isinstance(query, str):
                set_span_attribute(span, _GEN_AI_MEMORY_QUERY, query)

            # Scope path
            scope = kwargs.get("scope")
            if scope:
                set_span_attribute(span, "crewai.memory.scope_path", str(scope))

            source = kwargs.get("source")
            if source:
                set_span_attribute(span, _GEN_AI_MEMORY_NAMESPACE, str(source))

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                if isinstance(result, list):
                    set_span_attribute(span, _GEN_AI_MEMORY_SEARCH_RESULT_COUNT, len(result))
                return result
            except Exception as ex:
                error_type = _set_memory_error(span, ex)
                raise
            finally:
                _record_memory_duration(
                    duration_histogram, time.time() - start_time, operation, error_type
                )
    return _wrapper


def wrap_memory_forget(tracer: Tracer, duration_histogram: Histogram):
    """Wrap Memory.forget() → delete_memory span."""
    def _wrapper(wrapped, instance, args, kwargs):
        operation = "delete_memory"
        span_name = f"{operation} {_PROVIDER}"
        error_type = None
        start_time = time.time()
        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT,
            attributes={GenAIAttributes.GEN_AI_SYSTEM: _PROVIDER}
        ) as span:
            set_span_attribute(span, _GEN_AI_OPERATION_NAME, operation)
            set_span_attribute(span, _GEN_AI_PROVIDER_NAME, _PROVIDER)
            set_span_attribute(span, _GEN_AI_MEMORY_SCOPE, _infer_memory_scope(instance))

            scope = kwargs.get("scope")
            if scope:
                set_span_attribute(span, "crewai.memory.scope_path", str(scope))

            record_ids = kwargs.get("record_ids")
            if record_ids and isinstance(record_ids, list) and len(record_ids) == 1:
                set_span_attribute(span, "gen_ai.memory.id", str(record_ids[0]))

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                # forget() returns number of deleted records
                if isinstance(result, int):
                    set_span_attribute(span, "crewai.memory.deleted_count", result)
                return result
            except Exception as ex:
                error_type = _set_memory_error(span, ex)
                raise
            finally:
                _record_memory_duration(
                    duration_histogram, time.time() - start_time, operation, error_type
                )
    return _wrapper


def wrap_memory_reset(tracer: Tracer, duration_histogram: Histogram):
    """Wrap Memory.reset() → delete_memory span (scope-level wipe)."""
    def _wrapper(wrapped, instance, args, kwargs):
        operation = "delete_memory"
        span_name = f"{operation} {_PROVIDER}"
        error_type = None
        start_time = time.time()
        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT,
            attributes={GenAIAttributes.GEN_AI_SYSTEM: _PROVIDER}
        ) as span:
            set_span_attribute(span, _GEN_AI_OPERATION_NAME, operation)
            set_span_attribute(span, _GEN_AI_PROVIDER_NAME, _PROVIDER)
            set_span_attribute(span, _GEN_AI_MEMORY_SCOPE, _infer_memory_scope(instance))
            set_span_attribute(span, "crewai.memory.reset", True)

            scope = kwargs.get("scope")
            if scope:
                set_span_attribute(span, "crewai.memory.scope_path", str(scope))

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as ex:
                error_type = _set_memory_error(span, ex)
                raise
            finally:
                _record_memory_duration(
                    duration_histogram, time.time() - start_time, operation, error_type
                )
    return _wrapper
