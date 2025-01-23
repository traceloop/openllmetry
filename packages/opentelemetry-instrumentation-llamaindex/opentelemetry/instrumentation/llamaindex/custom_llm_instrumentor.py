import importlib
import pkgutil
from typing import Optional

from wrapt import wrap_function_wrapper
from inflection import underscore

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    dont_throw,
    start_as_current_span_async,
    should_send_prompts,
)
from opentelemetry.instrumentation.llamaindex.events import (
    create_prompt_event,
    create_completion_event,
)
from opentelemetry.instrumentation.llamaindex.config import Config

import llama_index.llms

try:
    from llama_index.core.llms.custom import CustomLLM
    MODULE_NAME = "llama_index.llms"
except ModuleNotFoundError:
    from llama_index.llms import CustomLLM
    MODULE_NAME = "llama_index.llms"


class CustomLLMInstrumentor:
    def __init__(self, tracer, event_logger: Optional[EventLogger] = None):
        self._tracer = tracer
        self._event_logger = event_logger
        self.config = Config()

    def instrument(self):
        packages = pkgutil.iter_modules(llama_index.llms.__path__)
        modules = [
            importlib.import_module(f"llama_index.llms.{p.name}") for p in packages
        ]
        custom_llms_classes = [
            cls
            for module in modules
            for name, cls in module.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, CustomLLM)
        ]

        for cls in custom_llms_classes:
            wrap_function_wrapper(
                cls.__module__,
                f"{cls.__name__}.complete",
                complete_wrapper(self._tracer, self._event_logger, self.config),
            )
            wrap_function_wrapper(
                cls.__module__,
                f"{cls.__name__}.acomplete",
                acomplete_wrapper(self._tracer, self._event_logger, self.config),
            )
            wrap_function_wrapper(
                cls.__module__,
                f"{cls.__name__}.chat",
                chat_wrapper(self._tracer, self._event_logger, self.config),
            )
            wrap_function_wrapper(
                cls.__module__,
                f"{cls.__name__}.achat",
                achat_wrapper(self._tracer, self._event_logger, self.config),
            )

    def uninstrument(self):
        pass

def _set_span_attribute(span, name, value):
    if value is not None:
        span.set_attribute(name, value)

@_with_tracer_wrapper
def chat_wrapper(tracer, event_logger, config, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span(
        name=f"{snake_case_class_name(instance)}.chat",
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, event_logger, config, LLMRequestTypeValues.CHAT, args, kwargs, instance)
        response = wrapped(*args, **kwargs)
        _handle_response(span, event_logger, config, LLMRequestTypeValues.CHAT, instance, response)
        return response

@_with_tracer_wrapper
async def achat_wrapper(tracer, event_logger, config, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    async with start_as_current_span_async(
        tracer,
        name=f"{snake_case_class_name(instance)}.chat",
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, event_logger, config, LLMRequestTypeValues.CHAT, args, kwargs, instance)
        response = await wrapped(*args, **kwargs)
        _handle_response(span, event_logger, config, LLMRequestTypeValues.CHAT, instance, response)
        return response

@_with_tracer_wrapper
def complete_wrapper(tracer, event_logger, config, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span(
        name=f"{snake_case_class_name(instance)}.complete",
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, event_logger, config, LLMRequestTypeValues.COMPLETION, args, kwargs, instance)
        response = wrapped(*args, **kwargs)
        _handle_response(span, event_logger, config, LLMRequestTypeValues.COMPLETION, instance, response)
        return response

@_with_tracer_wrapper
async def acomplete_wrapper(tracer, event_logger, config, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    async with start_as_current_span_async(
        tracer,
        name=f"{snake_case_class_name(instance)}.complete",
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, event_logger, config, LLMRequestTypeValues.COMPLETION, args, kwargs, instance)
        response = await wrapped(*args, **kwargs)
        _handle_response(span, event_logger, config, LLMRequestTypeValues.COMPLETION, instance, response)
        return response

@dont_throw
def _handle_request(span, event_logger, config, llm_request_type, args, kwargs, instance: CustomLLM):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, llm_request_type)
    _set_span_attribute(span, SpanAttributes.LLM_VENDOR, instance.metadata.get("vendor"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.metadata.get("model"))

    if should_send_prompts():
        prompt = kwargs.get("prompt") or (args[0] if args else None)
        if config.use_legacy_attributes:
            _set_span_attribute(span, SpanAttributes.LLM_PROMPTS, [prompt] if prompt else None)
        
        if event_logger and prompt:
            span_context = span.get_span_context()
            event_logger.emit(
                create_prompt_event(
                    {"prompt": prompt},
                    trace_id=span_context.trace_id,
                    span_id=span_context.span_id,
                    trace_flags=span_context.trace_flags,
                )
            )

@dont_throw
def _handle_response(span, event_logger, config, llm_request_type, instance, response):
    if config.use_legacy_attributes:
        _set_span_attribute(span, SpanAttributes.LLM_COMPLETIONS, [response] if response else None)
        _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, instance.metadata.get("model"))

    if event_logger and response:
        span_context = span.get_span_context()
        event_logger.emit(
            create_completion_event(
                {"completion": response},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

def snake_case_class_name(instance):
    return underscore(instance.__class__.__name__)
