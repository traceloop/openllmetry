import importlib
import json
import pkgutil

from wrapt import wrap_function_wrapper
from inflection import underscore

from opentelemetry import context as context_api

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.llamaindex._message_utils import (
    build_completion_output_message,
    build_input_messages,
    build_output_message,
)
from opentelemetry.instrumentation.llamaindex._response_utils import (
    detect_provider_name,
    extract_finish_reasons,
    extract_model_from_raw,
    extract_response_id,
    extract_token_usage,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    dont_throw,
    start_as_current_span_async,
    should_send_prompts,
)

import llama_index.llms

try:
    from llama_index.core.llms.custom import CustomLLM

    MODULE_NAME = "llama_index.llms"
except ModuleNotFoundError:
    from llama_index.llms import CustomLLM

    MODULE_NAME = "llama_index.llms"


class CustomLLMInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

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
                complete_wrapper(self._tracer),
            )
            wrap_function_wrapper(
                cls.__module__,
                f"{cls.__name__}.acomplete",
                acomplete_wrapper(self._tracer),
            )
            wrap_function_wrapper(
                cls.__module__, f"{cls.__name__}.chat", chat_wrapper(self._tracer)
            )
            wrap_function_wrapper(
                cls.__module__, f"{cls.__name__}.achat", achat_wrapper(self._tracer)
            )

    def unistrument(self):
        pass


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@_with_tracer_wrapper
def chat_wrapper(tracer, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    llm_request_type = LLMRequestTypeValues.CHAT

    with tracer.start_as_current_span(
        f"{snake_case_class_name(instance)}.chat"
    ) as span:
        _handle_request(span, llm_request_type, args, kwargs, instance)
        response = wrapped(*args, **kwargs)
        _handle_response(span, llm_request_type, instance, response)

        return response


@_with_tracer_wrapper
async def achat_wrapper(tracer, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    llm_request_type = LLMRequestTypeValues.CHAT

    async with start_as_current_span_async(
        tracer=tracer, name=f"{snake_case_class_name(instance)}.chat"
    ) as span:
        _handle_request(span, llm_request_type, args, kwargs, instance)
        response = await wrapped(*args, **kwargs)
        _handle_response(span, llm_request_type, instance, response)

        return response


@_with_tracer_wrapper
def complete_wrapper(tracer, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    llm_request_type = LLMRequestTypeValues.COMPLETION

    with tracer.start_as_current_span(
        f"{snake_case_class_name(instance)}.completion"
    ) as span:
        _handle_request(span, llm_request_type, args, kwargs, instance)
        response = wrapped(*args, **kwargs)
        _handle_response(span, llm_request_type, instance, response)

        return response


@_with_tracer_wrapper
async def acomplete_wrapper(tracer, wrapped, instance: CustomLLM, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    llm_request_type = LLMRequestTypeValues.COMPLETION

    async with start_as_current_span_async(
        tracer=tracer, name=f"{snake_case_class_name(instance)}.completion"
    ) as span:
        _handle_request(span, llm_request_type, args, kwargs, instance)
        response = await wrapped(*args, **kwargs)
        _handle_response(span, llm_request_type, instance, response)

        return response


@dont_throw
def _handle_request(span, llm_request_type, args, kwargs, instance: CustomLLM):
    op_name = "chat" if llm_request_type == LLMRequestTypeValues.CHAT else "text_completion"
    _set_span_attribute(span, GenAIAttributes.GEN_AI_OPERATION_NAME, op_name)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_PROVIDER_NAME, detect_provider_name(instance))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, instance.metadata.model_name)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, instance.metadata.num_output)

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT and args:
            messages = args[0]
            if messages:
                msgs = build_input_messages(messages)
                span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(msgs))
        elif llm_request_type == LLMRequestTypeValues.COMPLETION and args:
            prompt = args[0]
            text = prompt[0] if isinstance(prompt, list) else prompt
            msg = [{"role": "user", "parts": [{"type": "text", "content": text}]}]
            span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(msg))

        tools = kwargs.get("tools")
        if tools:
            span.set_attribute(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tools))


@dont_throw
def _handle_response(span, llm_request_type, instance, response):
    raw = getattr(response, "raw", None)

    response_model = extract_model_from_raw(raw) if raw else None
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model or instance.metadata.model_name)

    if raw:
        response_id = extract_response_id(raw)
        if response_id:
            _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_id)

        usage = extract_token_usage(raw)
        if usage.input_tokens is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, int(usage.input_tokens))
        if usage.output_tokens is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, int(usage.output_tokens))
        if usage.total_tokens is not None:
            span.set_attribute(SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, int(usage.total_tokens))

    # CRITICAL: finish_reasons is NOT gated by should_send_prompts()
    reasons = extract_finish_reasons(raw) if raw else []
    if reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, reasons)

    if should_send_prompts():
        fr = reasons[0] if reasons else None
        if llm_request_type == LLMRequestTypeValues.CHAT and hasattr(response, "message"):
            output_msg = build_output_message(response.message, finish_reason=fr)
        else:
            output_msg = build_completion_output_message(response.text, finish_reason=fr)
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps([output_msg]))


def snake_case_class_name(instance):
    return underscore(instance.__class__.__name__)
