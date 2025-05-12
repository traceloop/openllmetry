import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import (
    LLMResult,
)
from opentelemetry.context.context import Context
from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.span import Span


@dataclass
class SpanHolder:
    span: Span
    token: Any
    context: Context
    children: list[UUID]
    workflow_name: str
    entity_name: str
    entity_path: str
    start_time: float = field(default_factory=time.time)
    request_model: Optional[str] = None


def _message_type_to_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    else:
        return "unknown"


def _set_span_attribute(span, name, value):
    if value is not None:
        span.set_attribute(name, value)


def set_request_params(span, kwargs, span_holder: SpanHolder):
    if not span.is_recording():
        return

    for model_tag in ("model", "model_id", "model_name"):
        if (model := kwargs.get(model_tag)) is not None:
            span_holder.request_model = model
            break
        elif (
            model := (kwargs.get("invocation_params") or {}).get(model_tag)
        ) is not None:
            span_holder.request_model = model
            break
    else:
        model = "unknown"

    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if "invocation_params" in kwargs:
        params = (
            kwargs["invocation_params"].get("params") or kwargs["invocation_params"]
        )
    else:
        params = kwargs

    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        params.get("max_tokens") or params.get("max_new_tokens"),
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, params.get("temperature")
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, params.get("top_p"))


def set_llm_request(
    span: Span,
    serialized: dict[str, Any],
    prompts: list[str],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    if should_send_prompts():
        for i, msg in enumerate(prompts):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                "user",
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                msg,
            )


def set_chat_request(
    span: Span,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    if should_send_prompts():
        for i, function in enumerate(
            kwargs.get("invocation_params", {}).get("functions", [])
        ):
            prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"

            _set_span_attribute(span, f"{prefix}.name", function.get("name"))
            _set_span_attribute(
                span, f"{prefix}.description", function.get("description")
            )
            _set_span_attribute(
                span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
            )

        i = 0
        for message in messages:
            for msg in message:
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    _message_type_to_role(msg.type),
                )
                # if msg.content is string
                if isinstance(msg.content, str):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        msg.content,
                    )
                else:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        json.dumps(msg.content, cls=CallbackFilteredJSONEncoder),
                    )
                i += 1


def set_chat_response(span: Span, response: LLMResult) -> None:
    if not should_send_prompts():
        return

    i = 0
    for generations in response.generations:
        for generation in generations:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
            if hasattr(generation, "text") and generation.text != "":
                span.set_attribute(
                    f"{prefix}.content",
                    generation.text,
                )
                span.set_attribute(f"{prefix}.role", "assistant")
            else:
                span.set_attribute(
                    f"{prefix}.role",
                    _message_type_to_role(generation.type),
                )
                if generation.message.content is str:
                    span.set_attribute(
                        f"{prefix}.content",
                        generation.message.content,
                    )
                else:
                    span.set_attribute(
                        f"{prefix}.content",
                        json.dumps(
                            generation.message.content, cls=CallbackFilteredJSONEncoder
                        ),
                    )
                if generation.generation_info.get("finish_reason"):
                    span.set_attribute(
                        f"{prefix}.finish_reason",
                        generation.generation_info.get("finish_reason"),
                    )

                if generation.message.additional_kwargs.get("function_call"):
                    span.set_attribute(
                        f"{prefix}.tool_calls.0.name",
                        generation.message.additional_kwargs.get("function_call").get(
                            "name"
                        ),
                    )
                    span.set_attribute(
                        f"{prefix}.tool_calls.0.arguments",
                        generation.message.additional_kwargs.get("function_call").get(
                            "arguments"
                        ),
                    )

                if generation.message.additional_kwargs.get("tool_calls"):
                    for idx, tool_call in enumerate(
                        generation.message.additional_kwargs.get("tool_calls")
                    ):
                        tool_call_prefix = f"{prefix}.tool_calls.{idx}"

                        span.set_attribute(
                            f"{tool_call_prefix}.id", tool_call.get("id")
                        )
                        span.set_attribute(
                            f"{tool_call_prefix}.name",
                            tool_call.get("function").get("name"),
                        )
                        span.set_attribute(
                            f"{tool_call_prefix}.arguments",
                            tool_call.get("function").get("arguments"),
                        )
            i += 1


def set_chat_response_usage(span: Span, response: LLMResult):
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cache_read_tokens = 0

    i = 0
    for generations in response.generations:
        for generation in generations:
            if (
                hasattr(generation, "message")
                and hasattr(generation.message, "usage_metadata")
                and generation.message.usage_metadata is not None
            ):
                input_tokens += (
                    generation.message.usage_metadata.get("input_tokens")
                    or generation.message.usage_metadata.get("prompt_tokens")
                    or 0
                )
                output_tokens += (
                    generation.message.usage_metadata.get("output_tokens")
                    or generation.message.usage_metadata.get("completion_tokens")
                    or 0
                )
                total_tokens = input_tokens + output_tokens

                if generation.message.usage_metadata.get("input_token_details"):
                    input_token_details = generation.message.usage_metadata.get(
                        "input_token_details", {}
                    )
                    cache_read_tokens += input_token_details.get("cache_read", 0)
            i += 1

    if (
        input_tokens > 0
        or output_tokens > 0
        or total_tokens > 0
        or cache_read_tokens > 0
    ):
        span.set_attribute(
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            input_tokens,
        )
        span.set_attribute(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            output_tokens,
        )
        span.set_attribute(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            total_tokens,
        )
        span.set_attribute(
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            cache_read_tokens,
        )
