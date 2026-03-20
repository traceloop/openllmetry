from __future__ import annotations

from typing import Any

from openai._legacy_response import LegacyAPIResponse
from opentelemetry import context as context_api
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode

from opentelemetry.instrumentation.openai.v1.responses_safety import (
    apply_response_completion_safety,
    apply_response_prompt_safety,
)


def get_existing_response_data(response_store: dict, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    for key in ("response_id", "previous_response_id"):
        response_id = request_kwargs.get(key)
        if response_id and response_id in response_store:
            return response_store[response_id].model_dump()
    return {}


def get_parsed_response_output_text(parsed_response) -> str | None:
    if hasattr(parsed_response, "output_text"):
        return parsed_response.output_text
    try:
        return parsed_response.output[0].content[0].text
    except Exception:
        return None


def cache_legacy_parsed_response(response, parsed_response) -> None:
    if not isinstance(response, LegacyAPIResponse):
        return
    parsed_cache = getattr(response, "_parsed_by_type", None)
    if not isinstance(parsed_cache, dict):
        return
    cache_key = getattr(response, "_cast_to", type(parsed_response))
    parsed_cache[cache_key] = parsed_response


def create_error_traced_data(
    traced_data_cls,
    *,
    existing_data: dict[str, Any],
    request_kwargs: dict[str, Any],
    start_time: int,
    process_input,
    get_tools_from_kwargs,
):
    return traced_data_cls(
        start_time=existing_data.get("start_time", start_time),
        response_id=request_kwargs.get("response_id") or "",
        input=process_input(request_kwargs.get("input", existing_data.get("input", []))),
        instructions=request_kwargs.get("instructions", existing_data.get("instructions", "")),
        tools=get_tools_from_kwargs(request_kwargs) or existing_data.get("tools", []),
        output_blocks=existing_data.get("output_blocks", {}),
        usage=existing_data.get("usage"),
        output_text=request_kwargs.get("output_text", existing_data.get("output_text", "")),
        request_model=request_kwargs.get("model", existing_data.get("request_model", "")),
        response_model=existing_data.get("response_model", ""),
        request_reasoning_summary=request_kwargs.get("reasoning", {}).get(
            "summary", existing_data.get("request_reasoning_summary")
        ),
        request_reasoning_effort=request_kwargs.get("reasoning", {}).get(
            "effort", existing_data.get("request_reasoning_effort")
        ),
        response_reasoning_effort=request_kwargs.get("reasoning", {}).get("effort"),
        request_service_tier=request_kwargs.get("service_tier"),
        response_service_tier=existing_data.get("response_service_tier"),
        trace_context=existing_data.get("trace_context", context_api.get_current()),
    )


def create_success_traced_data(
    traced_data_cls,
    *,
    existing_data: dict[str, Any],
    request_kwargs: dict[str, Any],
    parsed_response,
    output_text: str | None,
    merged_tools,
    process_input,
):
    return traced_data_cls(
        start_time=existing_data.get("start_time"),
        response_id=parsed_response.id,
        input=process_input(existing_data.get("input", request_kwargs.get("input"))),
        instructions=existing_data.get("instructions", request_kwargs.get("instructions")),
        tools=merged_tools if merged_tools else None,
        output_blocks={block.id: block for block in parsed_response.output}
        | existing_data.get("output_blocks", {}),
        usage=existing_data.get("usage", parsed_response.usage),
        output_text=existing_data.get("output_text", output_text),
        request_model=existing_data.get("request_model", request_kwargs.get("model")),
        response_model=existing_data.get("response_model", parsed_response.model),
        request_reasoning_summary=request_kwargs.get("reasoning", {}).get(
            "summary", existing_data.get("request_reasoning_summary")
        ),
        request_reasoning_effort=request_kwargs.get("reasoning", {}).get(
            "effort", existing_data.get("request_reasoning_effort")
        ),
        response_reasoning_effort=request_kwargs.get("reasoning", {}).get("effort"),
        request_service_tier=existing_data.get(
            "request_service_tier", request_kwargs.get("service_tier")
        ),
        response_service_tier=existing_data.get(
            "response_service_tier", parsed_response.service_tier
        ),
        trace_context=existing_data.get("trace_context", context_api.get_current()),
    )


def create_stream_traced_data(
    traced_data_cls,
    *,
    existing_data: dict[str, Any],
    request_kwargs: dict[str, Any],
    start_time: int | None,
    process_input,
    get_tools_from_kwargs,
):
    return traced_data_cls(
        start_time=existing_data.get("start_time", start_time),
        response_id="",
        input=existing_data.get("input", process_input(request_kwargs.get("input", []))),
        instructions=request_kwargs.get("instructions", existing_data.get("instructions")),
        tools=get_tools_from_kwargs(request_kwargs) or existing_data.get("tools"),
        output_blocks=existing_data.get("output_blocks", {}),
        usage=existing_data.get("usage"),
        output_text=existing_data.get("output_text", ""),
        request_model=request_kwargs.get("model", existing_data.get("request_model", "")),
        response_model=existing_data.get("response_model", ""),
        request_reasoning_summary=request_kwargs.get("reasoning", {}).get(
            "summary", existing_data.get("request_reasoning_summary")
        ),
        request_reasoning_effort=request_kwargs.get("reasoning", {}).get(
            "effort", existing_data.get("request_reasoning_effort")
        ),
        response_reasoning_effort=existing_data.get("response_reasoning_effort"),
        request_service_tier=request_kwargs.get(
            "service_tier", existing_data.get("request_service_tier")
        ),
        response_service_tier=existing_data.get("response_service_tier"),
        trace_context=existing_data.get("trace_context"),
    )


def prepare_response_request(
    *,
    tracer,
    span_name: str,
    start_time: int,
    context,
    request_kwargs: dict[str, Any],
    instance,
    prepare_request_attributes,
    set_request_attributes,
):
    span = None
    updated_kwargs = request_kwargs
    if "input" in updated_kwargs or "instructions" in updated_kwargs:
        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            start_time=start_time,
            context=context,
        )
        updated_kwargs = apply_response_prompt_safety(span, updated_kwargs)
        set_request_attributes(
            span,
            prepare_request_attributes(updated_kwargs),
            instance,
        )
    return span, updated_kwargs


def handle_response_error(
    *,
    tracer,
    span_name: str,
    error,
    current_span,
    context,
    request_kwargs: dict[str, Any],
    instance,
    start_time: int,
    traced_data_cls,
    response_store: dict,
    process_input,
    get_tools_from_kwargs,
    prepare_request_attributes,
    set_request_attributes,
    set_data_attributes,
):
    existing_data = get_existing_response_data(response_store, request_kwargs)
    try:
        traced_data = create_error_traced_data(
            traced_data_cls,
            existing_data=existing_data,
            request_kwargs=request_kwargs,
            start_time=start_time,
            process_input=process_input,
            get_tools_from_kwargs=get_tools_from_kwargs,
        )
    except Exception:
        traced_data = None

    active_span = current_span or tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        start_time=(start_time if traced_data is None else int(traced_data.start_time)),
        context=(
            traced_data.trace_context
            if traced_data and traced_data.trace_context
            else context
        ),
    )
    if current_span is None:
        set_request_attributes(
            active_span,
            prepare_request_attributes(request_kwargs),
            instance,
        )
    active_span.set_attribute(ERROR_TYPE, error.__class__.__name__)
    active_span.record_exception(error)
    active_span.set_status(StatusCode.ERROR, str(error))
    if traced_data:
        set_data_attributes(traced_data, active_span)
    active_span.end()


def handle_response_success(
    *,
    tracer,
    span_name: str,
    current_span,
    context,
    request_kwargs: dict[str, Any],
    instance,
    start_time: int,
    response,
    parsed_response,
    traced_data_cls,
    response_store: dict,
    process_input,
    get_tools_from_kwargs,
    prepare_request_attributes,
    set_request_attributes,
    set_data_attributes,
):
    existing_data_model = response_store.get(parsed_response.id)
    if existing_data_model is None:
        existing_data = {}
    else:
        existing_data = existing_data_model.model_dump()

    request_tools = get_tools_from_kwargs(request_kwargs)
    merged_tools = existing_data.get("tools", []) + request_tools

    try:
        parsed_response_output_text = get_parsed_response_output_text(parsed_response)
        active_span = current_span or tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            start_time=int(existing_data.get("start_time", start_time)),
            context=existing_data.get("trace_context", context),
        )
        if current_span is None:
            set_request_attributes(
                active_span,
                prepare_request_attributes(request_kwargs),
                instance,
            )
        traced_data = create_success_traced_data(
            traced_data_cls,
            existing_data={
                **existing_data,
                "start_time": existing_data.get("start_time", start_time),
                "trace_context": existing_data.get("trace_context", context),
            },
            request_kwargs=request_kwargs,
            parsed_response=parsed_response,
            output_text=parsed_response_output_text,
            merged_tools=merged_tools,
            process_input=process_input,
        )
        masked_output_text = apply_response_completion_safety(active_span, parsed_response)
        if masked_output_text is not None:
            traced_data.output_text = masked_output_text
        cache_legacy_parsed_response(response, parsed_response)
        response_store[parsed_response.id] = traced_data
    except Exception:
        if current_span is not None:
            current_span.end()
        return False

    if active_span is not None:
        set_data_attributes(traced_data, active_span)
        active_span.end()
    return True
