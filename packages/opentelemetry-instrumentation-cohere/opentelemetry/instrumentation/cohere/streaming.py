from opentelemetry.instrumentation.cohere.event_emitter import emit_response_events
from opentelemetry.instrumentation.cohere.utils import (
    dont_throw,
    should_send_prompts,
    to_dict,
    should_emit_events,
)
from opentelemetry.instrumentation.cohere.span_utils import set_span_response_attributes, _set_span_chat_response
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

DEFAULT_MESSAGE = {
    "content": [],
    "role": "assistant",
    "tool_calls": [],
    "tool_plan": "",
    # TODO: Add citations
}


@dont_throw
def process_chat_v1_streaming_response(span, event_logger, llm_request_type, response):
    # This naive version assumes we've always successfully streamed till the end
    # and have received a StreamEndChatResponse, which includes the full response
    final_response = None
    try:
        for item in response:
            span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

            item_to_yield = item
            if getattr(item, "event_type", None) == "stream-end" and hasattr(item, "response"):
                final_response = item.response

            yield item_to_yield

        set_span_response_attributes(span, final_response)
        if should_emit_events():
            emit_response_events(event_logger, llm_request_type, final_response)
        elif should_send_prompts():
            _set_span_chat_response(span, final_response)
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end()


@dont_throw
async def aprocess_chat_v1_streaming_response(span, event_logger, llm_request_type, response):
    # This naive version assumes we've always successfully streamed till the end
    # and have received a StreamEndChatResponse, which includes the full response
    final_response = None
    try:
        async for item in response:
            span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

            item_to_yield = item
            if getattr(item, "event_type", None) == "stream-end" and hasattr(item, "response"):
                final_response = item.response

            yield item_to_yield
        set_span_response_attributes(span, final_response)
        if should_emit_events():
            emit_response_events(event_logger, llm_request_type, final_response)
        elif should_send_prompts():
            _set_span_chat_response(span, final_response)
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end()


@dont_throw
def process_chat_v2_streaming_response(span, event_logger, llm_request_type, response):
    final_response = {
        "finish_reason": None,
        "message": DEFAULT_MESSAGE,
        "usage": {},
        "id": "",
        "error": None,
    }
    current_content_item = {"type": "text", "thinking": None, "text": ""}
    current_tool_call_item = {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": "", "description": ""},
    }
    try:
        for item in response:
            span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")
            item_to_yield = item
            try:
                _accumulate_stream_item(item, current_content_item, current_tool_call_item, final_response)
            except Exception:
                pass
            yield item_to_yield

        set_span_response_attributes(span, final_response)
        if should_emit_events():
            emit_response_events(event_logger, llm_request_type, final_response)
        elif should_send_prompts():
            _set_span_chat_response(span, final_response)

        if final_response.get("error"):
            span.set_status(Status(StatusCode.ERROR, final_response.get("error")))
            span.record_exception(final_response.get("error"))
        else:
            span.set_status(Status(StatusCode.OK))
    finally:
        span.end()


@dont_throw
async def aprocess_chat_v2_streaming_response(span, event_logger, llm_request_type, response):
    final_response = {
        "finish_reason": None,
        "message": DEFAULT_MESSAGE,
        "usage": {},
        "id": "",
        "error": None,
    }
    current_content_item = {"type": "text", "thinking": None, "text": ""}
    current_tool_call_item = {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": "", "description": ""},
    }
    async for item in response:
        span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")
        item_to_yield = item
        try:
            _accumulate_stream_item(item, current_content_item, current_tool_call_item, final_response)
        except Exception:
            pass
        yield item_to_yield

    set_span_response_attributes(span, final_response)
    if should_emit_events():
        emit_response_events(event_logger, llm_request_type, final_response)
    elif should_send_prompts():
        _set_span_chat_response(span, final_response)
    if final_response.get("error"):
        span.set_status(Status(StatusCode.ERROR, final_response.get("error")))
        span.record_exception(final_response.get("error"))
    else:
        span.set_status(Status(StatusCode.OK))
        span.end()


# accumulated items are passed in by reference
def _accumulate_stream_item(item, current_content_item, current_tool_call_item, final_response):
    item_dict = to_dict(item)
    if item_dict.get("type") == "message-start":
        final_response["message"] = (item_dict.get("delta") or {}).get("message") or {**DEFAULT_MESSAGE}
        final_response["id"] = item_dict.get("id")
    elif item_dict.get("type") == "content-start":
        new_content_item = ((item_dict.get("delta") or {}).get("message") or {}).get("content")
        current_content_item.clear()
        current_content_item.update(new_content_item or {})
    elif item_dict.get("type") == "content-delta":
        new_thinking = (((item_dict.get("delta") or {}).get("message") or {}).get("content") or {}).get("thinking")
        if new_thinking:
            existing_thinking = current_content_item.get("thinking")
            current_content_item["thinking"] = (existing_thinking or "") + new_thinking
        new_text = (((item_dict.get("delta") or {}).get("message") or {}).get("content") or {}).get("text")
        if new_text:
            existing_text = current_content_item.get("text")
            current_content_item["text"] = (existing_text or "") + new_text
    elif item_dict.get("type") == "content-end":
        final_response["message"]["content"].append({**current_content_item})
    elif item_dict.get("type") == "tool-plan-delta":
        new_tool_plan = ((item_dict.get("delta") or {}).get("message") or {}).get("tool_plan")
        if new_tool_plan:
            existing_tool_plan = final_response["message"].get("tool_plan")
            final_response["message"]["tool_plan"] = (existing_tool_plan or "") + new_tool_plan
    elif item_dict.get("type") == "tool-call-start":
        new_tool_call_item = ((item_dict.get("delta") or {}).get("message") or {}).get("tool_calls")
        current_tool_call_item.update(new_tool_call_item or {})
    elif item_dict.get("type") == "tool-call-delta":
        message = (item_dict.get("delta") or {}).get("message") or {}
        new_arguments = ((message.get("tool_calls") or {}).get("function") or {}).get("arguments")
        if new_arguments:
            existing_arguments = (current_tool_call_item.get("function") or {}).get("arguments")
            current_tool_call_item["function"]["arguments"] = (existing_arguments or "") + new_arguments
    elif item_dict.get("type") == "tool-call-end":
        final_response["message"]["tool_calls"].append({**current_tool_call_item})
    elif item_dict.get("type") == "message-end":
        final_response["usage"] = (item_dict.get("delta") or {}).get("usage") or {}
        final_response["finish_reason"] = (item_dict.get("delta") or {}).get("finish_reason")
