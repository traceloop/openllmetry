"""Hook-based instrumentation for OpenAI Agents using the SDK's native callback system."""

from typing import Dict, Any
import json
import time
from collections import OrderedDict
from opentelemetry.trace import (
    Tracer,
    Status,
    StatusCode,
    SpanKind,
    get_current_span,
    set_span_in_context,
)
from opentelemetry import context
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agents.tracing.processors import TracingProcessor

from .utils import (
    should_send_prompts,
    dont_throw,
    GEN_AI_HANDOFF_FROM_AGENT,
    GEN_AI_HANDOFF_TO_AGENT,
)

try:
    # Attempt to import once, so that we aren't looking for it repeatedly.
    # Each failed lookup is somewhat expensive as it has to walk the path.
    from traceloop.sdk.tracing import set_agent_name
except ModuleNotFoundError:
    set_agent_name = None

# Import realtime span types at module level to avoid repeated lookups
try:
    from agents import SpeechSpanData, TranscriptionSpanData, SpeechGroupSpanData

    _has_realtime_spans = True
except ImportError:
    _has_realtime_spans = False
    SpeechSpanData = None
    TranscriptionSpanData = None
    SpeechGroupSpanData = None


# ---------------------------------------------------------------------------
# Finish-reason mapping: OpenAI → OTel GenAI semconv
# ---------------------------------------------------------------------------
_FINISH_REASON_MAP = {
    "stop": "stop",
    "tool_calls": "tool_call",    # plural → singular per OTel spec
    "function_call": "tool_call",  # legacy → OTel value
    "length": "length",
    "content_filter": "content_filter",
    "error": "error",
}


def _map_finish_reason(raw):
    """Map a provider-specific finish reason to the OTel enum value."""
    if raw is None:
        return None
    return _FINISH_REASON_MAP.get(raw, raw)


def _parse_arguments(args):
    """Best-effort parse of tool-call arguments to a dict (object) or None.

    Per OTel spec, arguments must be objects, never raw JSON strings.
    Falls back to ``{"_raw": args}`` when the string is not valid JSON
    or parses to a non-dict type.
    """
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args.strip():
            return None
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            # Parsed OK but not a dict (e.g. array, scalar) – wrap
            return {"_raw": args}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": args}
    return {"_raw": str(args)}


def _normalize_tool_call(tool_call):
    """Normalize a tool call (object or dict) into a flat {id, name, arguments} dict."""
    if isinstance(tool_call, dict):
        tc = dict(tool_call)
        if "function" in tc:
            function = tc["function"]
            if isinstance(function, dict):
                tc = {
                    "id": tc.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            else:
                tc = {
                    "id": tc.get("id"),
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                }
        return tc
    # Object with attributes
    tc_dict: dict = {}
    if hasattr(tool_call, "id"):
        tc_dict["id"] = tool_call.id
    if hasattr(tool_call, "function"):
        func = tool_call.function
        if hasattr(func, "name"):
            tc_dict["name"] = func.name
        if hasattr(func, "arguments"):
            tc_dict["arguments"] = func.arguments
    elif hasattr(tool_call, "name"):
        tc_dict["name"] = tool_call.name
    if hasattr(tool_call, "arguments") and "arguments" not in tc_dict:
        tc_dict["arguments"] = tool_call.arguments
    return tc_dict


_MESSAGE_ATTRS = (
    "role", "content", "tool_call_id", "tool_calls",
    "type", "name", "arguments", "call_id", "output",
)


def _msg_to_dict(message) -> dict:
    """Normalize a message (dict or SDK object) into a plain dict."""
    if isinstance(message, dict):
        return message
    return {
        attr: getattr(message, attr)
        for attr in _MESSAGE_ATTRS
        if hasattr(message, attr)
    }


def _stringify_content(content) -> str:
    """Coerce non-string content to a string for simple text parts."""
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _content_block_to_part(block) -> dict:
    """Convert a single multimodal content block to an OTel part.

    Handles dict blocks (OpenAI chat format) and SDK objects.
    """
    if isinstance(block, str):
        return {"type": "text", "content": block}

    if isinstance(block, dict):
        return _dict_block_to_part(block)

    return _object_block_to_part(block)


def _dict_block_to_part(block: dict) -> dict:
    """Map a dict-based content block (OpenAI format) to an OTel part.

    Spec mapping (openllmetry-semconv-review.md §1 / Part Types):
      OpenAI image_url  → OTel UriPart  {type:uri, modality:image, uri:...}
      OpenAI input_audio → OTel BlobPart {type:blob, modality:audio, ...}
    """
    btype = block.get("type", "text")
    if btype in ("text", "input_text", "output_text"):
        return {"type": "text", "content": block.get("text", "")}
    if btype == "image_url":
        url_info = block.get("image_url", {})
        url = (
            url_info.get("url", "")
            if isinstance(url_info, dict)
            else str(url_info)
        )
        return {"type": "uri", "modality": "image", "uri": url}
    if btype == "input_audio":
        audio_info = block.get("input_audio", {})
        return {
            "type": "blob",
            "modality": "audio",
            "content": audio_info.get("data", "") if isinstance(audio_info, dict) else str(audio_info),
        }
    return {"type": btype, "data": json.dumps(block)}


def _object_block_to_part(block) -> dict:
    """Map an SDK-object content block via getattr."""
    btype = getattr(block, "type", "text")
    if btype in ("text", "input_text", "output_text"):
        return {
            "type": "text",
            "content": getattr(block, "text", str(block)),
        }
    if btype == "image_url":
        url_obj = getattr(block, "image_url", None)
        url = getattr(url_obj, "url", str(url_obj)) if url_obj else ""
        return {"type": "uri", "modality": "image", "uri": url}
    if btype == "input_audio":
        audio_obj = getattr(block, "input_audio", None)
        data = getattr(audio_obj, "data", str(audio_obj)) if audio_obj else ""
        return {"type": "blob", "modality": "audio", "content": data}
    return {"type": btype, "data": str(block)}


def _content_to_parts(content) -> list:
    """Convert message content (str | list | scalar) into a list of OTel parts."""
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if isinstance(content, list):
        return [_content_block_to_part(block) for block in content]
    return [{"type": "text", "content": str(content)}]


def _tool_call_to_part(tool_call) -> dict:
    """Convert a single tool call to an OTel tool_call part."""
    tc = _normalize_tool_call(tool_call)
    part: dict = {"type": "tool_call"}
    if tc.get("id"):
        part["id"] = tc["id"]
    if tc.get("name"):
        part["name"] = tc["name"]
    if tc.get("arguments") is not None:
        part["arguments"] = _parse_arguments(tc["arguments"])
    return part


def _build_tool_response_part(call_id, content) -> dict:
    """Build a tool_call_response part from an id and optional content."""
    part: dict = {"type": "tool_call_response", "id": call_id}
    part["response"] = _stringify_content(content) if content is not None else ""
    return part


def _convert_chat_message(msg: dict):
    """Convert a role-based chat message to (role, parts) or None."""
    role = msg["role"]
    content = msg.get("content")
    tool_call_id = msg.get("tool_call_id")
    tool_calls = msg.get("tool_calls")

    if role == "tool" and tool_call_id:
        return role, [_build_tool_response_part(tool_call_id, content)]

    parts = []
    if tool_calls:
        if content is not None:
            if isinstance(content, list):
                parts.extend(_content_to_parts(content))
            else:
                text = _stringify_content(content)
                if text:
                    parts.append({"type": "text", "content": text})
        parts.extend(_tool_call_to_part(tc) for tc in tool_calls)
    elif content is not None:
        parts = _content_to_parts(content)

    return role, parts


def _convert_agents_sdk_message(msg: dict):
    """Convert an Agents SDK type-based message to (role, parts) or None."""
    msg_type = msg["type"]
    if msg_type == "function_call":
        part: dict = {
            "type": "tool_call",
            "id": msg.get("id", ""),
            "name": msg.get("name", ""),
        }
        if msg.get("arguments") is not None:
            part["arguments"] = _parse_arguments(msg["arguments"])
        return "assistant", [part]

    if msg_type == "function_call_output":
        part = _build_tool_response_part(
            msg.get("call_id"),
            msg.get("output"),
        )
        return "tool", [part]

    return None, []


def _extract_prompt_attributes(otel_span, input_data, trace_content: bool):
    """Set ``gen_ai.input.messages`` using the OTel parts-based schema.

    Handles both OpenAI chat format (role/content) and Agents SDK format
    (type/function_call/function_call_output).

    Only emitted when *trace_content* is True (opt-in content attribute).
    """
    if not input_data or not trace_content:
        return

    messages = []
    for message in input_data:
        msg = _msg_to_dict(message)

        if "role" in msg:
            role, parts = _convert_chat_message(msg)
        elif "type" in msg:
            role, parts = _convert_agents_sdk_message(msg)
        else:
            continue

        if role:
            messages.append({"role": role, "parts": parts})

    if messages:
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages)
        )


def _extract_response_attributes(otel_span, response, trace_content: bool):
    """
    Extract model settings, completions, and usage from a response object
    and set them as span attributes using the OTel parts-based schema.

    Returns a dict of model_settings for potential use by parent spans.
    """
    if not response:
        return {}

    model_settings = {}

    # Extract model settings
    if hasattr(response, "temperature") and response.temperature is not None:
        model_settings["temperature"] = response.temperature
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, response.temperature
        )

    if (
        hasattr(response, "max_output_tokens")
        and response.max_output_tokens is not None
    ):
        model_settings["max_tokens"] = response.max_output_tokens
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, response.max_output_tokens
        )

    if hasattr(response, "top_p") and response.top_p is not None:
        model_settings["top_p"] = response.top_p
        otel_span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_TOP_P, response.top_p)

    if hasattr(response, "model") and response.model:
        model_settings["model"] = response.model
        otel_span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, response.model)
        otel_span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.model)

    if hasattr(response, "id") and response.id:
        otel_span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)

    if (
        hasattr(response, "frequency_penalty")
        and response.frequency_penalty is not None
    ):
        model_settings["frequency_penalty"] = response.frequency_penalty
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
            response.frequency_penalty,
        )

    # Map finish reason (top-level)
    raw_finish_reason = getattr(response, "finish_reason", None)
    if raw_finish_reason is None:
        # Responses API uses status instead of finish_reason
        status = getattr(response, "status", None)
        if status == "completed":
            raw_finish_reason = "stop"
        elif status in ("failed", "cancelled", "incomplete"):
            raw_finish_reason = status
    mapped_finish_reason = _map_finish_reason(raw_finish_reason)

    # Set top-level finish_reasons attribute (even when trace_content=False)
    if mapped_finish_reason is not None:
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
            (mapped_finish_reason,),
        )

    # Extract completions from response.output as a JSON array (parts-based)
    # Only emit output messages when trace_content is enabled (opt-in content)
    if trace_content and hasattr(response, "output") and response.output:
        output_messages = []
        for output in response.output:
            if hasattr(output, "content") and output.content:
                # Text message with content array (ResponseOutputMessage)
                parts = []
                for content_item in output.content:
                    item_type = getattr(content_item, "type", None)
                    if item_type == "output_text" or (
                        hasattr(content_item, "text") and content_item.text
                    ):
                        parts.append({
                            "type": "text",
                            "content": content_item.text,
                        })
                    elif item_type == "refusal":
                        refusal_text = getattr(content_item, "refusal", "")
                        parts.append({
                            "type": "refusal",
                            "content": refusal_text,
                        })
                    elif item_type == "reasoning":
                        # Reasoning/thinking content
                        summary = getattr(content_item, "summary", None)
                        text = ""
                        if isinstance(summary, list):
                            text = " ".join(
                                getattr(s, "text", str(s))
                                for s in summary
                            )
                        elif summary:
                            text = str(summary)
                        parts.append({
                            "type": "reasoning",
                            "content": text,
                        })
                    elif item_type is not None:
                        # Unknown part type – preserve type and best-effort content
                        parts.append({
                            "type": item_type,
                            "data": str(content_item),
                        })

                msg = {
                    "role": getattr(output, "role", "assistant"),
                    "parts": parts,
                    "finish_reason": mapped_finish_reason if mapped_finish_reason else "",
                }
                output_messages.append(msg)

            elif hasattr(output, "name"):
                # Function/tool call (ResponseFunctionToolCall)
                tool_name = getattr(output, "name", "unknown_tool")
                tool_call_id = getattr(output, "call_id", "")
                part = {
                    "type": "tool_call",
                    "name": tool_name,
                    "id": tool_call_id,
                }
                raw_args = getattr(output, "arguments", None)
                part["arguments"] = _parse_arguments(raw_args)

                msg = {
                    "role": "assistant",
                    "parts": [part],
                    "finish_reason": _map_finish_reason("tool_calls"),
                }
                output_messages.append(msg)

            elif hasattr(output, "text"):
                # Direct text content
                parts = []
                if output.text:
                    parts.append({"type": "text", "content": output.text})
                msg = {
                    "role": getattr(output, "role", "assistant"),
                    "parts": parts,
                    "finish_reason": mapped_finish_reason if mapped_finish_reason else "",
                }
                output_messages.append(msg)

        if output_messages:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
            )

    # Extract usage data
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        if hasattr(usage, "input_tokens") and usage.input_tokens is not None:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens
            )
        elif hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
            )

        if hasattr(usage, "output_tokens") and usage.output_tokens is not None:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens
            )
        elif (
            hasattr(usage, "completion_tokens") and usage.completion_tokens is not None
        ):
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
            )

        if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
            otel_span.set_attribute(
                SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens
            )

    return model_settings


def _extract_tool_definitions(tools):
    """Extract tool/function specs into a JSON-serializable list.

    Handles both function-wrapped tools (tool.function.name) and
    direct function tools (tool.name).
    """
    if not tools:
        return []
    tool_defs = []
    for tool in tools:
        if hasattr(tool, "function"):
            function = tool.function
            func_def = {
                "name": getattr(function, "name", ""),
                "description": getattr(function, "description", ""),
            }
            if hasattr(function, "parameters"):
                func_def["parameters"] = function.parameters
            tool_def = {
                "type": getattr(tool, "type", "function"),
                "function": func_def,
            }
            tool_defs.append(tool_def)
        elif hasattr(tool, "name"):
            # Direct function format
            tool_def = {"name": tool.name}
            if hasattr(tool, "description"):
                tool_def["description"] = tool.description
            if hasattr(tool, "parameters"):
                tool_def["parameters"] = tool.parameters
            tool_defs.append(tool_def)
    return tool_defs


class OpenTelemetryTracingProcessor(TracingProcessor):
    """
    A tracing processor that creates OpenTelemetry spans for OpenAI Agents.

    This processor uses the OpenAI Agents SDK's native callback system to create
    proper OpenTelemetry spans with correct hierarchy and lifecycle management.
    """

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._root_spans: Dict[str, Any] = {}  # trace_id -> root span
        self._otel_spans: Dict[str, Any] = {}  # agents span -> otel span
        self._span_contexts: Dict[str, Any] = {}  # agents span -> context token
        self._last_model_settings: Dict[str, Any] = {}
        self._reverse_handoffs_dict: OrderedDict[str, str] = OrderedDict()

    @dont_throw
    def on_trace_start(self, trace):
        """Called when a new trace starts - create workflow span."""
        # Create a root "Agent Workflow" span for the entire trace
        workflow_span = self.tracer.start_span(
            "Agent Workflow",
            kind=SpanKind.INTERNAL,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
                GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
                SpanAttributes.TRACELOOP_WORKFLOW_NAME: "Agent Workflow",
            },
        )
        self._root_spans[trace.trace_id] = workflow_span

    @dont_throw
    def on_trace_end(self, trace):
        """Called when a trace ends - clean up workflow span."""
        if trace.trace_id in self._root_spans:
            workflow_span = self._root_spans[trace.trace_id]
            workflow_span.set_status(Status(StatusCode.OK))
            workflow_span.end()
            del self._root_spans[trace.trace_id]

    @dont_throw
    def on_span_start(self, span):
        """Called when a span starts - create appropriate OpenTelemetry span."""
        from agents import (
            AgentSpanData,
            HandoffSpanData,
            FunctionSpanData,
            GenerationSpanData,
        )

        if not span or not hasattr(span, "span_data"):
            return

        span_data = getattr(span, "span_data", None)
        if not span_data:
            return
        trace_id = getattr(span, "trace_id", None)
        parent_context = None
        if trace_id and trace_id in self._root_spans:
            workflow_span = self._root_spans[trace_id]
            parent_context = set_span_in_context(workflow_span)

        otel_span = None

        if isinstance(span_data, AgentSpanData):
            otel_span = self._start_agent_span(span_data, parent_context, trace_id)

        elif isinstance(span_data, HandoffSpanData):
            otel_span = self._start_handoff_span(span_data, parent_context, trace_id)

        elif isinstance(span_data, FunctionSpanData):
            agent_ctx = self._resolve_agent_parent(parent_context)
            otel_span = self._start_function_span(span_data, agent_ctx)

        elif (
            type(span_data).__name__ == "ResponseSpanData"
            or isinstance(span_data, GenerationSpanData)
        ):
            agent_ctx = self._resolve_agent_parent(parent_context)
            otel_span = self._start_generation_span(agent_ctx, span_data)

        elif (
            _has_realtime_spans
            and SpeechSpanData
            and isinstance(span_data, SpeechSpanData)
        ):
            agent_ctx = self._resolve_agent_parent(parent_context)
            otel_span = self._start_realtime_span(
                span_data, agent_ctx, "openai.realtime.speech", "speech",
            )

        elif (
            _has_realtime_spans
            and TranscriptionSpanData
            and isinstance(span_data, TranscriptionSpanData)
        ):
            agent_ctx = self._resolve_agent_parent(parent_context)
            otel_span = self._start_realtime_span(
                span_data, agent_ctx, "openai.realtime.transcription", "transcription",
            )

        elif (
            _has_realtime_spans
            and SpeechGroupSpanData
            and isinstance(span_data, SpeechGroupSpanData)
        ):
            agent_ctx = self._resolve_agent_parent(parent_context)
            otel_span = self._start_realtime_span(
                span_data, agent_ctx, "openai.realtime.speech_group", "speech_group",
            )

        if otel_span:
            self._otel_spans[span] = otel_span
            # Set as current span
            token = context.attach(set_span_in_context(otel_span))
            self._span_contexts[span] = token

    @dont_throw
    def on_span_end(self, span):
        """Called when a span ends - finish OpenTelemetry span."""
        from agents import FunctionSpanData, GenerationSpanData

        if not span or not hasattr(span, "span_data"):
            return

        if span in self._otel_spans:
            otel_span = self._otel_spans[span]
            span_data = getattr(span, "span_data", None)
            trace_content = should_send_prompts()
            if span_data and (
                type(span_data).__name__ == "ResponseSpanData"
                or isinstance(span_data, GenerationSpanData)
            ):
                self._end_generation_span(otel_span, span_data, trace_content)

            elif span_data and isinstance(span_data, FunctionSpanData):
                self._end_function_span(otel_span, span_data, trace_content)

            elif trace_content and span_data and _has_realtime_spans:
                if SpeechSpanData and isinstance(span_data, SpeechSpanData):
                    self._set_realtime_io_attributes(otel_span, span_data, has_output=True)
                elif TranscriptionSpanData and isinstance(span_data, TranscriptionSpanData):
                    self._set_realtime_io_attributes(otel_span, span_data, has_output=True)
                elif SpeechGroupSpanData and isinstance(span_data, SpeechGroupSpanData):
                    self._set_realtime_io_attributes(otel_span, span_data, has_output=False)

            if hasattr(span, "error") and span.error:
                otel_span.set_status(Status(StatusCode.ERROR, str(span.error)))
            else:
                otel_span.set_status(Status(StatusCode.OK))

            otel_span.end()
            del self._otel_spans[span]
            if span in self._span_contexts:
                context.detach(self._span_contexts[span])
                del self._span_contexts[span]

    # ------------------------------------------------------------------
    # on_span_start handlers (extracted from the former if-elif chain)
    # ------------------------------------------------------------------

    def _resolve_agent_parent(self, fallback_context):
        """Resolve parent context, preferring the current agent span."""
        current = self._find_current_agent_span()
        if current:
            return set_span_in_context(current)
        return fallback_context

    def _start_agent_span(self, span_data, parent_context, trace_id):
        """Create an OTel span for an AgentSpanData."""
        agent_name = getattr(span_data, "name", None) or "unknown_agent"

        if set_agent_name is not None:
            set_agent_name(agent_name)

        handoff_parent = None
        if trace_id:
            handoff_key = f"{agent_name}:{trace_id}"
            if parent_agent_name := self._reverse_handoffs_dict.pop(
                handoff_key, None
            ):
                handoff_parent = parent_agent_name

        attributes = {
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_AGENT_NAME: agent_name,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
        }

        if handoff_parent:
            attributes["gen_ai.agent.handoff_parent"] = handoff_parent

        if hasattr(span_data, "handoffs") and span_data.handoffs:
            for i, handoff_agent in enumerate(span_data.handoffs):
                handoff_info = {
                    "name": getattr(handoff_agent, "name", "unknown"),
                    "instructions": getattr(
                        handoff_agent, "instructions", "No instructions"
                    ),
                }
                attributes[f"openai.agent.handoff{i}"] = json.dumps(handoff_info)

        return self.tracer.start_span(
            f"{agent_name}.agent",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes=attributes,
        )

    def _start_handoff_span(self, span_data, parent_context, trace_id):
        """Create an OTel span for a HandoffSpanData."""
        from_agent = getattr(span_data, "from_agent", None) or "unknown"
        to_agent = getattr(span_data, "to_agent", None) or "unknown"

        if to_agent and to_agent != "unknown" and trace_id:
            handoff_key = f"{to_agent}:{trace_id}"
            self._reverse_handoffs_dict[handoff_key] = from_agent

            if len(self._reverse_handoffs_dict) > 1000:
                self._reverse_handoffs_dict.popitem(last=False)

        from_agent_span = self._find_agent_span(from_agent)
        if from_agent_span:
            parent_context = set_span_in_context(from_agent_span)

        handoff_attributes = {
            SpanAttributes.TRACELOOP_SPAN_KIND: "handoff",
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
        }

        if from_agent and from_agent != "unknown":
            handoff_attributes[GEN_AI_HANDOFF_FROM_AGENT] = from_agent
            handoff_attributes[GenAIAttributes.GEN_AI_AGENT_NAME] = from_agent
        if to_agent and to_agent != "unknown":
            handoff_attributes[GEN_AI_HANDOFF_TO_AGENT] = to_agent

        return self.tracer.start_span(
            f"{from_agent} → {to_agent}.handoff",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes=handoff_attributes,
        )

    def _start_function_span(self, span_data, parent_context):
        """Create an OTel span for a FunctionSpanData."""
        tool_name = getattr(span_data, "name", None) or "unknown_tool"

        tool_attributes = {
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
            GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
            GenAIAttributes.GEN_AI_TOOL_TYPE: "function",
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
        }

        if hasattr(span_data, "description") and span_data.description:
            # Only use description if it's not a generic class description
            desc = span_data.description
            if desc and not desc.startswith("Represents a Function Span"):
                tool_attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = desc

        return self.tracer.start_span(
            f"{tool_name}.tool",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes=tool_attributes,
        )

    def _start_generation_span(self, parent_context, span_data=None):
        """Create an OTel span for a GenerationSpanData or ResponseSpanData."""
        attributes = {
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
        }
        model = getattr(span_data, "model", None) if span_data else None
        if model:
            attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = model
        return self.tracer.start_span(
            "openai.response",
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes=attributes,
            start_time=time.time_ns(),
        )

    def _start_realtime_span(self, span_data, parent_context, span_name, operation):
        """Create an OTel span for a realtime span (Speech/Transcription/SpeechGroup).

        NOTE: "speech", "transcription", "speech_group" are OpenAI
        Realtime API-specific operations with no well-known OTel
        equivalents.  Kept as custom operation names intentionally.
        """
        attributes = {
            GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
        }

        model = getattr(span_data, "model", None)
        if model:
            attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = model

        return self.tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes=attributes,
            start_time=time.time_ns(),
        )

    # ------------------------------------------------------------------
    # on_span_end helpers (extracted from the former if-elif chain)
    # ------------------------------------------------------------------

    def _end_generation_span(self, otel_span, span_data, trace_content):
        """Handle on_span_end logic for generation/response spans."""
        input_data = getattr(span_data, "input", [])
        _extract_prompt_attributes(otel_span, input_data, trace_content)

        response = getattr(span_data, "response", None)
        if (
            trace_content
            and response
            and hasattr(response, "tools")
            and response.tools
        ):
            tool_defs = _extract_tool_definitions(response.tools)
            if tool_defs:
                otel_span.set_attribute(
                    GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs)
                )

        if response:
            model_settings = _extract_response_attributes(otel_span, response, trace_content)
            self._last_model_settings = model_settings

    def _end_function_span(self, otel_span, span_data, trace_content):
        """Handle on_span_end logic for function/tool spans.

        Sets ``gen_ai.tool.call.arguments`` and ``gen_ai.tool.call.result``
        from ``FunctionSpanData.input`` / ``.output``.  Both are content
        attributes and are only emitted when *trace_content* is True.
        """
        if not trace_content:
            return

        tool_input = getattr(span_data, "input", None)
        if tool_input is not None:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
                tool_input if isinstance(tool_input, str) else json.dumps(tool_input),
            )

        tool_output = getattr(span_data, "output", None)
        if tool_output is not None:
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
                tool_output if isinstance(tool_output, str) else json.dumps(tool_output),
            )

    def _set_realtime_io_attributes(self, otel_span, span_data, has_output=True):
        """Set input/output message attributes for realtime spans."""
        input_val = getattr(span_data, "input", None)
        if input_val and not isinstance(input_val, (bytes, bytearray)):
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": str(input_val)}]}]),
            )

        if not has_output:
            return

        output_val = getattr(span_data, "output", None)
        if output_val and not isinstance(output_val, (bytes, bytearray)):
            out_msg = {
                "role": "assistant",
                "parts": [{"type": "text", "content": str(output_val)}],
                "finish_reason": "",
            }
            otel_span.set_attribute(
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([out_msg]),
            )

    # ------------------------------------------------------------------
    # Span lookup helpers
    # ------------------------------------------------------------------

    def _find_agent_span(self, agent_name: str):
        """Find the OpenTelemetry span for a given agent."""
        for agents_span, otel_span in self._otel_spans.items():
            span_data = getattr(agents_span, "span_data", None)
            if span_data and getattr(span_data, "name", None) == agent_name:
                return otel_span
        return None

    def _find_current_agent_span(self):
        """Find the currently active agent span."""
        # This would need more sophisticated logic to find the current agent context
        # For now, return the current span if it's an agent span
        current = get_current_span()
        try:
            if (
                current
                and hasattr(current, "name")
                and current.name
                and current.name.endswith(".agent")
            ):
                return current
        except (AttributeError, TypeError):
            pass
        return None

    def force_flush(self):
        """Force flush any pending spans."""
        pass

    def shutdown(self):
        """Shutdown the processor and clean up resources."""
        # End any remaining spans
        for otel_span in self._otel_spans.values():
            if otel_span.is_recording():
                otel_span.end()

        # Clean up tracking dictionaries
        self._otel_spans.clear()
        self._span_contexts.clear()
        self._root_spans.clear()
        self._reverse_handoffs_dict.clear()
