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


def _extract_prompt_attributes(otel_span, input_data, trace_content: bool):
    """
    Extract prompt/input data from messages and set as a single GEN_AI_INPUT_MESSAGES
    JSON array attribute.

    Handles both OpenAI chat format (role/content) and Agents SDK format
    (type/function_call/function_call_output).
    """
    if not input_data:
        return

    messages = []

    for message in input_data:
        # Convert message to dict for unified handling
        if isinstance(message, dict):
            msg = message
        else:
            msg = {}
            for attr in [
                "role", "content", "tool_call_id", "tool_calls",
                "type", "name", "arguments", "call_id", "output",
            ]:
                if hasattr(message, attr):
                    msg[attr] = getattr(message, attr)

        role = None
        content = None
        tool_call_id = None
        tool_calls = None

        if "role" in msg:
            role = msg["role"]
            content = msg.get("content")
            tool_call_id = msg.get("tool_call_id")
            tool_calls = msg.get("tool_calls")
        elif "type" in msg:
            msg_type = msg["type"]
            if msg_type == "function_call":
                role = "assistant"
                tool_calls = [
                    {
                        "id": msg.get("id", ""),
                        "name": msg.get("name", ""),
                    } | (
                        {"arguments": msg.get("arguments", "")}
                        if trace_content else {}
                    )
                ]
            elif msg_type == "function_call_output" and trace_content:
                role = "tool"
                content = msg.get("output")
                tool_call_id = msg.get("call_id")

        if not role:
            continue

        msg_obj: dict = {"role": role}

        if content is not None and trace_content:
            if not isinstance(content, str):
                content = json.dumps(content)
            msg_obj["content"] = content

        if tool_call_id:
            msg_obj["tool_call_id"] = tool_call_id

        if tool_calls:
            processed_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
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
                    if hasattr(tool_call, "arguments"):
                        tc_dict["arguments"] = tool_call.arguments
                    tool_call = tc_dict

                if "function" in tool_call:
                    function = tool_call["function"]
                    tool_call = {
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                    }

                call_obj: dict = {}
                if tool_call.get("id"):
                    call_obj["id"] = tool_call["id"]
                if tool_call.get("name"):
                    call_obj["name"] = tool_call["name"]
                if tool_call.get("arguments") and trace_content:
                    args = tool_call["arguments"]
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    call_obj["arguments"] = args
                if call_obj:
                    processed_calls.append(call_obj)

            if processed_calls:
                msg_obj["tool_calls"] = processed_calls

        messages.append(msg_obj)

    if messages:
        otel_span.set_attribute(
            GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages)
        )


def _extract_response_attributes(otel_span, response, trace_content: bool):
    """
    Extract model settings, completions, and usage from a response object
    and set them as span attributes.

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

    if (
        hasattr(response, "frequency_penalty")
        and response.frequency_penalty is not None
    ):
        model_settings["frequency_penalty"] = response.frequency_penalty

    # Extract completions from response.output as a JSON array
    if hasattr(response, "output") and response.output:
        output_messages = []
        for output in response.output:
            if hasattr(output, "content") and output.content:
                # Text message with content array (ResponseOutputMessage)
                content_text = ""
                for content_item in output.content:
                    if hasattr(content_item, "text"):
                        content_text += content_item.text

                msg: dict = {"role": getattr(output, "role", "assistant")}
                if content_text and trace_content:
                    msg["content"] = content_text
                if hasattr(response, "finish_reason") and response.finish_reason:
                    msg["finish_reason"] = response.finish_reason
                output_messages.append(msg)

            elif hasattr(output, "name"):
                # Function/tool call (ResponseFunctionToolCall)
                tool_name = getattr(output, "name", "unknown_tool")
                tool_call_id = getattr(output, "call_id", "")
                call_obj: dict = {"name": tool_name, "id": tool_call_id}
                if trace_content:
                    call_obj["arguments"] = getattr(output, "arguments", "{}")
                msg = {
                    "role": "assistant",
                    "finish_reason": "tool_calls",
                    "tool_calls": [call_obj],
                }
                output_messages.append(msg)

            elif hasattr(output, "text"):
                # Direct text content
                msg = {"role": getattr(output, "role", "assistant")}
                if trace_content:
                    msg["content"] = output.text
                if hasattr(response, "finish_reason") and response.finish_reason:
                    msg["finish_reason"] = response.finish_reason
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
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
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
            agent_name = getattr(span_data, "name", None) or "unknown_agent"

            if set_agent_name is not None:
                set_agent_name(agent_name)

            handoff_parent = None
            trace_id = getattr(span, "trace_id", None)
            if trace_id:
                handoff_key = f"{agent_name}:{trace_id}"
                if parent_agent_name := self._reverse_handoffs_dict.pop(
                    handoff_key, None
                ):
                    handoff_parent = parent_agent_name

            attributes = {
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                GenAIAttributes.GEN_AI_AGENT_NAME: agent_name,
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
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

            otel_span = self.tracer.start_span(
                f"{agent_name}.agent",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=attributes,
            )

        elif isinstance(span_data, HandoffSpanData):
            from_agent = getattr(span_data, "from_agent", None)
            to_agent = getattr(span_data, "to_agent", None)

            from_agent = from_agent or "unknown"

            to_agent = to_agent or "unknown"

            trace_id = getattr(span, "trace_id", None)
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
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
            }

            if from_agent and from_agent != "unknown":
                handoff_attributes[GEN_AI_HANDOFF_FROM_AGENT] = from_agent
                handoff_attributes[GenAIAttributes.GEN_AI_AGENT_NAME] = from_agent
            if to_agent and to_agent != "unknown":
                handoff_attributes[GEN_AI_HANDOFF_TO_AGENT] = to_agent

            otel_span = self.tracer.start_span(
                f"{from_agent} → {to_agent}.handoff",
                kind=SpanKind.INTERNAL,
                context=parent_context,
                attributes=handoff_attributes,
            )

        elif isinstance(span_data, FunctionSpanData):
            tool_name = getattr(span_data, "name", None) or "unknown_tool"

            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            tool_attributes = {
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
                GenAIAttributes.GEN_AI_TOOL_TYPE: "function",
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
            }

            if hasattr(span_data, "description") and span_data.description:
                # Only use description if it's not a generic class description
                desc = span_data.description
                if desc and not desc.startswith("Represents a Function Span"):
                    tool_attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = desc

            otel_span = self.tracer.start_span(
                f"{tool_name}.tool",
                kind=SpanKind.INTERNAL,
                context=parent_context,
                attributes=tool_attributes,
            )

        elif type(span_data).__name__ == "ResponseSpanData":
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            response_attributes = {
                GenAIAttributes.GEN_AI_OPERATION_NAME: "response",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            }

            otel_span = self.tracer.start_span(
                "openai.response",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=response_attributes,
                start_time=time.time_ns(),
            )

        elif isinstance(span_data, GenerationSpanData):
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            response_attributes = {
                GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            }

            otel_span = self.tracer.start_span(
                "openai.response",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=response_attributes,
                start_time=time.time_ns(),
            )

        elif (
            _has_realtime_spans
            and SpeechSpanData
            and isinstance(span_data, SpeechSpanData)
        ):
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            speech_attributes = {
                GenAIAttributes.GEN_AI_OPERATION_NAME: "speech",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            }

            model = getattr(span_data, "model", None)
            if model:
                speech_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = model

            otel_span = self.tracer.start_span(
                "openai.realtime.speech",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=speech_attributes,
                start_time=time.time_ns(),
            )

        elif (
            _has_realtime_spans
            and TranscriptionSpanData
            and isinstance(span_data, TranscriptionSpanData)
        ):
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            transcription_attributes = {
                GenAIAttributes.GEN_AI_OPERATION_NAME: "transcription",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            }

            model = getattr(span_data, "model", None)
            if model:
                transcription_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = model

            otel_span = self.tracer.start_span(
                "openai.realtime.transcription",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=transcription_attributes,
                start_time=time.time_ns(),
            )

        elif (
            _has_realtime_spans
            and SpeechGroupSpanData
            and isinstance(span_data, SpeechGroupSpanData)
        ):
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            speech_group_attributes = {
                GenAIAttributes.GEN_AI_OPERATION_NAME: "speech_group",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            }

            otel_span = self.tracer.start_span(
                "openai.realtime.speech_group",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=speech_group_attributes,
                start_time=time.time_ns(),
            )

        if otel_span:
            self._otel_spans[span] = otel_span
            # Set as current span
            token = context.attach(set_span_in_context(otel_span))
            self._span_contexts[span] = token

    @dont_throw
    def on_span_end(self, span):
        """Called when a span ends - finish OpenTelemetry span."""
        from agents import GenerationSpanData

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
                # Extract prompt data from input
                input_data = getattr(span_data, "input", [])
                _extract_prompt_attributes(otel_span, input_data, trace_content)

                # Add function/tool specifications to the request using OpenAI semantic conventions
                response = getattr(span_data, "response", None)
                if (
                    response
                    and hasattr(response, "tools")
                    and response.tools
                ):
                    # Extract tool specifications as a JSON array
                    tool_defs = []
                    for tool in response.tools:
                        if hasattr(tool, "function"):
                            function = tool.function
                            tool_def = {
                                "name": getattr(function, "name", ""),
                                "description": getattr(function, "description", ""),
                            }
                            if hasattr(function, "parameters"):
                                tool_def["parameters"] = function.parameters
                            tool_defs.append(tool_def)
                        elif hasattr(tool, "name"):
                            # Direct function format
                            tool_def = {"name": tool.name}
                            if hasattr(tool, "description"):
                                tool_def["description"] = tool.description
                            if hasattr(tool, "parameters"):
                                tool_def["parameters"] = tool.parameters
                            tool_defs.append(tool_def)
                    if tool_defs:
                        otel_span.set_attribute(
                            GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs)
                        )

                if response:
                    model_settings = _extract_response_attributes(otel_span, response, trace_content)
                    self._last_model_settings = model_settings

            # Legacy fallback for other span types
            elif span_data:
                input_data = getattr(span_data, "input", [])
                _extract_prompt_attributes(otel_span, input_data, trace_content)

                response = getattr(span_data, "response", None)
                if response:
                    model_settings = _extract_response_attributes(otel_span, response, trace_content)
                    self._last_model_settings = model_settings

            elif (
                _has_realtime_spans
                and SpeechSpanData
                and isinstance(span_data, SpeechSpanData)
                and trace_content
            ):
                input_text = getattr(span_data, "input", None)
                if input_text:
                    otel_span.set_attribute(
                        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                        json.dumps([{"role": "user", "content": input_text}]),
                    )

                output_audio = getattr(span_data, "output", None)
                if output_audio and not isinstance(output_audio, (bytes, bytearray)):
                    otel_span.set_attribute(
                        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                        json.dumps([{"role": "assistant", "content": str(output_audio)}]),
                    )

            elif (
                _has_realtime_spans
                and TranscriptionSpanData
                and isinstance(span_data, TranscriptionSpanData)
                and trace_content
            ):
                input_audio = getattr(span_data, "input", None)
                if input_audio and not isinstance(input_audio, (bytes, bytearray)):
                    otel_span.set_attribute(
                        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                        json.dumps([{"role": "user", "content": str(input_audio)}]),
                    )

                output_text = getattr(span_data, "output", None)
                if output_text:
                    otel_span.set_attribute(
                        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                        json.dumps([{"role": "assistant", "content": output_text}]),
                    )

            elif (
                _has_realtime_spans
                and SpeechGroupSpanData
                and isinstance(span_data, SpeechGroupSpanData)
                and trace_content
            ):
                input_text = getattr(span_data, "input", None)
                if input_text:
                    otel_span.set_attribute(
                        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                        json.dumps([{"role": "user", "content": input_text}]),
                    )

            elif span_data and type(span_data).__name__ == "AgentSpanData":
                # For agent spans, add the model settings we stored from the response span
                if hasattr(self, "_last_model_settings") and self._last_model_settings:
                    for key, value in self._last_model_settings.items():
                        if key == "temperature":
                            otel_span.set_attribute(
                                GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, value
                            )
                        elif key == "max_tokens":
                            otel_span.set_attribute(
                                GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, value
                            )
                        elif key == "top_p":
                            otel_span.set_attribute(
                                GenAIAttributes.GEN_AI_REQUEST_TOP_P, value
                            )
                        elif key == "model":
                            otel_span.set_attribute(
                                GenAIAttributes.GEN_AI_REQUEST_MODEL, value
                            )
                        elif key == "frequency_penalty":
                            otel_span.set_attribute(
                                "openai.agent.model.frequency_penalty", value
                            )
                        # Note: prompt_attributes, completion_attributes, and usage tokens are now
                        # on response spans only

            if hasattr(span, "error") and span.error:
                otel_span.set_status(Status(StatusCode.ERROR, str(span.error)))
            else:
                otel_span.set_status(Status(StatusCode.OK))

            otel_span.end()
            del self._otel_spans[span]
            if span in self._span_contexts:
                context.detach(self._span_contexts[span])
                del self._span_contexts[span]

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
