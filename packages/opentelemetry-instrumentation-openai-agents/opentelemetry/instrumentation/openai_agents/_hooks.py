"""Hook-based instrumentation for OpenAI Agents using the SDK's native callback system."""

from typing import Dict, Any
import json
import time
from collections import OrderedDict
from opentelemetry.trace import Tracer, Status, StatusCode, SpanKind, get_current_span, set_span_in_context
from opentelemetry import context
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_COMPLETION
from agents.tracing.processors import TracingProcessor
from .utils import dont_throw


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
                "gen_ai.system": "openai_agents",
                "gen_ai.workflow.name": "Agent Workflow"
            }
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
        from agents import AgentSpanData, HandoffSpanData, FunctionSpanData, GenerationSpanData

        if not span or not hasattr(span, 'span_data'):
            return

        span_data = getattr(span, 'span_data', None)
        if not span_data:
            return
        trace_id = getattr(span, 'trace_id', None)
        parent_context = None
        if trace_id and trace_id in self._root_spans:
            workflow_span = self._root_spans[trace_id]
            parent_context = set_span_in_context(workflow_span)

        otel_span = None

        if isinstance(span_data, AgentSpanData):
            agent_name = getattr(span_data, 'name', None) or "unknown_agent"

            handoff_parent = None
            trace_id = getattr(span, 'trace_id', None)
            if trace_id:
                handoff_key = f"{agent_name}:{trace_id}"
                if parent_agent_name := self._reverse_handoffs_dict.pop(handoff_key, None):
                    handoff_parent = parent_agent_name

            attributes = {
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                "gen_ai.agent.name": agent_name,
                "gen_ai.system": "openai_agents"
            }

            if handoff_parent:
                attributes["gen_ai.agent.handoff_parent"] = handoff_parent

            if hasattr(span_data, 'handoffs') and span_data.handoffs:
                for i, handoff_agent in enumerate(span_data.handoffs):
                    handoff_info = {
                        "name": getattr(handoff_agent, 'name', 'unknown'),
                        "instructions": getattr(handoff_agent, 'instructions', 'No instructions')
                    }
                    attributes[f"openai.agent.handoff{i}"] = json.dumps(handoff_info)

            otel_span = self.tracer.start_span(
                f"{agent_name}.agent",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=attributes
            )

        elif isinstance(span_data, HandoffSpanData):
            from_agent = getattr(span_data, 'from_agent', None)
            to_agent = getattr(span_data, 'to_agent', None)

            from_agent = from_agent or 'unknown'

            to_agent = to_agent or 'unknown'

            trace_id = getattr(span, 'trace_id', None)
            if to_agent and to_agent != 'unknown' and trace_id:
                handoff_key = f"{to_agent}:{trace_id}"
                self._reverse_handoffs_dict[handoff_key] = from_agent

                if len(self._reverse_handoffs_dict) > 1000:
                    self._reverse_handoffs_dict.popitem(last=False)

            from_agent_span = self._find_agent_span(from_agent)
            if from_agent_span:
                parent_context = set_span_in_context(from_agent_span)

            handoff_attributes = {
                SpanAttributes.TRACELOOP_SPAN_KIND: "handoff",
                "gen_ai.system": "openai_agents"
            }

            if from_agent and from_agent != 'unknown':
                handoff_attributes["gen_ai.handoff.from_agent"] = from_agent
            if to_agent and to_agent != 'unknown':
                handoff_attributes["gen_ai.handoff.to_agent"] = to_agent

            otel_span = self.tracer.start_span(
                f"{from_agent} → {to_agent}.handoff",
                kind=SpanKind.INTERNAL,
                context=parent_context,
                attributes=handoff_attributes
            )

        elif isinstance(span_data, FunctionSpanData):
            tool_name = getattr(span_data, 'name', None) or "unknown_tool"

            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            tool_attributes = {
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                "gen_ai.tool.name": tool_name,
                "gen_ai.system": "openai_agents",
                f"{GEN_AI_COMPLETION}.tool.name": tool_name,
                f"{GEN_AI_COMPLETION}.tool.type": "FunctionTool",
                f"{GEN_AI_COMPLETION}.tool.strict_json_schema": True
            }

            if hasattr(span_data, 'description') and span_data.description:
                # Only use description if it's not a generic class description
                desc = span_data.description
                if desc and not desc.startswith("Represents a Function Span"):
                    tool_attributes[f"{GEN_AI_COMPLETION}.tool.description"] = desc

            otel_span = self.tracer.start_span(
                f"{tool_name}.tool",
                kind=SpanKind.INTERNAL,
                context=parent_context,
                attributes=tool_attributes
            )

        elif type(span_data).__name__ == 'ResponseSpanData':
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            response_attributes = {
                SpanAttributes.LLM_REQUEST_TYPE: "response",
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "response"
            }

            otel_span = self.tracer.start_span(
                "openai.response",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=response_attributes,
                start_time=time.time_ns()
            )

        elif isinstance(span_data, GenerationSpanData):
            current_agent_span = self._find_current_agent_span()
            if current_agent_span:
                parent_context = set_span_in_context(current_agent_span)

            response_attributes = {
                SpanAttributes.LLM_REQUEST_TYPE: "chat",
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat"
            }

            otel_span = self.tracer.start_span(
                "openai.response",
                kind=SpanKind.CLIENT,
                context=parent_context,
                attributes=response_attributes,
                start_time=time.time_ns()
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

        if not span or not hasattr(span, 'span_data'):
            return

        if span in self._otel_spans:
            otel_span = self._otel_spans[span]
            span_data = getattr(span, 'span_data', None)
            if span_data and (
                type(span_data).__name__ == 'ResponseSpanData' or isinstance(
                    span_data,
                    GenerationSpanData)):
                # Extract prompt data from input and add to response span using OpenAI semantic conventions
                input_data = getattr(span_data, 'input', [])
                if input_data:
                    for i, message in enumerate(input_data):
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            otel_span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.role)
                            otel_span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{i}.content", message.content)
                        elif isinstance(message, dict):
                            if 'role' in message and 'content' in message:
                                otel_span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message['role'])
                                otel_span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{i}.content", message['content'])

                # Add function/tool specifications to the request using OpenAI semantic conventions
                response = getattr(span_data, 'response', None)
                if response and hasattr(response, 'tools') and response.tools:
                    # Extract tool specifications
                    for i, tool in enumerate(response.tools):
                        if hasattr(tool, 'function'):
                            function = tool.function
                            otel_span.set_attribute(
                                f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.name", getattr(
                                    function, 'name', ''))
                            otel_span.set_attribute(
                                f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.description", getattr(
                                    function, 'description', ''))
                            if hasattr(function, 'parameters'):
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.parameters", json.dumps(
                                        function.parameters))
                        elif hasattr(tool, 'name'):
                            # Direct function format
                            otel_span.set_attribute(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.name", tool.name)
                            if hasattr(tool, 'description'):
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.description", tool.description)
                            if hasattr(tool, 'parameters'):
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.parameters", json.dumps(
                                        tool.parameters))

                if response:
                    # Extract model settings from the response
                    model_settings = {}

                    if hasattr(response, 'temperature') and response.temperature is not None:
                        model_settings['temperature'] = response.temperature
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, response.temperature)

                    if hasattr(response, 'max_output_tokens') and response.max_output_tokens is not None:
                        model_settings['max_tokens'] = response.max_output_tokens
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, response.max_output_tokens)

                    if hasattr(response, 'top_p') and response.top_p is not None:
                        model_settings['top_p'] = response.top_p
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, response.top_p)

                    if hasattr(response, 'model') and response.model:
                        model_settings['model'] = response.model
                        otel_span.set_attribute("gen_ai.request.model", response.model)

                    # Extract completions and add directly to response span using OpenAI semantic conventions
                    if hasattr(response, 'output') and response.output:
                        for i, output in enumerate(response.output):
                            # Handle different output types
                            if hasattr(output, 'content') and output.content:
                                # Text message with content array (ResponseOutputMessage)
                                content_text = ""
                                for content_item in output.content:
                                    if hasattr(content_item, 'text'):
                                        content_text += content_item.text

                                if content_text:
                                    otel_span.set_attribute(
                                        f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", content_text)
                                    otel_span.set_attribute(
                                        f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", getattr(
                                            output, 'role', 'assistant'))

                            elif hasattr(output, 'name'):
                                # Function/tool call (ResponseFunctionToolCall) - use OpenAI tool call format
                                tool_name = getattr(output, 'name', 'unknown_tool')
                                arguments = getattr(output, 'arguments', '{}')
                                tool_call_id = getattr(output, 'call_id', f"call_{i}")

                                # Set completion with tool call following OpenAI format
                                otel_span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", "assistant")
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.finish_reason", "tool_calls")
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.name", tool_name)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.arguments", arguments)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.id", tool_call_id)

                            elif hasattr(output, 'text'):
                                # Direct text content
                                otel_span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", output.text)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", getattr(
                                        output, 'role', 'assistant'))

                            # Add finish reason if available (for non-tool-call cases)
                            if hasattr(response, 'finish_reason') and not hasattr(output, 'name'):
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.finish_reason", response.finish_reason)

                    # Extract usage data and add directly to response span
                    if hasattr(response, 'usage') and response.usage:
                        usage = response.usage
                        # Try both naming conventions: input_tokens/output_tokens and prompt_tokens/completion_tokens
                        if hasattr(usage, 'input_tokens') and usage.input_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.input_tokens)
                        elif hasattr(usage, 'prompt_tokens') and usage.prompt_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)

                        if hasattr(usage, 'output_tokens') and usage.output_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.output_tokens)
                        elif hasattr(usage, 'completion_tokens') and usage.completion_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)

                        if hasattr(usage, 'total_tokens') and usage.total_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)

                    # Store model settings to add to the agent span (but NOT prompts/completions)
                    self._last_model_settings = model_settings

            # Legacy fallback for other span types
            elif span_data:
                # Extract prompt data from input and add to response span (legacy support)
                input_data = getattr(span_data, 'input', [])
                if input_data:
                    for i, message in enumerate(input_data):
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            otel_span.set_attribute(f"gen_ai.prompt.{i}.role", message.role)
                            otel_span.set_attribute(f"gen_ai.prompt.{i}.content", message.content)
                        elif isinstance(message, dict):
                            if 'role' in message and 'content' in message:
                                otel_span.set_attribute(f"gen_ai.prompt.{i}.role", message['role'])
                                otel_span.set_attribute(f"gen_ai.prompt.{i}.content", message['content'])

                response = getattr(span_data, 'response', None)
                if response:

                    # Extract model settings from the response
                    model_settings = {}

                    if hasattr(response, 'temperature') and response.temperature is not None:
                        model_settings['temperature'] = response.temperature
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, response.temperature)

                    if hasattr(response, 'max_output_tokens') and response.max_output_tokens is not None:
                        model_settings['max_tokens'] = response.max_output_tokens
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, response.max_output_tokens)

                    if hasattr(response, 'top_p') and response.top_p is not None:
                        model_settings['top_p'] = response.top_p
                        otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, response.top_p)

                    if hasattr(response, 'model') and response.model:
                        model_settings['model'] = response.model
                        otel_span.set_attribute("gen_ai.request.model", response.model)

                    # Extract completions and add directly to response span using OpenAI semantic conventions
                    if hasattr(response, 'output') and response.output:
                        for i, output in enumerate(response.output):
                            # Handle different output types
                            if hasattr(output, 'content') and output.content:
                                # Text message with content array (ResponseOutputMessage)
                                content_text = ""
                                for content_item in output.content:
                                    if hasattr(content_item, 'text'):
                                        content_text += content_item.text

                                if content_text:
                                    otel_span.set_attribute(
                                        f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", content_text)
                                    otel_span.set_attribute(
                                        f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", getattr(
                                            output, 'role', 'assistant'))

                            elif hasattr(output, 'name'):
                                # Function/tool call (ResponseFunctionToolCall) - use OpenAI tool call format
                                tool_name = getattr(output, 'name', 'unknown_tool')
                                arguments = getattr(output, 'arguments', '{}')
                                tool_call_id = getattr(output, 'call_id', f"call_{i}")

                                # Set completion with tool call following OpenAI format
                                otel_span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", "assistant")
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.finish_reason", "tool_calls")
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.name", tool_name)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.arguments", arguments)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls.0.id", tool_call_id)

                            elif hasattr(output, 'text'):
                                # Direct text content
                                otel_span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", output.text)
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", getattr(
                                        output, 'role', 'assistant'))

                            # Add finish reason if available (for non-tool-call cases)
                            if hasattr(response, 'finish_reason') and not hasattr(output, 'name'):
                                otel_span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{i}.finish_reason", response.finish_reason)

                    # Extract usage data and add directly to response span
                    if hasattr(response, 'usage') and response.usage:
                        usage = response.usage
                        # Try both naming conventions: input_tokens/output_tokens and prompt_tokens/completion_tokens
                        if hasattr(usage, 'input_tokens') and usage.input_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.input_tokens)
                        elif hasattr(usage, 'prompt_tokens') and usage.prompt_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)

                        if hasattr(usage, 'output_tokens') and usage.output_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.output_tokens)
                        elif hasattr(usage, 'completion_tokens') and usage.completion_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)

                        if hasattr(usage, 'total_tokens') and usage.total_tokens is not None:
                            otel_span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)

                    # Check for frequency_penalty
                    if hasattr(response, 'frequency_penalty') and response.frequency_penalty is not None:
                        model_settings['frequency_penalty'] = response.frequency_penalty

                    # Store model settings to add to the agent span (but NOT prompts/completions)
                    self._last_model_settings = model_settings

            elif span_data and type(span_data).__name__ == 'AgentSpanData':
                # For agent spans, add the model settings we stored from the response span
                if hasattr(self, '_last_model_settings') and self._last_model_settings:
                    for key, value in self._last_model_settings.items():
                        if key == 'temperature':
                            otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, value)
                        elif key == 'max_tokens':
                            otel_span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, value)
                        elif key == 'top_p':
                            otel_span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, value)
                        elif key == 'model':
                            otel_span.set_attribute("gen_ai.request.model", value)
                        elif key == 'frequency_penalty':
                            otel_span.set_attribute("openai.agent.model.frequency_penalty", value)
                        # Note: prompt_attributes, completion_attributes, and usage tokens are now
                        # on response spans only

            if hasattr(span, 'error') and span.error:
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
            span_data = getattr(agents_span, 'span_data', None)
            if span_data and getattr(span_data, 'name', None) == agent_name:
                return otel_span
        return None

    def _find_current_agent_span(self):
        """Find the currently active agent span."""
        # This would need more sophisticated logic to find the current agent context
        # For now, return the current span if it's an agent span
        current = get_current_span()
        try:
            if current and hasattr(current, 'name') and current.name and current.name.endswith('.agent'):
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
