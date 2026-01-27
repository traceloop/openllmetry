"""Wrapper instrumentation for OpenAI Agents SDK realtime sessions.

The openai-agents SDK's realtime functionality doesn't use its native tracing system,
so we need to patch the RealtimeSession class directly to add OpenTelemetry tracing.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from opentelemetry.trace import Tracer, Status, StatusCode, SpanKind, Span
from opentelemetry.trace import set_span_in_context
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
    error_attributes as ErrorAttributes,
)
from .utils import (
    dont_throw,
    should_send_prompts,
    GEN_AI_HANDOFF_FROM_AGENT,
    GEN_AI_HANDOFF_TO_AGENT,
)

_original_methods: Dict[str, Any] = {}
_tracing_states: Dict[int, "RealtimeTracingState"] = {}


class RealtimeTracingState:
    """Tracks OpenTelemetry spans for a realtime session."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.workflow_span: Optional[Span] = None
        self.agent_spans: Dict[str, Span] = {}
        self.tool_spans: Dict[str, Span] = {}
        self.audio_spans: Dict[str, Span] = {}
        self.current_agent_name: Optional[str] = None
        self.prompt_index: int = 0
        self.completion_index: int = 0
        self.pending_prompts: List[Tuple[str, str]] = []  # (role, content) tuples
        self.llm_call_index: int = 0
        self.agent_span_contexts: Dict[str, Any] = {}
        self.agent_start_times: Dict[str, int] = {}
        self.prompt_start_time: Optional[int] = None
        self.prompt_agent_name: Optional[str] = None
        self.starting_agent_name: Optional[str] = None
        self.model_name: str = "gpt-4o-realtime-preview"
        self.seen_completions: set = set()
        self.pending_usage: Optional[Dict[str, int]] = None

    def start_workflow_span(self, agent_name: str):
        """Start the root workflow span for the session."""
        self.starting_agent_name = agent_name
        self.workflow_span = self.tracer.start_span(
            "Realtime Session",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
                SpanAttributes.TRACELOOP_WORKFLOW_NAME: "Realtime Session",
            },
        )
        return self.workflow_span

    def end_workflow_span(self, error: Optional[Exception] = None):
        """End the workflow span."""
        if self.workflow_span:
            if error:
                self.workflow_span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                self.workflow_span.set_status(Status(StatusCode.OK))
            self.workflow_span.end()
            self.workflow_span = None

    def _get_parent_context(self, agent_name: Optional[str] = None):
        """Get parent context from agent spans or workflow span.

        Looks up the parent context in this order:
        1. Specified agent_name in active agent_spans
        2. Specified agent_name in agent_span_contexts
        3. Current agent in agent_span_contexts
        4. Workflow span
        """
        target = agent_name or self.current_agent_name
        if target:
            if target in self.agent_spans:
                return set_span_in_context(self.agent_spans[target])
            if target in self.agent_span_contexts:
                return self.agent_span_contexts[target]
        if self.workflow_span:
            return set_span_in_context(self.workflow_span)
        return None

    def start_agent_span(self, agent_name: str):
        """Start an agent span, or reuse existing one for the same agent."""
        if agent_name in self.agent_spans:
            self.current_agent_name = agent_name
            return self.agent_spans[agent_name]

        if agent_name in self.agent_span_contexts:
            self.current_agent_name = agent_name
            return None

        parent_context = None
        if self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        span = self.tracer.start_span(
            f"{agent_name}.agent",
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                GenAIAttributes.GEN_AI_AGENT_NAME: agent_name,
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
            },
        )
        self.agent_spans[agent_name] = span
        self.current_agent_name = agent_name
        self.agent_span_contexts[agent_name] = set_span_in_context(span)
        self.agent_start_times[agent_name] = time.time_ns()
        return span

    def end_agent_span(self, agent_name: str, error: Optional[Exception] = None):
        """Mark agent turn as ended but keep span open for multi-turn conversations."""
        if error and agent_name in self.agent_spans:
            span = self.agent_spans[agent_name]
            span.set_attribute(ErrorAttributes.ERROR_MESSAGE, str(error))

    def start_tool_span(self, tool_name: str, agent_name: Optional[str] = None):
        """Start a tool span."""
        parent_context = self._get_parent_context(agent_name)

        span = self.tracer.start_span(
            f"{tool_name}.tool",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
                GenAIAttributes.GEN_AI_TOOL_TYPE: "function",
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
            },
        )
        self.tool_spans[tool_name] = span
        return span

    def end_tool_span(
        self, tool_name: str, output: Any = None, error: Optional[Exception] = None
    ):
        """End a tool span."""
        if tool_name in self.tool_spans:
            span = self.tool_spans[tool_name]
            if output is not None:
                span.set_attribute(GenAIAttributes.GEN_AI_TOOL_CALL_RESULT, str(output))
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            del self.tool_spans[tool_name]

    def create_handoff_span(self, from_agent: str, to_agent: str):
        """Create a handoff span and end the from_agent's span."""
        parent_context = self._get_parent_context(from_agent)

        if from_agent in self.agent_spans:
            from_span = self.agent_spans[from_agent]
            from_span.set_status(Status(StatusCode.OK))
            from_span.end()
            del self.agent_spans[from_agent]

        span = self.tracer.start_span(
            f"{from_agent} → {to_agent}.handoff",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: "handoff",
                GenAIAttributes.GEN_AI_SYSTEM: "openai_agents",
                GEN_AI_HANDOFF_FROM_AGENT: from_agent,
                GEN_AI_HANDOFF_TO_AGENT: to_agent,
            },
        )
        span.set_status(Status(StatusCode.OK))
        span.end()
        return span

    def start_audio_span(self, item_id: str, content_index: int):
        """Start an audio/speech span."""
        parent_context = self._get_parent_context()

        span_key = f"{item_id}:{content_index}"
        span = self.tracer.start_span(
            "openai.realtime",
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes={
                SpanAttributes.LLM_REQUEST_TYPE: "realtime",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
            },
        )
        self.audio_spans[span_key] = span
        return span

    def end_audio_span(self, item_id: str, content_index: int):
        """End an audio span."""
        span_key = f"{item_id}:{content_index}"
        if span_key in self.audio_spans:
            span = self.audio_spans[span_key]
            span.set_status(Status(StatusCode.OK))
            span.end()
            del self.audio_spans[span_key]

    def record_error(self, error: Any):
        """Record an error on the current agent span or workflow span."""
        error_str = str(error)
        if self.current_agent_name and self.current_agent_name in self.agent_spans:
            span = self.agent_spans[self.current_agent_name]
            span.set_attribute(ErrorAttributes.ERROR_MESSAGE, error_str)
        elif self.workflow_span:
            self.workflow_span.set_attribute(ErrorAttributes.ERROR_MESSAGE, error_str)

    def record_prompt(self, role: str, content: str):
        """Record a prompt message - buffers it for the LLM span."""
        if not content:
            return
        if not self.pending_prompts:
            self.prompt_start_time = time.time_ns()
            self.prompt_agent_name = self.current_agent_name or self.starting_agent_name
        self.pending_prompts.append((role, content))

    def record_usage(self, usage: Any):
        """Record token usage from response.done event."""
        if not usage:
            return
        if isinstance(usage, dict):
            self.pending_usage = {
                "input_tokens": usage.get("input_tokens", 0) or 0,
                "output_tokens": usage.get("output_tokens", 0) or 0,
                "total_tokens": usage.get("total_tokens", 0) or 0,
            }
        else:
            self.pending_usage = {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }

    def record_completion(self, role: str, content: str):
        """Record a completion message - creates an LLM span with prompt and completion."""
        if not content:
            return
        content_hash = hash(content[:100])
        if content_hash in self.seen_completions:
            return
        self.seen_completions.add(content_hash)
        self.create_llm_span(content)

    def create_llm_span(self, completion_content: str):
        """Create an LLM span pairing the first pending prompt with a completion."""
        prompt_data = self.pending_prompts.pop(0) if self.pending_prompts else None
        prompt_role, prompt_content = prompt_data if prompt_data else (None, None)
        prompt_time = self.prompt_start_time
        self.prompt_start_time = time.time_ns() if self.pending_prompts else None
        agent_name = self.prompt_agent_name
        if not self.pending_prompts:
            self.prompt_agent_name = None

        parent_context = self._get_parent_context(agent_name)

        agent_start = self.agent_start_times.get(agent_name) if agent_name else None
        if agent_start and prompt_time:
            start_time = max(agent_start, prompt_time)
        elif agent_start:
            start_time = agent_start
        elif prompt_time:
            start_time = prompt_time
        else:
            start_time = time.time_ns()

        model_name_str = (
            str(self.model_name) if self.model_name else "gpt-4o-realtime-preview"
        )

        span = self.tracer.start_span(
            "openai.realtime",
            kind=SpanKind.CLIENT,
            context=parent_context,
            start_time=start_time,
            attributes={
                SpanAttributes.LLM_REQUEST_TYPE: "realtime",
                SpanAttributes.LLM_SYSTEM: "openai",
                GenAIAttributes.GEN_AI_SYSTEM: "openai",
                GenAIAttributes.GEN_AI_REQUEST_MODEL: model_name_str,
            },
        )

        if self.pending_usage:
            if self.pending_usage.get("input_tokens"):
                span.set_attribute(
                    GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                    self.pending_usage["input_tokens"],
                )
            if self.pending_usage.get("output_tokens"):
                span.set_attribute(
                    GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                    self.pending_usage["output_tokens"],
                )
            self.pending_usage = None

        if should_send_prompts():
            if prompt_content:
                span.set_attribute(
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", prompt_role or "user"
                )
                span.set_attribute(
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", prompt_content
                )

            span.set_attribute(
                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
            )
            span.set_attribute(
                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", completion_content
            )
            span.set_attribute(
                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason", "stop"
            )

        span.set_status(Status(StatusCode.OK))
        span.end()

        self.llm_call_index += 1
        return span

    def cleanup(self):
        """Clean up any remaining spans."""
        for span in list(self.tool_spans.values()):
            span.set_status(Status(StatusCode.OK))
            span.end()
        self.tool_spans.clear()

        for span in list(self.audio_spans.values()):
            span.set_status(Status(StatusCode.OK))
            span.end()
        self.audio_spans.clear()

        for span in list(self.agent_spans.values()):
            span.set_status(Status(StatusCode.OK))
            span.end()
        self.agent_spans.clear()
        self.agent_span_contexts.clear()

        if self.workflow_span:
            self.workflow_span.set_status(Status(StatusCode.OK))
            self.workflow_span.end()
            self.workflow_span = None


def wrap_realtime_session(tracer: Tracer):
    """Wrap the RealtimeSession class to add OpenTelemetry tracing."""
    if _original_methods:
        return

    try:
        from agents.realtime.session import RealtimeSession
    except ImportError:
        return

    _original_methods["__aenter__"] = RealtimeSession.__aenter__
    _original_methods["__aexit__"] = RealtimeSession.__aexit__
    _original_methods["_put_event"] = RealtimeSession._put_event
    if hasattr(RealtimeSession, "send_message"):
        _original_methods["send_message"] = RealtimeSession.send_message

    @dont_throw
    async def traced_aenter(self):
        """Wrapped __aenter__ that starts the workflow span."""
        result = await _original_methods["__aenter__"](self)

        try:
            state = RealtimeTracingState(tracer)
            _tracing_states[id(self)] = state
            agent_name = getattr(self._current_agent, "name", "Unknown Agent")
            state.start_workflow_span(agent_name)

            model_name_str = None
            model_obj = getattr(self, "_model", None) or getattr(self, "model", None)
            if model_obj:
                model_name_str = getattr(model_obj, "model", None)

            if not model_name_str:
                config = getattr(self, "_config", None) or getattr(self, "config", None)
                if config:
                    model_settings = getattr(config, "model_settings", None)
                    if model_settings:
                        model_name_str = getattr(
                            model_settings, "model_name", None
                        ) or getattr(model_settings, "model", None)
                    if not model_name_str:
                        model_name_str = getattr(config, "model_name", None) or getattr(
                            config, "model", None
                        )

            if model_name_str and isinstance(model_name_str, str):
                state.model_name = model_name_str
        except Exception:
            pass

        return result

    @dont_throw
    async def traced_aexit(self, exc_type, exc_val, exc_tb):
        """Wrapped __aexit__ that ends the workflow span."""
        result = await _original_methods["__aexit__"](self, exc_type, exc_val, exc_tb)

        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state:
                state.cleanup()
                state.end_workflow_span(error=exc_val if exc_type else None)
                del _tracing_states[session_id]
        except Exception:
            pass

        return result

    @dont_throw
    async def traced_put_event(self, event):
        """Wrapped _put_event that creates spans for key events."""
        result = await _original_methods["_put_event"](self, event)

        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state:
                event_type = getattr(event, "type", None)

                if event_type == "agent_start":
                    agent = getattr(event, "agent", None)
                    agent_name = (
                        getattr(agent, "name", "Unknown") if agent else "Unknown"
                    )
                    state.start_agent_span(agent_name)

                elif event_type == "agent_end":
                    agent = getattr(event, "agent", None)
                    agent_name = (
                        getattr(agent, "name", "Unknown") if agent else "Unknown"
                    )
                    state.end_agent_span(agent_name)

                elif event_type == "tool_start":
                    tool = getattr(event, "tool", None)
                    agent = getattr(event, "agent", None)
                    tool_name = (
                        getattr(tool, "name", "unknown_tool")
                        if tool
                        else "unknown_tool"
                    )
                    agent_name = getattr(agent, "name", None) if agent else None
                    state.start_tool_span(tool_name, agent_name)

                elif event_type == "tool_end":
                    tool = getattr(event, "tool", None)
                    tool_name = (
                        getattr(tool, "name", "unknown_tool")
                        if tool
                        else "unknown_tool"
                    )
                    output = getattr(event, "output", None)
                    state.end_tool_span(tool_name, output)

                elif event_type == "handoff":
                    from_agent = getattr(event, "from_agent", None)
                    to_agent = getattr(event, "to_agent", None)
                    from_name = (
                        getattr(from_agent, "name", "Unknown")
                        if from_agent
                        else "Unknown"
                    )
                    to_name = (
                        getattr(to_agent, "name", "Unknown") if to_agent else "Unknown"
                    )
                    state.create_handoff_span(from_name, to_name)

                elif event_type == "audio":
                    item_id = getattr(event, "item_id", "unknown")
                    content_index = getattr(event, "content_index", 0)
                    span_key = f"{item_id}:{content_index}"
                    if span_key not in state.audio_spans:
                        state.start_audio_span(item_id, content_index)

                elif event_type == "audio_end":
                    item_id = getattr(event, "item_id", "unknown")
                    content_index = getattr(event, "content_index", 0)
                    state.end_audio_span(item_id, content_index)

                elif event_type == "error":
                    error = getattr(event, "error", "Unknown error")
                    state.record_error(error)

                elif event_type == "history_added":
                    item = getattr(event, "item", None)
                    if item:
                        role = getattr(item, "role", None)
                        content = None

                        item_content = getattr(item, "content", None)
                        if item_content:
                            if isinstance(item_content, list):
                                for part in item_content:
                                    if hasattr(part, "text"):
                                        content = part.text
                                        break
                                    elif hasattr(part, "transcript"):
                                        content = part.transcript
                                        break
                            elif isinstance(item_content, str):
                                content = item_content

                        if not content:
                            content = getattr(item, "text", None) or getattr(
                                item, "transcript", None
                            )

                        if content and role == "assistant":
                            state.record_completion(role, content)

                elif event_type == "response":
                    output = getattr(event, "output", None)
                    if output and isinstance(output, list):
                        for item in output:
                            role = getattr(item, "role", None)
                            if role == "assistant":
                                item_content = getattr(item, "content", None)
                                if item_content:
                                    if isinstance(item_content, list):
                                        for part in item_content:
                                            text = getattr(part, "text", None)
                                            if text:
                                                state.record_completion(role, text)
                                                break
                                    elif isinstance(item_content, str):
                                        state.record_completion(role, item_content)

                elif event_type == "raw_model_event":
                    data = getattr(event, "data", None)
                    if data:
                        if isinstance(data, dict):
                            data_type = data.get("type")
                            raw_data = data.get("data", data)
                            if isinstance(raw_data, dict):
                                nested_type = raw_data.get("type")
                                if nested_type:
                                    data_type = nested_type
                        else:
                            data_type = getattr(data, "type", None)
                            nested_data = getattr(data, "data", None)
                            if nested_data:
                                if isinstance(nested_data, dict):
                                    nested_type = nested_data.get("type")
                                else:
                                    nested_type = getattr(nested_data, "type", None)
                                if nested_type:
                                    data_type = nested_type
                                    data = nested_data

                        if data_type == "response.done":
                            if isinstance(data, dict):
                                response = data.get("response", {})
                                usage = (
                                    response.get("usage")
                                    if isinstance(response, dict)
                                    else None
                                )
                            else:
                                response = getattr(data, "response", None)
                                usage = (
                                    getattr(response, "usage", None)
                                    if response
                                    else None
                                )
                            if usage:
                                state.record_usage(usage)

                                output = getattr(response, "output", None)
                                if output and isinstance(output, list):
                                    for item in output:
                                        item_type = getattr(item, "type", None)
                                        if item_type == "message":
                                            role = getattr(item, "role", None)
                                            if role == "assistant":
                                                item_content = getattr(
                                                    item, "content", None
                                                )
                                                if item_content and isinstance(
                                                    item_content, list
                                                ):
                                                    for part in item_content:
                                                        text = getattr(
                                                            part, "text", None
                                                        )
                                                        if text:
                                                            state.record_completion(
                                                                role, text
                                                            )
                                                            break

                        elif data_type == "item_updated":
                            item = getattr(data, "item", None)
                            if item:
                                role = getattr(item, "role", None)
                                item_content = getattr(item, "content", None)
                                if (
                                    role == "assistant"
                                    and item_content
                                    and isinstance(item_content, list)
                                ):
                                    for part in item_content:
                                        text = getattr(part, "text", None) or getattr(
                                            part, "transcript", None
                                        )
                                        if text:
                                            state.record_completion(role, text)
                                            break
        except Exception:
            pass

        return result

    @dont_throw
    async def traced_send_message(self, message):
        """Wrapped send_message that captures user input."""
        result = None
        if "send_message" in _original_methods:
            result = await _original_methods["send_message"](self, message)

        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state and message:
                if isinstance(message, str):
                    state.record_prompt("user", message)
                else:
                    content = getattr(message, "content", None)
                    if content:
                        if isinstance(content, str):
                            state.record_prompt("user", content)
                        elif isinstance(content, list):
                            for part in content:
                                text = getattr(part, "text", None)
                                if text:
                                    state.record_prompt("user", text)
                                    break
        except Exception:
            pass

        return result

    RealtimeSession.__aenter__ = traced_aenter
    RealtimeSession.__aexit__ = traced_aexit
    RealtimeSession._put_event = traced_put_event
    if "send_message" in _original_methods:
        RealtimeSession.send_message = traced_send_message


def unwrap_realtime_session():
    """Remove the instrumentation from RealtimeSession."""
    try:
        from agents.realtime.session import RealtimeSession
    except ImportError:
        return

    if "__aenter__" in _original_methods:
        RealtimeSession.__aenter__ = _original_methods["__aenter__"]
    if "__aexit__" in _original_methods:
        RealtimeSession.__aexit__ = _original_methods["__aexit__"]
    if "_put_event" in _original_methods:
        RealtimeSession._put_event = _original_methods["_put_event"]
    if "send_message" in _original_methods:
        RealtimeSession.send_message = _original_methods["send_message"]

    _original_methods.clear()
    _tracing_states.clear()
