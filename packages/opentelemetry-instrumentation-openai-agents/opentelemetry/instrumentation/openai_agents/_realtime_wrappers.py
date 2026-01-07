"""Wrapper instrumentation for OpenAI Agents SDK realtime sessions.

The openai-agents SDK's realtime functionality doesn't use its native tracing system,
so we need to patch the RealtimeSession class directly to add OpenTelemetry tracing.
"""

import time
from typing import Dict, Any, Optional, List
from opentelemetry.trace import Tracer, Status, StatusCode, SpanKind, Span
from opentelemetry.trace import set_span_in_context
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from .utils import dont_throw

# Store original methods for uninstrumentation
_original_methods: Dict[str, Any] = {}


class RealtimeTracingState:
    """Tracks OpenTelemetry spans for a realtime session."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.workflow_span: Optional[Span] = None
        self.agent_spans: Dict[str, Span] = {}  # agent_name -> span
        self.tool_spans: Dict[str, Span] = {}  # tool_name -> span
        self.audio_spans: Dict[str, Span] = {}  # item_id -> span
        self.llm_spans: Dict[str, Span] = {}  # llm_call_id -> span
        self.span_contexts: Dict[str, Any] = {}  # span_id -> context token
        self.current_agent_name: Optional[str] = None
        self.prompt_index: int = 0  # Track prompt message index
        self.completion_index: int = 0  # Track completion message index
        self.pending_prompts: List[str] = []  # Buffer for user prompts awaiting completion
        self.llm_call_index: int = 0  # Track LLM call index
        # Store agent span contexts even after spans end (for async tool calls)
        self.agent_span_contexts: Dict[str, Any] = {}  # agent_name -> context
        # Track agent start times (for LLM span duration)
        self.agent_start_times: Dict[str, int] = {}  # agent_name -> start time in ns
        # Track when prompt was sent (for response time calculation)
        self.prompt_start_time: Optional[int] = None  # nanoseconds timestamp
        self.prompt_agent_name: Optional[str] = None  # agent name when prompt was recorded
        self.starting_agent_name: Optional[str] = None  # the initial agent for fallback

    def start_workflow_span(self, agent_name: str):
        """Start the root workflow span for the session."""
        self.starting_agent_name = agent_name  # Save for fallback when recording prompts
        self.workflow_span = self.tracer.start_span(
            "Realtime Session",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
                "gen_ai.system": "openai_agents",
                "gen_ai.workflow.name": "Realtime Session",
                "gen_ai.agent.starting_agent": agent_name,
            }
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

    def start_agent_span(self, agent_name: str):
        """Start an agent span."""
        parent_context = None
        if self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        span = self.tracer.start_span(
            f"{agent_name}.agent",
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                "gen_ai.agent.name": agent_name,
                "gen_ai.system": "openai_agents",
            }
        )
        self.agent_spans[agent_name] = span
        self.current_agent_name = agent_name
        # Store context for async tool calls that may come after span ends
        self.agent_span_contexts[agent_name] = set_span_in_context(span)
        # Record start time for LLM span duration
        self.agent_start_times[agent_name] = time.time_ns()
        return span

    def end_agent_span(self, agent_name: str, error: Optional[Exception] = None):
        """End an agent span."""
        if agent_name in self.agent_spans:
            span = self.agent_spans[agent_name]
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            del self.agent_spans[agent_name]

    def start_tool_span(self, tool_name: str, agent_name: Optional[str] = None):
        """Start a tool span."""
        parent_context = None
        # Try to parent under the current agent span (or its saved context for async tools)
        if agent_name and agent_name in self.agent_spans:
            parent_context = set_span_in_context(self.agent_spans[agent_name])
        elif agent_name and agent_name in self.agent_span_contexts:
            # Agent span ended but we have its context (async tool call)
            parent_context = self.agent_span_contexts[agent_name]
        elif self.current_agent_name and self.current_agent_name in self.agent_span_contexts:
            # Fall back to current agent's saved context
            parent_context = self.agent_span_contexts[self.current_agent_name]
        elif self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        span = self.tracer.start_span(
            f"{tool_name}.tool",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                "gen_ai.tool.name": tool_name,
                "gen_ai.system": "openai_agents",
            }
        )
        self.tool_spans[tool_name] = span
        return span

    def end_tool_span(self, tool_name: str, output: Any = None, error: Optional[Exception] = None):
        """End a tool span."""
        if tool_name in self.tool_spans:
            span = self.tool_spans[tool_name]
            if output is not None:
                span.set_attribute("gen_ai.tool.output", str(output)[:4096])
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            del self.tool_spans[tool_name]

    def create_handoff_span(self, from_agent: str, to_agent: str):
        """Create a handoff span."""
        parent_context = None
        if from_agent and from_agent in self.agent_spans:
            parent_context = set_span_in_context(self.agent_spans[from_agent])
        elif from_agent and from_agent in self.agent_span_contexts:
            # Agent span ended but we have its context
            parent_context = self.agent_span_contexts[from_agent]
        elif self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        span = self.tracer.start_span(
            f"{from_agent} → {to_agent}.handoff",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: "handoff",
                "gen_ai.system": "openai_agents",
                "gen_ai.handoff.from_agent": from_agent,
                "gen_ai.handoff.to_agent": to_agent,
            }
        )
        span.set_status(Status(StatusCode.OK))
        span.end()
        return span

    def start_audio_span(self, item_id: str, content_index: int):
        """Start an audio/speech span."""
        parent_context = None
        if self.current_agent_name and self.current_agent_name in self.agent_spans:
            parent_context = set_span_in_context(self.agent_spans[self.current_agent_name])
        elif self.current_agent_name and self.current_agent_name in self.agent_span_contexts:
            # Agent span ended but we have its context
            parent_context = self.agent_span_contexts[self.current_agent_name]
        elif self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        span_key = f"{item_id}:{content_index}"
        # Use "openai.realtime" to match the OpenAI instrumentation span name
        span = self.tracer.start_span(
            "openai.realtime",
            kind=SpanKind.CLIENT,
            context=parent_context,
            attributes={
                SpanAttributes.LLM_REQUEST_TYPE: "realtime",
                "gen_ai.system": "openai",
                "gen_ai.audio.item_id": item_id,
                "gen_ai.audio.content_index": content_index,
            }
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
            span.set_attribute("gen_ai.error", error_str)
        elif self.workflow_span:
            self.workflow_span.set_attribute("gen_ai.error", error_str)

    def record_prompt(self, role: str, content: str):
        """Record a prompt message - buffers it for the LLM span."""
        if not content:
            return
        # Record start time and agent name when first prompt is received
        if not self.pending_prompts:
            self.prompt_start_time = time.time_ns()
            # Use current agent, or fall back to starting agent if no agent is active yet
            self.prompt_agent_name = self.current_agent_name or self.starting_agent_name
        self.pending_prompts.append(content)

    def record_completion(self, role: str, content: str):
        """Record a completion message - creates an LLM span with prompt and completion."""
        if not content:
            return

        # Create a dedicated LLM span for this exchange
        self.create_llm_span(content)

    def create_llm_span(self, completion_content: str):
        """Create an LLM span that shows the prompt and completion."""
        # Get pending prompts, start time, and agent name
        prompts = self.pending_prompts.copy()
        self.pending_prompts.clear()
        prompt_time = self.prompt_start_time
        self.prompt_start_time = None  # Reset for next exchange
        agent_name = self.prompt_agent_name
        self.prompt_agent_name = None  # Reset for next exchange

        # Calculate response time if we have prompt time
        response_time_ms = None
        if prompt_time:
            response_time_ms = (time.time_ns() - prompt_time) / 1_000_000  # Convert to ms

        # Use the agent name from when the prompt was sent (not current)
        parent_context = None
        if agent_name and agent_name in self.agent_spans:
            parent_context = set_span_in_context(self.agent_spans[agent_name])
        elif agent_name and agent_name in self.agent_span_contexts:
            # Agent span ended but we have its context
            parent_context = self.agent_span_contexts[agent_name]
        elif self.workflow_span:
            parent_context = set_span_in_context(self.workflow_span)

        # Use agent start time as LLM span start time (so it fits within parent)
        start_time = None
        if agent_name and agent_name in self.agent_start_times:
            start_time = self.agent_start_times[agent_name]

        # Create the LLM span with agent's start time
        # Use "openai.realtime" to match the OpenAI instrumentation span name
        span = self.tracer.start_span(
            "openai.realtime",
            kind=SpanKind.CLIENT,
            context=parent_context,
            start_time=start_time,
            attributes={
                SpanAttributes.LLM_REQUEST_TYPE: "realtime",
                SpanAttributes.LLM_SYSTEM: "openai",
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-realtime-preview",
            }
        )

        # Record actual response time as attribute
        if response_time_ms is not None:
            span.set_attribute("gen_ai.response_time_ms", response_time_ms)

        # Set prompt attributes
        for i, prompt in enumerate(prompts):
            span.set_attribute(f"gen_ai.prompt.{i}.role", "user")
            span.set_attribute(f"gen_ai.prompt.{i}.content", prompt[:4096])

        # Set completion attributes
        span.set_attribute("gen_ai.completion.0.role", "assistant")
        span.set_attribute("gen_ai.completion.0.content", completion_content[:4096])

        # Finish the span immediately
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

        # Clear saved agent contexts
        self.agent_span_contexts.clear()

        if self.workflow_span:
            self.workflow_span.set_status(Status(StatusCode.OK))
            self.workflow_span.end()
            self.workflow_span = None


def wrap_realtime_session(tracer: Tracer):
    """Wrap the RealtimeSession class to add OpenTelemetry tracing."""
    try:
        from agents.realtime.session import RealtimeSession
    except ImportError:
        return  # Realtime not available

    # Store original methods
    _original_methods['__aenter__'] = RealtimeSession.__aenter__
    _original_methods['__aexit__'] = RealtimeSession.__aexit__
    _original_methods['_put_event'] = RealtimeSession._put_event
    if hasattr(RealtimeSession, 'send_message'):
        _original_methods['send_message'] = RealtimeSession.send_message

    # Store tracing state on the session instance
    _tracing_states: Dict[int, RealtimeTracingState] = {}

    @dont_throw
    async def traced_aenter(self):
        """Wrapped __aenter__ that starts the workflow span."""
        # Always call original first to ensure session works
        result = await _original_methods['__aenter__'](self)

        # Then do tracing (failures here shouldn't break the session)
        try:
            state = RealtimeTracingState(tracer)
            _tracing_states[id(self)] = state

            # Get agent name
            agent_name = getattr(self._current_agent, 'name', 'Unknown Agent')

            # Start workflow span
            state.start_workflow_span(agent_name)
        except Exception:
            pass  # Tracing failure shouldn't break the app

        return result

    @dont_throw
    async def traced_aexit(self, exc_type, exc_val, exc_tb):
        """Wrapped __aexit__ that ends the workflow span."""
        # Always call original first to ensure session closes properly
        result = await _original_methods['__aexit__'](self, exc_type, exc_val, exc_tb)

        # Then do tracing cleanup (failures here shouldn't break the app)
        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state:
                # Cleanup any remaining spans
                state.cleanup()

                # End workflow span
                state.end_workflow_span(error=exc_val if exc_type else None)

                # Remove state
                del _tracing_states[session_id]
        except Exception:
            pass  # Tracing failure shouldn't break the app

        return result

    @dont_throw
    async def traced_put_event(self, event):
        """Wrapped _put_event that creates spans for key events."""
        # Always call original first to ensure event gets queued
        result = await _original_methods['_put_event'](self, event)

        # Then do tracing (failures here shouldn't break the app)
        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state:
                event_type = getattr(event, 'type', None)

                if event_type == 'agent_start':
                    agent = getattr(event, 'agent', None)
                    agent_name = getattr(agent, 'name', 'Unknown') if agent else 'Unknown'
                    state.start_agent_span(agent_name)

                elif event_type == 'agent_end':
                    agent = getattr(event, 'agent', None)
                    agent_name = getattr(agent, 'name', 'Unknown') if agent else 'Unknown'
                    state.end_agent_span(agent_name)

                elif event_type == 'tool_start':
                    tool = getattr(event, 'tool', None)
                    agent = getattr(event, 'agent', None)
                    tool_name = getattr(tool, 'name', 'unknown_tool') if tool else 'unknown_tool'
                    agent_name = getattr(agent, 'name', None) if agent else None
                    state.start_tool_span(tool_name, agent_name)

                elif event_type == 'tool_end':
                    tool = getattr(event, 'tool', None)
                    tool_name = getattr(tool, 'name', 'unknown_tool') if tool else 'unknown_tool'
                    output = getattr(event, 'output', None)
                    state.end_tool_span(tool_name, output)

                elif event_type == 'handoff':
                    from_agent = getattr(event, 'from_agent', None)
                    to_agent = getattr(event, 'to_agent', None)
                    from_name = getattr(from_agent, 'name', 'Unknown') if from_agent else 'Unknown'
                    to_name = getattr(to_agent, 'name', 'Unknown') if to_agent else 'Unknown'
                    state.create_handoff_span(from_name, to_name)

                elif event_type == 'audio':
                    item_id = getattr(event, 'item_id', 'unknown')
                    content_index = getattr(event, 'content_index', 0)
                    # Only start if we don't have an existing span for this item
                    span_key = f"{item_id}:{content_index}"
                    if span_key not in state.audio_spans:
                        state.start_audio_span(item_id, content_index)

                elif event_type == 'audio_end':
                    item_id = getattr(event, 'item_id', 'unknown')
                    content_index = getattr(event, 'content_index', 0)
                    state.end_audio_span(item_id, content_index)

                elif event_type == 'error':
                    error = getattr(event, 'error', 'Unknown error')
                    state.record_error(error)

                elif event_type == 'history_added':
                    # Capture text content from history items
                    item = getattr(event, 'item', None)
                    if item:
                        role = getattr(item, 'role', None)
                        content = None

                        # Try to extract text content from the item
                        # The item may have a 'content' attribute (list) or 'text' attribute
                        item_content = getattr(item, 'content', None)
                        if item_content:
                            # Content is usually a list of content parts
                            if isinstance(item_content, list):
                                for part in item_content:
                                    if hasattr(part, 'text'):
                                        content = part.text
                                        break
                                    elif hasattr(part, 'transcript'):
                                        content = part.transcript
                                        break
                            elif isinstance(item_content, str):
                                content = item_content

                        # Also check for direct text/transcript attributes
                        if not content:
                            content = getattr(item, 'text', None) or getattr(item, 'transcript', None)

                        if content and role:
                            if role == 'user':
                                state.record_prompt(role, content)
                            elif role == 'assistant':
                                state.record_completion(role, content)

                elif event_type == 'raw_model_event':
                    # Handle raw model events to capture response text
                    data = getattr(event, 'data', None)
                    if data:
                        data_type = getattr(data, 'type', None)
                        if data_type == 'item_updated':
                            item = getattr(data, 'item', None)
                            if item:
                                role = getattr(item, 'role', None)
                                item_content = getattr(item, 'content', None)
                                if item_content and isinstance(item_content, list):
                                    for part in item_content:
                                        text = getattr(part, 'text', None) or getattr(part, 'transcript', None)
                                        if text and role == 'assistant':
                                            state.record_completion(role, text)
        except Exception:
            pass  # Tracing failure shouldn't break the app

        return result

    @dont_throw
    async def traced_send_message(self, message):
        """Wrapped send_message that captures user input.

        Args:
            message: RealtimeUserInput - can be str or RealtimeUserInputMessage
        """
        # Always call original first to ensure message is sent
        result = None
        if 'send_message' in _original_methods:
            result = await _original_methods['send_message'](self, message)

        # Then do tracing (failures here shouldn't break the app)
        try:
            session_id = id(self)
            state = _tracing_states.get(session_id)

            if state and message:
                # Extract text content from message
                if isinstance(message, str):
                    state.record_prompt("user", message)
                else:
                    # message is RealtimeUserInputMessage - extract text from content
                    content = getattr(message, 'content', None)
                    if content:
                        if isinstance(content, str):
                            state.record_prompt("user", content)
                        elif isinstance(content, list):
                            for part in content:
                                text = getattr(part, 'text', None)
                                if text:
                                    state.record_prompt("user", text)
                                    break
        except Exception:
            pass  # Tracing failure shouldn't break the app

        return result

    # Replace the methods
    RealtimeSession.__aenter__ = traced_aenter
    RealtimeSession.__aexit__ = traced_aexit
    RealtimeSession._put_event = traced_put_event
    if 'send_message' in _original_methods:
        RealtimeSession.send_message = traced_send_message


def unwrap_realtime_session():
    """Remove the instrumentation from RealtimeSession."""
    try:
        from agents.realtime.session import RealtimeSession
    except ImportError:
        return

    if '__aenter__' in _original_methods:
        RealtimeSession.__aenter__ = _original_methods['__aenter__']
    if '__aexit__' in _original_methods:
        RealtimeSession.__aexit__ = _original_methods['__aexit__']
    if '_put_event' in _original_methods:
        RealtimeSession._put_event = _original_methods['_put_event']
    if 'send_message' in _original_methods:
        RealtimeSession.send_message = _original_methods['send_message']

    _original_methods.clear()
