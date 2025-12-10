"""
Instrumentation wrappers for OpenAI Realtime API.

The Realtime API uses WebSocket connections for low-latency, multi-modal conversations.
This module provides wrappers to instrument:
- Session lifecycle (connect/disconnect)
- Response cycles (response.create -> response.done)
- Tool/function calls

Key concepts:
- A "session" span covers the entire WebSocket connection lifecycle
- A "response" span covers each response.create -> response.done cycle
- Token usage is captured from response.done events
"""

import json
import time
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from opentelemetry.instrumentation.openai.shared import (
    _set_span_attribute,
    model_as_dict,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    dont_throw,
    should_send_prompts,
)


SPAN_NAME_SESSION = "openai.session"
SPAN_NAME_RESPONSE = "openai.realtime"


class RealtimeSessionState:
    """Tracks state for a realtime session to correlate events with spans."""

    def __init__(self, tracer: Tracer, model: str):
        self.tracer = tracer
        self.model = model
        self.session_span = None
        self.response_span = None
        self.response_start_time = None
        self.session_config: dict = {}
        self.current_response_id: Optional[str] = None
        self.accumulated_text: str = ""
        self.accumulated_audio_transcript: str = ""
        self.function_calls: list = []
        self.input_text: str = ""
        self.trace_context = None


class RealtimeEventProcessor:
    """Shared event processing logic for realtime connections."""

    def __init__(self, state: RealtimeSessionState):
        self._state = state

    @dont_throw
    def process_event(self, event):
        """Process incoming server events."""
        event_type = getattr(event, "type", None)
        if not event_type:
            return

        if event_type == "session.created":
            self.handle_session_created(event)
        elif event_type == "session.updated":
            self.handle_session_updated(event)
        elif event_type == "response.created":
            self.handle_response_created(event)
        elif event_type == "response.text.delta":
            self.handle_text_delta(event)
        elif event_type == "response.audio_transcript.delta":
            self.handle_audio_transcript_delta(event)
        elif event_type == "response.function_call_arguments.done":
            self.handle_function_call_done(event)
        elif event_type == "response.done":
            self.handle_response_done(event)
        elif event_type == "error":
            self.handle_error_event(event)

    @dont_throw
    def handle_session_created(self, event):
        """Handle session.created event."""
        if hasattr(event, "session"):
            session = event.session
            if hasattr(session, "model"):
                self._state.model = session.model
            if hasattr(session, "modalities"):
                self._state.session_config["modalities"] = session.modalities
            if hasattr(session, "instructions"):
                self._state.session_config["instructions"] = session.instructions

        if self._state.session_span and self._state.session_span.is_recording():
            _set_span_attribute(
                self._state.session_span,
                GenAIAttributes.GEN_AI_REQUEST_MODEL,
                self._state.model,
            )

    @dont_throw
    def handle_session_updated(self, event):
        """Handle session.updated event."""
        if hasattr(event, "session"):
            session = event.session
            session_dict = (
                model_as_dict(session) if hasattr(session, "__dict__") else {}
            )
            self._state.session_config.update(session_dict)

            if self._state.session_span and self._state.session_span.is_recording():
                if hasattr(session, "modalities"):
                    _set_span_attribute(
                        self._state.session_span,
                        f"{SpanAttributes.LLM_REQUEST_TYPE}.modalities",
                        json.dumps(session.modalities) if session.modalities else None,
                    )
                if hasattr(session, "temperature") and session.temperature is not None:
                    _set_span_attribute(
                        self._state.session_span,
                        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
                        session.temperature,
                    )

    @dont_throw
    def handle_response_created(self, event, start_span_if_none: bool = False):
        """Handle response.created event - response cycle started."""
        if hasattr(event, "response") and hasattr(event.response, "id"):
            self._state.current_response_id = event.response.id

        # Optionally start span if not already started (used by iterator)
        if start_span_if_none and self._state.response_span is None:
            self.start_response_span()

    @dont_throw
    def handle_text_delta(self, event):
        """Handle response.text.delta event - accumulate text."""
        if hasattr(event, "delta"):
            self._state.accumulated_text += event.delta

    @dont_throw
    def handle_audio_transcript_delta(self, event):
        """Handle response.audio_transcript.delta event."""
        if hasattr(event, "delta"):
            self._state.accumulated_audio_transcript += event.delta

    @dont_throw
    def handle_function_call_done(self, event):
        """Handle response.function_call_arguments.done event."""
        self._state.function_calls.append({
            "name": getattr(event, "name", None),
            "call_id": getattr(event, "call_id", None),
            "arguments": getattr(event, "arguments", None),
        })

    @dont_throw
    def handle_response_done(self, event):
        """Handle response.done event - end the response span."""
        if self._state.response_span is None:
            return

        span = self._state.response_span

        if span.is_recording():
            # Set response attributes
            if hasattr(event, "response"):
                response = event.response
                _set_span_attribute(
                    span,
                    GenAIAttributes.GEN_AI_RESPONSE_ID,
                    getattr(response, "id", None),
                )

                # Set usage metrics
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    _set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                        getattr(usage, "input_tokens", None),
                    )
                    _set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                        getattr(usage, "output_tokens", None),
                    )
                    total = (getattr(usage, "input_tokens", 0) or 0) + (
                        getattr(usage, "output_tokens", 0) or 0
                    )
                    _set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        total,
                    )

                # Set output content if tracing is enabled
                if should_send_prompts():
                    # Always set role for completions
                    _set_span_attribute(
                        span,
                        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role",
                        "assistant",
                    )

                    # Set content (text or audio transcript)
                    if self._state.accumulated_text:
                        _set_span_attribute(
                            span,
                            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
                            self._state.accumulated_text,
                        )
                    elif self._state.accumulated_audio_transcript:
                        _set_span_attribute(
                            span,
                            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
                            self._state.accumulated_audio_transcript,
                        )

                    # Set tool calls and finish_reason
                    if self._state.function_calls:
                        _set_span_attribute(
                            span,
                            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason",
                            "tool_calls",
                        )
                        for i, call in enumerate(self._state.function_calls):
                            _set_span_attribute(
                                span,
                                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{i}.type",
                                "function",
                            )
                            _set_span_attribute(
                                span,
                                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{i}.name",
                                call.get("name"),
                            )
                            _set_span_attribute(
                                span,
                                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{i}.id",
                                call.get("call_id"),
                            )
                            _set_span_attribute(
                                span,
                                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{i}.arguments",
                                call.get("arguments"),
                            )
                    else:
                        _set_span_attribute(
                            span,
                            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason",
                            "stop",
                        )

            span.set_status(Status(StatusCode.OK))

        span.end()
        self.reset_response_state()

    @dont_throw
    def handle_error_event(self, event):
        """Handle error event."""
        error_msg = str(getattr(event, "error", "Unknown error"))

        if self._state.response_span and self._state.response_span.is_recording():
            self._state.response_span.set_status(Status(StatusCode.ERROR, error_msg))
            self._state.response_span.record_exception(Exception(error_msg))
            self._state.response_span.end()
            self.reset_response_state()

    def start_response_span(self, end_existing: bool = False, set_input: bool = False):
        """Start a new response span."""
        if self._state.response_span is not None:
            if end_existing:
                # End any existing response span with OK status
                self._state.response_span.set_status(Status(StatusCode.OK))
                self._state.response_span.end()
                self.reset_response_state()
            else:
                return  # Don't start a new span if one exists and end_existing=False

        self._state.response_start_time = time.time_ns()

        ctx = self._state.trace_context or context_api.get_current()

        self._state.response_span = self._state.tracer.start_span(
            SPAN_NAME_RESPONSE,
            kind=SpanKind.CLIENT,
            start_time=self._state.response_start_time,
            context=ctx,
        )

        if self._state.response_span.is_recording():
            _set_span_attribute(
                self._state.response_span,
                GenAIAttributes.GEN_AI_SYSTEM,
                "openai",
            )
            _set_span_attribute(
                self._state.response_span,
                GenAIAttributes.GEN_AI_REQUEST_MODEL,
                self._state.model,
            )
            _set_span_attribute(
                self._state.response_span,
                SpanAttributes.LLM_REQUEST_TYPE,
                "realtime",
            )

            # Set input if available and requested
            if set_input and should_send_prompts() and self._state.input_text:
                _set_span_attribute(
                    self._state.response_span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
                    self._state.input_text,
                )
                _set_span_attribute(
                    self._state.response_span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.role",
                    "user",
                )

    def reset_response_state(self):
        """Reset state for the next response cycle."""
        self._state.response_span = None
        self._state.response_start_time = None
        self._state.current_response_id = None
        self._state.accumulated_text = ""
        self._state.accumulated_audio_transcript = ""
        self._state.function_calls = []
        self._state.input_text = ""


class RealtimeConnectionWrapper:
    """
    Wrapper for OpenAI Realtime connection that instruments events.

    Wraps the connection object returned by client.beta.realtime.connect()
    to capture telemetry for session lifecycle and response cycles.
    """

    def __init__(self, connection, state: RealtimeSessionState):
        self._connection = connection
        self._state = state
        self._processor = RealtimeEventProcessor(state)
        self._closed = False

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped connection."""
        return getattr(self._connection, name)

    @property
    def session(self):
        """Return a wrapped session object."""
        return RealtimeSessionWrapper(self._connection.session, self._state)

    @property
    def conversation(self):
        """Return a wrapped conversation object."""
        return RealtimeConversationWrapper(self._connection.conversation, self._state)

    @property
    def response(self):
        """Return a wrapped response object."""
        return RealtimeResponseWrapper(
            self._connection.response, self._state, self._processor
        )

    async def recv(self):
        """Receive and process an event."""
        event = await self._connection.recv()
        self._processor.process_event(event)
        return event

    def recv_bytes(self):
        """Delegate to wrapped connection."""
        return self._connection.recv_bytes()

    async def send(self, event):
        """Send an event, tracking response.create."""
        self._process_outgoing_event(event)
        return await self._connection.send(event)

    def __aiter__(self):
        """Return async iterator for events."""
        return RealtimeEventIterator(self._connection, self._state, self._processor)

    async def close(self):
        """Close the connection and end the session span."""
        if not self._closed:
            self._closed = True
            self._end_session_span()
        return await self._connection.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if not self._closed:
            self._closed = True
            if exc_type is not None:
                self._handle_error(exc_val)
            self._end_session_span()
        if hasattr(self._connection, "__aexit__"):
            return await self._connection.__aexit__(exc_type, exc_val, exc_tb)
        await self._connection.close()
        return False

    @dont_throw
    def _process_outgoing_event(self, event):
        """Process outgoing client events."""
        if isinstance(event, dict):
            event_type = event.get("type")
        else:
            event_type = getattr(event, "type", None)

        if event_type == "response.create":
            self._processor.start_response_span(end_existing=True, set_input=True)
        elif event_type == "conversation.item.create":
            self._track_input(event)

    @dont_throw
    def _track_input(self, event):
        """Track input from conversation.item.create events."""
        if isinstance(event, dict):
            item = event.get("item", {})
        else:
            item = getattr(event, "item", {})

        if isinstance(item, dict):
            content = item.get("content", [])
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "input_text":
                        self._state.input_text = block.get("text", "")
                    elif block.get("type") == "input_audio":
                        # Audio input - could capture transcript if available
                        pass

    def _handle_error(self, error):
        """Handle connection errors."""
        if self._state.response_span and self._state.response_span.is_recording():
            self._state.response_span.set_status(
                Status(StatusCode.ERROR, str(error))
            )
            self._state.response_span.record_exception(error)
            self._state.response_span.end()
            self._processor.reset_response_state()

        if self._state.session_span and self._state.session_span.is_recording():
            self._state.session_span.set_status(
                Status(StatusCode.ERROR, str(error))
            )
            self._state.session_span.record_exception(error)

    def _end_session_span(self):
        """End the session span."""
        if self._state.session_span and self._state.session_span.is_recording():
            self._state.session_span.set_status(Status(StatusCode.OK))
            self._state.session_span.end()


class RealtimeEventIterator:
    """Async iterator that wraps connection events and processes them."""

    def __init__(
        self,
        connection,
        state: RealtimeSessionState,
        processor: RealtimeEventProcessor,
    ):
        self._connection = connection
        self._state = state
        self._processor = processor
        self._iterator = connection.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self._iterator.__anext__()
        self._process_event(event)
        return event

    @dont_throw
    def _process_event(self, event):
        """Process server events to track response lifecycle."""
        event_type = getattr(event, "type", None)
        if not event_type:
            return

        # For response.created, start span if not already started
        if event_type == "response.created":
            self._processor.handle_response_created(event, start_span_if_none=True)
        else:
            # Delegate all other events to the shared processor
            self._processor.process_event(event)


class RealtimeSessionWrapper:
    """Wrapper for connection.session to track session updates."""

    def __init__(self, session, state: RealtimeSessionState):
        self._session = session
        self._state = state

    def __getattr__(self, name):
        return getattr(self._session, name)

    async def update(self, **kwargs):
        """Wrap session.update to track configuration changes."""
        result = await self._session.update(**kwargs)

        # Track session config updates
        if "session" in kwargs:
            session_config = kwargs["session"]
            if isinstance(session_config, dict):
                self._state.session_config.update(session_config)

                # Update session span attributes
                if self._state.session_span and self._state.session_span.is_recording():
                    if "temperature" in session_config:
                        _set_span_attribute(
                            self._state.session_span,
                            GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
                            session_config["temperature"],
                        )
                    if "instructions" in session_config:
                        _set_span_attribute(
                            self._state.session_span,
                            GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                            session_config["instructions"],
                        )

        return result


class RealtimeConversationWrapper:
    """Wrapper for connection.conversation to track input items."""

    def __init__(self, conversation, state: RealtimeSessionState):
        self._conversation = conversation
        self._state = state

    def __getattr__(self, name):
        return getattr(self._conversation, name)

    @property
    def item(self):
        """Return wrapped item accessor."""
        return RealtimeConversationItemWrapper(self._conversation.item, self._state)


class RealtimeConversationItemWrapper:
    """Wrapper for connection.conversation.item to track input."""

    def __init__(self, item, state: RealtimeSessionState):
        self._item = item
        self._state = state

    def __getattr__(self, name):
        return getattr(self._item, name)

    async def create(self, **kwargs):
        """Wrap item.create to track user input."""
        result = await self._item.create(**kwargs)

        # Track input for the span
        if "item" in kwargs:
            item = kwargs["item"]
            if isinstance(item, dict):
                content = item.get("content", [])
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "input_text":
                            self._state.input_text = block.get("text", "")

        return result


class RealtimeResponseWrapper:
    """Wrapper for connection.response to track response lifecycle."""

    def __init__(
        self,
        response,
        state: RealtimeSessionState,
        processor: RealtimeEventProcessor,
    ):
        self._response = response
        self._state = state
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._response, name)

    async def create(self, **kwargs):
        """Wrap response.create to start a response span."""
        # Start the response span before making the call
        self._processor.start_response_span(end_existing=True, set_input=True)

        return await self._response.create(**kwargs)

    async def cancel(self):
        """Wrap response.cancel to handle span cleanup."""
        result = await self._response.cancel()

        # End the response span with cancellation status
        if self._state.response_span and self._state.response_span.is_recording():
            self._state.response_span.set_status(
                Status(StatusCode.OK, "Response cancelled")
            )
            self._state.response_span.set_attribute("response.cancelled", True)
            self._state.response_span.end()
            self._processor.reset_response_state()

        return result


class RealtimeConnectionManagerWrapper:
    """
    Wrapper for the connection manager returned by client.beta.realtime.connect().

    This wraps the async context manager to:
    1. Start a session span when the connection is established
    2. Return a wrapped connection that instruments events
    3. End the session span when the connection closes
    """

    def __init__(self, connection_manager, tracer: Tracer, model: str):
        self._connection_manager = connection_manager
        self._tracer = tracer
        self._model = model
        self._state = None
        self._connection = None

    async def __aenter__(self):
        """Enter the connection manager and start session span."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self._connection_manager.__aenter__()

        # Start the session span
        self._state = RealtimeSessionState(self._tracer, self._model)
        parent_context = context_api.get_current()

        self._state.session_span = self._tracer.start_span(
            SPAN_NAME_SESSION,
            kind=SpanKind.CLIENT,
            context=parent_context,
        )

        # Set trace_context to include the session span as parent for response spans
        self._state.trace_context = trace.set_span_in_context(
            self._state.session_span, parent_context
        )

        if self._state.session_span.is_recording():
            _set_span_attribute(
                self._state.session_span,
                GenAIAttributes.GEN_AI_SYSTEM,
                "openai",
            )
            _set_span_attribute(
                self._state.session_span,
                GenAIAttributes.GEN_AI_REQUEST_MODEL,
                self._model,
            )
            _set_span_attribute(
                self._state.session_span,
                SpanAttributes.LLM_REQUEST_TYPE,
                "realtime",
            )

        # Enter the underlying connection manager
        connection = await self._connection_manager.__aenter__()

        # Wrap the connection
        self._connection = RealtimeConnectionWrapper(connection, self._state)
        return self._connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the connection manager and end session span."""
        if self._state is None:
            return await self._connection_manager.__aexit__(exc_type, exc_val, exc_tb)

        try:
            # Handle any errors
            if exc_type is not None and self._state.session_span:
                if self._state.session_span.is_recording():
                    self._state.session_span.set_status(
                        Status(StatusCode.ERROR, str(exc_val))
                    )
                    self._state.session_span.record_exception(exc_val)
            elif self._state.session_span and self._state.session_span.is_recording():
                self._state.session_span.set_status(Status(StatusCode.OK))

            # End the session span
            if self._state.session_span:
                self._state.session_span.end()

        finally:
            # Exit the underlying connection manager
            # Don't return the result - let any exception propagate naturally
            await self._connection_manager.__aexit__(exc_type, exc_val, exc_tb)


@_with_tracer_wrapper
def realtime_connect_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    """
    Wrapper for client.beta.realtime.connect().

    Returns a wrapped connection manager that instruments the session.
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    model = kwargs.get("model", "gpt-4o-realtime-preview")
    connection_manager = wrapped(*args, **kwargs)
    return RealtimeConnectionManagerWrapper(connection_manager, tracer, model)
