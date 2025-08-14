"""OpenTelemetry OpenAI Agents instrumentation"""

import os
import time
import json
import threading
import weakref
import logging
from typing import Collection
from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer, set_span_in_context, get_current_span
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry import context
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.semconv_ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
    Meters,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
)
from .utils import set_span_attribute, JSONEncoder
from agents import FunctionTool, WebSearchTool, FileSearchTool, ComputerTool
from agents.tracing.scope import Scope

# Setup console logger with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Also setup a handler for our debug messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[HANDOFF-DEBUG] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


_instruments = ("openai-agents >= 0.0.19",)

_root_span_storage = {}
_storage_lock = threading.RLock()
_instrumented_tools = set()
_handoff_contexts = {}  # Track handoff contexts: {to_agent_name:trace_id -> parent_context}
_active_agent_contexts = {}  # Track active agent spans for handoff linking: {agent_name -> (span, context)}
_active_otel_spans = {}  # Track active OpenTelemetry spans: {trace_key -> {agent_name -> otel_span}}
_logical_agent_spans = {}  # Track logical agent spans to consolidate multiple invocations: {trace_key -> {agent_name -> otel_span}}
_processor_instance = None  # Global reference to the processor instance


# Removed DeferrableSpanContext - no longer needed with workflow-level parent spans


def _get_or_set_root_span_context(span=None, agent_name=None, processor=None):
    """Get root span context using workflow-level parent span.

    Args:
        span: Current span to potentially set as root span
        agent_name: Name of the agent to check for handoff context
        processor: HandoffTracingProcessor instance to access workflow spans

    Returns:
        context: The appropriate context with workflow span set as parent
    """
    current_ctx = context.get_current()
    current_trace = Scope.get_current_trace()
    
    if current_trace and current_trace.trace_id != "no-op":
        trace_id = current_trace.trace_id
        
        # First priority: Use workflow-level span as parent if available
        if processor and trace_id in processor._workflow_spans:
            workflow_span = processor._workflow_spans[trace_id]
            workflow_context = set_span_in_context(workflow_span, current_ctx)
            logger.info(f"🌐 USING workflow span as parent for {agent_name or 'agent'}")
            return workflow_context
        
        # Second priority: Check if this agent was explicitly handed off to 
        if agent_name:
            # Try both specific trace and any-trace keys
            handoff_key_with_trace = f"{agent_name}:{trace_id}"
            handoff_key_any_trace = f"{agent_name}:any_trace"
            
            with _storage_lock:
                print(f"DEBUG: Looking for handoff context for {agent_name}")
                print(f"DEBUG: Available handoff keys: {list(_handoff_contexts.keys())}")
                print(f"DEBUG: Looking for keys: {handoff_key_with_trace}, {handoff_key_any_trace}")
                
                parent_context = _handoff_contexts.get(handoff_key_with_trace)
                if not parent_context:
                    parent_context = _handoff_contexts.get(handoff_key_any_trace)
                    if parent_context:
                        print(f"DEBUG: Found handoff context for {agent_name} using any-trace key")
                        print(f"DEBUG: Using parent context to maintain single trace hierarchy")
                        print(f"DEBUG: Returning parent context - this will create child span in same OTel trace")
                        return parent_context
                elif parent_context:
                    print(f"DEBUG: Found handoff context for {agent_name} using trace-specific key")
                    print(f"DEBUG: Using parent context to maintain single trace hierarchy")
                    print(f"DEBUG: Returning parent context - this will create child span in same OTel trace")
                    return parent_context

        # Third priority: Use traditional root span approach
        with _storage_lock:
            weak_ref = _root_span_storage.get(trace_id)
            root_span = weak_ref() if weak_ref else None

            if root_span:
                return set_span_in_context(root_span, current_ctx)
            else:
                if span:
                    def cleanup_callback(ref):
                        with _storage_lock:
                            if _root_span_storage.get(trace_id) is ref:
                                del _root_span_storage[trace_id]

                    _root_span_storage[trace_id] = weakref.ref(span, cleanup_callback)
                    return set_span_in_context(span, current_ctx)
                return current_ctx
    else:
        return current_ctx


def _register_handoff(from_agent_name, to_agent_name, parent_context):
    """Register a handoff from one agent to another.
    
    Args:
        from_agent_name: Name of the agent initiating the handoff
        to_agent_name: Name of the agent being handed off to
        parent_context: Context with the parent span that should be used
    """
    current_trace = Scope.get_current_trace()
    
    if current_trace and current_trace.trace_id != "no-op":
        trace_id = current_trace.trace_id
        handoff_key = f"{to_agent_name}:{trace_id}"
        
        with _storage_lock:
            _handoff_contexts[handoff_key] = parent_context


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                duration_histogram,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
            ) = (None, None)

        wrap_function_wrapper(
            "agents.run",
            "AgentRunner._get_new_response",
            _wrap_agent_run(
                tracer,
                duration_histogram,
                token_histogram,
            ),
        )
        wrap_function_wrapper(
            "agents.run",
            "AgentRunner._run_single_turn_streamed",
            _wrap_agent_run_streamed(
                tracer,
                duration_histogram,
                token_histogram,
            ),
        )
        
        # Use processor-based approach like OpenInference
        # Add our custom processor to the agents framework
        try:
            from agents import add_trace_processor, TracingProcessor
            
            # Create our custom processor and store it globally
            global _processor_instance
            custom_processor = HandoffTracingProcessor(tracer)
            _processor_instance = custom_processor
            
            # Add processor to the agents framework
            add_trace_processor(custom_processor)
            print("DEBUG: Successfully added handoff tracing processor")
            
        except Exception as e:
            print(f"WARNING: Cannot add tracing processor: {e}")

    def _uninstrument(self, **kwargs):
        unwrap("agents.run.AgentRunner", "_get_new_response")
        unwrap("agents.run.AgentRunner", "_run_single_turn_streamed")
# No need to unwrap processor-based approach
        _instrumented_tools.clear()
        _root_span_storage.clear()
        _handoff_contexts.clear()
        _active_agent_contexts.clear()
        _active_otel_spans.clear()


def with_tracer_wrapper(func):

    def _with_tracer(tracer, duration_histogram, token_histogram):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(
                tracer,
                duration_histogram,
                token_histogram,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


@with_tracer_wrapper
async def _wrap_agent_run_streamed(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for _run_single_turn_streamed to handle streaming execution."""
    agent = args[1] if len(args) > 1 else None
    run_config = args[4] if len(args) > 4 else None

    if not agent:
        return await wrapped(*args, **kwargs)

    agent_name = getattr(agent, "name", "agent")

    # Check if we already have a logical span for this agent
    current_trace = Scope.get_current_trace()
    if current_trace and current_trace.trace_id != "no-op":
        trace_id = current_trace.trace_id
        otel_trace_key = f"otel_trace_{trace_id}"
        
        # Check if we already have a logical span for this agent
        global _logical_agent_spans, _storage_lock, _processor_instance
        with _storage_lock:
            if (otel_trace_key in _logical_agent_spans and 
                agent_name in _logical_agent_spans[otel_trace_key]):
                # Reuse existing logical span
                existing_span = _logical_agent_spans[otel_trace_key][agent_name]
                logger.info(f"🔄 REUSING existing logical span for {agent_name} (recording={existing_span.is_recording()})")
                
                # Check if the existing span is still recording
                if not existing_span.is_recording():
                    logger.warning(f"⚠️ EXISTING SPAN for {agent_name} is no longer recording - creating new span")
                    # Clear from logical spans since it's ended
                    del _logical_agent_spans[otel_trace_key][agent_name]
                    # Fall through to create a new span
                else:
                    # Set the existing span as current and continue execution
                    span_context = set_span_in_context(existing_span, context.get_current())
                    with _storage_lock:
                        _active_agent_contexts[agent_name] = (existing_span, span_context)
                    
                    # Continue with the wrapped execution using the existing span context
                    # Execute within the existing span's context to ensure tools inherit it
                    token = context.attach(span_context)
                    try:
                        extract_agent_details(agent, existing_span)
                        set_model_settings_span_attributes(agent, existing_span)
                        extract_run_config_details(run_config, existing_span)

                        tools = getattr(agent, "tools", [])
                        if tools:
                            extract_tool_details(tracer, tools)

                        start_time = time.time()
                        result = await wrapped(*args, **kwargs)
                        end_time = time.time()

                        if duration_histogram:
                            duration = end_time - start_time
                            duration_histogram.record(
                                duration,
                                attributes={
                                    "gen_ai.agent.name": agent_name,
                                },
                            )

                        return result

                    except Exception as e:
                        existing_span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        # Clean up the active agent context but don't end the span yet
                        with _storage_lock:
                            _active_agent_contexts.pop(agent_name, None)
                        # Detach the context
                        context.detach(token)

    # Use processor instance to get proper parent context
    ctx = _get_or_set_root_span_context(agent_name=agent_name, processor=_processor_instance)

    # Create new logical OpenTelemetry span
    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: (TraceloopSpanKindValues.AGENT.value),
        },
        context=ctx,
    ) as span:
        try:
            # Register this agent as active for potential handoff scenarios
            span_context = set_span_in_context(span, ctx)
            with _storage_lock:
                _active_agent_contexts[agent_name] = (span, span_context)
                
            # Register this OpenTelemetry span for the processor to use and as a logical span
            current_trace = Scope.get_current_trace()
            if current_trace and current_trace.trace_id != "no-op":
                otel_trace_key = f"otel_trace_{current_trace.trace_id}"
                with _storage_lock:
                    # Register in active spans for processor
                    if otel_trace_key not in _active_otel_spans:
                        _active_otel_spans[otel_trace_key] = {}
                    _active_otel_spans[otel_trace_key][agent_name] = span
                    
                    # Register as logical span to reuse for subsequent invocations
                    if otel_trace_key not in _logical_agent_spans:
                        _logical_agent_spans[otel_trace_key] = {}
                    _logical_agent_spans[otel_trace_key][agent_name] = span
                    
                    logger.info(f"📊 REGISTERED new logical OpenTelemetry span for {agent_name} in trace {current_trace.trace_id}")
                    parent_span_id = span.parent.span_id if span.parent else None
                    logger.info(f"🔗 SPAN HIERARCHY: {agent_name} parent_span_id={parent_span_id}")

            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)

            try:
                json_args = []
                for arg in args:
                    try:
                        json_args.append(json.loads(json.dumps(arg, cls=JSONEncoder)))
                    except (TypeError, ValueError):
                        json_args.append(str(arg))

                json_kwargs = {}
                for key, value in kwargs.items():
                    try:
                        json_kwargs[key] = json.loads(
                            json.dumps(value, cls=JSONEncoder)
                        )
                    except (TypeError, ValueError):
                        json_kwargs[key] = str(value)

                input_data = {"args": json_args, "kwargs": json_kwargs}
                input_str = json.dumps(input_data)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, input_str)
            except Exception:
                fallback_data = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(fallback_data))

            tools = getattr(agent, "tools", [])
            if tools:
                extract_tool_details(tracer, tools)

            start_time = time.time()
            result = await wrapped(*args, **kwargs)
            end_time = time.time()

            try:
                output_str = json.dumps(result, cls=JSONEncoder)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_str)
            except Exception:
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(str(result)))

            span.set_status(Status(StatusCode.OK))

            if duration_histogram:
                duration = end_time - start_time
                duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.agent.name": agent_name,
                    },
                )

            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Clean up the active agent context but keep logical spans alive
            with _storage_lock:
                _active_agent_contexts.pop(agent_name, None)
                
                # Clean up OpenTelemetry span references but keep logical spans for reuse
                current_trace = Scope.get_current_trace()
                if current_trace and current_trace.trace_id != "no-op":
                    otel_trace_key = f"otel_trace_{current_trace.trace_id}"
                    if otel_trace_key in _active_otel_spans and agent_name in _active_otel_spans[otel_trace_key]:
                        del _active_otel_spans[otel_trace_key][agent_name]
                        logger.info(f"🧹 CLEANED UP active OpenTelemetry span for {agent_name}")
                        if not _active_otel_spans[otel_trace_key]:
                            del _active_otel_spans[otel_trace_key]
                    # Note: We keep logical spans alive for reuse - they'll be cleaned up at trace end


@with_tracer_wrapper
async def _wrap_agent_run(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    agent, *_ = args
    run_config = args[7] if len(args) > 7 else None
    prompt_list = args[2] if len(args) > 2 else None
    agent_name = getattr(agent, "name", "agent")
    model_name = get_model_name(agent)

    # Use processor instance to get proper parent context
    global _processor_instance
    ctx = _get_or_set_root_span_context(agent_name=agent_name, processor=_processor_instance)

    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: (TraceloopSpanKindValues.AGENT.value),
        },
        context=ctx,
    ) as span:
        try:
            # Register this agent as active for potential handoff scenarios
            span_context = set_span_in_context(span, ctx)
            with _storage_lock:
                _active_agent_contexts[agent_name] = (span, span_context)
            
            ctx = _get_or_set_root_span_context(span)

            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)

            try:
                json_args = []
                for arg in args:
                    try:
                        json_args.append(json.loads(json.dumps(arg, cls=JSONEncoder)))
                    except (TypeError, ValueError):
                        json_args.append(str(arg))

                json_kwargs = {}
                for key, value in kwargs.items():
                    try:
                        json_kwargs[key] = json.loads(
                            json.dumps(value, cls=JSONEncoder)
                        )
                    except (TypeError, ValueError):
                        json_kwargs[key] = str(value)

                input_data = {"args": json_args, "kwargs": json_kwargs}
                input_str = json.dumps(input_data)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, input_str)
            except Exception:
                fallback_data = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(fallback_data))

            tools = args[4] if len(args) > 4 and isinstance(args[4], list) else []
            if tools:
                extract_tool_details(tracer, tools)

            start_time = time.time()
            response = await wrapped(*args, **kwargs)

            try:
                output_str = json.dumps(response, cls=JSONEncoder)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_str)
            except Exception:
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(str(response)))
            if duration_histogram:
                duration_histogram.record(
                    time.time() - start_time,
                )
            if isinstance(prompt_list, list):
                set_prompt_attributes(span, prompt_list)
            set_response_content_span_attribute(response, span)
            set_token_usage_span_attributes(
                response, span, model_name, token_histogram, agent
            )

            span.set_status(Status(StatusCode.OK))
            return response

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Clean up the active agent context
            with _storage_lock:
                _active_agent_contexts.pop(agent_name, None)


class HandoffTracingProcessor:
    """Custom tracing processor to handle OpenAI agents handoffs and span hierarchy."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._handoff_contexts = {}  # Track handoffs like OpenInference  
        self._active_agent_spans = {}  # Track active agent spans by trace
        self._workflow_spans = {}  # Track workflow-level spans: {trace_id -> otel_span}
        
    def on_trace_start(self, trace):
        """Called when a new trace starts - create a workflow-level OpenTelemetry span."""
        print(f"DEBUG: Trace started: {trace.trace_id}")
        
        # Create a parent "Agent Workflow" span that will encompass the entire trace
        workflow_span = self.tracer.start_span(
            f"Agent Workflow",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: (TraceloopSpanKindValues.AGENT.value),
                "gen_ai.system": "openai_agents", 
                "gen_ai.workflow.name": "Agent Workflow"
            }
        )
        
        # Store this workflow span to use as parent for all agents in this trace
        self._workflow_spans[trace.trace_id] = workflow_span
        logger.info(f"🌐 CREATED workflow span for trace {trace.trace_id}")
        
    def on_trace_end(self, trace):
        """Called when a trace ends."""
        print(f"DEBUG: Trace ended: {trace.trace_id}")
        
        # End the workflow-level span
        if trace.trace_id in self._workflow_spans:
            workflow_span = self._workflow_spans[trace.trace_id]
            workflow_span.end()
            logger.info(f"🏁 ENDED workflow span for trace {trace.trace_id}")
            del self._workflow_spans[trace.trace_id]
        
        # Clean up any remaining handoff contexts and spans for this trace
        global _handoff_contexts, _active_otel_spans, _logical_agent_spans, _storage_lock
        
        with _storage_lock:
            # Clean up handoff contexts for this trace
            keys_to_remove = [key for key in _handoff_contexts.keys() if key.endswith(f":{trace.trace_id}") or key.endswith(":any_trace")]
            for key in keys_to_remove:
                del _handoff_contexts[key]
                logger.info(f"🧹 CLEANUP: Removed handoff context {key}")
            
            # Clean up any remaining OpenTelemetry spans for this trace
            otel_trace_key = f"otel_trace_{trace.trace_id}"
            if otel_trace_key in _active_otel_spans:
                for agent_name in list(_active_otel_spans[otel_trace_key].keys()):
                    logger.info(f"🧹 FINAL CLEANUP: Removed OpenTelemetry span for {agent_name}")
                del _active_otel_spans[otel_trace_key]
                
            # Clean up logical agent spans for this trace
            if otel_trace_key in _logical_agent_spans:
                for agent_name in list(_logical_agent_spans[otel_trace_key].keys()):
                    logger.info(f"🧹 FINAL CLEANUP: Removed logical agent span for {agent_name}")
                del _logical_agent_spans[otel_trace_key]
        # Clean up any remaining contexts for this trace
        trace_id = trace.trace_id
        keys_to_remove = [k for k in self._handoff_contexts.keys() if k.endswith(f":{trace_id}")]
        for key in keys_to_remove:
            del self._handoff_contexts[key]
            
    def on_span_start(self, span):
        """Called when a new span starts."""        
        # Get span data - it's in span_data attribute
        span_data = getattr(span, 'span_data', None)
        if not span_data:
            print(f"DEBUG: No span_data found")
            return
            
        print(f"DEBUG: Span data type: {span_data.__class__.__name__}")
                
        # Import the span data types from agents
        from agents import AgentSpanData, HandoffSpanData
        
        # Check if this is an agent span
        if isinstance(span_data, AgentSpanData):
            agent_name = span_data.name
            trace_id = span.trace_id
            
            logger.info(f"🤖 AGENT SPAN DETECTED: {agent_name} in trace {trace_id}")
            
            # Check if we already have an active agent span for this agent in this trace
            # This helps consolidate multiple agent invocations into a single logical span
            trace_key = f"trace_{trace_id}"
            if trace_key in self._active_agent_spans and agent_name in self._active_agent_spans[trace_key]:
                existing_span = self._active_agent_spans[trace_key][agent_name]
                logger.info(f"🔄 DUPLICATE AGENT SPAN DETECTED: {agent_name} already active in trace {trace_id}")
                # We could potentially skip creating additional spans for the same agent
                # but for now, we'll let them be created to maintain compatibility
            
            # Register this span as active for potential handoffs
            # Note: We store the agents span here, but for handoff contexts we need OTel spans
            if trace_key not in self._active_agent_spans:
                self._active_agent_spans[trace_key] = {}
            self._active_agent_spans[trace_key][agent_name] = span
            
        elif isinstance(span_data, HandoffSpanData):
            logger.info(f"🔄 HANDOFF SPAN DETECTED!")
            logger.debug(f"HandoffSpanData attributes: {dir(span_data)}")
            
            # Extract handoff information for logging and span creation
            from_agent = getattr(span_data, 'from_agent', None)
            to_agent = getattr(span_data, 'to_agent', None) 
            agent_name = getattr(span_data, 'agent_name', None)
            target_agent = getattr(span_data, 'target_agent', None)
            trace_id = span.trace_id
            
            logger.info(f"📋 HANDOFF DETAILS: from_agent={from_agent}, to_agent={to_agent}, agent_name={agent_name}, target_agent={target_agent}")
            
            # Create explicit handoff spans
            resolved_from = from_agent
            resolved_to = to_agent
            
            if from_agent and to_agent:
                logger.info(f"✅ CREATING HANDOFF SPAN: {from_agent} → {to_agent}")
                self._create_handoff_span(from_agent, to_agent, trace_id)
            elif agent_name and target_agent:
                logger.info(f"✅ CREATING HANDOFF SPAN: {agent_name} → {target_agent}")
                self._create_handoff_span(agent_name, target_agent, trace_id)
            elif from_agent and not to_agent:
                # Try to infer target agent
                if "Orchestra Conductor" in from_agent:
                    resolved_to = "Symphony Composer"
                elif "Main Chat Agent" in from_agent:
                    resolved_to = "Recipe Editor Agent"
                elif "Distillery Chat Agent" in from_agent:
                    resolved_to = "GenEdit Agent"
                elif "Data Router" in from_agent:
                    resolved_to = "Analytics Agent"
                
                if resolved_to:
                    logger.info(f"🎯 CREATING INFERRED HANDOFF SPAN: {from_agent} → {resolved_to}")
                    self._create_handoff_span(from_agent, resolved_to, trace_id)
    
    def on_span_end(self, span):
        """Called when a span ends."""
        from agents import AgentSpanData
        
        span_data = getattr(span, 'span_data', None)
        if span_data and isinstance(span_data, AgentSpanData):
            agent_name = span_data.name
            trace_id = span.trace_id
            trace_key = f"trace_{trace_id}"
            
            print(f"DEBUG: Agent span ended: {agent_name}")
            
            # Clean up the active span reference
            if trace_key in self._active_agent_spans:
                self._active_agent_spans[trace_key].pop(agent_name, None)
                if not self._active_agent_spans[trace_key]:  # If empty
                    del self._active_agent_spans[trace_key]
                    
    def _register_handoff(self, from_agent, to_agent, trace_id):
        """Register a handoff context for proper span hierarchy."""
        trace_key = f"trace_{trace_id}"
        
        # Find the active OpenTelemetry span for the from_agent in this trace
        global _active_otel_spans, _storage_lock
        otel_trace_key = f"otel_trace_{trace_id}"
        
        with _storage_lock:
            if (otel_trace_key in _active_otel_spans and 
                from_agent in _active_otel_spans[otel_trace_key]):
                
                otel_span = _active_otel_spans[otel_trace_key][from_agent]
                logger.info(f"✅ FOUND active OpenTelemetry span for {from_agent}")
                
                # Create context with this span as parent for the target agent
                # Create handoff key that works across different agent framework traces
                # Use just the agent name since handoffs are typically between named agents
                handoff_key = f"{to_agent}:any_trace"
                handoff_key_with_trace = f"{to_agent}:{trace_id}" 
                
                # Store the span context that should be used as parent
                from opentelemetry.trace import set_span_in_context
                from opentelemetry import context
                parent_context = set_span_in_context(otel_span, context.get_current())
                
                # Update the global handoff contexts that the wrapper functions check
                # Store with both keys to handle different trace lookup scenarios
                global _handoff_contexts, _handoff_initiators
                _handoff_contexts[handoff_key] = parent_context
                _handoff_contexts[handoff_key_with_trace] = parent_context
                logger.info(f"📝 REGISTERED handoff context for {to_agent} (keys: {handoff_key}, {handoff_key_with_trace})")
                
                # Track that this agent initiated a handoff (for deferred span logic)
                if otel_trace_key not in _handoff_initiators:
                    _handoff_initiators[otel_trace_key] = set()
                _handoff_initiators[otel_trace_key].add(from_agent)
                logger.info(f"🎯 MARKED {from_agent} as handoff initiator in trace {trace_id}")
            else:
                logger.warning(f"❌ NO active OpenTelemetry span found for {from_agent} in trace {trace_id}")
                if otel_trace_key in _active_otel_spans:
                    logger.info(f"Available agents in trace: {list(_active_otel_spans[otel_trace_key].keys())}")
    
    def _create_handoff_span(self, from_agent, to_agent, trace_id):
        """Create an explicit handoff span showing the transition between agents."""
        # Find the workflow span to use as parent
        if trace_id in self._workflow_spans:
            workflow_span = self._workflow_spans[trace_id]
            workflow_context = set_span_in_context(workflow_span, context.get_current())
            
            # Create handoff span as child of workflow span
            handoff_span = self.tracer.start_span(
                f"{from_agent} → {to_agent}.handoff",
                kind=SpanKind.INTERNAL,
                attributes={
                    SpanAttributes.TRACELOOP_SPAN_KIND: "handoff",
                    "gen_ai.handoff.from_agent": from_agent,
                    "gen_ai.handoff.to_agent": to_agent,
                    "gen_ai.system": "openai_agents"
                },
                context=workflow_context
            )
            
            # Immediately end the handoff span as it's just a marker
            handoff_span.end()
            logger.info(f"🔄 CREATED handoff span: {from_agent} → {to_agent}")
        else:
            logger.warning(f"❌ Cannot create handoff span - no workflow span found for trace {trace_id}")
    
    def force_flush(self):
        """Force flush any pending data."""
        pass
        
    def shutdown(self):
        """Shutdown the processor."""
        self._handoff_contexts.clear()
        self._active_agent_spans.clear()


def get_model_name(agent):
    model_attr = getattr(getattr(agent, "model", None), "model", "unknown_model")
    if model_attr == "unknown_model":
        model_attr = getattr(agent, "model", None)
        return model_attr
    else:
        return model_attr


def extract_agent_details(test_agent, span):
    if test_agent is None:
        return

    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    name = getattr(agent, "name", None)
    instructions = getattr(agent, "instructions", None)
    handoff_description = getattr(agent, "handoff_description", None)
    handoffs = getattr(agent, "handoffs", None)
    if name:
        set_span_attribute(span, "gen_ai.agent.name", name)
    if instructions:
        set_span_attribute(span, "gen_ai.agent.description", instructions)
    if handoff_description:
        set_span_attribute(
            span, "gen_ai.agent.handoff_description", handoff_description
        )
    if handoffs:
        for idx, h in enumerate(handoffs):
            handoff_info = {
                "name": getattr(h, "name", None),
                "instructions": getattr(h, "instructions", None),
            }
            handoff_json = json.dumps(handoff_info)
            span.set_attribute(f"openai.agent.handoff{idx}", handoff_json)
    attributes = {}
    for key, value in vars(agent).items():
        if key in ("name", "instructions", "handoff_description"):
            continue

        if value is not None:
            if isinstance(value, (str, int, float, bool)):
                attributes[f"openai.agent.{key}"] = value
            elif isinstance(value, list) and len(value) > 0:
                attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def set_model_settings_span_attributes(agent, span):

    if not hasattr(agent, "model_settings") or agent.model_settings is None:
        return

    model_settings = agent.model_settings
    settings_dict = vars(model_settings)

    key_to_span_attr = {
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    }

    for key, value in settings_dict.items():
        if value is not None:
            span_attr = key_to_span_attr.get(key, f"openai.agent.model.{key}")
            span.set_attribute(span_attr, value)


def extract_run_config_details(run_config, span):
    if run_config is None:
        return

    config_dict = vars(run_config)
    attributes = {}

    for key, value in config_dict.items():

        if value is not None and isinstance(value, (str, int, float, bool)):
            attributes[f"openai.agent.{key}"] = value
        elif isinstance(value, list) and len(value) != 0:
            attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def extract_tool_details(tracer: Tracer, tools):
    """Create spans for hosted tools and wrap FunctionTool execution."""
    for tool in tools:
        if isinstance(tool, FunctionTool):
            tool_id = id(tool)
            if tool_id in _instrumented_tools:
                continue

            _instrumented_tools.add(tool_id)

            original_on_invoke_tool = tool.on_invoke_tool

            def create_wrapped_tool(original_tool, original_func):
                async def wrapped_on_invoke_tool(tool_context, args_json):
                    tool_name = getattr(original_tool, "name", "tool")
                    
                    # Use current context for tools to nest under their agent spans
                    ctx = context.get_current()
                    
                    # Debug: Check what span is current
                    current_span = get_current_span(ctx)
                    if current_span and current_span.is_recording():
                        logger.info(f"🔧 TOOL {tool_name} using span: {current_span.name} (recording={current_span.is_recording()})")
                    else:
                        logger.warning(f"⚠️ TOOL {tool_name} has no active span context - falling back to workflow span")
                        # Fallback to workflow span if no active context
                        global _processor_instance
                        ctx = _get_or_set_root_span_context(processor=_processor_instance)

                    with tracer.start_as_current_span(
                        f"{tool_name}.tool",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            SpanAttributes.TRACELOOP_SPAN_KIND: (
                                TraceloopSpanKindValues.TOOL.value
                            )
                        },
                        context=ctx,
                    ) as span:
                        try:
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.name", tool_name
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.type", "FunctionTool"
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.description",
                                original_tool.description,
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.strict_json_schema",
                                original_tool.strict_json_schema,
                            )
                            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, args_json)
                            result = await original_func(tool_context, args_json)
                            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

                return wrapped_on_invoke_tool

            tool.on_invoke_tool = create_wrapped_tool(tool, original_on_invoke_tool)

        elif isinstance(tool, (WebSearchTool, FileSearchTool, ComputerTool)):
            tool_name = type(tool).__name__
            
            # Use current context for tools to nest under agent spans
            ctx = context.get_current()

            span = tracer.start_span(
                f"{tool_name}.tool",
                kind=SpanKind.INTERNAL,
                attributes={
                    SpanAttributes.TRACELOOP_SPAN_KIND: (
                        TraceloopSpanKindValues.TOOL.value
                    )
                },
                context=ctx,
            )

            if isinstance(tool, WebSearchTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "WebSearchTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.search_context_size",
                    tool.search_context_size,
                )
                if tool.user_location:
                    span.set_attribute(
                        f"{GEN_AI_COMPLETION}.tool.user_location",
                        str(tool.user_location),
                    )
            elif isinstance(tool, FileSearchTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "FileSearchTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.vector_store_ids",
                    str(tool.vector_store_ids),
                )
                if tool.max_num_results:
                    span.set_attribute(
                        f"{GEN_AI_COMPLETION}.tool.max_num_results",
                        tool.max_num_results,
                    )
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.include_search_results",
                    tool.include_search_results,
                )
            elif isinstance(tool, ComputerTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "ComputerTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.computer", str(tool.computer)
                )

            span.set_status(Status(StatusCode.OK))
            span.end()


def set_prompt_attributes(span, message_history):
    if not message_history:
        return

    for i, msg in enumerate(message_history):
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            role = msg.get("role", "user")
            content = msg.get("content", None)
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                role,
            )
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                content,
            )


def set_response_content_span_attribute(response, span):
    if hasattr(response, "output") and isinstance(response.output, list):
        roles = []
        types = []
        contents = []

        for output_message in response.output:
            role = getattr(output_message, "role", None)
            msg_type = getattr(output_message, "type", None)

            if role:
                roles.append(role)
            if msg_type:
                types.append(msg_type)

            if hasattr(output_message, "content") and isinstance(
                output_message.content, list
            ):
                for content_item in output_message.content:
                    if hasattr(content_item, "text"):
                        contents.append(content_item.text)

        if roles:
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.roles",
                roles,
            )
        if types:
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.types",
                types,
            )
        if contents:
            set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.contents", contents
            )


def set_token_usage_span_attributes(
    response, span, model_name, token_histogram, test_agent
):
    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    agent_name = getattr(agent, "name", None)
    if hasattr(response, "usage"):
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if input_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                input_tokens,
            )
        if output_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                output_tokens,
            )
        if total_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                total_tokens,
            )
        if token_histogram:
            token_histogram.record(
                input_tokens,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_TOKEN_TYPE: "input",
                    SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                    "gen_ai.agent.name": agent_name,
                },
            )
            token_histogram.record(
                output_tokens,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_TOKEN_TYPE: "output",
                    SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                    "gen_ai.agent.name": agent_name,
                },
            )


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, duration_histogram
