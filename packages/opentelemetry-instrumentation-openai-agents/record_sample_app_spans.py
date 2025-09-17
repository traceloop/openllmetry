#!/usr/bin/env python3
"""
Span Recording Script for Sample App Analysis

This script captures real OpenTelemetry spans from the sample app execution
to analyze agent name propagation in complex real-world scenarios.
"""

import json
import time
import asyncio
from typing import Dict, List, Any
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor


class SpanRecorder:
    """Custom span exporter that records spans for analysis."""

    def __init__(self):
        self.recorded_spans = []
        self.start_time = time.time()

    def export(self, spans):
        """Export spans and record them for analysis."""
        for span in spans:
            span_data = {
                "name": span.name,
                "span_id": format(span.context.span_id, '016x'),
                "trace_id": format(span.context.trace_id, '032x'),
                "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "duration_ns": span.end_time - span.start_time if span.end_time else None,
                "attributes": dict(span.attributes) if span.attributes else {},
                "status": {
                    "status_code": span.status.status_code.name,
                    "description": span.status.description
                },
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {}
                    }
                    for event in span.events
                ],
                "relative_timestamp": time.time() - self.start_time
            }
            self.recorded_spans.append(span_data)

        return 0  # Success

    def shutdown(self):
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush any pending spans."""
        return True


def setup_recording_infrastructure():
    """Set up OpenTelemetry infrastructure for recording spans."""

    # Create span recorder
    recorder = SpanRecorder()

    # Set up tracer provider
    provider = TracerProvider()
    processor = SimpleSpanProcessor(recorder)
    provider.add_span_processor(processor)
    set_tracer_provider(provider)

    # Instrument OpenAI Agents
    instrumentor = OpenAIAgentsInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()

    return recorder


def save_recorded_spans(recorder: SpanRecorder, filename: str):
    """Save recorded spans to a JSON file for analysis."""

    output_data = {
        "metadata": {
            "total_spans": len(recorder.recorded_spans),
            "recording_duration": time.time() - recorder.start_time,
            "timestamp": time.time(),
        },
        "spans": recorder.recorded_spans
    }

    output_path = Path(__file__).parent / "recordings" / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"✅ Recorded {len(recorder.recorded_spans)} spans to {output_path}")
    return output_path


def analyze_agent_name_propagation(spans: List[Dict[str, Any]]):
    """Analyze spans to identify missing agent name propagation."""

    print(f"\n📊 SPAN ANALYSIS REPORT")
    print(f"{'='*50}")

    # Categorize spans
    workflow_spans = [s for s in spans if "Agent Workflow" in s["name"]]
    agent_spans = [s for s in spans if s["name"].endswith(".agent")]
    tool_spans = [s for s in spans if ".tool" in s["name"]]
    response_spans = [s for s in spans if s["name"] == "openai.response"]
    handoff_spans = [s for s in spans if "handoff" in s["name"].lower()]
    other_spans = [s for s in spans if s not in workflow_spans + agent_spans + tool_spans + response_spans + handoff_spans]

    print(f"Total spans: {len(spans)}")
    print(f"  Workflow spans: {len(workflow_spans)}")
    print(f"  Agent spans: {len(agent_spans)}")
    print(f"  Tool spans: {len(tool_spans)}")
    print(f"  Response spans: {len(response_spans)}")
    print(f"  Handoff spans: {len(handoff_spans)}")
    print(f"  Other spans: {len(other_spans)}")

    # Check agent name propagation
    print(f"\n🔍 AGENT NAME PROPAGATION ANALYSIS")
    print(f"{'='*50}")

    def has_agent_name(span):
        return "gen_ai.agent.name" in span["attributes"]

    def get_agent_name(span):
        return span["attributes"].get("gen_ai.agent.name", "NO_AGENT_NAME")

    # Workflow spans should NOT have agent name
    workflow_with_agent = [s for s in workflow_spans if has_agent_name(s)]
    if workflow_with_agent:
        print(f"❌ {len(workflow_with_agent)} workflow spans incorrectly have agent name")
    else:
        print(f"✅ {len(workflow_spans)} workflow spans correctly have no agent name")

    # Agent spans should have agent name
    agent_without_name = [s for s in agent_spans if not has_agent_name(s)]
    if agent_without_name:
        print(f"❌ {len(agent_without_name)} agent spans missing agent name")
        for span in agent_without_name:
            print(f"   - {span['name']}")
    else:
        print(f"✅ {len(agent_spans)} agent spans have agent name")

    # Tool spans should have agent name
    tool_without_name = [s for s in tool_spans if not has_agent_name(s)]
    if tool_without_name:
        print(f"❌ {len(tool_without_name)}/{len(tool_spans)} tool spans missing agent name")
        for span in tool_without_name:
            print(f"   - {span['name']}")
    else:
        print(f"✅ {len(tool_spans)} tool spans have agent name")

    # Response spans should have agent name
    response_without_name = [s for s in response_spans if not has_agent_name(s)]
    if response_without_name:
        print(f"❌ {len(response_without_name)}/{len(response_spans)} response spans missing agent name")
    else:
        print(f"✅ {len(response_spans)} response spans have agent name")

    # Print detailed span information
    print(f"\n📋 DETAILED SPAN INFORMATION")
    print(f"{'='*50}")

    for i, span in enumerate(spans):
        agent_name = get_agent_name(span)
        parent_id = span["parent_span_id"] or "ROOT"

        print(f"Span {i+1:2d}: {span['name']:<30} | Agent: {agent_name:<20} | Parent: {parent_id}")

        if span["attributes"]:
            relevant_attrs = {k: v for k, v in span["attributes"].items()
                            if "gen_ai" in k or "traceloop" in k or "openai" in k}
            if relevant_attrs:
                for key, value in relevant_attrs.items():
                    print(f"         {key}: {value}")

    # Summary of issues
    print(f"\n🚨 ISSUES SUMMARY")
    print(f"{'='*50}")

    total_issues = len(agent_without_name) + len(tool_without_name) + len(response_without_name) + len(workflow_with_agent)

    if total_issues == 0:
        print("✅ No agent name propagation issues found!")
    else:
        print(f"❌ Found {total_issues} agent name propagation issues:")
        if agent_without_name:
            print(f"   - {len(agent_without_name)} agent spans missing names")
        if tool_without_name:
            print(f"   - {len(tool_without_name)} tool spans missing names")
        if response_without_name:
            print(f"   - {len(response_without_name)} response spans missing names")
        if workflow_with_agent:
            print(f"   - {len(workflow_with_agent)} workflow spans incorrectly have names")


async def record_sample_app_execution():
    """Record the actual sample app execution."""

    print("🎬 Starting sample app execution recording...")
    print("⚠️  Make sure you have valid OpenAI API keys set in your environment!")

    # Set up recording infrastructure
    recorder = setup_recording_infrastructure()

    try:
        # Import the sample app components
        import sys
        sys.path.append(str(Path(__file__).parent / ".." / "sample-app" / "sample_app"))

        # Try to import the sample app
        try:
            from openai_agents_example import run_streaming_chat

            print("📱 Running sample app workflow...")

            # Run the sample app
            user_input = "Can you edit the carbonara recipe to be vegetarian?"
            await run_streaming_chat(user_input)

        except ImportError as e:
            print(f"❌ Could not import sample app: {e}")
            print("💡 Falling back to simplified workflow...")

            # Fallback: create a simple workflow
            from agents import Agent, Runner

            class TestAgent(Agent):
                def __init__(self):
                    super().__init__(
                        name="Test Recording Agent",
                        instructions="You are a test agent for recording spans.",
                        model="gpt-4o"
                    )

            test_agent = TestAgent()
            messages = [{"role": "user", "content": "Hello, test agent!"}]

            try:
                Runner.run_sync(test_agent, messages)
            except Exception as e:
                print(f"Expected error: {e}")

        except Exception as e:
            print(f"❌ Error during execution: {e}")

    finally:
        # Allow some time for spans to be processed
        await asyncio.sleep(1)

        # Save the recorded spans
        filename = f"sample_app_spans_{int(time.time())}.json"
        output_path = save_recorded_spans(recorder, filename)

        # Analyze the spans
        analyze_agent_name_propagation(recorder.recorded_spans)

        return output_path, recorder.recorded_spans


if __name__ == "__main__":
    """Run the span recording script."""

    print("🔍 OpenAI Agents Span Recording Tool")
    print("=" * 50)
    print("This script records real OpenTelemetry spans from sample app execution")
    print("to analyze agent name propagation issues.")
    print()

    # Run the recording
    asyncio.run(record_sample_app_execution())