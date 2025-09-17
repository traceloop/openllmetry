#!/usr/bin/env python3
"""
Simple Recording Test

Uses our existing test infrastructure to record spans and identify issues.
"""

import json
import time
from pathlib import Path

# Import our existing test setup
import pytest
from unittest.mock import Mock, patch

from agents import Agent, function_tool, Runner, RunContextWrapper
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider


def setup_span_collection():
    """Set up span collection infrastructure."""
    exporter = InMemorySpanExporter()

    # Clear any existing instrumentation
    instrumentor = OpenAIAgentsInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    # Set up fresh instrumentation
    processor = SimpleSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    set_tracer_provider(provider)

    instrumentor.instrument()

    return exporter


def record_complex_workflow():
    """Record a complex workflow similar to the sample app."""

    print("🎬 Recording Complex Workflow")
    print("=" * 50)

    # Set up span collection
    exporter = setup_span_collection()
    exporter.clear()

    # Create function tools (like the sample app)
    @function_tool
    def search_recipes(query: str) -> str:
        """Search for recipes (mock)."""
        return f"Found recipes for: {query}"

    @function_tool
    def modify_recipe(recipe_name: str, modification: str) -> str:
        """Modify a recipe (mock)."""
        return f"Modified {recipe_name} with {modification}"

    # Create agents like the sample app
    class RecipeAgent(Agent):
        def __init__(self):
            super().__init__(
                name="Recipe Editor Agent",
                instructions="You edit recipes using tools.",
                model="gpt-4o",
                tools=[search_recipes, modify_recipe]
            )

    class MainAgent(Agent):
        def __init__(self, recipe_agent):
            super().__init__(
                name="Main Chat Agent",
                instructions="You handle conversations and hand off to specialists.",
                model="gpt-4o",
                handoffs=[recipe_agent]
            )

    # Create agents
    recipe_agent = RecipeAgent()
    main_agent = MainAgent(recipe_agent)

    try:
        # Scenario 1: Main agent
        print("🤖 Testing Main Agent...")
        messages = [{"role": "user", "content": "Edit carbonara to be vegetarian"}]

        try:
            Runner.run_sync(main_agent, messages)
        except Exception as e:
            print(f"Main agent error (expected): {str(e)[:100]}...")

        # Scenario 2: Recipe agent with tools
        print("🍝 Testing Recipe Agent...")

        try:
            Runner.run_sync(recipe_agent, messages)
        except Exception as e:
            print(f"Recipe agent error (expected): {str(e)[:100]}...")

        # Scenario 3: Simple agent for comparison
        print("🧪 Testing Simple Agent...")

        simple_agent = Agent(
            name="Simple Agent",
            instructions="You are simple.",
            model="gpt-4o"
        )

        try:
            Runner.run_sync(simple_agent, [{"role": "user", "content": "Hello"}])
        except Exception as e:
            print(f"Simple agent error (expected): {str(e)[:100]}...")

    finally:
        # Collect spans
        spans = exporter.get_finished_spans()
        print(f"\n📊 Collected {len(spans)} spans")

        return spans


def analyze_recorded_spans(spans):
    """Analyze recorded spans for agent name propagation issues."""

    print("\n🔍 SPAN ANALYSIS")
    print("=" * 50)

    if not spans:
        print("❌ No spans were recorded!")
        return []

    issues = []

    # Print all spans for analysis
    print(f"Recorded {len(spans)} spans:")

    for i, span in enumerate(spans):
        agent_name = span.attributes.get("gen_ai.agent.name", "NO_AGENT_NAME")
        span_kind = span.attributes.get("traceloop.span.kind", "unknown")

        print(f"  {i+1:2d}. {span.name:<30} | Agent: {agent_name:<20} | Kind: {span_kind}")

        # Check for issues
        span_name = span.name

        # Spans that should have agent name
        should_have_agent = (
            span_name.endswith(".agent") or
            ".tool" in span_name or
            span_name == "openai.response"
        )

        # Spans that should NOT have agent name
        should_not_have_agent = span_name == "Agent Workflow"

        has_agent_name = "gen_ai.agent.name" in span.attributes

        if should_have_agent and not has_agent_name:
            issues.append(f"❌ {span_name} missing agent name")
        elif should_not_have_agent and has_agent_name:
            issues.append(f"❌ {span_name} incorrectly has agent name")

    # Report issues
    print(f"\n🚨 ISSUES FOUND:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No issues found!")

    return issues


def save_recording(spans, issues):
    """Save the recording data for analysis."""

    recording_data = {
        "timestamp": time.time(),
        "total_spans": len(spans),
        "issues_found": len(issues),
        "issues": issues,
        "spans": [
            {
                "name": span.name,
                "attributes": dict(span.attributes) if span.attributes else {},
                "has_agent_name": "gen_ai.agent.name" in span.attributes
            }
            for span in spans
        ]
    }

    # Save to file
    recordings_dir = Path(__file__).parent / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    filename = f"simple_recording_{int(time.time())}.json"
    filepath = recordings_dir / filename

    with open(filepath, 'w') as f:
        json.dump(recording_data, f, indent=2)

    print(f"\n💾 Recording saved to: {filepath}")
    return filepath


def main():
    """Main recording function."""

    print("🎯 Simple Span Recording Tool")
    print("=" * 50)
    print("Recording spans from agent execution to identify agent name propagation issues")
    print()

    try:
        # Record workflow
        spans = record_complex_workflow()

        # Analyze spans
        issues = analyze_recorded_spans(spans)

        # Save recording
        filepath = save_recording(spans, issues)

        # Summary
        print(f"\n🎊 RECORDING COMPLETE")
        print("=" * 50)
        print(f"📊 Total spans: {len(spans)}")
        print(f"🚨 Issues found: {len(issues)}")
        print(f"📄 Data saved to: {filepath}")

        if issues:
            print(f"\n❌ AGENT NAME PROPAGATION ISSUES DETECTED!")
            print("These issues need to be fixed in the instrumentation.")
            return 1
        else:
            print(f"\n✅ NO ISSUES DETECTED!")
            print("Agent name propagation appears to be working correctly.")
            return 0

    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)