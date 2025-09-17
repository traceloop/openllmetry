#!/usr/bin/env python3
"""
Mocked Recording Test

Uses mocked OpenAI responses to record complete agent workflows including
tool calls and response spans to identify agent name propagation issues.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Set a dummy API key to allow client initialization
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

from agents import Agent, function_tool, Runner
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


def create_mock_openai_response(content: str, tool_calls=None):
    """Create a mock OpenAI response object."""

    class MockChoice:
        def __init__(self, content, tool_calls=None):
            self.message = Mock()
            self.message.content = content
            self.message.tool_calls = tool_calls or []
            self.finish_reason = "stop" if not tool_calls else "tool_calls"

    class MockResponse:
        def __init__(self, content, tool_calls=None):
            self.choices = [MockChoice(content, tool_calls)]
            self.id = "chatcmpl-mock123"
            self.model = "gpt-4o"
            self.usage = Mock()
            self.usage.prompt_tokens = 100
            self.usage.completion_tokens = 50

    return MockResponse(content, tool_calls)


def create_mock_tool_call(name: str, args: dict):
    """Create a mock tool call object."""
    tool_call = Mock()
    tool_call.id = f"call_{name}_123"
    tool_call.type = "function"
    tool_call.function = Mock()
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(args)
    return tool_call


def record_mocked_workflow():
    """Record a workflow with mocked OpenAI responses to capture all span types."""

    print("🎭 Recording Mocked Workflow")
    print("=" * 50)

    # Set up span collection
    exporter = setup_span_collection()
    exporter.clear()

    # Create function tools
    @function_tool
    def search_recipes(query: str) -> str:
        """Search for recipes."""
        print(f"🔍 Tool called: search_recipes('{query}')")
        return f"Found recipe: {query}"

    @function_tool
    def modify_recipe(recipe_name: str, modification: str) -> str:
        """Modify a recipe."""
        print(f"🔧 Tool called: modify_recipe('{recipe_name}', '{modification}')")
        return f"Modified {recipe_name}: {modification}"

    # Create agents
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

    # Mock the OpenAI client chat.completions.create method to return predictable responses
    with patch('openai.resources.chat.completions.Completions.create') as mock_create:

        # Set up mock responses for different scenarios
        def mock_chat_create(*args, **kwargs):
            messages = kwargs.get('messages', [])
            tools = kwargs.get('tools', [])

            # Determine which agent is calling based on messages or context
            if tools and any('search_recipes' in str(tool) for tool in tools):
                # Recipe agent - should make tool calls
                search_tool_call = create_mock_tool_call("search_recipes", {"query": "carbonara"})
                modify_tool_call = create_mock_tool_call("modify_recipe", {
                    "recipe_name": "carbonara",
                    "modification": "vegetarian"
                })
                return create_mock_openai_response("I'll help you modify the carbonara recipe.",
                                                 [search_tool_call, modify_tool_call])
            else:
                # Main agent - just respond
                return create_mock_openai_response("I'll hand this off to the recipe specialist.")

        mock_create.side_effect = mock_chat_create

        try:
            # Test Recipe Agent (should generate tool and response spans)
            print("🍝 Testing Recipe Agent with Tools...")
            messages = [{"role": "user", "content": "Make carbonara vegetarian"}]

            try:
                result = Runner.run_sync(recipe_agent, messages)
                print(f"Recipe agent result: {result}")
            except Exception as e:
                print(f"Recipe agent error: {str(e)[:200]}...")

            # Test Main Agent (should generate response spans)
            print("🤖 Testing Main Agent...")
            messages = [{"role": "user", "content": "Help me cook"}]

            try:
                result = Runner.run_sync(main_agent, messages)
                print(f"Main agent result: {result}")
            except Exception as e:
                print(f"Main agent error: {str(e)[:200]}...")

        finally:
            # Collect spans
            spans = exporter.get_finished_spans()
            print(f"\n📊 Collected {len(spans)} spans")
            return spans


def analyze_mocked_spans(spans):
    """Analyze mocked spans for agent name propagation issues."""

    print("\n🔍 MOCKED SPAN ANALYSIS")
    print("=" * 50)

    if not spans:
        print("❌ No spans were recorded!")
        return []

    issues = []

    print(f"Recorded {len(spans)} spans:")

    for i, span in enumerate(spans):
        agent_name = span.attributes.get("gen_ai.agent.name", "NO_AGENT_NAME")
        span_kind = span.attributes.get("traceloop.span.kind", "unknown")

        print(f"  {i+1:2d}. {span.name:<35} | Agent: {agent_name:<25} | Kind: {span_kind}")

        # Check for specific propagation issues
        span_name = span.name

        # These should have agent name
        needs_agent_name = (
            span_name.endswith(".agent") or
            ".tool" in span_name or
            span_name == "openai.response" or
            span_name.startswith("openai.") or
            "tool_call" in span_name.lower()
        )

        # These should NOT have agent name
        should_not_have_agent = span_name == "Agent Workflow"

        has_agent_name = "gen_ai.agent.name" in span.attributes

        if needs_agent_name and not has_agent_name:
            issues.append(f"❌ {span_name} missing agent name")
        elif should_not_have_agent and has_agent_name:
            issues.append(f"❌ {span_name} incorrectly has agent name")

    # Report issues
    print(f"\n🚨 PROPAGATION ISSUES:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No agent name propagation issues found!")

    return issues


def save_mocked_recording(spans, issues):
    """Save the mocked recording data for analysis."""

    recording_data = {
        "test_type": "mocked_workflow",
        "timestamp": time.time(),
        "total_spans": len(spans),
        "issues_found": len(issues),
        "issues": issues,
        "spans": [
            {
                "name": span.name,
                "attributes": dict(span.attributes) if span.attributes else {},
                "has_agent_name": "gen_ai.agent.name" in span.attributes,
                "span_kind": span.attributes.get("traceloop.span.kind", "unknown")
            }
            for span in spans
        ]
    }

    # Save to file
    recordings_dir = Path(__file__).parent / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    filename = f"mocked_recording_{int(time.time())}.json"
    filepath = recordings_dir / filename

    with open(filepath, 'w') as f:
        json.dump(recording_data, f, indent=2)

    print(f"\n💾 Mocked recording saved to: {filepath}")
    return filepath


def main():
    """Main mocked recording function."""

    print("🎭 Mocked Span Recording Tool")
    print("=" * 50)
    print("Recording spans from mocked agent execution to identify agent name propagation issues")
    print()

    try:
        # Record mocked workflow
        spans = record_mocked_workflow()

        # Analyze spans
        issues = analyze_mocked_spans(spans)

        # Save recording
        filepath = save_mocked_recording(spans, issues)

        # Summary
        print(f"\n🎊 MOCKED RECORDING COMPLETE")
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
        print(f"❌ Error during mocked recording: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)