"""
Auto-generated test based on recorded span patterns.
Generated from actual mocked workflow execution with 6 recorded spans.

Key findings from recording:
- Agent spans (*.agent): ✅ Have gen_ai.agent.name
- Response spans (openai.response): ✅ Have gen_ai.agent.name
- Workflow spans (Agent Workflow): ✅ Correctly do NOT have gen_ai.agent.name
- Tool spans: Not captured in this recording (API auth prevented execution)
"""

import os
import pytest
from unittest.mock import Mock, patch

from agents import Agent, function_tool, Runner

def test_recorded_span_patterns(exporter):
    """Test based on recorded span patterns from mocked workflow execution."""

    # Set dummy API key for client initialization
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"

    try:
        # Clear exporter
        exporter.clear()

        # Expected span patterns from recording:
        # - Agent spans: 2 (Recipe Editor Agent.agent, Main Chat Agent.agent)
        # - Response spans: 2 (openai.response)
        # - Workflow spans: 2 (Agent Workflow)

        # Create test agents matching the recording
        @function_tool
        def search_recipes(query: str) -> str:
            """Search for recipes."""
            return f"Found recipe: {query}"

        class RecipeAgent(Agent):
            def __init__(self):
                super().__init__(
                    name="Recipe Editor Agent",
                    instructions="You edit recipes using tools.",
                    model="gpt-4o",
                    tools=[search_recipes]
                )

        class MainAgent(Agent):
            def __init__(self, recipe_agent):
                super().__init__(
                    name="Main Chat Agent",
                    instructions="You handle conversations and hand off to specialists.",
                    model="gpt-4o",
                    handoffs=[recipe_agent]
                )

        # Create test agents
        recipe_agent = RecipeAgent()
        main_agent = MainAgent(recipe_agent)

        # Mock OpenAI API calls to simulate the recorded behavior
        with patch('openai.resources.chat.completions.Completions.create') as mock_create:

            # Mock response that simulates what happened in recording
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "I'll help you with that."
            mock_response.choices[0].message.tool_calls = []
            mock_response.choices[0].finish_reason = "stop"
            mock_response.id = "chatcmpl-test123"
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            # Configure the mock to return 401 error (like in recording)
            from openai import AuthenticationError
            mock_create.side_effect = AuthenticationError("Mocked auth error", response=Mock(), body=None)

            # Test Recipe Agent (should generate agent + response + workflow spans)
            messages = [{"role": "user", "content": "Make carbonara vegetarian"}]

            try:
                Runner.run_sync(recipe_agent, messages)
            except (AuthenticationError, Exception):
                pass  # Expected - auth error like in recording

            # Test Main Agent (should generate agent + response + workflow spans)
            messages = [{"role": "user", "content": "Help me cook"}]

            try:
                Runner.run_sync(main_agent, messages)
            except (AuthenticationError, Exception):
                pass  # Expected - auth error like in recording

        # Get spans
        spans = exporter.get_finished_spans()

        # Verify we got the expected span types (from recording analysis)
        span_names = [span.name for span in spans]
        agent_spans = [s for s in span_names if s.endswith(".agent")]
        response_spans = [s for s in span_names if s == "openai.response"]
        workflow_spans = [s for s in span_names if s == "Agent Workflow"]

        # Verify span counts match recording expectations
        assert len(agent_spans) >= 2, f"Expected at least 2 agent spans, got {len(agent_spans)}: {agent_spans}"
        assert len(workflow_spans) >= 2, f"Expected at least 2 workflow spans, got {len(workflow_spans)}: {workflow_spans}"

        # Response spans depend on whether OpenAI calls are made, should have some if auth fails after span creation
        print(f"📊 Captured spans: Agent={len(agent_spans)}, Response={len(response_spans)}, Workflow={len(workflow_spans)}")

        # Verify agent name propagation based on recorded patterns
        for span in spans:
            span_name = span.name
            has_agent_name = "gen_ai.agent.name" in span.attributes

            if span_name.endswith(".agent"):
                # Agent spans MUST have agent name (verified in recording)
                assert has_agent_name, f"Agent span missing agent name: {span_name}"
                agent_name = span.attributes["gen_ai.agent.name"]
                assert agent_name in ["Recipe Editor Agent", "Main Chat Agent"], f"Unexpected agent name: {agent_name}"

            elif span_name == "openai.response":
                # Response spans MUST have agent name (verified in recording)
                assert has_agent_name, f"Response span missing agent name: {span_name}"
                agent_name = span.attributes["gen_ai.agent.name"]
                assert agent_name in ["Recipe Editor Agent", "Main Chat Agent"], f"Unexpected agent name in response: {agent_name}"

            elif ".tool" in span_name:
                # Tool spans SHOULD have agent name (not captured in this recording but requirement)
                assert has_agent_name, f"Tool span missing agent name: {span_name}"

            elif span_name == "Agent Workflow":
                # Workflow spans should NOT have agent name (verified in recording)
                assert not has_agent_name, f"Workflow span should not have agent name: {span_name}"

        print("✅ Recorded pattern test passed!")

    finally:
        # Restore original API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
