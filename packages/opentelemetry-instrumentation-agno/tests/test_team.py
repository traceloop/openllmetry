import pytest
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_team_discussion(instrument, span_exporter, reader):
    """Test team with multiple agents having a discussion."""
    technical_expert = Agent(
        name="TechnicalExpert",
        role="Technical perspective",
        model=OpenAIChat(id="gpt-4o-mini"),
        add_name_to_context=True,
        instructions=dedent("""
        Provide technical perspective. Keep responses brief.
        """),
    )

    business_expert = Agent(
        name="BusinessExpert",
        role="Business perspective",
        model=OpenAIChat(id="gpt-4o-mini"),
        add_name_to_context=True,
        instructions=dedent("""
        Provide business perspective. Keep responses brief.
        """),
    )

    discussion_team = Team(
        name="DiscussionTeam",
        delegate_task_to_all_members=True,
        model=OpenAIChat(id="gpt-4o-mini"),
        members=[technical_expert, business_expert],
        instructions=[
            "Gather perspectives from all members.",
            "Provide a brief synthesis.",
        ],
        markdown=True,
    )

    discussion_team.run("Should we use microservices? Brief answer only.")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    team_span = spans[-1]
    assert team_span.name == "DiscussionTeam.team"
    assert team_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"


@pytest.mark.vcr
def test_team_basic(instrument, span_exporter, reader):
    """Test basic team functionality."""
    agent1 = Agent(
        name="Agent1",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="First agent",
    )

    agent2 = Agent(
        name="Agent2",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="Second agent",
    )

    team = Team(
        name="BasicTeam",
        members=[agent1, agent2],
        model=OpenAIChat(id="gpt-4o-mini"),
    )

    team.run("What is 1 + 1?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    team_span = spans[-1]
    assert team_span.name == "BasicTeam.team"
    assert team_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
