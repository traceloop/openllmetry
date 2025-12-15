import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from traceloop.sdk import Traceloop

from dotenv import load_dotenv

load_dotenv()


Traceloop.init(app_name="agno_discussion_team")


def run_discussion_team():
    """
    A multi-agent discussion team where different experts discuss
    and provide perspectives on a given topic.
    """

    technical_expert = Agent(
        name="Technical Expert",
        role="Provide technical perspective and implementation details",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        add_name_to_context=True,
        instructions=dedent(
            """
        You are a technical expert focused on implementation details.
        Analyze topics from a technical feasibility perspective.
        Discuss technical challenges, solutions, and best practices.
        Keep responses concise and focused.
        """
        ),
    )

    business_expert = Agent(
        name="Business Expert",
        role="Provide business perspective and practical implications",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        add_name_to_context=True,
        instructions=dedent(
            """
        You are a business expert focused on practical value.
        Analyze topics from a business impact and ROI perspective.
        Discuss market implications, costs, and business benefits.
        Keep responses concise and focused.
        """
        ),
    )

    user_experience_expert = Agent(
        name="UX Expert",
        role="Provide user experience perspective",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        add_name_to_context=True,
        instructions=dedent(
            """
        You are a user experience expert focused on usability.
        Analyze topics from an end-user perspective.
        Discuss user needs, pain points, and user satisfaction.
        Keep responses concise and focused.
        """
        ),
    )

    discussion_team = Team(
        name="Expert Discussion Team",
        delegate_task_to_all_members=True,
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        members=[
            technical_expert,
            business_expert,
            user_experience_expert,
        ],
        instructions=[
            "You are a discussion facilitator.",
            "Gather perspectives from all team members.",
            "Synthesize their viewpoints into a balanced conclusion.",
            "Stop the discussion when team consensus is reached.",
        ],
        markdown=True,
        show_members_responses=True,
    )

    print("Starting expert discussion team...\n")
    result = discussion_team.run(
        "Discuss the topic: 'Should we adopt AI-powered code review tools in our development workflow?'"
    )
    print(f"\n{'='*80}\nFinal Team Consensus:\n{'='*80}")
    print(result.content)

    return result


if __name__ == "__main__":
    run_discussion_team()
