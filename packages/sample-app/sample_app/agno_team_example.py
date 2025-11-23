import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow


Traceloop.init(app_name="agno_team_example")


@workflow(name="agno_team_example")
def run_team():
    researcher = Agent(
        name="Researcher",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        description="Gathers information and does research",
        instructions=["Research topics thoroughly", "Provide factual information"],
    )

    writer = Agent(
        name="Writer",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        description="Writes content based on research",
        instructions=[
            "Write clear and engaging content",
            "Use information from the researcher",
        ],
    )

    team = Team(
        name="ContentTeam",
        members=[researcher, writer],
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        instructions=[
            "First use the Researcher to gather information",
            "Then use the Writer to create engaging content",
        ],
    )

    print("Running multi-agent team...")
    result = team.run("Create a short article about OpenTelemetry observability")
    print(f"\nTeam response: {result.content}")

    return result


if __name__ == "__main__":
    run_team()
