from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMMathChain
from langchain_community.llms.openai import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper


from traceloop.sdk import Traceloop

Traceloop.init(app_name="langchain_agent")


def langchain_app():
    llm = OpenAI(temperature=0, streaming=True)
    search = DuckDuckGoSearchAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about "
            + "current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]

    # Initialize agent
    mrkl = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    print(
        mrkl.run(
            "What is the full name of the artist who recently released an album called "
            + "'The Storm Before the Calm' and are they in the FooBar database? "
            + "If so, what albums of theirs are in the FooBar database?"
        )
    )


langchain_app()
