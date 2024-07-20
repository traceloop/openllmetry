import pytest
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI


@pytest.mark.vcr
def test_agents(exporter):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_executor.invoke({"input": "What is OpenLLMetry?"})

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "RunnableLambda.task",
        "RunnableParallel<agent_scratchpad>.task",
        "RunnableAssign<agent_scratchpad>.task",
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "ToolsAgentOutputParser.task",
        "RunnableSequence.task",
        "tavily_search_results_json.tool",
        "RunnableLambda.task",
        "RunnableParallel<agent_scratchpad>.task",
        "RunnableAssign<agent_scratchpad>.task",
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "ToolsAgentOutputParser.task",
        "RunnableSequence.task",
        "AgentExecutor.workflow",
    ]
