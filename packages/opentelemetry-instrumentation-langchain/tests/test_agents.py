from typing import Tuple

import pytest
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Constant prompt template to replace hub.pull("hwchase17/openai-functions-agent")
OPENAI_FUNCTIONS_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


@pytest.mark.vcr
def test_agents(instrument_legacy, span_exporter, log_exporter):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = OPENAI_FUNCTIONS_AGENT_PROMPT

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_executor.invoke({"input": "What is OpenLLMetry?"})

    spans = span_exporter.get_finished_spans()

    assert set([span.name for span in spans]) == {
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
    }

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_agents_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = OPENAI_FUNCTIONS_AGENT_PROMPT

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    prompt = "What is OpenLLMetry?"
    response = agent_executor.invoke({"input": prompt})

    spans = span_exporter.get_finished_spans()

    assert set([span.name for span in spans]) == {
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
    }

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 8

    # Validate that the user message Event exists
    assert_message_in_logs(logs, "gen_ai.user.message", {"content": prompt})

    # validate that the system message Event exists
    assert_message_in_logs(
        logs, "gen_ai.system.message", {"content": "You are a helpful assistant"}
    )

    # Validate that the assistant message Event exists
    assert_message_in_logs(
        logs,
        "gen_ai.assistant.message",
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_RQXCf1bdiBiFrtwfOTP3GCCd",
                    "function": {
                        "name": "tavily_search_results_json",
                        "arguments": {"query": "OpenLLMetry"},
                    },
                    "type": "function",
                }
            ],
        },
    )

    # Validate that the ai calls the tool
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": ""},
        "tool_calls": [
            {
                "id": "call_RQXCf1bdiBiFrtwfOTP3GCCd",
                "function": {
                    "name": "tavily_search_results_json",
                    "arguments": {"query": "OpenLLMetry"},
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs, "gen_ai.choice", choice_event)

    # Validate that the final ai response exists
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["output"]},
    }
    assert_message_in_logs(logs, "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_agents_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = OPENAI_FUNCTIONS_AGENT_PROMPT

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_executor.invoke({"input": "What is OpenLLMetry?"})

    spans = span_exporter.get_finished_spans()

    assert set([span.name for span in spans]) == {
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
    }

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 8
    assert all(
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"
        for log in logs
    )

    # Validate that the user message Event exists
    assert_message_in_logs(logs, "gen_ai.user.message", {})

    # validate that the system message Event exists
    assert_message_in_logs(logs, "gen_ai.system.message", {})

    # Validate that the assistant message Event exists
    assert_message_in_logs(
        logs,
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "id": "call_Joct6NnIJFGGWfvMqZJPRB4C",
                    "function": {"name": "tavily_search_results_json"},
                    "type": "function",
                }
            ]
        },
    )

    # Validate that the ai calls the tool
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "id": "call_Joct6NnIJFGGWfvMqZJPRB4C",
                "function": {"name": "tavily_search_results_json"},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs, "gen_ai.choice", choice_event)

    # Validate that the final ai response exists
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs, "gen_ai.choice", choice_event)


def assert_message_in_logs(
    logs: Tuple[LogData], event_name: str, expected_content: dict
):
    assert any(
        log.log_record.event_name == event_name
        for log in logs
    )
    assert any(dict(log.log_record.body) == expected_content for log in logs)
