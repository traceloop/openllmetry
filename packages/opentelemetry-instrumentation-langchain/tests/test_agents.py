import os
from typing import Tuple

import pytest
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_agents(instrument_legacy, span_exporter, log_exporter):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = hub.pull(
        "hwchase17/openai-functions-agent",
        api_key=os.environ["LANGSMITH_API_KEY"],
    )

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

    prompt = hub.pull(
        "hwchase17/openai-functions-agent",
        api_key=os.environ["LANGSMITH_API_KEY"],
    )

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

    # Validate that the assistant message Event exists (with flexible tool call ID)
    assistant_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.assistant.message"]
    assert len(assistant_logs) > 0, "Should have at least one gen_ai.assistant.message log"
    assistant_log = assistant_logs[0]
    assistant_body = dict(assistant_log.log_record.body)
    assert assistant_body["content"] == ""
    assert len(assistant_body["tool_calls"]) == 1
    tool_call = assistant_body["tool_calls"][0]
    assert tool_call["function"]["name"] == "tavily_search_results_json"
    assert tool_call["function"]["arguments"] == {"query": "OpenLLMetry"}
    assert tool_call["type"] == "function"
    assert tool_call["id"].startswith("call_")  # Tool call ID is random, just check prefix

    # Validate that the ai calls the tool (with flexible tool call ID)
    choice_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.choice" and dict(log.log_record.body).get("finish_reason") == "tool_calls"]
    assert len(choice_logs) > 0, "Should have at least one gen_ai.choice log with tool_calls"
    choice_log = choice_logs[0]
    choice_body = dict(choice_log.log_record.body)
    assert choice_body["index"] == 0
    assert choice_body["finish_reason"] == "tool_calls"
    assert choice_body["message"]["content"] == ""
    assert len(choice_body["tool_calls"]) == 1
    tool_call = choice_body["tool_calls"][0]
    assert tool_call["function"]["name"] == "tavily_search_results_json"
    assert tool_call["function"]["arguments"] == {"query": "OpenLLMetry"}
    assert tool_call["type"] == "function"
    assert tool_call["id"].startswith("call_")  # Tool call ID is random

    # Validate that the final ai response exists
    final_choice_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.choice" and dict(log.log_record.body).get("finish_reason") == "stop"]
    assert len(final_choice_logs) > 0, "Should have at least one gen_ai.choice log with stop"
    final_choice_log = final_choice_logs[0]
    final_choice_body = dict(final_choice_log.log_record.body)
    assert final_choice_body["index"] == 0
    assert final_choice_body["finish_reason"] == "stop"
    assert final_choice_body["message"]["content"] == response["output"]


@pytest.mark.vcr
def test_agents_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = hub.pull(
        "hwchase17/openai-functions-agent",
        api_key=os.environ["LANGSMITH_API_KEY"],
    )

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

    # Validate that the assistant message Event exists (with flexible tool call ID, no content mode)
    assistant_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.assistant.message"]
    assert len(assistant_logs) > 0, "Should have at least one gen_ai.assistant.message log"
    assistant_log = assistant_logs[0]
    assistant_body = dict(assistant_log.log_record.body)
    # In no content mode, content field may not be present
    assert len(assistant_body["tool_calls"]) == 1
    tool_call = assistant_body["tool_calls"][0]
    assert tool_call["function"]["name"] == "tavily_search_results_json"
    assert tool_call["type"] == "function"
    assert tool_call["id"].startswith("call_")  # Tool call ID is random

    # Validate that the ai calls the tool (with flexible tool call ID, no content mode)
    choice_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.choice" and dict(log.log_record.body).get("finish_reason") == "tool_calls"]
    assert len(choice_logs) > 0, "Should have at least one gen_ai.choice log with tool_calls"
    choice_log = choice_logs[0]
    choice_body = dict(choice_log.log_record.body)
    assert choice_body["index"] == 0
    assert choice_body["finish_reason"] == "tool_calls"
    assert "message" in choice_body
    assert len(choice_body["tool_calls"]) == 1
    tool_call = choice_body["tool_calls"][0]
    assert tool_call["function"]["name"] == "tavily_search_results_json"
    assert tool_call["type"] == "function"
    assert tool_call["id"].startswith("call_")  # Tool call ID is random

    # Validate that the final ai response exists (no content mode)
    final_choice_logs = [log for log in logs if log.log_record.attributes.get('event.name') == "gen_ai.choice" and dict(log.log_record.body).get("finish_reason") == "stop"]
    assert len(final_choice_logs) > 0, "Should have at least one gen_ai.choice log with stop"
    final_choice_log = final_choice_logs[0]
    final_choice_body = dict(final_choice_log.log_record.body)
    assert final_choice_body["index"] == 0
    assert final_choice_body["finish_reason"] == "stop"
    assert "message" in final_choice_body


def assert_message_in_logs(
    logs: Tuple[LogData], event_name: str, expected_content: dict
):
    assert any(
        log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
        for log in logs
    )
    assert any(dict(log.log_record.body) == expected_content for log in logs)
