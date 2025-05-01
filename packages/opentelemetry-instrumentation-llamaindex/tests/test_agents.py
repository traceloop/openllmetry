import json

import pytest
from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.core import SQLDatabase
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert


def make_sql_table():
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()
    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    rows = [
        {"city": "Toronto", "population": 2930000, "country": "Canada"},
        {"city": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city": "Berlin", "population": 3645000, "country": "Germany"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)
    return SQLDatabase(engine, include_tables=["city_stats"])


@pytest.mark.vcr
def test_agents_and_tools(instrument_legacy, span_exporter, log_exporter):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

    agent.chat("What is 2 times 3?")

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "FunctionTool.task",
        "openai.chat",
        "ReActOutputParser.task",
        "ReActAgentWorker.task",
    } == {span.name for span in spans}

    agent_workflow_span = next(
        span for span in spans if span.name == "AgentRunner.workflow"
    )
    function_tool_span = next(
        span for span in spans if span.name == "FunctionTool.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_workflow_span.parent is None
    assert function_tool_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_1.attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
    ].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 43
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 479
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 522

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == (
        "Observation: 6"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Thought: I can answer without using any more tools. I'll use the user's "
        "language to answer.\nAnswer: 2 times 3 is 6."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 32
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 535
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 567

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_agents_and_tools_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

    agent.chat("What is 2 times 3?")

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "FunctionTool.task",
        "openai.chat",
        "ReActOutputParser.task",
        "ReActAgentWorker.task",
    } == {span.name for span in spans}

    agent_workflow_span = next(
        span for span in spans if span.name == "AgentRunner.workflow"
    )
    function_tool_span = next(
        span for span in spans if span.name == "FunctionTool.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_workflow_span.parent is None
    assert function_tool_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_1.attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
    ].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 43
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 479
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 522

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == (
        "Observation: 6"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Thought: I can answer without using any more tools. I'll use the user's "
        "language to answer.\nAnswer: 2 times 3 is 6."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 32
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 535
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 567

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 8
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": 'You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.\n\n## Tools\n\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\nThis may require breaking the task into subtasks and using different tools to complete each subtask.\n\nYou have access to the following tools:\n> Tool Name: multiply\nTool Description: multiply(a: int, b: int) -> int\nMultiply two integers and returns the result integer\nTool Args: {"properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}, "required": ["a", "b"], "type": "object"}\n\n\n\n## Output Format\n\nPlease answer in the same language as the question and use the following format:\n\n```\nThought: The current language of the user is: (user\'s language). I need to use a tool to help me answer the question.\nAction: tool name (one of multiply) if using a tool.\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})\n```\n\nPlease ALWAYS start with a Thought.\n\nNEVER surround your response with markdown code markers. You may use code markers within your response if you need to.\n\nPlease use a valid JSON format for the Action Input. Do NOT do this {\'input\': \'hello world\', \'num_beams\': 5}.\n\nIf this format is used, the tool will respond in the following format:\n\n```\nObservation: tool response\n```\n\nYou should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:\n\n```\nThought: I can answer without using any more tools. I\'ll use the user\'s language to answer\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n```\nThought: I cannot answer the question with the provided tools.\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n## Current Conversation\n\nBelow is the current conversation consisting of interleaving human and assistant messages.\n',
            "role": "system",
        },
    )
    assert_message_in_logs(
        logs[1],
        "gen_ai.user.message",
        {"content": "What is 2 times 3?", "role": "user"},
    )
    assert_message_in_logs(
        logs[2],
        "gen_ai.choice",
        {
            "index": 0,
            "finish_reason": "unknown",
            "message": {
                "content": 'Thought: The current language of the user is English. I need to use a tool to help me answer the question.\nAction: multiply\nAction Input: {"a": 2, "b": 3}',
                "role": "assistant",
            },
        },
    )
    assert_message_in_logs(
        logs[3],
        "gen_ai.system.message",
        {
            "content": 'You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.\n\n## Tools\n\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\nThis may require breaking the task into subtasks and using different tools to complete each subtask.\n\nYou have access to the following tools:\n> Tool Name: multiply\nTool Description: multiply(a: int, b: int) -> int\nMultiply two integers and returns the result integer\nTool Args: {"properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}, "required": ["a", "b"], "type": "object"}\n\n\n\n## Output Format\n\nPlease answer in the same language as the question and use the following format:\n\n```\nThought: The current language of the user is: (user\'s language). I need to use a tool to help me answer the question.\nAction: tool name (one of multiply) if using a tool.\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})\n```\n\nPlease ALWAYS start with a Thought.\n\nNEVER surround your response with markdown code markers. You may use code markers within your response if you need to.\n\nPlease use a valid JSON format for the Action Input. Do NOT do this {\'input\': \'hello world\', \'num_beams\': 5}.\n\nIf this format is used, the tool will respond in the following format:\n\n```\nObservation: tool response\n```\n\nYou should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:\n\n```\nThought: I can answer without using any more tools. I\'ll use the user\'s language to answer\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n```\nThought: I cannot answer the question with the provided tools.\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n## Current Conversation\n\nBelow is the current conversation consisting of interleaving human and assistant messages.\n',
            "role": "system",
        },
    )
    assert_message_in_logs(
        logs[4],
        "gen_ai.user.message",
        {"content": "What is 2 times 3?", "role": "user"},
    )
    assert_message_in_logs(
        logs[5],
        "gen_ai.assistant.message",
        {
            "content": "Thought: The current language of the user is English. I need to use a tool to help me answer the question.\nAction: multiply\nAction Input: {'a': 2, 'b': 3}",
            "role": "assistant",
        },
    )
    assert_message_in_logs(
        logs[6], "gen_ai.user.message", {"content": "Observation: 6", "role": "user"}
    )
    assert_message_in_logs(
        logs[7],
        "gen_ai.choice",
        {
            "index": 0,
            "finish_reason": "unknown",
            "message": {
                "content": "Thought: I can answer without using any more tools. I'll use the user's language to answer.\nAnswer: 2 times 3 is 6.",
                "role": "assistant",
            },
        },
    )


@pytest.mark.vcr
def test_agents_and_tools_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

    agent.chat("What is 2 times 3?")

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "FunctionTool.task",
        "openai.chat",
        "ReActOutputParser.task",
        "ReActAgentWorker.task",
    } == {span.name for span in spans}

    agent_workflow_span = next(
        span for span in spans if span.name == "AgentRunner.workflow"
    )
    function_tool_span = next(
        span for span in spans if span.name == "FunctionTool.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_workflow_span.parent is None
    assert function_tool_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_1.attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
    ].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 43
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 479
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 522

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == (
        "Observation: 6"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Thought: I can answer without using any more tools. I'll use the user's "
        "language to answer.\nAnswer: 2 times 3 is 6."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 32
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 535
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 567

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 8

    assert_message_in_logs(logs[0], "gen_ai.system.message", {})
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})
    assert_message_in_logs(
        logs[2],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "unknown", "message": {}},
    )
    assert_message_in_logs(logs[3], "gen_ai.system.message", {})
    assert_message_in_logs(logs[4], "gen_ai.user.message", {})
    assert_message_in_logs(logs[5], "gen_ai.assistant.message", {})
    assert_message_in_logs(logs[6], "gen_ai.user.message", {})
    assert_message_in_logs(
        logs[7],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "unknown", "message": {}},
    )


@pytest.mark.vcr
def test_agent_with_query_tool(instrument_legacy, span_exporter, log_exporter):
    sql_database = make_sql_table()

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )

    agent = OpenAIAssistantAgent.from_new(
        name="City bot",
        instructions="You are a bot designed to answer questions about cities (both unstructured and structured data)",
        tools=[sql_tool],
        verbose=True,
    )

    agent.chat("Which city has the highest population?")

    spans = span_exporter.get_finished_spans()

    assert {
        "OpenAIAssistantAgent.workflow",
        "BaseSynthesizer.task",
        "LLM.task",
        "openai.chat",
        "TokenTextSplitter.task",
    }.issubset({span.name for span in spans})

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, first create a syntactically correct sqlite"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 68
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 224
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 292

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, synthesize a response from the query results."
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "The city with the highest population in the city_stats table is Tokyo, "
        "with a population of 13,960,000."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 25
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 63
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 88

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_agent_with_query_tool_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    sql_database = make_sql_table()

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )

    agent = OpenAIAssistantAgent.from_new(
        name="City bot",
        instructions="You are a bot designed to answer questions about cities (both unstructured and structured data)",
        tools=[sql_tool],
        verbose=True,
    )

    agent.chat("Which city has the highest population?")

    spans = span_exporter.get_finished_spans()

    assert {
        "OpenAIAssistantAgent.workflow",
        "BaseSynthesizer.task",
        "LLM.task",
        "openai.chat",
        "TokenTextSplitter.task",
    }.issubset({span.name for span in spans})

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, first create a syntactically correct sqlite"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 68
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 224
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 292

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, synthesize a response from the query results."
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "The city with the highest population in the city_stats table is Tokyo, "
        "with a population of 13,960,000."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 25
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 63
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 88

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": "Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.\n\nNever query for all the columns from a specific table, only ask for a few relevant columns given the question.\n\nPay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:\n\nQuestion: Question here\nSQLQuery: SQL Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here\n\nOnly use tables listed below.\nTable 'city_stats' has columns: city (VARCHAR(16)), population (INTEGER), country (VARCHAR(16)), .\n\nQuestion: SELECT city, MAX(population) FROM city_stats\nSQLQuery: ",
            "role": "user",
        },
    )
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {
            "index": 0,
            "finish_reason": "unknown",
            "message": {
                "content": "SELECT city_name, MAX(population) FROM city_stats\nSQLResult: \ncity_name | MAX(population)\nParis     | 2245000\nTokyo     | 13929286\nNew York  | 8175133\nAnswer: The city with the highest population is Tokyo with 13,929,286 people.",
                "role": "assistant",
            },
        },
    )
    assert_message_in_logs(
        logs[2],
        "gen_ai.user.message",
        {
            "content": "Given an input question, synthesize a response from the query results.\nQuery: SELECT city, MAX(population) FROM city_stats\nSQL: SELECT city_name, MAX(population) FROM city_stats\nSQL Response: Error: Statement 'SELECT city_name, MAX(population) FROM city_stats' is invalid SQL.\nError: no such column: city_name\nResponse: ",
            "role": "user",
        },
    )
    assert_message_in_logs(
        logs[3],
        "gen_ai.choice",
        {
            "index": 0,
            "finish_reason": "unknown",
            "message": {
                "content": "The city with the highest population in the city_stats table is Tokyo, with a population of 13,960,000.",
                "role": "assistant",
            },
        },
    )


@pytest.mark.vcr
def test_agent_with_query_tool_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    sql_database = make_sql_table()

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )

    agent = OpenAIAssistantAgent.from_new(
        name="City bot",
        instructions="You are a bot designed to answer questions about cities (both unstructured and structured data)",
        tools=[sql_tool],
        verbose=True,
    )

    agent.chat("Which city has the highest population?")

    spans = span_exporter.get_finished_spans()

    assert {
        "OpenAIAssistantAgent.workflow",
        "BaseSynthesizer.task",
        "LLM.task",
        "openai.chat",
        "TokenTextSplitter.task",
    }.issubset({span.name for span in spans})

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, first create a syntactically correct sqlite"
    )
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 68
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 224
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 292

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "Given an input question, synthesize a response from the query results."
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "The city with the highest population in the city_stats table is Tokyo, "
        "with a population of 13,960,000."
    )
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 25
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 63
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 88

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "unknown", "message": {}},
    )
    assert_message_in_logs(logs[2], "gen_ai.user.message", {})
    assert_message_in_logs(
        logs[3],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "unknown", "message": {}},
    )


@pytest.mark.vcr
def test_agent_with_multiple_tools(instrument_legacy, span_exporter, log_exporter):
    def calculate_years_to_target_population(
        target_population: int,
        current_population: int,
        yearly_increase: int,
    ) -> int:
        return round((target_population - current_population) / yearly_increase)

    calc_tool = FunctionTool.from_defaults(
        fn=calculate_years_to_target_population,
        name="calc_tool",
        description=(
            "Useful for calculating the number of years until a city reaches a target population."
        ),
    )

    sql_database = make_sql_table()
    llm = Cohere()
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
        llm=llm,
    )
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table which contains the names of cities, together with their population"
            " and country"
        ),
    )

    agent = ReActAgent.from_tools(tools=[calc_tool, sql_tool], llm=llm, verbose=True)

    agent.chat(
        "Which city has the highest population and how many years will it take to reach"
        " 20 million inhabitants if it's population increases by 1 million a year?"
    )

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "BaseQueryEngine.task",
        "BaseSQLTableQueryEngine.task",
        "BaseSynthesizer.task",
        "Cohere.task",
        "CompactAndRefine.task",
        "DefaultRefineProgram.task",
        "DefaultSQLParser.task",
        "FunctionTool.task",
        "LLM.task",
        "QueryEngineTool.task",
        "ReActAgentWorker.task",
        "ReActOutputParser.task",
        "Refine.task",
        "TokenTextSplitter.task",
    } == {span.name for span in spans}

    agent_span = next(span for span in spans if span.name == "AgentRunner.workflow")
    _, sql_tool_span, calc_tool_span, _, _ = [
        span for span in spans if span.name == "ReActAgentWorker.task"
    ]

    assert agent_span.parent is None

    assert sql_tool_span.attributes["tool.name"] == "sql_tool"
    assert json.loads(sql_tool_span.attributes["tool.arguments"]) == {
        "input": "SELECT city, population FROM table ORDER BY population DESC LIMIT 1"
    }
    assert calc_tool_span.attributes["tool.name"] == "calc_tool"
    assert json.loads(calc_tool_span.attributes["tool.arguments"]) == {
        "current_population": 13960000,
        "target_population": 20000000,
        "yearly_increase": 1000000,
    }

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_agent_with_multiple_tools_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    def calculate_years_to_target_population(
        target_population: int,
        current_population: int,
        yearly_increase: int,
    ) -> int:
        return round((target_population - current_population) / yearly_increase)

    calc_tool = FunctionTool.from_defaults(
        fn=calculate_years_to_target_population,
        name="calc_tool",
        description=(
            "Useful for calculating the number of years until a city reaches a target population."
        ),
    )

    sql_database = make_sql_table()
    llm = Cohere()
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
        llm=llm,
    )
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table which contains the names of cities, together with their population"
            " and country"
        ),
    )

    agent = ReActAgent.from_tools(tools=[calc_tool, sql_tool], llm=llm, verbose=True)

    agent.chat(
        "Which city has the highest population and how many years will it take to reach"
        " 20 million inhabitants if it's population increases by 1 million a year?"
    )

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "BaseQueryEngine.task",
        "BaseSQLTableQueryEngine.task",
        "BaseSynthesizer.task",
        "Cohere.task",
        "CompactAndRefine.task",
        "DefaultRefineProgram.task",
        "DefaultSQLParser.task",
        "FunctionTool.task",
        "LLM.task",
        "QueryEngineTool.task",
        "ReActAgentWorker.task",
        "ReActOutputParser.task",
        "Refine.task",
        "TokenTextSplitter.task",
    } == {span.name for span in spans}

    agent_span = next(span for span in spans if span.name == "AgentRunner.workflow")
    _, sql_tool_span, calc_tool_span, _, _ = [
        span for span in spans if span.name == "ReActAgentWorker.task"
    ]

    assert agent_span.parent is None

    assert sql_tool_span.attributes["tool.name"] == "sql_tool"
    assert json.loads(sql_tool_span.attributes["tool.arguments"]) == {
        "input": "SELECT city, population FROM table ORDER BY population DESC LIMIT 1"
    }
    assert calc_tool_span.attributes["tool.name"] == "calc_tool"
    assert json.loads(calc_tool_span.attributes["tool.arguments"]) == {
        "current_population": 13960000,
        "target_population": 20000000,
        "yearly_increase": 1000000,
    }

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 19


@pytest.mark.vcr
def test_agent_with_multiple_tools_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    def calculate_years_to_target_population(
        target_population: int,
        current_population: int,
        yearly_increase: int,
    ) -> int:
        return round((target_population - current_population) / yearly_increase)

    calc_tool = FunctionTool.from_defaults(
        fn=calculate_years_to_target_population,
        name="calc_tool",
        description=(
            "Useful for calculating the number of years until a city reaches a target population."
        ),
    )

    sql_database = make_sql_table()
    llm = Cohere()
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
        llm=llm,
    )
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table which contains the names of cities, together with their population"
            " and country"
        ),
    )

    agent = ReActAgent.from_tools(tools=[calc_tool, sql_tool], llm=llm, verbose=True)

    agent.chat(
        "Which city has the highest population and how many years will it take to reach"
        " 20 million inhabitants if it's population increases by 1 million a year?"
    )

    spans = span_exporter.get_finished_spans()

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "BaseQueryEngine.task",
        "BaseSQLTableQueryEngine.task",
        "BaseSynthesizer.task",
        "Cohere.task",
        "CompactAndRefine.task",
        "DefaultRefineProgram.task",
        "DefaultSQLParser.task",
        "FunctionTool.task",
        "LLM.task",
        "QueryEngineTool.task",
        "ReActAgentWorker.task",
        "ReActOutputParser.task",
        "Refine.task",
        "TokenTextSplitter.task",
    } == {span.name for span in spans}

    agent_span = next(span for span in spans if span.name == "AgentRunner.workflow")
    _, sql_tool_span, calc_tool_span, _, _ = [
        span for span in spans if span.name == "ReActAgentWorker.task"
    ]

    assert agent_span.parent is None

    assert sql_tool_span.attributes["tool.name"] == "sql_tool"
    assert json.loads(sql_tool_span.attributes["tool.arguments"]) == {
        "input": "SELECT city, population FROM table ORDER BY population DESC LIMIT 1"
    }
    assert calc_tool_span.attributes["tool.name"] == "calc_tool"
    assert json.loads(calc_tool_span.attributes["tool.arguments"]) == {
        "current_population": 13960000,
        "target_population": 20000000,
        "yearly_increase": 1000000,
    }

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 19

    assert all(
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "llamaindex"
        for log in logs
    )
    assert all(
        log.log_record.body == {}
        or log.log_record.body
        == {"index": 0, "finish_reason": "unknown", "message": {}}
        for log in logs
    )


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "llamaindex"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
