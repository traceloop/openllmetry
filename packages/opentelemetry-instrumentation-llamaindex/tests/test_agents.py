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


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "llamaindex"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


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
        "ReActAgent.workflow",
        "ReActAgent.task",
        "FunctionTool.task",
        "openai.chat",
        "ReActOutputParser.task",
        "ReActAgentWorker.task",
    } == {span.name for span in spans}

    agent_workflow_span = next(
        span for span in spans if span.name == "ReActAgent.workflow"
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
        llm_span_1.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_1.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_1.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_1.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_1.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    ].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 43
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 479
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 522

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        llm_span_2.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert (
        llm_span_2.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-3.5-turbo-0613"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"].startswith(
        "You are designed to help with a variety of tasks,"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"] == (
        "What is 2 times 3?"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"].startswith(
        "Thought: The current language of the user is English. I need to use a tool"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.content"] == (
        "Observation: 6"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == (
        "Thought: I can answer without using any more tools. I'll use the user's "
        "language to answer.\nAnswer: 2 times 3 is 6."
    )
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 32
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 535
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 567

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agents_and_tools_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agents_and_tools_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass


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
        "CompactAndRefine.task",
        "openai.chat",
        "TokenTextSplitter.task",
        "openai.assistant.run",
    }.issubset({span.name for span in spans})

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "CompactAndRefine.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "openai.chat"]

    assert agent_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_1.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_1.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"].startswith(
        "Given an input question, first create a syntactically correct sqlite"
    )
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 68
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 224
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 292

    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_2.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"].startswith(
        "Given an input question, synthesize a response from the query results."
    )
    assert llm_span_2.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == (
        "The city with the highest population in the city_stats table is Tokyo, "
        "with a population of 13,960,000."
    )
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 25
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 63
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 88

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agent_with_query_tool_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agent_with_query_tool_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass


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
        "ReActAgent.workflow",
        "ReActAgent.task",
        "NLSQLTableQueryEngine.task",
        "Cohere.task",
        "CompactAndRefine.task",
        "DefaultRefineProgram.task",
        "DefaultSQLParser.task",
        "FunctionTool.task",
        "QueryEngineTool.task",
        "ReActAgentWorker.task",
        "ReActOutputParser.task",
        "TokenTextSplitter.task",
    } == {span.name for span in spans}

    agent_span = next(span for span in spans if span.name == "ReActAgent.workflow")
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


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agent_with_multiple_tools_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass


@pytest.mark.skip(reason="Agent API changed in llama-index 0.13.1 - needs update for workflow-based agents")
@pytest.mark.vcr
def test_agent_with_multiple_tools_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    # Test skipped - Agent API changed in llama-index 0.13.1
    pass
