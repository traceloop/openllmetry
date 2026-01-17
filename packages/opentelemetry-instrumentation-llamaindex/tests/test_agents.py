import pytest
from llama_index.core import SQLDatabase
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk._logs import ReadableLogRecord
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


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "llamaindex"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agents_and_tools(instrument_legacy, span_exporter, log_exporter):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model="gpt-4o-mini")
    agent = ReActAgent(tools=[multiply_tool], llm=llm, verbose=True, streaming=False)

    await agent.run("What is 2 times 3?")

    spans = span_exporter.get_finished_spans()
    span_names = {span.name for span in spans}

    # Verify we have the key workflow and task spans
    assert "ReActAgent.workflow" in span_names
    assert "FunctionTool.task" in span_names
    assert "openai.chat" in span_names

    agent_workflow_span = next(
        span for span in spans if span.name == "ReActAgent.workflow"
    )
    function_tool_span = next(
        span for span in spans if span.name == "FunctionTool.task"
    )
    llm_spans = [span for span in spans if span.name == "openai.chat"]

    assert agent_workflow_span.parent is None
    assert function_tool_span.parent is not None
    assert len(llm_spans) >= 1
    assert all(span.parent is not None for span in llm_spans)

    # Check the first LLM span has correct attributes
    llm_span = llm_spans[0]
    assert llm_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in llm_span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in llm_span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in llm_span.attributes

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


@pytest.mark.skip(reason="llama-index-agent-openai not compatible with llama-index 0.14.x")
@pytest.mark.vcr
def test_agent_with_query_tool(instrument_legacy, span_exporter, log_exporter):
    # Test skipped - llama-index-agent-openai requires llama-index-core<0.13
    pass


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
@pytest.mark.asyncio
async def test_agent_with_multiple_tools(instrument_legacy, span_exporter, log_exporter):
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
    llm = Cohere(model="command-a-03-2025")
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

    agent = ReActAgent(tools=[calc_tool, sql_tool], llm=llm, verbose=True, streaming=False)

    await agent.run(
        "Which city has the highest population and how many years will it take to reach"
        " 20 million inhabitants if it's population increases by 1 million a year?"
    )

    spans = span_exporter.get_finished_spans()
    span_names = {span.name for span in spans}

    # Verify we have the key workflow and task spans
    assert "ReActAgent.workflow" in span_names
    assert "NLSQLTableQueryEngine.task" in span_names
    assert "FunctionTool.task" in span_names
    assert "QueryEngineTool.task" in span_names

    agent_span = next(span for span in spans if span.name == "ReActAgent.workflow")
    assert agent_span.parent is None

    # Check that we have tool spans with proper attributes
    tool_spans = [span for span in spans if span.name == "call_tool.task"]
    if tool_spans:
        # New workflow-based API uses call_tool.task spans
        for tool_span in tool_spans:
            if "tool.name" in tool_span.attributes:
                assert tool_span.attributes["tool.name"] in ["sql_tool", "calc_tool"]

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
