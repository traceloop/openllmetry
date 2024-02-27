import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, insert
from llama_index.core import SQLDatabase
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI


@pytest.mark.vcr
def test_agents_and_tools(exporter):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

    agent.chat("What is 2 times 3?")

    spans = exporter.get_finished_spans()

    assert set(
        [
            "ReActAgent.agent",
            "FunctionTool.tool",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    agent_span = next(span for span in spans if span.name == "ReActAgent.agent")
    function_tool = next(span for span in spans if span.name == "FunctionTool.tool")
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert function_tool.parent.span_id == agent_span.context.span_id
    assert openai_span.parent.span_id == agent_span.context.span_id


@pytest.mark.vcr
def test_agent_with_query_tool(exporter):
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()

    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)

    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

    sql_database = SQLDatabase(engine, include_tables=["city_stats"])

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

    spans = exporter.get_finished_spans()

    assert set(
        [
            "OpenAIAssistantAgent.agent",
            "QueryEngineTool.tool",
            "synthesize.task",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.agent"
    )
    query_engine_tool = next(
        span for span in spans if span.name == "QueryEngineTool.tool"
    )
    synthesize_task = next(span for span in spans if span.name == "synthesize.task")

    assert query_engine_tool.parent.span_id == agent_span.context.span_id
    assert synthesize_task.parent.span_id == query_engine_tool.context.span_id
