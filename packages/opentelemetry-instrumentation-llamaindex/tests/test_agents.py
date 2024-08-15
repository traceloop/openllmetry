import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, insert
from llama_index.core import SQLDatabase
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI
from opentelemetry.semconv_ai import SpanAttributes


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

    assert {
        "AgentRunner.workflow",
        "AgentRunner.task",
        "FunctionTool.task",
        "OpenAI.task",
        "ReActOutputParser.task",
        "ReActAgentWorker.task",
    } == {span.name for span in spans}

    agent_workflow_span = next(
        span for span in spans if span.name == "AgentRunner.workflow"
    )
    function_tool_span = next(
        span for span in spans if span.name == "FunctionTool.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "OpenAI.task"]

    assert agent_workflow_span.parent is None
    assert function_tool_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None

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

    assert {
        "OpenAIAssistantAgent.workflow",
        "BaseSynthesizer.task",
        "LLM.task",
        "OpenAI.task",
        "TokenTextSplitter.task",
    }.issubset({span.name for span in spans})

    agent_span = next(
        span for span in spans if span.name == "OpenAIAssistantAgent.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "OpenAI.task"]

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
