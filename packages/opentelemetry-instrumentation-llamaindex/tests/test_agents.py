import pytest
from llama_index.core import SQLDatabase
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
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

    # Verify we have the key workflow and task spans (some span names changed in llama-index 0.14.x)
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
    # In llama-index 0.14.x, there may be multiple LLM spans depending on the agent's reasoning
    assert len(llm_spans) >= 1
    assert all(span.parent is not None for span in llm_spans)

    # Check the first LLM span has correct attributes (same fields as before, values may differ)
    llm_span_1 = llm_spans[0]
    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in llm_span_1.attributes
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in llm_span_1.attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in llm_span_1.attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.1.content" in llm_span_1.attributes
    assert llm_span_1.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"] == (
        "What is 2 times 3?"
    )
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" in llm_span_1.attributes
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 43
    assert llm_span_1.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 479
    assert llm_span_1.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 522

    # Verify second LLM span
    assert len(llm_spans) >= 2, "Expected at least 2 LLM spans"
    llm_span_2 = llm_spans[1]
    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in llm_span_2.attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" in llm_span_2.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" in llm_span_2.attributes
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 32
    assert llm_span_2.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 535
    assert llm_span_2.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 567

    # Verify tool.name and tool.arguments are set on call_tool spans
    call_tool_spans = [span for span in spans if span.name == "call_tool.task"]
    assert len(call_tool_spans) >= 1, "Expected at least one call_tool.task span"
    for tool_span in call_tool_spans:
        assert (
            "tool.name" in tool_span.attributes
        ), "Expected tool.name attribute on call_tool span"
        assert tool_span.attributes["tool.name"] == "multiply"
        assert (
            "tool.arguments" in tool_span.attributes
        ), "Expected tool.arguments attribute on call_tool span"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_with_multiple_tools(
    instrument_legacy, span_exporter, log_exporter
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

    agent = ReActAgent(
        tools=[calc_tool, sql_tool], llm=llm, verbose=True, streaming=False
    )

    await agent.run(
        "Which city has the highest population and how many years will it take to reach"
        " 20 million inhabitants if it's population increases by 1 million a year?"
    )

    spans = span_exporter.get_finished_spans()
    span_names = {span.name for span in spans}

    # Verify we have the key workflow and task spans (some names changed in llama-index 0.14.x)
    assert "ReActAgent.workflow" in span_names
    assert "NLSQLTableQueryEngine.task" in span_names
    assert "FunctionTool.task" in span_names
    assert "QueryEngineTool.task" in span_names

    # These spans should exist from the SQL query workflow
    task_span_names = {
        "CompactAndRefine.task",
        "TokenTextSplitter.task",
        "DefaultSQLParser.task",
    }
    assert (
        len(task_span_names & span_names) > 0
    ), "Expected at least one task span from the workflow"

    agent_span = next(span for span in spans if span.name == "ReActAgent.workflow")
    assert agent_span.parent is None

    # Verify Cohere LLM spans have the expected gen_ai attributes (same fields as before)
    cohere_spans = [span for span in spans if span.name == "Cohere.task"]
    assert len(cohere_spans) >= 1, "Expected at least one Cohere LLM span"

    # In llama-index 0.14.x, there are two types of Cohere.task spans:
    # 1. LLM call spans with gen_ai.request.model, gen_ai.prompt.X.content, gen_ai.completion.X.content
    # 2. Text processing spans with gen_ai.completion.content only
    # We verify that at least one span has the full set of LLM attributes
    llm_spans_with_model = [
        span
        for span in cohere_spans
        if GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
        or "gen_ai.request.model" in span.attributes
    ]
    assert (
        len(llm_spans_with_model) >= 1
    ), "Expected at least one Cohere span with gen_ai.request.model"

    # Check that LLM spans with gen_ai.request.model have the expected attributes
    for cohere_span in llm_spans_with_model:
        # Check for gen_ai.request.model attribute
        assert (
            GenAIAttributes.GEN_AI_REQUEST_MODEL in cohere_span.attributes
            or "gen_ai.request.model" in cohere_span.attributes
        ), f"Expected gen_ai.request.model in {cohere_span.name}"

        # Check for prompt content attributes (gen_ai.prompt.X.content)
        prompt_keys = [
            k for k in cohere_span.attributes if k.startswith("gen_ai.prompt.")
        ]
        assert len(prompt_keys) > 0, f"Expected prompt attributes in {cohere_span.name}"

        # Check for completion content attributes (gen_ai.completion.X.content)
        completion_keys = [
            k for k in cohere_span.attributes if k.startswith("gen_ai.completion")
        ]
        assert (
            len(completion_keys) > 0
        ), f"Expected completion attributes in {cohere_span.name}"

        # Check that llm.request.type exists
        assert (
            SpanAttributes.LLM_REQUEST_TYPE in cohere_span.attributes
            or "llm.request.type" in cohere_span.attributes
        ), f"Expected llm.request.type in {cohere_span.name}"

        # Check for token usage attributes (restored with Cohere format support)
        assert (
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in cohere_span.attributes
        ), f"Expected gen_ai.usage.output_tokens in {cohere_span.name}"
        assert cohere_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
        assert (
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in cohere_span.attributes
        ), f"Expected gen_ai.usage.input_tokens in {cohere_span.name}"
        assert cohere_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
        assert (
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS in cohere_span.attributes
        ), f"Expected llm.usage.total_tokens in {cohere_span.name}"
        assert cohere_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0

    # Verify tool-related spans exist (FunctionTool.task and QueryEngineTool.task)
    function_tool_spans = [span for span in spans if span.name == "FunctionTool.task"]
    query_engine_tool_spans = [
        span for span in spans if span.name == "QueryEngineTool.task"
    ]

    assert (
        len(function_tool_spans) >= 1
    ), "Expected at least one FunctionTool.task span (calc_tool)"
    assert (
        len(query_engine_tool_spans) >= 1
    ), "Expected at least one QueryEngineTool.task span (sql_tool)"

    # Check that task spans have traceloop.entity.name attribute
    for span in function_tool_spans + query_engine_tool_spans:
        assert (
            "traceloop.entity.name" in span.attributes
        ), f"Expected traceloop.entity.name in {span.name}"

    # Verify tool.name and tool.arguments are set on call_tool spans
    call_tool_spans = [span for span in spans if span.name == "call_tool.task"]
    assert (
        len(call_tool_spans) >= 2
    ), "Expected at least 2 call_tool.task spans (sql_tool and calc_tool)"

    tool_names_found = set()
    for tool_span in call_tool_spans:
        assert (
            "tool.name" in tool_span.attributes
        ), "Expected tool.name attribute on call_tool span"
        tool_name = tool_span.attributes["tool.name"]
        tool_names_found.add(tool_name)
        assert tool_name in [
            "sql_tool",
            "calc_tool",
        ], f"Unexpected tool name: {tool_name}"
        assert (
            "tool.arguments" in tool_span.attributes
        ), "Expected tool.arguments attribute on call_tool span"
        # tool.arguments should be a JSON string
        assert isinstance(tool_span.attributes["tool.arguments"], str)

    # Verify both tools were called
    assert "sql_tool" in tool_names_found, "Expected sql_tool to be called"
    assert "calc_tool" in tool_names_found, "Expected calc_tool to be called"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"
