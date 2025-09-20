from unittest.mock import patch
from typing import TypedDict
import pytest
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import Meters, SpanAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from langgraph.graph import StateGraph
from openai import OpenAI


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def llm():
    return ChatOpenAI(temperature=0)


@pytest.fixture
def chain(llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


@pytest.mark.vcr
def test_llm_chain_metrics(instrument_legacy, reader, chain):
    chain.run(product="colorful socks")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

    assert found_token_metric is True
    assert found_duration_metric is True


@pytest.mark.vcr
def test_llm_chain_streaming_metrics(instrument_legacy, reader, llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for _ in chain.stream({"product": "colorful socks"}):
        pass

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_choices_metric = False
    found_ttft_metric = False
    found_streaming_time_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choices_metric = True
                    assert any(
                        data_point.value >= 1 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_ttft_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_streaming_time_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

    assert found_token_metric is True
    assert found_duration_metric is True
    assert found_choices_metric is True
    assert found_ttft_metric is True
    assert found_streaming_time_metric is True
    # Note: TTFT and streaming time metrics may only be present with streaming responses


def verify_token_metrics(data_points):
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
            "output",
            "input",
        ]
        assert data_point.sum > 0
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"


def verify_duration_metrics(data_points):
    assert any(data_point.count > 0 for data_point in data_points)
    assert any(data_point.sum > 0 for data_point in data_points)
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"


def verify_langchain_metrics(reader):
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    verify_token_metrics(metric.data.data_points)

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    verify_duration_metrics(metric.data.data_points)

    return found_token_metric, found_duration_metric


@pytest.mark.vcr
def test_llm_chain_metrics_with_none_llm_output(instrument_legacy, reader, chain, llm):
    """
    This test verifies that the metrics system correctly handles edge cases where the
    LLM response contains a None value in the llm_output field, ensuring that token
    usage and operation duration metrics are still properly recorded.
    """
    original_generate = llm._generate

    # Create a patched version that returns results with None llm_output
    def patched_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        result.llm_output = None
        return result

    with patch.object(llm, '_generate', side_effect=patched_generate):
        chain.run(product="colorful socks")

    found_token_metric, found_duration_metric = verify_langchain_metrics(reader)

    assert found_token_metric is True, "Token usage metrics not found"
    assert found_duration_metric is True, "Operation duration metrics not found"


@pytest.mark.vcr
def test_langgraph_metrics(instrument_legacy, reader, openai_client):
    class State(TypedDict):
        request: str
        result: str

    def calculate(state: State):
        request = state["request"]
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mathematician."},
                {"role": "user", "content": request}
            ]
        )
        return {"result": completion.choices[0].message.content}
    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    langgraph.invoke(input={"request": user_request})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) == 1

    metric_data = resource_metrics[0].scope_metrics[-1].metrics
    assert len(metric_data) == 3

    token_usage_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_TOKEN_USAGE
        ),
        None,
    )
    assert token_usage_metric is not None
    token_usage_data_point = token_usage_metric.data.data_points[0]
    assert token_usage_data_point.sum > 0
    assert (
        token_usage_data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
        and token_usage_data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in ["input", "output"]
    )

    duration_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_OPERATION_DURATION
        ),
        None,
    )
    assert duration_metric is not None
    duration_data_point = duration_metric.data.data_points[0]
    assert duration_data_point.sum > 0
    assert duration_data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"

    generation_choices_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_GENERATION_CHOICES
        ),
        None
    )
    assert generation_choices_metric is not None
    generation_choices_data_points = generation_choices_metric.data.data_points
    for data_point in generation_choices_data_points:
        assert (
            data_point.attributes[SpanAttributes.LLM_SYSTEM]
            == "openai"
        )
        assert data_point.value > 0


@pytest.mark.vcr
def test_streaming_with_ttft_and_generation_time_metrics(instrument_legacy, reader):
    """Test streaming metrics with actual token-by-token response."""
    from langchain_core.prompts import ChatPromptTemplate

    # Use a model that supports streaming
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_template("Write a very short story about {topic}")
    chain = prompt | llm

    # Stream the response to trigger on_llm_new_token calls
    response_chunks = []
    for chunk in chain.stream({"topic": "a robot learning to paint"}):
        response_chunks.append(chunk)

    # Verify we got streaming chunks
    assert len(response_chunks) > 1, "Should have multiple chunks for streaming"

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_choices_metric = False
    found_ttft_metric = False
    found_streaming_time_metric = False
    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True

                elif metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                elif metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choices_metric = True
                    assert any(
                        data_point.value >= 1 for data_point in metric.data.data_points
                    )

                elif metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_ttft_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                elif metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_streaming_time_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )

                elif metric.name == "llm.langchain.completions.exceptions":
                    found_exception_metric = True

    # Basic metrics should always be present
    assert found_token_metric is True
    assert found_duration_metric is True
    assert found_choices_metric is True
    assert found_ttft_metric is True
    assert found_streaming_time_metric is True
    assert found_exception_metric is True

    # Streaming-specific metrics should be present with actual streaming
    # Note: These might not appear if the LLM doesn't actually stream tokens individually
    # This depends on the model and provider implementation


def test_exception_metrics(instrument_legacy, reader):
    """Test that exception metrics are recorded when LLM calls fail."""
    from unittest.mock import patch

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
    )

    # Mock the LLM to raise an exception
    with patch.object(llm, '_generate', side_effect=Exception("API Error")):
        try:
            chain.run(product="test")
        except Exception:
            pass  # Expected to fail

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "llm.langchain.completions.exceptions":
                    found_exception_metric = True
                    assert any(
                        data_point.value >= 1 for data_point in metric.data.data_points
                    )
                    # Check that error attributes are set
                    for data_point in metric.data.data_points:
                        assert "error.type" in data_point.attributes or ERROR_TYPE in data_point.attributes

    assert found_exception_metric is True
