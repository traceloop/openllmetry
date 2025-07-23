from unittest.mock import patch
import pytest
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import Meters, SpanAttributes


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


def verify_token_metrics(data_points):
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
            "output",
            "input",
        ]
        assert data_point.sum > 0
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "Langchain"


def verify_duration_metrics(data_points):
    assert any(data_point.count > 0 for data_point in data_points)
    assert any(data_point.sum > 0 for data_point in data_points)
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "Langchain"


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
