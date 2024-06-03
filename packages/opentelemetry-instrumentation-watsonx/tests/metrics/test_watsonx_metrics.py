import pytest
import sys


@pytest.mark.skipif(sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10")
@pytest.mark.vcr
def test_generate_metrics(metrics_test_context, watson_ai_model):
    if watson_ai_model is None:
        print("test_generate_metrics test skipped.")
        return

    provider, reader = metrics_test_context

    watson_ai_model.generate(prompt="What is 1 + 1?")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_response_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == "gen_ai.client.token.usage":
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.token.type"] in [
                            "output",
                            "input",
                        ]
                        assert data_point.value > 0

                if metric.name == "llm.watsonx.completions.responses":
                    found_response_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1

                if metric.name == "gen_ai.client.operation.duration":
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                assert metric.data.data_points[0].attributes["gen_ai.system"] == "watsonx"

    assert found_token_metric is True
    assert found_response_metric is True
    assert found_duration_metric is True


@pytest.mark.skipif(sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10")
@pytest.mark.vcr
def test_generate_stream_metrics(metrics_test_context, watson_ai_model):
    if watson_ai_model is None:
        print("test_generate_stream_metrics test skipped.")
        return

    provider, reader = metrics_test_context

    response = watson_ai_model.generate_text_stream(prompt="Write an epigram about the sun")
    generated_text = ""
    for chunk in response:
        generated_text += chunk

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_response_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == "gen_ai.client.token.usage":
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.token.type"] in [
                            "output",
                            "input",
                        ]
                        assert data_point.value > 0

                if metric.name == "llm.watsonx.completions.responses":
                    found_response_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1

                if metric.name == "gen_ai.client.operation.duration":
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                assert metric.data.data_points[0].attributes["gen_ai.system"] == "watsonx"

    assert found_token_metric is True
    assert found_response_metric is True
    assert found_duration_metric is True
