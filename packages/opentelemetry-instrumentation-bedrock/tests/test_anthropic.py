import pytest
import json


@pytest.mark.vcr
def test_anthropic_2_completion(exporter, brt):
    body = json.dumps(
        {
            "prompt": "Human: Tell me a joke about opentelemetry Assistant:",
            "max_tokens_to_sample": 200,
            "temperature": 0.5,
        }
    )

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-v2:1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("completion")

    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes["llm.prompts.0.user"]
        == "Human: Tell me a joke about opentelemetry Assistant:"
    )
    assert anthropic_span.attributes.get("llm.completions.0.content") == completion


@pytest.mark.vcr
def test_anthropic_3_completion(exporter, brt):
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me a joke about opentelemetry"},
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.5,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    response = brt.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    completion = response_body.get("content")

    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    anthropic_span = spans[0]
    assert json.loads(anthropic_span.attributes["llm.prompts.0.content"]) == [
        {"type": "text", "text": "Tell me a joke about opentelemetry"},
    ]

    assert (
        json.loads(anthropic_span.attributes.get("llm.completions.0.content"))
        == completion
    )
