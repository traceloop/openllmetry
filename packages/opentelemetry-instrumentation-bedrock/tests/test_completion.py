import json
import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.instrumentation.bedrock.config import Config


def find_event(span, event_name):
    """Helper function to find an event by name in a span."""
    return next((event for event in span.events if event.name == event_name), None)


def mock_bedrock_response(content, model_id, token_counts=None):
    """Helper to create a mock Bedrock response."""
    response = {"ResponseMetadata": {"HTTPStatusCode": 200}}

    if "anthropic" in model_id:
        if "messages" in content:
            response["body"] = json.dumps(
                {
                    "content": content["messages"],
                    "model": model_id,
                    "usage": {
                        "input_tokens": token_counts["input"] if token_counts else 10,
                        "output_tokens": token_counts["output"] if token_counts else 20,
                    },
                    "stop_reason": "stop_sequence",
                }
            )
        else:
            response["body"] = json.dumps(
                {
                    "completion": content["completion"],
                    "model": model_id,
                    "usage": {
                        "input_tokens": token_counts["input"] if token_counts else 10,
                        "output_tokens": token_counts["output"] if token_counts else 20,
                    },
                    "stop_reason": "stop_sequence",
                }
            )
    elif "cohere" in model_id:
        response["body"] = json.dumps(
            {
                "generations": [{"text": text} for text in content["generations"]],
                "token_count": {
                    "prompt_tokens": token_counts["input"] if token_counts else 10,
                    "response_tokens": token_counts["output"] if token_counts else 20,
                },
            }
        )
    elif "ai21" in model_id:
        response["body"] = json.dumps(
            {
                "completions": [
                    {"data": {"text": text}} for text in content["completions"]
                ],
                "prompt": {
                    "tokens": ["t"] * (token_counts["input"] if token_counts else 10)
                },
                "completions": [
                    {
                        "data": {
                            "tokens": ["t"]
                            * (token_counts["output"] if token_counts else 20)
                        }
                    }
                ],
            }
        )
    elif "meta" in model_id:
        if "generations" in content:
            response["body"] = json.dumps(
                {
                    "generations": content["generations"],
                    "prompt_token_count": token_counts["input"] if token_counts else 10,
                    "generation_token_count": token_counts["output"]
                    if token_counts
                    else 20,
                }
            )
        else:
            response["body"] = json.dumps(
                {
                    "generation": content["generation"],
                    "prompt_token_count": token_counts["input"] if token_counts else 10,
                    "generation_token_count": token_counts["output"]
                    if token_counts
                    else 20,
                }
            )
    elif "amazon" in model_id:
        response["body"] = json.dumps(
            {
                "inputTextTokenCount": token_counts["input"] if token_counts else 10,
                "results": [
                    {
                        "outputText": text,
                        "tokenCount": token_counts["output"] if token_counts else 20,
                    }
                    for text in content["results"]
                ],
            }
        )

    return response


@pytest.mark.parametrize("use_legacy_attributes", [True, False])
def test_cohere_completion(exporter, use_legacy_attributes):
    """Test Cohere completion in both legacy and event-based modes."""
    Config.use_legacy_attributes = use_legacy_attributes

    prompt = "Tell me a joke about OpenTelemetry"
    generations = [
        "Why did the OpenTelemetry trace cross the road?",
        "To reach the other service!",
    ]

    with patch("botocore.client.BaseClient._make_api_call") as mock_api:
        mock_api.return_value = mock_bedrock_response(
            {"generations": generations},
            "cohere.command",
            {"input": 15, "output": 25},
        )

        client = MagicMock()
        client.invoke_model(
            body=json.dumps(
                {
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "p": 0.8,
                }
            ),
            modelId="cohere.command",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    if use_legacy_attributes:
        # Check legacy attribute-based instrumentation
        assert span.attributes.get("gen_ai.prompt.0.user") == prompt
        for i, text in enumerate(generations):
            assert span.attributes.get(f"gen_ai.completion.{i}.content") == text
    else:
        # Check event-based instrumentation
        prompt_event = find_event(span, "prompt")
        assert prompt_event is not None
        assert prompt_event.attributes["llm.prompt.content"] == prompt

        for i, text in enumerate(generations):
            completion_event = find_event(span, "completion")
            assert completion_event is not None
            assert completion_event.attributes["llm.completion.content"] == text
            assert completion_event.attributes["llm.completion.index"] == i

    # Token usage metrics should be set in both modes
    assert span.attributes.get("gen_ai.usage.prompt_tokens") == 15
    assert span.attributes.get("gen_ai.usage.completion_tokens") == 25
    assert span.attributes.get("gen_ai.usage.total_tokens") == 40


@pytest.mark.parametrize("use_legacy_attributes", [True, False])
def test_anthropic_chat(exporter, use_legacy_attributes):
    """Test Anthropic chat in both legacy and event-based modes."""
    Config.use_legacy_attributes = use_legacy_attributes

    messages = [
        {"role": "user", "content": "Tell me a joke"},
        {
            "role": "assistant",
            "content": "Why did the OpenTelemetry trace cross the road?",
        },
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "To reach the other service!"},
    ]

    with patch("botocore.client.BaseClient._make_api_call") as mock_api:
        mock_api.return_value = mock_bedrock_response(
            {"messages": messages},
            "anthropic.claude-3",
            {"input": 20, "output": 30},
        )

        client = MagicMock()
        client.invoke_model(
            body=json.dumps(
                {
                    "messages": messages[:2],
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.8,
                }
            ),
            modelId="anthropic.claude-3",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    if use_legacy_attributes:
        # Check legacy attribute-based instrumentation
        for i, msg in enumerate(messages[:2]):
            assert span.attributes.get(f"gen_ai.prompt.{i}.role") == msg["role"]
            assert span.attributes.get(f"gen_ai.prompt.{i}.content") == msg["content"]
        assert (
            span.attributes.get("gen_ai.completion.0.content") == messages[3]["content"]
        )
        assert span.attributes.get("gen_ai.completion.0.role") == "assistant"
    else:
        # Check event-based instrumentation
        prompt_event = find_event(span, "prompt")
        assert prompt_event is not None
        assert prompt_event.attributes["llm.prompt.type"] == "chat"

        completion_event = find_event(span, "completion")
        assert completion_event is not None
        assert (
            completion_event.attributes["llm.completion.content"]
            == messages[3]["content"]
        )
        assert completion_event.attributes["llm.completion.role"] == "assistant"

    # Token usage metrics should be set in both modes
    assert span.attributes.get("gen_ai.usage.prompt_tokens") == 20
    assert span.attributes.get("gen_ai.usage.completion_tokens") == 30
    assert span.attributes.get("gen_ai.usage.total_tokens") == 50


@pytest.mark.parametrize("use_legacy_attributes", [True, False])
def test_amazon_completion(exporter, use_legacy_attributes):
    """Test Amazon completion in both legacy and event-based modes."""
    Config.use_legacy_attributes = use_legacy_attributes

    prompt = "Tell me a joke about OpenTelemetry"
    results = [
        "Why did the OpenTelemetry trace cross the road?",
        "To reach the other service!",
    ]

    with patch("botocore.client.BaseClient._make_api_call") as mock_api:
        mock_api.return_value = mock_bedrock_response(
            {"results": results},
            "amazon.titan-text",
            {"input": 12, "output": 18},
        )

        client = MagicMock()
        client.invoke_model(
            body=json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 100,
                        "temperature": 0.7,
                        "topP": 0.8,
                    },
                }
            ),
            modelId="amazon.titan-text",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    if use_legacy_attributes:
        # Check legacy attribute-based instrumentation
        assert span.attributes.get("gen_ai.prompt.0.user") == prompt
        for i, text in enumerate(results):
            assert span.attributes.get(f"gen_ai.completion.{i}.content") == text
    else:
        # Check event-based instrumentation
        prompt_event = find_event(span, "prompt")
        assert prompt_event is not None
        assert prompt_event.attributes["llm.prompt.content"] == prompt

        for i, text in enumerate(results):
            completion_event = find_event(span, "completion")
            assert completion_event is not None
            assert completion_event.attributes["llm.completion.content"] == text
            assert completion_event.attributes["llm.completion.index"] == i

    # Token usage metrics should be set in both modes
    assert span.attributes.get("gen_ai.usage.prompt_tokens") == 12
    assert span.attributes.get("gen_ai.usage.completion_tokens") == 18
    assert span.attributes.get("gen_ai.usage.total_tokens") == 30


@pytest.mark.parametrize("use_legacy_attributes", [True, False])
def test_meta_completion(exporter, use_legacy_attributes):
    """Test Meta completion in both legacy and event-based modes."""
    Config.use_legacy_attributes = use_legacy_attributes

    prompt = "Tell me a joke about OpenTelemetry"
    generation = (
        "Why did the OpenTelemetry trace cross the road? To reach the other service!"
    )

    with patch("botocore.client.BaseClient._make_api_call") as mock_api:
        mock_api.return_value = mock_bedrock_response(
            {"generation": generation},
            "meta.llama2",
            {"input": 14, "output": 22},
        )

        client = MagicMock()
        client.invoke_model(
            body=json.dumps(
                {
                    "prompt": prompt,
                    "max_gen_len": 100,
                    "temperature": 0.7,
                    "top_p": 0.8,
                }
            ),
            modelId="meta.llama2",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    if use_legacy_attributes:
        # Check legacy attribute-based instrumentation
        assert span.attributes.get("gen_ai.prompt.0.content") == prompt
        assert span.attributes.get("gen_ai.prompt.0.role") == "user"
        assert span.attributes.get("gen_ai.completion.0.content") == generation
        assert span.attributes.get("gen_ai.completion.0.role") == "assistant"
    else:
        # Check event-based instrumentation
        prompt_event = find_event(span, "prompt")
        assert prompt_event is not None
        assert prompt_event.attributes["llm.prompt.content"] == prompt

        completion_event = find_event(span, "completion")
        assert completion_event is not None
        assert completion_event.attributes["llm.completion.content"] == generation
        assert completion_event.attributes["llm.completion.role"] == "assistant"

    # Token usage metrics should be set in both modes
    assert span.attributes.get("gen_ai.usage.prompt_tokens") == 14
    assert span.attributes.get("gen_ai.usage.completion_tokens") == 22
    assert span.attributes.get("gen_ai.usage.total_tokens") == 36


@pytest.mark.parametrize("use_legacy_attributes", [True, False])
def test_ai21_completion(exporter, use_legacy_attributes):
    """Test AI21 completion in both legacy and event-based modes."""
    Config.use_legacy_attributes = use_legacy_attributes

    prompt = "Tell me a joke about OpenTelemetry"
    completions = [
        "Why did the OpenTelemetry trace cross the road?",
        "To reach the other service!",
    ]

    with patch("botocore.client.BaseClient._make_api_call") as mock_api:
        mock_api.return_value = mock_bedrock_response(
            {"completions": completions},
            "ai21.j2-ultra",
            {"input": 16, "output": 24},
        )

        client = MagicMock()
        client.invoke_model(
            body=json.dumps(
                {
                    "prompt": prompt,
                    "maxTokens": 100,
                    "temperature": 0.7,
                    "topP": 0.8,
                }
            ),
            modelId="ai21.j2-ultra",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    if use_legacy_attributes:
        # Check legacy attribute-based instrumentation
        assert span.attributes.get("gen_ai.prompt.0.user") == prompt
        for i, text in enumerate(completions):
            assert span.attributes.get(f"gen_ai.completion.{i}.content") == text
    else:
        # Check event-based instrumentation
        prompt_event = find_event(span, "prompt")
        assert prompt_event is not None
        assert prompt_event.attributes["llm.prompt.content"] == prompt

        for i, text in enumerate(completions):
            completion_event = find_event(span, "completion")
            assert completion_event is not None
            assert completion_event.attributes["llm.completion.content"] == text
            assert completion_event.attributes["llm.completion.index"] == i

    # Token usage metrics should be set in both modes
    assert span.attributes.get("gen_ai.usage.prompt_tokens") == 16
    assert span.attributes.get("gen_ai.usage.completion_tokens") == 24
    assert span.attributes.get("gen_ai.usage.total_tokens") == 40
