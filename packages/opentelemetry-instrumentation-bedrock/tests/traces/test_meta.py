import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr
def test_meta_llama2_completion_string_content(test_context, brt):
    model_id = "meta.llama2-13b-chat-v1"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden  What should I do? [/INST]"""
    # Create request body.
    body = json.dumps(
        {"prompt": prompt, "max_gen_len": 128, "temperature": 0.1, "top_p": 0.9}
    )

    response = brt.invoke_model(body=body, modelId=model_id)

    response_body = json.loads(response.get("body").read())

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_meta_llama3_completion(test_context, brt):
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = "Tell me a joke about opentelemetry"
    # Create request body.
    body = json.dumps(
        {"prompt": prompt, "max_gen_len": 128, "temperature": 0.1, "top_p": 0.9}
    )

    response = brt.invoke_model(body=body, modelId=model_id)

    response_body = json.loads(response.get("body").read())

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == prompt
    assert (
        meta_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response_body["generation"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_meta_converse(test_context, brt):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Tell me a joke about opentelemetry"
                },
            ],
        }
    ]
    system_prompt = "You are an app that knows about everything."
    system = [{"text": system_prompt}]
    response = brt.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inference_config,
    )

    generated_text = response["output"]["message"]["content"]

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
            == response["usage"]['inputTokens']
    )
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
            == response["usage"]['outputTokens']
    )
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
            == response["usage"]['totalTokens']
    )
    assert meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
            meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
            == system_prompt
    )
    assert meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert (
            meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
            == json.dumps(messages[0]["content"])
    )
    for i in range(0, len(generated_text)):
        assert meta_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role"] == "assistant"
        assert (
                meta_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content"]
                == generated_text[i]["text"]
        )
    assert meta_span.attributes.get("gen_ai.response.id") is None


@pytest.mark.vcr
def test_meta_converse_stream(test_context, brt):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Tell me a joke about opentelemetry"
                },
            ],
        }
    ]
    system_prompt = "You are an app that knows about everything."
    system = [{"text": system_prompt}]
    response = brt.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inference_config,
    )

    stream = response.get("stream")

    response_role = None
    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:

            if "messageStart" in event:
                response_role = event["messageStart"]["role"]

            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    exporter, _, _ = test_context
    spans = exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
            == inputTokens
    )
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
            == outputTokens
    )
    assert (
            meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
            == inputTokens + outputTokens
    )
    assert meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
            meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
            == system_prompt
    )
    assert meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert (
            meta_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
            == json.dumps(messages[0]["content"])
    )

    assert meta_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == response_role
    assert (
            meta_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
            == content
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None