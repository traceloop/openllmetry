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
