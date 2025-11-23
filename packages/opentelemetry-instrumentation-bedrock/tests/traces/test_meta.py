import json

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_meta_llama2_completion_string_content(
    instrument_legacy, brt, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_meta_llama2_completion_string_content_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": response_body["generation"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_llama2_completion_string_content_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_llama3_completion(instrument_legacy, brt, span_exporter, log_exporter):
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = "Tell me a joke about opentelemetry"
    # Create request body.
    body = json.dumps(
        {"prompt": prompt, "max_gen_len": 128, "temperature": 0.1, "top_p": 0.9}
    )

    response = brt.invoke_model(body=body, modelId=model_id)

    response_body = json.loads(response.get("body").read())

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == prompt
    assert (
        meta_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response_body["generation"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_meta_llama3_completion_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = "Tell me a joke about opentelemetry"
    # Create request body.
    body = json.dumps(
        {"prompt": prompt, "max_gen_len": 128, "temperature": 0.1, "top_p": 0.9}
    )

    response = brt.invoke_model(body=body, modelId=model_id)

    response_body = json.loads(response.get("body").read())

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {"content": prompt})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response_body["generation"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_llama3_completion_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = "Tell me a joke about opentelemetry"
    # Create request body.
    body = json.dumps(
        {"prompt": prompt, "max_gen_len": 128, "temperature": 0.1, "top_p": 0.9}
    )

    response = brt.invoke_model(body=body, modelId=model_id)

    response_body = json.loads(response.get("body").read())

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.completion" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response_body["prompt_token_count"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response_body["generation_token_count"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response_body["generation_token_count"] + response_body["prompt_token_count"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_converse(instrument_legacy, brt, span_exporter, log_exporter):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response["usage"]["inputTokens"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response["usage"]["outputTokens"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response["usage"]["totalTokens"]
    )
    assert meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system_prompt
    )
    assert meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"
    assert meta_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
    ] == json.dumps(messages[0]["content"])
    for i in range(0, len(generated_text)):
        assert (
            meta_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.role"]
            == "assistant"
        )
        assert (
            meta_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{i}.content"]
            == generated_text[i]["text"]
        )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_meta_converse_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response["usage"]["inputTokens"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response["usage"]["outputTokens"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response["usage"]["totalTokens"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": system_prompt},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": messages[0]["content"]}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_converse_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response["usage"]["inputTokens"]
    )
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response["usage"]["outputTokens"]
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response["usage"]["totalTokens"]
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_converse_stream(instrument_legacy, brt, span_exporter, log_exporter):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == outputTokens
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )
    assert meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == system_prompt
    )
    assert meta_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"
    assert meta_span.attributes[
        f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"
    ] == json.dumps(messages[0]["content"])

    assert (
        meta_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"]
        == response_role
    )
    assert (
        meta_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == content
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_meta_converse_stream_with_events_with_content(
    instrument_with_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == outputTokens
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {"content": system_prompt},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": messages[0]["content"]}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {"content": content},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_meta_converse_stream_with_events_with_no_content(
    instrument_with_no_content, brt, span_exporter, log_exporter
):
    model_id = "meta.llama3-2-1b-instruct-v1:0"
    inference_config = {"temperature": 0.5}
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Tell me a joke about opentelemetry"},
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

    content = ""
    inputTokens = 0
    outputTokens = 0

    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                content += event["contentBlockDelta"]["delta"]["text"]

            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    inputTokens = metadata["usage"]["inputTokens"]
                    outputTokens = metadata["usage"]["outputTokens"]

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "bedrock.converse" for span in spans)

    meta_span = spans[0]
    assert meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == inputTokens
    assert (
        meta_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == outputTokens
    )
    assert (
        meta_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == inputTokens + outputTokens
    )
    assert meta_span.attributes.get("gen_ai.response.id") is None

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.system.message",
        {},
    )

    # Validate user message Event
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "end_turn",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.AWS_BEDROCK.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
