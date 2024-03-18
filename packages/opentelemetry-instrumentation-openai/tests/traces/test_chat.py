import os

import pytest


@pytest.mark.vcr
def test_chat_streaming(exporter, openai_client, vcr):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            stream=True,
        )

        for _ in response:
            pass

        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.chat",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes["llm.prompts.0.content"]
            == "Tell me a joke about opentelemetry"
        )
        assert open_ai_span.attributes.get("llm.completions.0.content")
        assert open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get("llm.usage.completion_tokens")
        prompt_tokens = open_ai_span.attributes.get("llm.usage.prompt_tokens")
        total_tokens = open_ai_span.attributes.get("llm.usage.total_tokens")
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens

    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value
