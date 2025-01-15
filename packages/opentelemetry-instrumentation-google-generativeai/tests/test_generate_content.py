# import pytest
# import os

# import google.generativeai as genai
# from opentelemetry.semconv_ai import SpanAttributes

def test_gemini_generate_content(exporter):
    # This test is working, but since Gemini uses gRPC,
    # vcr does not record it, therefore we cannot test this without
    # setting the API key in a shared secret store like GitHub secrets
    pass

    # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # model.generate_content(
    #     "The opposite of hot is",
    # )
    # spans = exporter.get_finished_spans()
    # assert all(span.name == "gemini.generate_content" for span in spans)

    # gemini_span = spans[0]
    # assert (
    #     gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    #     == "The opposite of hot is\n"
    # )
    # assert (
    #     gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
    #     == "user"
    # )
    # assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == "cold\n"
    # assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    # assert gemini_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 6
    # assert (
    #     gemini_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
    #     + gemini_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
    #     == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    # )

    # assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "models/gemini-1.5-flash"
    # assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "models/gemini-1.5-flash"
