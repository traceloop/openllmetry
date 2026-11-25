import os
import pytest
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_haystack(exporter):

    prompt_builder = DynamicChatPromptBuilder()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model="gpt-4")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")
    query = "OpenTelemetry"
    messages = [ChatMessage.from_user("Tell me a joke about {{query}}")]
    pipe.run(
        data={
            "prompt_builder": {
                "template_variables": {"query": query},
                "prompt_source": messages,
            }
        }
    )

    spans = exporter.get_finished_spans()
    assert {
        "haystack.openai.chat",
        "haystack_pipeline.workflow",
    } == {span.name for span in spans}
    span_workflow = next(
        span for span in spans if span.name == "haystack_pipeline.workflow"
    )
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span_workflow.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span_workflow.attributes


def _run_wrapped_generator(exporter, object_name, fake_run, kwargs):
    from opentelemetry.trace import get_tracer

    from opentelemetry.instrumentation.haystack.wrap_openai import wrap

    to_wrap = {"object": object_name, "method": "run"}
    wrap(get_tracer("opentelemetry.instrumentation.haystack.test"), to_wrap)(
        fake_run, None, (), kwargs
    )

    return exporter.get_finished_spans()[0]


def test_chat_generator_records_reply_content(exporter):
    # haystack 2.x OpenAIChatGenerator.run() returns {"replies": [ChatMessage, ...]};
    # the span must record the actual replies, not the "replies" key.
    def fake_run(*args, **kwargs):
        return {
            "replies": [
                ChatMessage.from_assistant("Hello there"),
                ChatMessage.from_assistant("General Kenobi"),
            ]
        }

    span = _run_wrapped_generator(
        exporter, "OpenAIChatGenerator", fake_run, {"messages": [ChatMessage.from_user("hi")]}
    )

    assert span.attributes["gen_ai.completion.0.content"] == "Hello there"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.completion.1.content"] == "General Kenobi"


def test_generator_records_reply_content(exporter):
    # OpenAIGenerator.run() returns {"replies": [str, ...]}; same key must be unpacked.
    def fake_run(*args, **kwargs):
        return {"replies": ["first reply", "second reply"]}

    span = _run_wrapped_generator(
        exporter, "OpenAIGenerator", fake_run, {"prompt": "hi"}
    )

    assert span.attributes["gen_ai.completion.0.content"] == "first reply"
    assert span.attributes["gen_ai.completion.1.content"] == "second reply"
