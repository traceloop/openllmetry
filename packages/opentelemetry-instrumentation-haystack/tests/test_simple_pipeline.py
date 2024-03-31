import os
import pytest
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


@pytest.mark.vcr
def test_haystack(exporter):

    prompt_builder = DynamicChatPromptBuilder()
    api_key=os.getenv("OPENAI_API_KEY")
    llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model="gpt-4")


    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")
    query = "OpenTelemetry"
    messages = [ChatMessage.from_user("Tell me a joke about {{query}}")]
    pipe.run(data={"prompt_builder": {"template_variables":{"query": query}, "prompt_source": messages}})

    spans = exporter.get_finished_spans()
    assert set(
        [
            "openai.chat",
            "haystack_pipeline.workflow",
        ]
    ).issubset([span.name for span in spans])
