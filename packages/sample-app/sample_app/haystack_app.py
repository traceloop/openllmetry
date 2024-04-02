import os
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from traceloop.sdk import Traceloop

Traceloop.init(app_name="haystack_example")

def haystack_app():
    
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


haystack_app()