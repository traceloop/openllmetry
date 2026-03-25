# AG2 (formerly AutoGen) provides the 'autogen' module
from autogen import AssistantAgent, UserProxyAgent
from traceloop.sdk import Traceloop

Traceloop.init(app_name="ag2-example")

assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o-mini"},
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
)

result = user_proxy.initiate_chat(
    assistant,
    message="What are 3 key trends in AI for 2025? Be concise.",
)

print("Chat Result:", result.summary)
