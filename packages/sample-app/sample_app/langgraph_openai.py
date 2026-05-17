from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph
from traceloop.sdk import Traceloop

Traceloop.init(app_name="langgraph_example")

# Define the tools for the agent to use
client = OpenAI()


class State(TypedDict):
    request: str
    result: str


def calculate(state: State):
    request = state["request"]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a mathematician."},
            {"role": "user", "content": request}
        ]
    )
    return {"result": completion.choices[0].message.content}


workflow = StateGraph(State)
workflow.add_node("calculate", calculate)
workflow.set_entry_point("calculate")

langgraph = workflow.compile()

user_request = "What's 5 + 5?"
langgraph.invoke(input={"request": user_request})
