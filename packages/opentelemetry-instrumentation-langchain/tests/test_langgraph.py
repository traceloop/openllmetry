import pytest
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph


@pytest.mark.vcr
def test_langgraph_invoke(exporter):
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
    spans = exporter.get_finished_spans()
    assert set(
        [
            "LangGraph.workflow",
            "__start__.task",
            "calculate.task",
            "ChannelWrite<calculate,request,result>.task",
            "openai.chat"
        ]
    ) == set([span.name for span in spans])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_langgraph_ainvoke(exporter):
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
    await langgraph.ainvoke(input={"request": user_request})
    spans = exporter.get_finished_spans()
    assert set(
        [
            "LangGraph.workflow",
            "__start__.task",
            "calculate.task",
            "_write.task",
            "openai.chat"
        ]
    ) == set([span.name for span in spans])
