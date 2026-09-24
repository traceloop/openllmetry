"""Async + streaming LiteLLM completion with tool calling, instrumented by Traceloop.

This mirrors a realistic agent loop using the primary call shape from
agent-orchestrator-v2 (`await litellm.acompletion(..., stream=True)`), plus tool
(function) calling:

  1. Stream a completion that offers the model a `get_current_weather` tool.
  2. Accumulate the streamed tool-call deltas, then execute the tool locally.
  3. Feed the tool result back and stream the model's final natural-language answer.

Each `litellm.acompletion` call produces one `litellm.chat` span; the instrumentation
accumulates the streamed deltas (content *and* tool calls) into that span.

Pass `stream_options={"include_usage": True}` so the provider reports token usage on
the final chunk.

Requires: OPENAI_API_KEY, and TRACELOOP_API_KEY unless a custom exporter/endpoint
is configured (the default `Traceloop.init()` path needs it to emit spans).
"""

import ast
import asyncio
import json
import operator

import litellm
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, tool, workflow
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="litellm_async_streaming_example",
    instruments={Instruments.LITELLM},
)

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. '23 * 19'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression using + - * / and parentheses.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


@tool(name="get_current_weather")
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Pretend tool implementation — a real one would call a weather API."""
    fake_db = {"paris": 18, "london": 14, "new york": 22}
    temp = fake_db.get(location.lower(), 20)
    return json.dumps({"location": location, "temperature": temp, "unit": unit})


@tool(name="get_current_time")
def get_current_time(location: str) -> str:
    """Pretend tool implementation — a real one would resolve the city's timezone."""
    fake_clock = {"paris": "14:05", "london": "13:05", "tokyo": "22:05"}
    return json.dumps({"location": location, "time": fake_clock.get(location.lower(), "12:00")})


# Only the four basic binary operators and unary +/- are allowed — note the absence
# of ast.Pow, so model-supplied expressions like ``9**9**9`` can't hang the sample.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arith(node):
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError("unsupported expression")


@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression (+, -, *, / on numbers)."""
    if len(expression) > 100:
        return json.dumps({"error": "expression too long"})
    try:
        result = _eval_arith(ast.parse(expression, mode="eval"))
    except Exception as exc:  # noqa: BLE001 - report the failure back to the model
        return json.dumps({"error": str(exc)})
    return json.dumps({"expression": expression, "result": result})


# Map tool name -> implementation, so the agent loop can dispatch any requested tool.
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "get_current_time": get_current_time,
    "calculate": calculate,
}


async def _consume_stream(stream):
    """Consume a streamed response, accumulating text content and tool-call deltas."""
    content = ""
    tool_calls: list[dict] = []
    async for chunk in stream:
        # The final `include_usage` chunk carries token usage with an empty choices
        # array — skip it so we don't index into a non-existent choice.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)
        for tc in delta.tool_calls or []:
            while len(tool_calls) <= tc.index:
                tool_calls.append({"id": "", "name": "", "arguments": ""})
            slot = tool_calls[tc.index]
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return content, tool_calls


@task(name="request_with_tools")
async def request_with_tools(messages: list) -> tuple[str, list]:
    stream = await litellm.acompletion(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
    )
    return await _consume_stream(stream)


@task(name="execute_tools")
def execute_tools(tool_calls: list) -> list:
    """Run every requested tool inside one span.

    Wrapping the loop in a task makes the individual tool spans children of
    `execute_tools` instead of flat siblings of the model-call tasks. The parent
    inherits the first tool's start time, so it sorts ahead of `final_answer`
    and the tools can never scatter after it in the waterfall.
    """
    tool_messages = []
    for tc in tool_calls:
        impl = TOOL_REGISTRY.get(tc["name"])
        try:
            args = json.loads(tc["arguments"] or "{}")
            if impl is None:
                result = json.dumps({"error": f"unknown tool {tc['name']}"})
            else:
                result = impl(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool failures back to the model
            result = json.dumps({"error": str(exc)})
        print(f"\n[tool] {tc['name']}({tc['arguments']}) -> {result}")
        tool_messages.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": result}
        )
    return tool_messages


@task(name="final_answer")
async def final_answer(messages: list) -> str:
    stream = await litellm.acompletion(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    content, _ = await _consume_stream(stream)
    return content


@workflow(name="litellm_tool_calling")
async def main():
    messages = [
        {
            "role": "user",
            "content": (
                "What's the weather in Paris, what time is it in Tokyo, "
                "and what is 23 * 19?"
            ),
        }
    ]

    print("--- first call (may request a tool) ---")
    content, tool_calls = await request_with_tools(messages)
    print()

    if not tool_calls:
        print("\nModel answered directly:", content)
        return

    # Record the assistant's tool-call turn.
    messages.append(
        {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        }
    )

    # Execute each requested tool (dispatching by name) and append its result.
    messages.extend(execute_tools(tool_calls))

    print("\n--- final answer ---")
    answer = await final_answer(messages)
    print("\n\nFinal:", answer)


if __name__ == "__main__":
    asyncio.run(main())
