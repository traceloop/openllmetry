import json
import os

from writerai import Writer
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import tool
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

"""---------------------------------------- Initializing environment ----------------------------------------"""

if "TRACELOOP_METRICS_ENABLED" not in os.environ:
    os.environ["TRACELOOP_METRICS_ENABLED"] = "true"

load_dotenv()

Traceloop.init()
writer_client = Writer()

messages = [{"role": "system", "content": "Answer with jokes"},
            {"role": "user", "content": "What is the weather like today in Barcelona?"}]
tools = [
        {
            "function": {
                "description": "Return weather in the specific location.",
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "location": {
                            "description": "Location to return weather at",
                            "type": "string",
                        },
                    },
                    "required": ["location"],
                    "type": "object",
                },
            },
            "type": "function",
        }
    ]

"""---------------------------------------- Chat tool definition ----------------------------------------"""


@tool(name="get_weather")
def get_weather(location: str) -> str:
    return f"Weather in {location}: sunny, 25 deg above 0, wind is 2 m/s south."


"""---------------------------------------- Subsidiary functions ----------------------------------------"""


@dataclass
class StreamResult:
    content: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: Optional[str] = None


def handle_streaming_response(response_stream):
    accumulated_content = ""
    tool_calls_accumulator = {}
    finish_reason = None

    for chunk in response_stream:
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_content += delta.content

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    _process_tool_call_chunk(tool_call_chunk, tool_calls_accumulator)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            print(StreamResult(
                content=accumulated_content,
                tool_calls=list(tool_calls_accumulator.values()),
                finish_reason=finish_reason
            ))

    return StreamResult(
        content=accumulated_content,
        tool_calls=list(tool_calls_accumulator.values()),
        finish_reason=finish_reason
    )


def _process_tool_call_chunk(
        tool_call_chunk,
        accumulator
):
    index = tool_call_chunk.index

    if index is None:
        return

    if index not in accumulator:
        accumulator[index] = {
            'id': '',
            'type': 'function',
            'function': {
                'name': '',
                'arguments': ''
            }
        }

    current_tool_call = accumulator[index]

    if tool_call_chunk.id:
        current_tool_call["id"] += tool_call_chunk.id

    if tool_call_chunk.function and tool_call_chunk.function.name:
        current_tool_call['function']['name'] += tool_call_chunk.function.name

    if tool_call_chunk.function and tool_call_chunk.function.arguments:
        current_tool_call['function']['arguments'] += tool_call_chunk.function.arguments


"""---------------------------------------- Usage example ----------------------------------------"""

gen = writer_client.chat.chat(
    model="palmyra-x4",
    messages=messages,
    tools=tools,
    stream=True,
    temperature=0.7,
    top_p=0.9,
    stream_options={"include_usage": True}
)

print(20 * "-" + "Processing stream" + 20 * "-")
response = handle_streaming_response(gen)
messages += [{"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}]

print(20 * "-" + "Received response" + 20 * "-")
print(response)

print(20 * "-" + "Calling tool 'get_weather'" + 20 * "-")
tool_call_arguments = json.loads(response.tool_calls[0]["function"]["arguments"])
tool_call_result = get_weather(tool_call_arguments["location"])
print(f"Arguments: {tool_call_arguments}. Tool call result: {tool_call_result}")
messages += [{"role": "tool", "content": tool_call_result}]

gen = writer_client.chat.chat(
    model="palmyra-x4",
    messages=messages,
    tools=tools,
    stream=True,
    temperature=0.7,
    top_p=0.9,
    stream_options={"include_usage": True}
)

print(20 * "-" + "Processing stream" + 20 * "-")
response = handle_streaming_response(gen)
messages += [{"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}]

print(20 * "-" + "Received response" + 20 * "-")
print(response)
