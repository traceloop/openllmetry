"""Synchronous LiteLLM completion with tool calling, instrumented by Traceloop.

A small customer-support agent loop built on the blocking `litellm.completion`
API (vs. the async/streaming sibling example). It exposes three tools that have
nothing to do with weather:

  1. `lookup_order`     — fetch an order's status from a fake order DB.
  2. `check_inventory`  — check how many units of a SKU are in stock.
  3. `issue_refund`     — issue a refund for an order.

The loop keeps calling the model until it stops requesting tools, dispatching
each requested tool by name and feeding results back. Every `litellm.completion`
call produces one `litellm.chat` span with `gen_ai.*` attributes; the tool
functions are wrapped with `@tool` so they appear as their own spans.

Requires: OPENAI_API_KEY, and TRACELOOP_API_KEY unless a custom exporter/endpoint
is configured (the default `Traceloop.init()` path needs it to emit spans).
"""

import json

import litellm
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, tool, workflow
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="litellm_completion_example",
    instruments={Instruments.LITELLM},
)

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up the status and total of a customer order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. 'A1001'.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check how many units of a product SKU are currently in stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "The product SKU, e.g. 'WIDGET-BLUE'.",
                    },
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order. Returns a confirmation id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to refund.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "The amount to refund, in USD.",
                    },
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]

# Fake backing data — a real agent would hit a database or internal API.
_ORDERS = {
    "A1001": {"status": "delivered", "total": 49.90, "sku": "WIDGET-BLUE"},
    "A1002": {"status": "in_transit", "total": 19.95, "sku": "GADGET-RED"},
}
_INVENTORY = {"WIDGET-BLUE": 0, "GADGET-RED": 137}


@tool(name="lookup_order")
def lookup_order(order_id: str) -> str:
    order = _ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": f"no order {order_id}"})
    return json.dumps({"order_id": order_id, **order})


@tool(name="check_inventory")
def check_inventory(sku: str) -> str:
    units = _INVENTORY.get(sku)
    if units is None:
        return json.dumps({"error": f"unknown sku {sku}"})
    return json.dumps({"sku": sku, "in_stock": units})


@tool(name="issue_refund")
def issue_refund(order_id: str, amount: float) -> str:
    # A real implementation would call a payments API; here we just confirm.
    return json.dumps(
        {"order_id": order_id, "amount": amount, "refund_id": f"RF-{order_id}", "status": "refunded"}
    )


# Map tool name -> implementation so the agent loop can dispatch any requested tool.
TOOL_REGISTRY = {
    "lookup_order": lookup_order,
    "check_inventory": check_inventory,
    "issue_refund": issue_refund,
}


@task(name="chat_step")
def chat_step(messages: list) -> object:
    """One blocking model call that may return content and/or tool calls."""
    response = litellm.completion(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message


@task(name="execute_tools")
def execute_tools(tool_calls: list) -> list:
    """Run every requested tool inside one span.

    Wrapping the loop in a task makes the individual tool spans children of
    `execute_tools` instead of flat siblings of `chat_step`, so they stay grouped
    and correctly ordered between the model calls in the waterfall.
    """
    tool_messages = []
    for tc in tool_calls:
        impl = TOOL_REGISTRY.get(tc.function.name)
        try:
            args = json.loads(tc.function.arguments or "{}")
            if impl is None:
                result = json.dumps({"error": f"unknown tool {tc.function.name}"})
            else:
                result = impl(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool failures back to the model
            result = json.dumps({"error": str(exc)})
        print(f"[tool] {tc.function.name}({tc.function.arguments}) -> {result}")
        tool_messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )
    return tool_messages


@workflow(name="litellm_completion")
def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer-support agent. Use the available tools to look up "
                "orders and inventory before answering, and issue refunds when a customer "
                "asks for one on a delivered order."
            ),
        },
        {
            "role": "user",
            "content": (
                "I want a refund for order A1001 — the widget arrived broken. "
                "Also, can I reorder the same item right away?"
            ),
        },
    ]

    # Agent loop: keep calling until the model answers without requesting tools.
    for _ in range(5):
        message = chat_step(messages)
        tool_calls = message.tool_calls or []

        if not tool_calls:
            print("\nFinal:", message.content)
            return

        # Record the assistant's tool-call turn.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute each requested tool (dispatching by name) and append its result.
        messages.extend(execute_tools(tool_calls))

    print("\nStopped: reached max tool-calling iterations.")


if __name__ == "__main__":
    main()
