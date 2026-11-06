"""tools/call output must keep every content block, in order.

A tool result can carry several TextContent blocks. The client span used to
record only result.content[0], so everything after the first block was missing
from traceloop.entity.output while the server span kept the full result.
"""

from types import SimpleNamespace

from opentelemetry.instrumentation.mcp.instrumentation import McpInstrumentor


def _tool_result(*texts):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text) for text in texts],
        isError=False,
    )


def test_tools_call_keeps_every_content_block():
    output = McpInstrumentor()._extract_clean_output(
        "tools/call", _tool_result("alpha-block", "omega-block")
    )

    assert output["result"] == ["alpha-block", "omega-block"]


def test_tools_call_single_block_keeps_existing_shape():
    output = McpInstrumentor()._extract_clean_output(
        "tools/call", _tool_result("only-block")
    )

    assert output["result"] == "only-block"


def test_tools_call_without_content_returns_empty_output():
    result = SimpleNamespace(content=[], isError=False)

    assert McpInstrumentor()._extract_clean_output("tools/call", result) == {}
