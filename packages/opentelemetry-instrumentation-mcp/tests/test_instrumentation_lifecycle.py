"""Run the lifecycle scenario directly in a fresh interpreter to isolate patches."""

import asyncio
from collections import Counter
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("import_when", "fresh_clients"), [
    ("before", False), ("before", True), ("disabled", False), ("re-enabled", False),
])
def test_instrumentation_lifecycle(import_when, fresh_clients):
    package = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(package), str(package.parent / "opentelemetry-semantic-conventions-ai"), env.get("PYTHONPATH", ""),
    ])
    # Running this file directly avoids the suite's autouse instrumentation.
    result = subprocess.run(
        [sys.executable, __file__, import_when, str(fresh_clients)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _original_methods():
    from wrapt import resolve_path

    targets = [
        ("fastmcp", "FastMCP.__init__"),
        ("fastmcp.tools.tool_manager", "ToolManager.call_tool"),
        ("fastmcp.client", "Client.__aenter__"),
        ("fastmcp.client", "Client.__aexit__"),
        ("mcp.client.sse", "sse_client"),
        ("mcp.server.sse", "SseServerTransport.connect_sse"),
        ("mcp.client.stdio", "stdio_client"),
        ("mcp.server.stdio", "stdio_server"),
        ("mcp.server.session", "ServerSession.__init__"),
        ("mcp.client.streamable_http", "streamablehttp_client"),
        ("mcp.server.streamable_http", "StreamableHTTPServerTransport.connect"),
        ("mcp.shared.session", "BaseSession.send_request"),
    ]
    return [resolve_path(importlib.import_module(module), name) for module, name in targets]


async def _exercise_lifecycle(import_when, fresh_clients):
    from opentelemetry import trace
    from opentelemetry.instrumentation.mcp import McpInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from wrapt import resolve_path

    exporter = InMemorySpanExporter()
    providers = []

    def provider(cycle):
        current = TracerProvider(resource=Resource.create({"cycle": cycle}))
        current.add_span_processor(SimpleSpanProcessor(exporter))
        providers.append(current)
        return current

    current_provider = provider(0)
    instrumentor = McpInstrumentor()
    try:
        if import_when != "before":
            for _ in range(3):
                instrumentor.instrument(tracer_provider=current_provider)
                modules_before = set(sys.modules)
                instrumentor.uninstrument()
                assert set(sys.modules) == modules_before, "teardown imported a module"
            assert "fastmcp" not in sys.modules
        originals = _original_methods() if import_when == "before" else []
        if import_when == "re-enabled":
            instrumentor.instrument(tracer_provider=current_provider)
        from fastmcp import Client, FastMCP

        server = FastMCP("lifecycle-regression")
        executions = 0

        @server.tool()
        async def add_numbers(a: int, b: int) -> int:
            nonlocal executions
            executions += 1
            return a + b

        client = Client(server)

        async def call(enabled, cycle):
            exporter.clear()
            before = executions
            async with (Client(server) if fresh_clients else client) as connected:
                result = await connected.call_tool("add_numbers", {"a": 5, "b": 3})
                assert result.content[0].text == "8"
            assert executions == before + 1
            assert not trace.get_current_span().get_span_context().is_valid
            spans = exporter.get_finished_spans()
            assert all(s.resource.attributes["cycle"] == cycle for s in spans)
            assert len({s.context.trace_id for s in spans}) == int(enabled)
            return Counter(s.name for s in spans)

        if import_when == "disabled":
            assert not await call(False, 0), "deferred hooks traced a disabled call"
        baseline = None
        for cycle in range(3):
            if cycle:
                current_provider = provider(cycle)
            if not instrumentor.is_instrumented_by_opentelemetry:
                McpInstrumentor().instrument(tracer_provider=current_provider)
            assert instrumentor.is_instrumented_by_opentelemetry
            for parent, attribute, original in originals:
                assert resolve_path(parent, attribute)[2] is not original, attribute
            counts = await call(True, cycle)
            assert counts["add_numbers.tool"] == 2
            assert counts["mcp.client.session"] == 1
            if baseline is None:
                baseline = counts
            assert counts == baseline, (cycle, counts, baseline)
            # Reconstructing the singleton must retain the installed patches.
            McpInstrumentor().uninstrument()
            assert not instrumentor.is_instrumented_by_opentelemetry
            for parent, attribute, original in originals:
                assert resolve_path(parent, attribute)[2] is original, attribute
            assert not await call(False, cycle), "disabled calls exported spans"
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        for current in providers:
            current.shutdown()


if __name__ == "__main__":
    asyncio.run(_exercise_lifecycle(sys.argv[1], sys.argv[2] == "True"))
