import pytest
import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from opentelemetry.trace import Tracer
from mcp import ClientSession
from mcp.shared.session import RequestResponder
from mcp.types import ClientResult, ServerNotification, ServerRequest
from tests.whoami import TestClientResult, TestServerRequest, WhoamiResult

from tests.trace_collector import OTLPServer, Telemetry


@asynccontextmanager
async def mcp_client(
    transport: str, tracer: Tracer, otlp_endpoint: str
) -> AsyncGenerator[ClientSession, None]:
    async def message_handler(
        message: (
            RequestResponder[ServerRequest, ClientResult]
            | ServerNotification
            | Exception
        ),
    ) -> None:
        if (
            not isinstance(message, RequestResponder)
            or message.request.root.method != "whoami"
        ):
            return
        with message as responder:
            await responder.respond(TestClientResult(WhoamiResult(name="World")))

    server_script = str(Path(__file__).parent / "mcp_server.py")
    pythonpath = str(Path(__file__).parent.parent)

    match transport:
        case "sse":
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                server_script,
                env={
                    "MCP_TRANSPORT": "sse",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
                    "PYTHONPATH": pythonpath,
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                from mcp.client.sse import sse_client

                stderr = proc.stderr
                port = None
                for i in range(10):
                    line = str(await stderr.readline())
                    if "Uvicorn running" in line:
                        _, rest = line.split("http://127.0.0.1:", 1)
                        port, _ = rest.split(" ", 1)
                        break
                async with (
                    sse_client(f"http://localhost:{port}/sse") as (reader, writer),
                    ClientSession(
                        reader, writer, message_handler=message_handler
                    ) as client,
                ):
                    client._receive_request_type = TestServerRequest
                    await client.initialize()
                    yield client
            finally:
                proc.kill()
                await proc.wait()
        case "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            async with (
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[server_script],
                        env={
                            "MCP_TRANSPORT": "stdio",
                            "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
                            "PYTHONPATH": pythonpath,
                        },
                    )
                ) as (reader, writer),
                ClientSession(
                    reader, writer, message_handler=message_handler
                ) as client,
            ):
                client._receive_request_type = TestServerRequest
                await client.initialize()
                yield client


@pytest.mark.parametrize("transport", ["sse", "stdio"])
async def test_mcp_instrumentor(
    transport: str, tracer: Tracer, telemetry: Telemetry, otlp_collector: OTLPServer
) -> None:
    async with mcp_client(
        transport, tracer, f"http://localhost:{otlp_collector.server_port}/"
    ) as client:
        tools_res = await client.list_tools()
        assert len(tools_res.tools) == 1
        assert tools_res.tools[0].name == "hello"

    traceids = set()
    for spans in telemetry.traces:
        for scope_spans in spans.scope_spans:
            for scope_span in scope_spans.spans:
                traceids.add(scope_span.trace_id)
    assert len(traceids) == 3
