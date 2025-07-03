import os
from typing import Literal, cast
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from traceloop.sdk import Traceloop
from mcp.server.fastmcp import Context, FastMCP
from tests.whoami import TestClientResult, WhoamiRequest

transport = cast(
    Literal["sse", "stdio", "streamable-http"], os.environ.get("MCP_TRANSPORT")
)
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
span_exporter = OTLPSpanExporter(f"{otlp_endpoint}/v1/traces")

Traceloop.init(app_name="StockMarketAgent", exporter=span_exporter)
server = FastMCP(port=0)


@server.tool()
async def hello(ctx: Context) -> int:
    response = await ctx.session.send_request(
        WhoamiRequest(method="whoami"), TestClientResult
    )
    name = response.root.name
    return f"Hello {name}!"


server.run(transport=transport)
