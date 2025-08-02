from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans


@dataclass
class Telemetry:
    traces: list[ResourceSpans] = field(default_factory=list)

    def clear(self) -> None:
        self.traces.clear()


class OTLPServer(HTTPServer):
    def __init__(self, server_address: tuple[str, int], telemetry: Telemetry):
        super().__init__(server_address, OTLPHandler, True)
        self.telemetry = telemetry


class OTLPHandler(BaseHTTPRequestHandler):
    server: OTLPServer

    def do_POST(self) -> None:
        request_url = urlparse(self.path)
        if request_url.path == "/v1/traces":
            self._handle_traces()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_traces(self) -> None:
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        self.server.telemetry.traces.extend(request.resource_spans)
        self.send_response(200, "OK")
        self.end_headers()
