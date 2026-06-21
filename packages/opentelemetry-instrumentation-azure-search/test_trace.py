from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
from unittest.mock import patch, MagicMock
from azure.search.documents import SearchClient

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

AzureSearchInstrumentor().instrument()

client = SearchClient.__new__(SearchClient)
client._index_name = "demo-index"
client._endpoint = "https://demo.search.windows.net"

with patch("azure.core.pipeline.Pipeline.run", return_value=MagicMock(
    http_response=MagicMock(status_code=200, text=lambda encoding=None: '{"value": []}', headers={})
)):
    try:
        client.search("azure openai", top=5)
    except Exception:
        pass

print("Done")