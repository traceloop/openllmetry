import pytest
from unittest.mock import Mock, patch
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.trace import Span
from opentelemetry import context as context_api
from opentelemetry.trace import set_tracer_provider, TracerProvider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from contextlib import contextmanager

# Mock the Azure Search SDK
class MockSearchClient:
    def search(self, *args, **kwargs):
        return {"value": [], "count": 0}

class AzureSearchInstrumentor:
    def __init__(self, meter, tracer):
        self.meter = meter
        self.tracer = tracer
        self.search_counter = self.meter.create_counter(
            "azure_search_requests",
            description="Number of Azure Search requests"
        )
        self.latency_histogram = self.meter.create_histogram(
            "azure_search_latency",
            description="Azure Search request latency"
        )
        self.enabled = True

    def instrument_search_client(self, client):
        original_search = client.search

        def instrumented_search(*args, **kwargs):
            if not self.enabled:
                return original_search(*args, **kwargs)

            with self.tracer.start_span("azure_search") as span:
                try:
                    span.set_attribute("search.query", kwargs.get("search_text", ""))
                    span.set_attribute("search.top", kwargs.get("top", 0))
                    
                    result = original_search(*args, **kwargs)
                    
                    self.search_counter.add(1, {"status": "success"})
                    self.latency_histogram.record(0.1)
                    
                    return result
                except Exception as e:
                    self.search_counter.add(1, {"status": "error"})
                    span.record_exception(e)
                    raise

        client.search = instrumented_search
        return client

    def suppress_instrumentation(self):
        self.enabled = False

    def enable_instrumentation(self):
        self.enabled = True

# Test fixtures
@pytest.fixture
def mock_meter():
    meter = Mock()
    meter.create_counter.return_value = Mock(spec=Counter)
    meter.create_histogram.return_value = Mock(spec=Histogram)
    return meter

@pytest.fixture
def mock_span():
    span = MagicMock(spec=Span)
    return span

@pytest.fixture
def mock_tracer(mock_span):
    tracer = MagicMock()
    
    @contextmanager
    def mock_span_context(*args, **kwargs):
        yield mock_span
    
    tracer.start_span = mock_span_context
    return tracer

@pytest.fixture
def search_client():
    return MockSearchClient()

@pytest.fixture
def instrumentation(mock_meter, mock_tracer):
    return AzureSearchInstrumentor(mock_meter, mock_tracer)

# Tests
def test_instrumentation_suppression(instrumentation, search_client, mock_meter):
    instrumented_client = instrumentation.instrument_search_client(search_client)
    instrumentation.suppress_instrumentation()
    
    instrumented_client.search(search_text="test")
    
    # Verify metrics weren't recorded when suppressed
    assert mock_meter.create_counter().add.call_count == 0
    assert mock_meter.create_histogram().record.call_count == 0

def test_normal_instrumentation(instrumentation, search_client, mock_meter):
    instrumented_client = instrumentation.instrument_search_client(search_client)
    
    instrumented_client.search(search_text="test", top=10)
    
    # Verify metrics were recorded
    mock_meter.create_counter().add.assert_called_once_with(1, {"status": "success"})
    mock_meter.create_histogram().record.assert_called_once_with(0.1)

def test_exception_handling(instrumentation, search_client, mock_meter):
    instrumented_client = instrumentation.instrument_search_client(search_client)
    
    # Make the search method raise an exception
    def raise_error(*args, **kwargs):
        raise Exception("Search failed")
    
    search_client.search = raise_error
    
    with pytest.raises(Exception):
        instrumented_client.search(search_text="test")
    
    # Verify error metrics were recorded
    mock_meter.create_counter().add.assert_called_once_with(1, {"status": "error"})

def test_metrics_recording(instrumentation, search_client, mock_meter):
    instrumented_client = instrumentation.instrument_search_client(search_client)
    
    instrumented_client.search(search_text="test")
    
    # Verify both counter and histogram were recorded
    mock_meter.create_counter().add.assert_called_once()
    mock_meter.create_histogram().record.assert_called_once()

def test_search_parameters(instrumentation, search_client, mock_span):
    instrumented_client = instrumentation.instrument_search_client(search_client)
    
    instrumented_client.search(search_text="test query", top=5)
    
    # Verify span attributes were set correctly
    mock_span.set_attribute.assert_any_call("search.query", "test query")
    mock_span.set_attribute.assert_any_call("search.top", 5)