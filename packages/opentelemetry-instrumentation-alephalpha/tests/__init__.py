"""unit tests."""
import unittest
from unittest.mock import MagicMock
from opentelemetry.trace import Span
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor
from opentelemetry.instrumentation.alephalpha.utils import dont_throw
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.trace.status import Status, StatusCode


class TestAlephAlphaInstrumentor(unittest.TestCase):

    def setUp(self):
        """Set up necessary mocks and test environment."""
        self.tracer_mock = MagicMock()
        self.instrumentor = AlephAlphaInstrumentor()

    def test_legacy_attributes_true(self):
        """Test when use_legacy_attributes is True."""
        # Simulate that the flag is True
        use_legacy_attributes = True
        
        # Mock the span to capture the method calls
        span_mock = MagicMock(spec=Span)
        self.tracer_mock.start_span.return_value = span_mock

        # Wrap the function with legacy attribute flag set to True
        @dont_throw
        def _wrap_with_legacy(tracer, to_wrap, wrapped, instance, args, kwargs):
            span = tracer.start_span("test_span", kind=1, attributes={})
            if span.is_recording():
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
            return wrapped(*args, **kwargs)
        
        # Simulate a tracing event
        _wrap_with_legacy(self.tracer_mock, {"span_name": "test_span"}, MagicMock(), None, [], {"model": "TestModel"})
        
        # Assert that set_attribute is called
        span_mock.set_attribute.assert_called_with(SpanAttributes.LLM_REQUEST_MODEL, "TestModel")

    def test_legacy_attributes_false(self):
        """Test when use_legacy_attributes is False."""
        # Simulate that the flag is False
        use_legacy_attributes = False
        
        # Mock the span to capture the method calls
        span_mock = MagicMock(spec=Span)
        self.tracer_mock.start_span.return_value = span_mock

        # Wrap the function with legacy attribute flag set to False
        @dont_throw
        def _wrap_with_event(tracer, to_wrap, wrapped, instance, args, kwargs):
            span = tracer.start_span("test_span", kind=1, attributes={})
            if span.is_recording():
                span.add_event("llm_request_model", {"model": kwargs.get("model")})
            return wrapped(*args, **kwargs)
        
        # Simulate a tracing event
        _wrap_with_event(self.tracer_mock, {"span_name": "test_span"}, MagicMock(), None, [], {"model": "TestModel"})
        
        # Assert that add_event is called instead of set_attribute
        span_mock.add_event.assert_called_with("llm_request_model", {"model": "TestModel"})

    def test_instrumentation_dependencies(self):
        """Test the instrumentation dependencies are correctly returned."""
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIn("aleph_alpha_client >= 7.1.0, <8", dependencies)

    def test_should_send_prompts(self):
        """Test the should_send_prompts logic."""
        # Mock the environment variable
        with unittest.mock.patch.dict('os.environ', {"TRACELOOP_TRACE_CONTENT": "false"}):
            self.assertFalse(self.instrumentor.should_send_prompts())

        with unittest.mock.patch.dict('os.environ', {"TRACELOOP_TRACE_CONTENT": "true"}):
            self.assertTrue(self.instrumentor.should_send_prompts())

    def test_wrap_function(self):
        """Test the _wrap function to ensure it starts and ends a span correctly."""
        # Set up necessary mocks
        tracer_mock = MagicMock()
        wrapped_method = {"method": "complete", "span_name": "alephalpha.completion"}
        wrapped_function = MagicMock()
        args = ["Test prompt"]
        kwargs = {"model": "TestModel"}

        # Mock the span object
        span_mock = MagicMock(spec=Span)
        tracer_mock.start_span.return_value = span_mock

        # Execute the wrap function
        _wrap = self.instrumentor._wrap(tracer_mock, wrapped_method)
        _wrap(wrapped_function, None, args, kwargs)
        
        # Check if span methods are called
        span_mock.start_span.assert_called_with("alephalpha.completion", kind=1, attributes={})
        span_mock.set_attribute.assert_called_with(SpanAttributes.LLM_REQUEST_MODEL, "TestModel")


if __name__ == "__main__":
    unittest.main()
