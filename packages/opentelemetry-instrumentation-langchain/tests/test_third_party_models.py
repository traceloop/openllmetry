from unittest.mock import Mock
from uuid import uuid4
from opentelemetry.instrumentation.langchain.callback_handler import TraceloopCallbackHandler
from opentelemetry.instrumentation.langchain.span_utils import SpanHolder
from opentelemetry.semconv_ai import SpanAttributes
from langchain_core.outputs import LLMResult, Generation


class TestThirdPartyModels:
    """Test enhanced model extraction for third-party LangChain integrations like ChatDeepSeek."""

    def setup_method(self):
        self.tracer = Mock()
        self.duration_histogram = Mock()
        self.token_histogram = Mock()
        self.ttft_histogram = Mock()
        self.streaming_time_histogram = Mock()
        self.choices_counter = Mock()
        self.exception_counter = Mock()

        self.handler = TraceloopCallbackHandler(
            self.tracer,
            self.duration_histogram,
            self.token_histogram,
            self.ttft_histogram,
            self.streaming_time_histogram,
            self.choices_counter,
            self.exception_counter,
        )

    def test_chatdeepseek_support(self):
        """Test model extraction and streaming metrics for models that store info in serialized kwargs."""
        run_id = uuid4()

        # Test model extraction from serialized kwargs
        serialized = {
            "id": ["langchain_deepseek", "chat_models", "ChatDeepSeek"],
            "kwargs": {
                "model": "deepseek-reasoner",
                "api_base": "https://api.deepseek.com/beta",
                "temperature": 0.7
            }
        }

        kwargs = {
            "invocation_params": {
                "temperature": 0.7
            }
        }

        # Start chat model
        self.handler.on_chat_model_start(
            serialized=serialized,
            messages=[],
            run_id=run_id,
            kwargs=kwargs
        )

        # Verify model extraction
        assert run_id in self.handler.spans
        span_holder = self.handler.spans[run_id]
        assert span_holder.request_model == "deepseek-reasoner"

        span = span_holder.span
        span.set_attribute.assert_any_call(SpanAttributes.LLM_REQUEST_MODEL, "deepseek-reasoner")

        span_attrs = {
            SpanAttributes.LLM_SYSTEM: "Langchain",
            SpanAttributes.LLM_REQUEST_TYPE: "chat"
        }
        span.attributes = Mock()
        span.attributes.get = lambda key, default=None: span_attrs.get(key, default)

        # Test fallback behavior when no model in serialized kwargs
        fallback_serialized = {
            "id": ["langchain_deepseek", "chat_models", "ChatDeepSeek"],
            "kwargs": {
                "api_base": "https://api.deepseek.com/beta"
                # No model field
            }
        }

        run_id_fallback = uuid4()
        self.handler.on_chat_model_start(
            serialized=fallback_serialized,
            messages=[],
            run_id=run_id_fallback,
            kwargs={}
        )

        span_holder_fallback = self.handler.spans[run_id_fallback]
        assert span_holder_fallback.request_model == "deepseek-unknown"

        # Test response processing without model info in llm_output
        response = LLMResult(
            generations=[[Generation(text="Response")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
        )

        self.handler.on_llm_end(response, run_id=run_id)

        # Verify that metrics use correct model name from request fallback
        token_calls = self.token_histogram.record.call_args_list

        # Verify both calls use correct model name from request fallback
        assert len(token_calls) == 2, f"Expected 2 token calls, got {len(token_calls)}"

        for call in token_calls:
            attributes = call[1]["attributes"]
            assert attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "deepseek-reasoner"

        # Test model variant extraction
        from opentelemetry.instrumentation.langchain.span_utils import _extract_model_name_from_request

        variant_serialized = {
            "id": ["langchain_deepseek", "ChatDeepSeek"],
            "kwargs": {
                "model": "deepseek-coder-v2",
                "api_base": "https://api.deepseek.com/beta"
            }
        }

        test_span_holder = SpanHolder(
            span=Mock(),
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test"
        )

        test_models = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner", "deepseek-coder-v2"]
        for model in test_models:
            variant_serialized["kwargs"]["model"] = model
            result = _extract_model_name_from_request({}, test_span_holder, variant_serialized)
            assert result == model, f"Failed to extract {model}"
