from unittest.mock import Mock
from opentelemetry.instrumentation.langchain.span_utils import (
    _extract_model_name_from_request,
    _infer_model_from_class_name,
    extract_model_name_from_response_metadata,
    SpanHolder,
)
from langchain_core.outputs import LLMResult, Generation, ChatGeneration
from langchain_core.messages import AIMessage


class TestModelExtraction:
    """Test enhanced model name extraction for third-party integrations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_span = Mock()
        self.span_holder = SpanHolder(
            span=self.mock_span,
            token=None,
            context=None,
            children=[],
            workflow_name="test",
            entity_name="test",
            entity_path="test"
        )

    def test_standard_model_extraction_from_kwargs(self):
        """Test standard model extraction from kwargs."""
        kwargs = {"model": "gpt-4"}

        result = _extract_model_name_from_request(kwargs, self.span_holder)
        assert result == "gpt-4"

    def test_model_extraction_from_invocation_params(self):
        """Test model extraction from invocation_params."""
        kwargs = {
            "invocation_params": {
                "model": "claude-3-sonnet"
            }
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder)
        assert result == "claude-3-sonnet"

    def test_deepseek_model_extraction_from_serialized(self):
        """Test ChatDeepSeek model extraction from serialized data."""
        kwargs = {}
        serialized = {
            "id": ["langchain", "chat_models", "ChatDeepSeek"],
            "kwargs": {
                "model": "deepseek-coder",
                "api_base": "https://api.deepseek.com/beta"
            }
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized)
        assert result == "deepseek-coder"

    def test_deepseek_fallback_when_no_model_in_serialized(self):
        """Test ChatDeepSeek fallback to default model."""
        kwargs = {}
        serialized = {
            "id": ["langchain", "chat_models", "ChatDeepSeek"],
            "kwargs": {
                "api_base": "https://api.deepseek.com/beta"
                # No model field
            }
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized)
        assert result == "deepseek-unknown"

    def test_class_name_inference_for_known_models(self):
        """Test model inference from class names for known integrations."""
        test_cases = [
            ("ChatOpenAI", {}, "gpt-unknown"),
            ("ChatAnthropic", {}, "claude-unknown"),
            ("ChatCohere", {}, "command-unknown"),
            ("ChatOllama", {}, "ollama-unknown"),
        ]

        for class_name, serialized, expected in test_cases:
            result = _infer_model_from_class_name(class_name, serialized)
            assert result == expected, f"Failed for {class_name}"

    def test_unknown_class_returns_unknown(self):
        """Test that unknown class names return unknown."""
        result = _infer_model_from_class_name("SomeUnknownModel", {})
        assert result == "unknown"

    def test_enhanced_response_metadata_extraction(self):
        """Test enhanced response metadata extraction."""

        # Test with response_metadata
        message = AIMessage(
            content="Test response",
            response_metadata={"model": "deepseek-v2"}
        )
        generation = ChatGeneration(message=message)
        response = LLMResult(generations=[[generation]])

        result = extract_model_name_from_response_metadata(response)
        assert result == "deepseek-v2"

    def test_response_extraction_from_llm_output(self):
        """Test model extraction from llm_output."""
        response = LLMResult(
            generations=[[Generation(text="Test")]],
            llm_output={"model": "deepseek-coder-v2"}
        )

        result = extract_model_name_from_response_metadata(response)
        assert result == "deepseek-coder-v2"

    def test_response_extraction_from_generation_info(self):
        """Test model extraction from generation_info."""
        generation = Generation(
            text="Test response",
            generation_info={"model_name": "deepseek-reasoner"}
        )
        response = LLMResult(generations=[[generation]])

        result = extract_model_name_from_response_metadata(response)
        assert result == "deepseek-reasoner"

    def test_model_extraction_priority_order(self):
        """Test that model extraction follows correct priority order."""
        # kwargs should have highest priority
        kwargs = {"model": "from-kwargs"}
        serialized = {
            "id": ["ChatDeepSeek"],
            "kwargs": {"model": "from-serialized"}
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized)
        assert result == "from-kwargs"

        # invocation_params should be second priority
        kwargs = {
            "invocation_params": {"model": "from-invocation-params"},
            "kwargs": {"model": "from-nested-kwargs"}
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized)
        assert result == "from-invocation-params"

    def test_nested_kwargs_extraction(self):
        """Test extraction from nested kwargs structures."""
        kwargs = {
            "kwargs": {"model": "nested-model"}
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder)
        assert result == "nested-model"

    def test_model_kwargs_extraction(self):
        """Test extraction from model_kwargs."""
        kwargs = {
            "model_kwargs": {"model_name": "model-from-kwargs"}
        }

        result = _extract_model_name_from_request(kwargs, self.span_holder)
        assert result == "model-from-kwargs"

    def test_no_model_info_returns_unknown(self):
        """Test that missing model info returns unknown."""
        kwargs = {}
        serialized = {"id": ["UnknownModel"]}

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized)
        assert result == "unknown"

    def test_association_metadata_extraction(self):
        """Test model extraction from association metadata (ChatDeepSeek pattern)."""
        kwargs = {}
        metadata = {"ls_model_name": "deepseek-reasoner"}

        result = _extract_model_name_from_request(kwargs, self.span_holder, None, metadata)
        assert result == "deepseek-reasoner"

    def test_metadata_has_priority_over_class_inference(self):
        """Test that association metadata has higher priority than class inference."""
        kwargs = {}
        serialized = {
            "id": ["ChatDeepSeek"],
            "kwargs": {}  # No model in serialized kwargs
        }
        metadata = {"ls_model_name": "deepseek-v3"}

        result = _extract_model_name_from_request(kwargs, self.span_holder, serialized, metadata)
        assert result == "deepseek-v3"  # Should use metadata, not fallback to "deepseek-chat"
