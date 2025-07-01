import pytest
import openai
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_azure_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test that Azure OpenAI is detected as 'Azure' vendor"""
    azure_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://test.openai.azure.com/openai/",
    )
    
    azure_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test Azure vendor detection"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "Azure"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4"


@pytest.mark.vcr 
def test_aws_bedrock_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test that AWS Bedrock via OpenAI SDK is detected as 'AWS' vendor"""
    aws_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/",
    )
    
    aws_client.chat.completions.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "Test AWS vendor detection"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"
    # Should strip vendor prefix from model name
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "claude-3-5-sonnet-20241022-v2:0"


@pytest.mark.vcr
def test_google_vertex_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test that Google Vertex AI via OpenAI SDK is detected as 'Google' vendor"""
    google_client = openai.OpenAI(
        api_key="test-key", 
        base_url="https://us-central1-aiplatform.googleapis.com/v1/",
    )
    
    google_client.chat.completions.create(
        model="gemini-pro",
        messages=[{"role": "user", "content": "Test Google vendor detection"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "Google"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-pro"


@pytest.mark.vcr
def test_openrouter_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test that OpenRouter is detected as 'OpenRouter' vendor"""
    openrouter_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1/",
    )
    
    openrouter_client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Test OpenRouter vendor detection"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "OpenRouter"
    # Should strip provider prefix from model name
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o"


@pytest.mark.vcr
def test_default_openai_vendor_detection(instrument_legacy, span_exporter, log_exporter, openai_client):
    """Test that default OpenAI is detected as 'openai' vendor"""
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test default OpenAI vendor detection"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4"


@pytest.mark.vcr
def test_aws_regional_model_stripping(instrument_legacy, span_exporter, log_exporter):
    """Test that regional AWS model IDs are properly stripped"""
    aws_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/",
    )
    
    aws_client.chat.completions.create(
        model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "Test regional model stripping"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "AWS"
    # Should strip regional prefix: us.anthropic.model -> model
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "claude-3-5-sonnet-20241022-v2:0"


@pytest.mark.vcr
def test_embeddings_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test vendor detection works for embeddings endpoint"""
    azure_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://test.openai.azure.com/openai/",
    )
    
    azure_client.embeddings.create(
        model="text-embedding-ada-002",
        input="Test Azure embeddings vendor detection",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.embeddings"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "Azure"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "text-embedding-ada-002"


@pytest.mark.vcr
def test_completions_vendor_detection(instrument_legacy, span_exporter, log_exporter):
    """Test vendor detection works for legacy completions endpoint"""
    openrouter_client = openai.OpenAI(
        api_key="test-key", 
        base_url="https://openrouter.ai/api/v1/",
    )
    
    openrouter_client.completions.create(
        model="anthropic/claude-3-sonnet",
        prompt="Test OpenRouter completions vendor detection",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.completion"
    assert open_ai_span.attributes[SpanAttributes.LLM_SYSTEM] == "OpenRouter"
    # Should strip provider prefix: anthropic/model -> model
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "claude-3-sonnet" 
