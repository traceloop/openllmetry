import os
import pytest
from aleph_alpha_client import Client, Prompt, CompletionRequest

@pytest.mark.vcr
def test_alephalpha_completion(exporter):
    """Test the Aleph Alpha completion request and validate OpenTelemetry spans."""
    
    # Initialize the Aleph Alpha client with the API token from environment variables
    client = Client(token=os.environ.get("AA_TOKEN"))
    
    # Define the prompt for the completion request
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    
    # Create the completion request
    request = CompletionRequest(**params)
    response = client.complete(request, model="luminous-base")

    # Get the finished spans from the exporter
    spans = exporter.get_finished_spans()
    assert spans, "No spans were returned from the exporter"

    together_span = spans[0]
    
    # Validate the span name
    assert together_span.name == "alephalpha.completion"

    # Validate the attributes of the span
    assert together_span.attributes.get("gen_ai.system") == "AlephAlpha"
    assert together_span.attributes.get("llm.request.type") == "completion"
    assert together_span.attributes.get("gen_ai.request.model") == "luminous-base"
    assert (
        together_span.attributes.get("gen_ai.prompt.0.content")
        == prompt_text
    )
    assert (
        together_span.attributes.get("gen_ai.completion.0.content")
        == response.completions[0].completion
    )
    
    # Validate token usage attributes
    assert together_span.attributes.get("gen_ai.usage.prompt_tokens") is not None, "Prompt tokens should not be None"
    assert together_span.attributes.get("llm.usage.total_tokens") == together_span.attributes.get(
        "gen_ai.usage.completion_tokens"
    ) + together_span.attributes.get("gen_ai.usage.prompt_tokens")
