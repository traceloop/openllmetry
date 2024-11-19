import pytest
from opentelemetry.instrumentation.azureai import AzureSearchIntegration
def test_azure_search_integration():
    integration = AzureSearchIntegration(
        endpoint="https://test-search.search.windows.net",
        key="test-key",
        index_name="test-index"
    )
    
    # Test basic search
    results = integration.search("test query")
    assert "results" in results
    assert "count" in results
    
    # Test search with filter
    filtered_results = integration.search(
        "test query",
        filter="category eq 'test'"
    )
    assert "results" in filtered_results
    
    # Test error handling
    with pytest.raises(Exception):
        integration.search("")