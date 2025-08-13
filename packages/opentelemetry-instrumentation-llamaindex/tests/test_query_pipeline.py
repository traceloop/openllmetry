import pytest
# QueryPipeline module not available in llama-index 0.13.1
# from llama_index.core.query_pipeline import QueryPipeline


@pytest.mark.skip(reason="QueryPipeline not available in llama-index 0.13.1")
@pytest.mark.vcr
def test_query_pipeline(instrument_legacy, span_exporter):
    # Test skipped - QueryPipeline not available in llama-index 0.13.1
    pass
