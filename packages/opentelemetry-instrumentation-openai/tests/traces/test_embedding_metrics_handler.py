import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.instrumentation.openai.shared.embeddings_wrappers import _set_embeddings_metrics


@pytest.mark.vcr
def test_set_embeddings_metrics_handles_none_values():
    # Mock the necessary arguments
    instance = MagicMock()
    token_counter = MagicMock()
    vector_size_counter = MagicMock()
    duration_histogram = MagicMock()
    response_dict = {
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": 10,
        },
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
    }
    duration = 1.23

    expected_attributes = {
        'gen_ai.system': 'openai',
        'gen_ai.response.model': 'text-embedding-ada-002',
        'gen_ai.operation.name': 'embeddings',
        'server.address': '',
        'stream': False,
        'gen_ai.token.type': 'output'
    }

    with patch("logging.error") as mock_logging_error:
        _set_embeddings_metrics(
            instance,
            token_counter,
            vector_size_counter,
            duration_histogram,
            response_dict,
            duration,
        )

        # Check that logging.error was called for the None value
        mock_logging_error.assert_called_with("Received None value for prompt_tokens in usage")

        # Ensure token_counter.record was called with the correct attributes
        token_counter.record.assert_called_once_with(10, attributes=expected_attributes)
