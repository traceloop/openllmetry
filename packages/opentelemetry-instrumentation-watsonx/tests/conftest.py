"""Unit tests configuration module."""

import pytest


pytest_plugins = []


@pytest.fixture
def watson_ai_model():
    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference
    except ImportError:
        print("no supported ibm_watsonx_ai package found, model creating skipped.")
        return None

    watsonx_ai_model = ModelInference(
        model_id="google/flan-ul2",
        project_id="c1234567-2222-2222-3333-444444444444",
        credentials={
                "apikey": "test_api_key",
                "url": "https://us-south.ml.cloud.ibm.com"
                },
    )
    return watsonx_ai_model
