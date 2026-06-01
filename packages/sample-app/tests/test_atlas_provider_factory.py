from sample_app.atlas_models import (
    ATLASCLOUD_DEFAULT_MODEL,
    ATLASCLOUD_VALIDATED_MODELS,
)
from sample_app.provider_factory import get_atlascloud_validated_models, get_provider_config


def test_atlascloud_default_model_is_first_validated_model():
    assert ATLASCLOUD_DEFAULT_MODEL == ATLASCLOUD_VALIDATED_MODELS[0]
    assert ATLASCLOUD_DEFAULT_MODEL == "deepseek-ai/DeepSeek-V3-0324"


def test_provider_factory_exposes_validated_atlas_model_pool(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    config = get_provider_config("atlascloud")

    assert config.default_model == ATLASCLOUD_DEFAULT_MODEL
    assert get_atlascloud_validated_models() == ATLASCLOUD_VALIDATED_MODELS
    assert len(ATLASCLOUD_VALIDATED_MODELS) == 50
