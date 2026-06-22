"""Helpers for selecting and instantiating OpenAI-compatible providers."""

import os
from dataclasses import dataclass, field

from openai import OpenAI

from sample_app.atlas_models import (
    ATLASCLOUD_DEFAULT_MODEL,
    ATLASCLOUD_VALIDATED_MODELS,
)


ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    """Configuration required to create an OpenAI-compatible client."""

    name: str
    api_key: str = field(repr=False)
    base_url: str | None
    default_model: str


def get_atlascloud_validated_models() -> tuple[str, ...]:
    """Return the validated Atlas model pool exposed by the sample app."""

    return ATLASCLOUD_VALIDATED_MODELS


def _get_env_value(name: str) -> str | None:
    """Return a stripped env var value, treating blank values as unset."""

    value = os.getenv(name)
    if value is None:
        return None

    value = value.strip()
    return value or None


def get_provider_config(provider_name: str | None = None) -> OpenAICompatibleProviderConfig:
    """Resolve provider configuration from explicit input or environment variables."""

    provider = (provider_name or _get_env_value("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "atlascloud":
        api_key = _get_env_value("ATLASCLOUD_API_KEY")
        if not api_key:
            raise ValueError("Missing Atlas Cloud API key. Set ATLASCLOUD_API_KEY in your environment.")

        return OpenAICompatibleProviderConfig(
            name="atlascloud",
            api_key=api_key,
            base_url=_get_env_value("ATLASCLOUD_BASE_URL") or ATLASCLOUD_BASE_URL,
            default_model=_get_env_value("ATLASCLOUD_MODEL") or ATLASCLOUD_DEFAULT_MODEL,
        )

    if provider == "openai":
        api_key = _get_env_value("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in your environment.")

        return OpenAICompatibleProviderConfig(
            name="openai",
            api_key=api_key,
            base_url=_get_env_value("OPENAI_BASE_URL"),
            default_model=_get_env_value("OPENAI_MODEL") or OPENAI_DEFAULT_MODEL,
        )

    raise ValueError(f"Unsupported provider '{provider}'. Supported values: atlascloud, openai.")


def create_openai_compatible_client(
    provider_name: str | None = None,
) -> tuple[OpenAICompatibleProviderConfig, OpenAI]:
    """Create an OpenAI SDK client using the resolved provider configuration."""

    config = get_provider_config(provider_name)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    return config, client
