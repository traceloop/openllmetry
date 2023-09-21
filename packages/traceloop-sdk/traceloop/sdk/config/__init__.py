import os


def is_prompt_registry_enabled() -> bool:
    return os.getenv("TRACELOOP_PROMPT_REGISTRY_ENABLED") == "true"


def base_url() -> str:
    return os.getenv("TRACELOOP_BASE_URL") or "https://api.traceloop.com"
