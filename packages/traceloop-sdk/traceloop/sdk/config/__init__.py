import os


def is_tracing_enabled() -> bool:
    return (os.getenv("TRACELOOP_TRACING_ENABLED") or "true") == "true"


def is_prompt_registry_enabled() -> bool:
    return (os.getenv("TRACELOOP_PROMPT_REGISTRY_ENABLED") or "false") == "true"


def base_url() -> str:
    return os.getenv("TRACELOOP_BASE_URL") or "https://api.traceloop.com"
