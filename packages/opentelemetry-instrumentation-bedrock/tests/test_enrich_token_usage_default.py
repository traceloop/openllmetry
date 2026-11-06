"""Unit tests for the `enrich_token_usage` default (#3168).

Standalone instrumentation must collect token usage by default, matching
``Traceloop.init()`` behavior. Constructing the instrumentor only touches
global ``Config`` (no network), so these tests are hermetic.
"""

from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.instrumentation.bedrock.config import Config

ENV_VAR = "TRACELOOP_ENRICH_TOKEN_USAGE"


def test_enrich_token_usage_defaults_to_true(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    previous = Config.enrich_token_usage
    try:
        BedrockInstrumentor()
        assert Config.enrich_token_usage is True
    finally:
        Config.enrich_token_usage = previous


def test_enrich_token_usage_env_override(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "false")
    previous = Config.enrich_token_usage
    try:
        BedrockInstrumentor()
        assert Config.enrich_token_usage is False
    finally:
        Config.enrich_token_usage = previous


def test_enrich_token_usage_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "false")
    previous = Config.enrich_token_usage
    try:
        BedrockInstrumentor(enrich_token_usage=True)
        assert Config.enrich_token_usage is True
    finally:
        Config.enrich_token_usage = previous
