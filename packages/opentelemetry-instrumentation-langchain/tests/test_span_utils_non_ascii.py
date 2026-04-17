"""Verify span_utils json.dumps preserves non-ASCII characters (complements
test_non_ascii_content.py which covers callback_handler.py)."""

import json
from unittest.mock import MagicMock

from opentelemetry.instrumentation.langchain.span_utils import (
    _content_to_parts,
)


def test_content_to_parts_preserves_cjk():
    """Non-ASCII text in multimodal content blocks must not be escaped."""
    content = [{"type": "custom", "data": "日本語テスト"}]
    parts = _content_to_parts(content)
    # The unknown-block-type branch serialises via json.dumps
    assert len(parts) == 1
    assert parts[0]["type"] == "text"
    raw = parts[0]["content"]
    # Must contain the original characters, not \\uXXXX escapes
    assert "日本語テスト" in raw
    assert "\\u" not in raw


def test_content_to_parts_preserves_cyrillic():
    content = [{"type": "custom", "data": "Привет мир"}]
    parts = _content_to_parts(content)
    raw = parts[0]["content"]
    assert "Привет мир" in raw
    assert "\\u" not in raw


def test_content_to_parts_preserves_arabic():
    content = [{"type": "custom", "data": "مرحبا بالعالم"}]
    parts = _content_to_parts(content)
    raw = parts[0]["content"]
    assert "مرحبا" in raw
    assert "\\u" not in raw


def test_content_to_parts_plain_string_unchanged():
    """Plain string content should pass through without json.dumps."""
    parts = _content_to_parts("hello world")
    assert parts == [{"type": "text", "content": "hello world"}]
