"""
Unit tests for multimodal content block → OTel part conversion.

Covers _dict_block_to_part and _object_block_to_part helpers in _hooks.py:
  - data: URLs in image_url must produce BlobPart, not UriPart
  - input_audio must include mime_type when format is provided
"""

import pytest
from types import SimpleNamespace


def _dict_block(block: dict):
    from opentelemetry.instrumentation.openai_agents._hooks import _dict_block_to_part
    return _dict_block_to_part(block)


def _object_block(block):
    from opentelemetry.instrumentation.openai_agents._hooks import _object_block_to_part
    return _object_block_to_part(block)


class TestImageUrlDataUrlBecomesBlob:
    """image_url with a data: URL must produce BlobPart, not UriPart."""

    def test_data_url_png_produces_blob_part(self):
        block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123=="}}
        part = _dict_block(block)
        assert part["type"] == "blob", f"Expected blob, got {part['type']}"
        assert part["modality"] == "image"
        assert part["content"] == "abc123=="
        assert part.get("mime_type") == "image/png"

    def test_data_url_jpeg_produces_blob_part(self):
        block = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ=="}}
        part = _dict_block(block)
        assert part["type"] == "blob"
        assert part["content"] == "/9j/4AAQ=="
        assert part.get("mime_type") == "image/jpeg"

    def test_data_url_no_mime_produces_blob_without_mime_type(self):
        """data: URL with no detectable MIME must still be BlobPart (no mime_type key)."""
        block = {"type": "image_url", "image_url": {"url": "data:base64,abc123"}}
        part = _dict_block(block)
        assert part["type"] == "blob"
        # mime_type may be absent or empty — must NOT be a URI
        assert "uri" not in part

    def test_https_url_still_produces_uri_part(self):
        """Regular https: URLs must remain UriPart."""
        block = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        part = _dict_block(block)
        assert part["type"] == "uri"
        assert part["modality"] == "image"
        assert part["uri"] == "https://example.com/image.png"

    def test_http_url_still_produces_uri_part(self):
        block = {"type": "image_url", "image_url": {"url": "http://cdn.example.com/photo.jpg"}}
        part = _dict_block(block)
        assert part["type"] == "uri"
        assert part["uri"] == "http://cdn.example.com/photo.jpg"

    def test_object_block_data_url_produces_blob_part(self):
        """SDK object path (_object_block_to_part) must also handle data: URLs."""
        image_url_obj = SimpleNamespace(url="data:image/png;base64,xyz789==")
        block = SimpleNamespace(type="image_url", image_url=image_url_obj)
        part = _object_block(block)
        assert part["type"] == "blob", f"Expected blob, got {part['type']}"
        assert part["modality"] == "image"
        assert part["content"] == "xyz789=="
        assert part.get("mime_type") == "image/png"

    def test_object_block_https_url_produces_uri_part(self):
        image_url_obj = SimpleNamespace(url="https://example.com/photo.png")
        block = SimpleNamespace(type="image_url", image_url=image_url_obj)
        part = _object_block(block)
        assert part["type"] == "uri"
        assert part["uri"] == "https://example.com/photo.png"


class TestInputAudioMimeType:
    """input_audio BlobPart must include mime_type when format is provided."""

    def test_wav_format_produces_audio_wav_mime_type(self):
        block = {"type": "input_audio", "input_audio": {"data": "audiobase64==", "format": "wav"}}
        part = _dict_block(block)
        assert part["type"] == "blob"
        assert part["modality"] == "audio"
        assert part["content"] == "audiobase64=="
        assert part.get("mime_type") == "audio/wav", (
            f"Expected audio/wav, got {part.get('mime_type')!r}"
        )

    def test_mp3_format_produces_audio_mpeg_mime_type(self):
        """MP3 format maps to audio/mpeg per IANA media types."""
        block = {"type": "input_audio", "input_audio": {"data": "mp3base64==", "format": "mp3"}}
        part = _dict_block(block)
        assert part["type"] == "blob"
        assert part.get("mime_type") == "audio/mpeg", (
            f"Expected audio/mpeg, got {part.get('mime_type')!r}"
        )

    def test_ogg_format_produces_audio_ogg_mime_type(self):
        block = {"type": "input_audio", "input_audio": {"data": "oggdata==", "format": "ogg"}}
        part = _dict_block(block)
        assert part.get("mime_type") == "audio/ogg"

    def test_no_format_omits_mime_type(self):
        """When format is absent, mime_type should not be present (don't fabricate it)."""
        block = {"type": "input_audio", "input_audio": {"data": "rawdata=="}}
        part = _dict_block(block)
        assert part["type"] == "blob"
        assert part["modality"] == "audio"
        assert "mime_type" not in part

    def test_object_block_wav_format_produces_mime_type(self):
        """SDK object path (_object_block_to_part) must also include mime_type."""
        audio_obj = SimpleNamespace(data="audiobase64==", format="wav")
        block = SimpleNamespace(type="input_audio", input_audio=audio_obj)
        part = _object_block(block)
        assert part["type"] == "blob"
        assert part["modality"] == "audio"
        assert part["content"] == "audiobase64=="
        assert part.get("mime_type") == "audio/wav", (
            f"Expected audio/wav, got {part.get('mime_type')!r}"
        )

    def test_object_block_mp3_format_produces_mpeg_mime_type(self):
        audio_obj = SimpleNamespace(data="mp3data==", format="mp3")
        block = SimpleNamespace(type="input_audio", input_audio=audio_obj)
        part = _object_block(block)
        assert part.get("mime_type") == "audio/mpeg"
