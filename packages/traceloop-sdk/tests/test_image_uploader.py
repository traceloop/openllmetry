from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from traceloop.sdk import (
    ImageUploader as SDKImageUploader,
    Traceloop,
    TraceloopImageUploader as SDKTraceloopImageUploader,
)
from traceloop.sdk.images import ImageUploader, TraceloopImageUploader
from traceloop.sdk.images.image_uploader import (
    ImageUploader as LegacyPathImageUploader,
)


class RecordingImageUploader(ImageUploader):
    def __init__(self) -> None:
        self.calls = []

    async def aupload_base64_image(self, trace_id: str, span_id: str, image_name: str, image_file: str) -> str:
        self.calls.append((trace_id, span_id, image_name, image_file))
        return f"custom://{trace_id}/{span_id}/{image_name}"


class FalseyImageUploader(RecordingImageUploader):
    def __bool__(self) -> bool:
        return False


def test_image_uploader_is_abstract() -> None:
    with pytest.raises(TypeError):
        ImageUploader()  # type: ignore[abstract]


def test_sync_upload_delegates_to_async_implementation() -> None:
    uploader = RecordingImageUploader()

    url = uploader.upload_base64_image("trace", "span", "image.png", "base64")

    assert url == "custom://trace/span/image.png"
    assert uploader.calls == [("trace", "span", "image.png", "base64")]


@pytest.mark.asyncio
async def test_async_upload_uses_custom_implementation() -> None:
    uploader = RecordingImageUploader()

    url = await uploader.aupload_base64_image("trace", "span", "image.png", "base64")

    assert url == "custom://trace/span/image.png"
    assert uploader.calls == [("trace", "span", "image.png", "base64")]


@pytest.mark.asyncio
async def test_traceloop_uploader_preserves_backend_request_contract() -> None:
    uploader = TraceloopImageUploader("https://api.example.com", "secret")

    with (
        patch("traceloop.sdk.images.traceloop_image_uploader.requests.post") as post,
        patch.object(uploader, "_async_upload", new_callable=AsyncMock) as upload,
    ):
        post.return_value.json.return_value = {"url": "https://uploads.example.com/image"}

        url = await uploader.aupload_base64_image("trace", "span", "image.png", "base64")

    assert url == "https://uploads.example.com/image"
    post.assert_called_once_with(
        "https://api.example.com/v2/traces/trace/spans/span/images",
        json={"image_name": "image.png"},
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
    )
    upload.assert_awaited_once_with("https://uploads.example.com/image", "base64")


def test_public_imports_remain_available() -> None:
    assert SDKImageUploader is ImageUploader
    assert LegacyPathImageUploader is ImageUploader
    assert SDKTraceloopImageUploader is TraceloopImageUploader


def test_custom_uploader_is_injected_even_when_falsey() -> None:
    uploader = FalseyImageUploader()

    with (
        patch("traceloop.sdk.TracerWrapper") as tracer_wrapper,
        patch("traceloop.sdk.is_metrics_enabled", return_value=False),
        patch("traceloop.sdk.is_logging_enabled", return_value=False),
    ):
        Traceloop.init(
            exporter=InMemorySpanExporter(),
            image_uploader=uploader,
            resource_attributes={},
        )

    assert tracer_wrapper.call_args.kwargs["image_uploader"] is uploader


def test_traceloop_uploader_is_the_default() -> None:
    with (
        patch("traceloop.sdk.TracerWrapper") as tracer_wrapper,
        patch("traceloop.sdk.is_metrics_enabled", return_value=False),
        patch("traceloop.sdk.is_logging_enabled", return_value=False),
    ):
        Traceloop.init(
            exporter=InMemorySpanExporter(),
            api_endpoint="https://api.example.com",
            api_key="secret",
            resource_attributes={},
        )

    uploader = tracer_wrapper.call_args.kwargs["image_uploader"]
    assert isinstance(uploader, TraceloopImageUploader)
    assert uploader.base_url == "https://api.example.com"
    assert uploader.api_key == "secret"
