from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_upload_base64_image(image_uploader):
    trace_id = "test_trace_id"
    span_id = "test_span_id"
    image_name = "test_image.jpeg"
    base64_image = "dGVzdF9pbWFnZV9kYXRh"

    with patch.object(image_uploader, '_get_image_url', return_value="https://example.com/uploaded_image.jpg") as mock_get_image_url:
        with patch.object(image_uploader, '_async_upload', new_callable=AsyncMock) as mock_async_upload:
            url = image_uploader.upload_base64_image(trace_id, span_id, image_name, base64_image)

            mock_get_image_url.assert_called_once_with(trace_id, span_id, image_name)
            mock_async_upload.assert_awaited_once_with("https://example.com/uploaded_image.jpg", base64_image)
            assert url == "https://example.com/uploaded_image.jpg"
