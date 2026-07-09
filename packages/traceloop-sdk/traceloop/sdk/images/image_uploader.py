import aiohttp
import asyncio
import logging

import requests


class ImageUploader:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def upload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> None:
        asyncio.run(self.aupload_base64_image(trace_id, span_id, image_name, image_file))

    async def aupload_base64_image(
        self, trace_id: str, span_id: str, image_name: str, image_file: str
    ) -> str:
        url = self._get_image_url(trace_id, span_id, image_name)

        await self._async_upload(url, image_file)

        return url

    def _get_image_url(self, trace_id: str, span_id: str, image_name: str) -> str:
        response = requests.post(
            f"{self.base_url}/v2/traces/{trace_id}/spans/{span_id}/images",
            json={
                "image_name": image_name,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        return response.json()["url"]  # type: ignore[no-any-return]

    async def _async_upload(self, url: str, base64_image: str) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "image_data": base64_image,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status < 200 or response.status >= 300:
                    self.logger.error(
                        f"Failed to upload image. Status code: {response.status}"
                    )
                    self.logger.error(await response.text())
                else:
                    self.logger.info(f"Successfully uploaded image {url}")
