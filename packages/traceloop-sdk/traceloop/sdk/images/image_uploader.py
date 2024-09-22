import aiohttp
import asyncio
import logging

class ImageUploader:
    def __init__(self, base_url, api_key):
        self.base_url = "http://localhost:3000" # base_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Create a handler and set its formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(handler)

    def get_image_format(self, base64_image_url):
        return base64_image_url.split('/')[1].split(';')[0]

    def upload_base64_image(self, trace_id, span_id, image_name, base64_image):
        url = f"{self.base_url}/traces/{trace_id}/spans/{span_id}/images/{image_name}"
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._async_upload(trace_id, span_id, image_name, base64_image))
        
        return url

    async def _async_upload(self, trace_id, span_id, image_name, base64_image):
        url = f"{self.base_url}/v2/traces/{trace_id}/spans/{span_id}/images/"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "image_name": image_name,
            "image_data": base64_image,
            "image_format": self.get_image_format(base64_image)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status < 200 or response.status >= 300:
                    self.logger.error(f"Failed to upload image. Status code: {response.status}")
                    self.logger.error(await response.text())
                else:
                    self.logger.info(f"Successfully uploaded image {image_name}")
