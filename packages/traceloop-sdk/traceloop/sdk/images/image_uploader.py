import asyncio
from abc import ABC, abstractmethod


class ImageUploader(ABC):
    """Interface for storing base64-encoded images referenced by spans.

    Custom implementations can be supplied through the ``image_uploader`` argument
    of :meth:`traceloop.sdk.Traceloop.init`.
    """

    def upload_base64_image(self, trace_id: str, span_id: str, image_name: str, image_file: str) -> str:
        """Upload an image synchronously and return its destination URL."""
        return asyncio.run(self.aupload_base64_image(trace_id, span_id, image_name, image_file))

    @abstractmethod
    async def aupload_base64_image(self, trace_id: str, span_id: str, image_name: str, image_file: str) -> str:
        """Upload an image asynchronously and return its destination URL."""
        raise NotImplementedError
