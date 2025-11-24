from typing import Dict, Any, Optional, TYPE_CHECKING
import os

from .base import BaseDatasetEntity
from traceloop.sdk.client.http import HTTPClient
from .model import (
    FileCellType,
    FileStorageType,
    UploadURLRequest,
    UploadURLResponse,
    UploadStatusRequest,
    ExternalURLRequest,
)

if TYPE_CHECKING:
    from .dataset import Dataset


class Row(BaseDatasetEntity):
    id: str
    values: Dict[str, Any]
    _dataset: "Dataset"
    deleted: bool = False

    def __init__(
        self,
        http: HTTPClient,
        dataset: "Dataset",
        id: str,
        values: Dict[str, Any],
    ):
        super().__init__(http)
        self._dataset = dataset
        self.id = id
        self.values = values

    def delete(self) -> None:
        """Remove this row from dataset"""
        if self.deleted:
            raise Exception(f"Row {self.id} already deleted")

        result = self._http.delete(f"datasets/{self._dataset.slug}/rows/{self.id}")
        if result is None:
            raise Exception(f"Failed to delete row {self.id}")
        if self._dataset.rows and self in self._dataset.rows:
            self._dataset.rows.remove(self)
            self.deleted = True

    def update(self, values: Dict[str, Any]) -> None:
        """Update this row's values"""
        if self.deleted:
            raise Exception(f"Row {self.id} already deleted")

        data = {"values": values}
        result = self._http.put(f"datasets/{self._dataset.slug}/rows/{self.id}", data)
        if result is None:
            raise Exception(f"Failed to update row {self.id}")
        self.values.update(values)

    def _request_file_upload(
        self,
        column_slug: str,
        file_name: str,
        file_type: FileCellType,
        content_type: Optional[str] = None,
        with_thumbnail: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UploadURLResponse:
        """Request presigned upload URL for file cell (private method)."""
        request_data = UploadURLRequest(
            type=file_type,
            file_name=file_name,
            content_type=content_type,
            with_thumbnail=with_thumbnail,
            metadata=metadata or {},
        )

        result = self._http.post(
            f"datasets/{self._dataset.slug}/rows/{self.id}/cells/{column_slug}/upload-url",
            request_data.model_dump(),
        )
        if result is None:
            raise Exception(f"Failed to request upload URL for cell {column_slug}")

        return UploadURLResponse(**result)

    def _upload_file_to_presigned_url(
        self,
        upload_url: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """Upload file to presigned S3 URL (private method)."""
        import requests

        with open(file_path, "rb") as f:
            headers = {}
            if content_type:
                headers["Content-Type"] = content_type
            response = requests.put(upload_url, data=f, headers=headers)
            return response.status_code in [200, 201, 204]

    def _confirm_file_upload(
        self,
        column_slug: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Confirm file upload completion (private method)."""
        request_data = UploadStatusRequest(status=status, metadata=metadata or {})

        result = self._http.put(
            f"datasets/{self._dataset.slug}/rows/{self.id}/cells/{column_slug}/upload-status",
            request_data.model_dump(),
        )

        if result is None:
            raise Exception(f"Failed to confirm upload for cell {column_slug}")

        return result.get("success", False)

    def _set_external_file_url(
        self,
        column_slug: str,
        url: str,
        file_type: FileCellType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set external URL for file cell (private method)."""
        request_data = ExternalURLRequest(
            type=file_type, url=url, metadata=metadata or {}
        )

        result = self._http.post(
            f"datasets/{self._dataset.slug}/rows/{self.id}/cells/{column_slug}/external-url",
            request_data.model_dump(),
        )

        if result is None:
            raise Exception(f"Failed to set external URL for cell {column_slug}")

        # Update local row values
        self.values[column_slug] = {
            "type": file_type.value,
            "status": "success",
            "storage": FileStorageType.EXTERNAL.value,
            "url": url,
            "metadata": metadata or {},
        }

    def set_file_cell(
        self,
        column_slug: str,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        file_type: FileCellType = FileCellType.FILE,
        content_type: Optional[str] = None,
        with_thumbnail: bool = False,
        thumbnail_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set a file cell value by either uploading a local file or linking to an external URL.

        Args:
            column_slug: The column to update
            file_path: Path to local file to upload (mutually exclusive with url)
            url: External URL to link to (mutually exclusive with file_path)
            file_type: Type of file (IMAGE, VIDEO, AUDIO, FILE)
            content_type: MIME type for uploads
            with_thumbnail: Whether to include thumbnail for uploads
            thumbnail_path: Path to thumbnail image
            metadata: Additional metadata

        Raises:
            ValueError: If both file_path and url are provided, or neither is provided
            Exception: If upload or URL setting fails

        Examples:
            # Upload local image
            row.set_file_cell("screenshot", file_path="/path/to/image.png", file_type=FileCellType.IMAGE)

            # Link to YouTube video
            row.set_file_cell("demo_video", url="https://youtube.com/watch?v=xyz", file_type=FileCellType.VIDEO)

            # Upload audio file
            row.set_file_cell("podcast", file_path="/path/to/audio.mp3", file_type=FileCellType.AUDIO)
        """
        if self.deleted:
            raise Exception(f"Row {self.id} already deleted")

        # Validate parameters
        if file_path and url:
            raise ValueError("Cannot provide both file_path and url. Choose one.")
        if not file_path and not url:
            raise ValueError("Must provide either file_path or url.")

        if url:
            # External URL mode
            self._set_external_file_url(
                column_slug=column_slug,
                url=url,
                file_type=file_type,
                metadata=metadata,
            )
        else:
            # Internal storage mode (upload file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_name = os.path.basename(file_path)

            # Step 1: Request upload URL
            upload_response = self._request_file_upload(
                column_slug=column_slug,
                file_name=file_name,
                file_type=file_type,
                content_type=content_type,
                with_thumbnail=with_thumbnail,
                metadata=metadata,
            )

            # Step 2: Upload main file
            success = self._upload_file_to_presigned_url(
                upload_response.upload_url, file_path, content_type
            )

            if not success:
                self._confirm_file_upload(column_slug, status="failed", metadata=metadata)
                raise Exception(f"Failed to upload file to S3: {file_name}")

            # Step 3: Upload thumbnail if provided
            if with_thumbnail and thumbnail_path and upload_response.thumbnail_upload_url:
                if not os.path.exists(thumbnail_path):
                    print(f"Warning: Thumbnail file not found: {thumbnail_path}")
                else:
                    thumb_success = self._upload_file_to_presigned_url(
                        upload_response.thumbnail_upload_url,
                        thumbnail_path,
                        "image/png",  # Thumbnails are typically images
                    )
                    if not thumb_success:
                        print(f"Warning: Failed to upload thumbnail for {file_name}")

            # Step 4: Confirm upload
            final_metadata = metadata or {}
            final_metadata["size_bytes"] = os.path.getsize(file_path)

            self._confirm_file_upload(
                column_slug=column_slug, status="success", metadata=final_metadata
            )

            # Update local row values
            self.values[column_slug] = {
                "type": file_type.value,
                "status": "success",
                "storage": FileStorageType.INTERNAL.value,
                "storage_key": upload_response.storage_key,
                "url": None,  # Will be generated on next fetch
                "metadata": final_metadata,
            }
