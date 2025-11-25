"""
Attachment classes for handling file uploads and downloads in datasets.
Simplified implementation inspired by Braintrust's attachment pattern.
"""

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from traceloop.sdk.client.http import HTTPClient

from .model import (
    ExternalURLRequest,
    FileCellType,
    FileStorageType,
    UploadStatusRequest,
    UploadURLRequest,
    UploadURLResponse,
)


class Attachment:
    """
    Represents a file to be uploaded to a dataset cell.
    Supports both file paths and in-memory data.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        data: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        file_type: Optional[FileCellType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        with_thumbnail: bool = False,
        thumbnail_path: Optional[str] = None,
        thumbnail_data: Optional[bytes] = None,
    ):
        # Validate input
        if file_path and data:
            raise ValueError("Cannot provide both file_path and data")
        if not file_path and not data:
            raise ValueError("Must provide either file_path or data")

        self.file_path = file_path
        self.data = data
        self.metadata = metadata or {}
        self.with_thumbnail = with_thumbnail
        self.thumbnail_path = thumbnail_path
        self.thumbnail_data = thumbnail_data

        # Set filename
        if filename:
            self.filename = filename
        elif file_path:
            self.filename = os.path.basename(file_path)
        else:
            self.filename = "attachment"

        # Set content type
        if content_type:
            self.content_type = content_type
        elif file_path:
            self.content_type = (
                mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            )
        else:
            self.content_type = "application/octet-stream"

        # Set file type
        if file_type:
            self.file_type = file_type
        else:
            self.file_type = self._guess_file_type()

    def _guess_file_type(self) -> FileCellType:
        """Guess file type from content type."""
        if self.content_type.startswith("image/"):
            return FileCellType.IMAGE
        elif self.content_type.startswith("video/"):
            return FileCellType.VIDEO
        elif self.content_type.startswith("audio/"):
            return FileCellType.AUDIO
        else:
            return FileCellType.FILE

    def _get_file_data(self) -> bytes:
        """Get file data as bytes."""
        if self.data is not None:
            return self.data
        elif self.file_path:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("No file data available")

    def _get_file_size(self) -> int:
        """Get file size in bytes."""
        if self.data is not None:
            return len(self.data)
        elif self.file_path and os.path.exists(self.file_path):
            return os.path.getsize(self.file_path)
        return 0

    def upload(
        self,
        http_client: HTTPClient,
        dataset_slug: str,
        row_id: str,
        column_slug: str,
    ) -> "AttachmentReference":
        """Upload the attachment to a dataset cell."""
        # Request upload URL
        request = UploadURLRequest(
            type=self.file_type,
            file_name=self.filename,
            content_type=self.content_type,
            with_thumbnail=self.with_thumbnail,
            metadata=self.metadata,
        )

        result = http_client.post(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/upload-url",
            request.model_dump(),
        )

        if not result:
            raise Exception(f"Failed to get upload URL for {column_slug}")

        upload_response = UploadURLResponse(**result)

        # Upload to S3
        if not self._upload_to_s3(upload_response.upload_url):
            raise Exception(f"Failed to upload {self.filename}")

        # Upload thumbnail if provided
        if self.with_thumbnail and upload_response.thumbnail_upload_url:
            if self.thumbnail_data is not None:
                thumb_bytes = self.thumbnail_data
            elif self.thumbnail_path:
                with open(self.thumbnail_path, "rb") as f:
                    thumb_bytes = f.read()
            else:
                thumb_bytes = None
            if thumb_bytes is not None:
                requests.put(upload_response.thumbnail_upload_url, data=thumb_bytes)

        # Confirm upload
        metadata = self.metadata.copy()
        metadata["size_bytes"] = self._get_file_size()

        status_request = UploadStatusRequest(status="success", metadata=metadata)
        http_client.put(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/upload-status",
            status_request.model_dump(),
        )

        return AttachmentReference(
            storage_type=FileStorageType.INTERNAL,
            storage_key=upload_response.storage_key,
            file_type=self.file_type,
            metadata=metadata,
        )

    def _upload_to_s3(self, upload_url: str) -> bool:
        """Upload file to S3."""
        try:
            file_data = self._get_file_data()
            response = requests.put(
                upload_url, data=file_data, headers={"Content-Type": self.content_type}
            )
            return response.status_code in [200, 201, 204]
        except Exception:
            return False


class ExternalAttachment:
    """
    Represents an external file URL to be linked to a dataset cell.
    """

    def __init__(
        self,
        url: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        file_type: FileCellType = FileCellType.FILE,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.filename = filename or url.split("/")[-1]
        self.content_type = content_type
        self.file_type = file_type
        self.metadata = metadata or {}

    def attach(
        self,
        http_client: HTTPClient,
        dataset_slug: str,
        row_id: str,
        column_slug: str,
    ) -> "AttachmentReference":
        """Attach external URL to a dataset cell."""
        request = ExternalURLRequest(
            type=self.file_type,
            url=self.url,
            metadata=self.metadata,
        )

        result = http_client.post(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/external-url",
            request.model_dump(),
        )

        if not result:
            raise Exception(f"Failed to set external URL for {column_slug}")

        return AttachmentReference(
            storage_type=FileStorageType.EXTERNAL,
            url=self.url,
            file_type=self.file_type,
            metadata=self.metadata,
        )


class AttachmentReference:
    """
    Reference to an attachment in a dataset cell.
    """

    def __init__(
        self,
        storage_type: FileStorageType,
        storage_key: Optional[str] = None,
        url: Optional[str] = None,
        file_type: Optional[FileCellType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        http_client: Optional[HTTPClient] = None,
        dataset_slug: Optional[str] = None,
    ):
        self.storage_type = storage_type
        self.storage_key = storage_key
        self.url = url
        self.file_type = file_type
        self.metadata = metadata or {}
        self.http_client = http_client
        self.dataset_slug = dataset_slug
        self._cached_data: Optional[bytes] = None

    @property
    def data(self) -> bytes:
        """Download and return attachment data as bytes."""
        if self._cached_data is None:
            download_url = self.get_url()
            if not download_url:
                raise Exception("No download URL available")
            response = requests.get(download_url)
            response.raise_for_status()
            self._cached_data = response.content
        return self._cached_data

    def download(self, file_path: Optional[str] = None) -> Optional[bytes]:
        """Download the attachment."""
        file_data = self.data
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file_data)
            return None
        return file_data

    def get_url(self) -> Optional[str]:
        """Get download URL for the attachment."""
        if self.storage_type == FileStorageType.EXTERNAL:
            return self.url
        # For internal storage, would need to implement presigned URL generation
        return None

    def __repr__(self) -> str:
        """String representation."""
        if self.storage_type == FileStorageType.EXTERNAL:
            return f"<AttachmentReference(external, url={self.url})>"
        else:
            return f"<AttachmentReference(internal, key={self.storage_key})>"
