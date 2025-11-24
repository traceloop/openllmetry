"""
Attachment classes for handling file uploads and downloads in datasets.
Inspired by Braintrust's attachment pattern for declarative file handling.
"""

import os
import mimetypes
import requests
from typing import Optional, Dict, Any, Union, BinaryIO
from pathlib import Path

from traceloop.sdk.client.http import HTTPClient
from .model import (
    FileCellType,
    FileStorageType,
    UploadURLRequest,
    UploadURLResponse,
    UploadStatusRequest,
    ExternalURLRequest,
)


class Attachment:
    """
    Represents a file to be uploaded to a dataset cell.

    Supports both file paths and in-memory data.

    Examples:
        # Upload from file path
        attachment = Attachment(file_path="/path/to/image.png")
        row.set_file_cell("image", attachment)

        # Upload from in-memory data
        attachment = Attachment(data=image_bytes, filename="photo.jpg", content_type="image/jpeg")
        row.set_file_cell("photo", attachment)
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        data: Optional[Union[bytes, BinaryIO]] = None,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        file_type: Optional[FileCellType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        with_thumbnail: bool = False,
        thumbnail_path: Optional[str] = None,
        thumbnail_data: Optional[bytes] = None,
    ):
        """
        Initialize an Attachment.

        Args:
            file_path: Path to file on disk (mutually exclusive with data)
            data: In-memory file data (mutually exclusive with file_path)
            filename: Display name for the file
            content_type: MIME type (auto-detected if not provided)
            file_type: Type of file (IMAGE, VIDEO, AUDIO, FILE)
            metadata: Additional metadata to store with the file
            with_thumbnail: Whether to include a thumbnail
            thumbnail_path: Path to thumbnail image file
            thumbnail_data: In-memory thumbnail data
        """
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

        # Determine filename
        if filename:
            self.filename = filename
        elif file_path:
            self.filename = os.path.basename(file_path)
        else:
            self.filename = "attachment"

        # Determine content type
        if content_type:
            self.content_type = content_type
        elif file_path:
            self.content_type = (
                mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            )
        else:
            self.content_type = "application/octet-stream"

        # Determine file type
        if file_type:
            self.file_type = file_type
        else:
            self.file_type = self._guess_file_type()

    def _guess_file_type(self) -> FileCellType:
        """Guess the file type based on content type."""
        if self.content_type.startswith("image/"):
            return FileCellType.IMAGE
        elif self.content_type.startswith("video/"):
            return FileCellType.VIDEO
        elif self.content_type.startswith("audio/"):
            return FileCellType.AUDIO
        else:
            return FileCellType.FILE

    def _get_file_data(self) -> bytes:
        """Get the file data as bytes."""
        if self.data:
            if isinstance(self.data, bytes):
                return self.data
            else:
                # Assume it's a file-like object
                return self.data.read()
        elif self.file_path:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            with open(self.file_path, "rb") as f:
                return f.read()
        else:
            raise ValueError("No file data available")

    def _get_file_size(self) -> int:
        """Get the file size in bytes."""
        if self.file_path:
            return os.path.getsize(self.file_path)
        elif self.data:
            if isinstance(self.data, bytes):
                return len(self.data)
            else:
                # For file-like objects, try to get size
                pos = self.data.tell()
                self.data.seek(0, 2)  # Seek to end
                size = self.data.tell()
                self.data.seek(pos)  # Restore position
                return size
        return 0

    def upload(
        self,
        http_client: HTTPClient,
        dataset_slug: str,
        row_id: str,
        column_slug: str,
    ) -> "AttachmentReference":
        """
        Upload the attachment to a dataset cell.

        Args:
            http_client: HTTP client for API communication
            dataset_slug: Dataset identifier
            row_id: Row identifier
            column_slug: Column identifier

        Returns:
            AttachmentReference for the uploaded file
        """
        # Step 1: Request upload URL
        upload_response = self._request_upload_url(
            http_client, dataset_slug, row_id, column_slug
        )

        # Step 2: Upload file to S3
        success = self._upload_to_s3(upload_response.upload_url)
        if not success:
            self._confirm_upload(
                http_client, dataset_slug, row_id, column_slug, status="failed"
            )
            raise Exception(f"Failed to upload file to S3: {self.filename}")

        # Step 3: Upload thumbnail if provided
        if self.with_thumbnail and upload_response.thumbnail_upload_url:
            self._upload_thumbnail(upload_response.thumbnail_upload_url)

        # Step 4: Confirm upload
        final_metadata = self.metadata.copy()
        final_metadata["size_bytes"] = self._get_file_size()

        self._confirm_upload(
            http_client,
            dataset_slug,
            row_id,
            column_slug,
            status="success",
            metadata=final_metadata,
        )

        # Return reference to uploaded file
        return AttachmentReference(
            storage_type=FileStorageType.INTERNAL,
            storage_key=upload_response.storage_key,
            file_type=self.file_type,
            metadata=final_metadata,
            http_client=http_client,
            dataset_slug=dataset_slug,
        )

    def _request_upload_url(
        self,
        http_client: HTTPClient,
        dataset_slug: str,
        row_id: str,
        column_slug: str,
    ) -> UploadURLResponse:
        """Request presigned upload URL from backend."""
        request_data = UploadURLRequest(
            type=self.file_type,
            file_name=self.filename,
            content_type=self.content_type,
            with_thumbnail=self.with_thumbnail,
            metadata=self.metadata,
        )

        result = http_client.post(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/upload-url",
            request_data.model_dump(),
        )

        if result is None:
            raise Exception(f"Failed to request upload URL for cell {column_slug}")

        return UploadURLResponse(**result)

    def _upload_to_s3(self, upload_url: str) -> bool:
        """Upload file to S3 using presigned URL."""
        try:
            file_data = self._get_file_data()
            headers = {"Content-Type": self.content_type}
            response = requests.put(upload_url, data=file_data, headers=headers)
            return response.status_code in [200, 201, 204]
        except Exception:
            return False

    def _upload_thumbnail(self, thumbnail_url: str) -> bool:
        """Upload thumbnail to S3."""
        try:
            if self.thumbnail_data:
                thumb_data = self.thumbnail_data
            elif self.thumbnail_path:
                if not os.path.exists(self.thumbnail_path):
                    return False
                with open(self.thumbnail_path, "rb") as f:
                    thumb_data = f.read()
            else:
                return False

            headers = {"Content-Type": "image/png"}
            response = requests.put(thumbnail_url, data=thumb_data, headers=headers)
            return response.status_code in [200, 201, 204]
        except Exception:
            return False

    def _confirm_upload(
        self,
        http_client: HTTPClient,
        dataset_slug: str,
        row_id: str,
        column_slug: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Confirm upload completion with backend."""
        request_data = UploadStatusRequest(status=status, metadata=metadata or {})

        result = http_client.put(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/upload-status",
            request_data.model_dump(),
        )

        if result is None:
            raise Exception(f"Failed to confirm upload for cell {column_slug}")

        return result.get("success", False)


class ExternalAttachment:
    """
    Represents an external file URL to be linked to a dataset cell.

    Examples:
        # Link to YouTube video
        attachment = ExternalAttachment(
            url="https://youtube.com/watch?v=example",
            file_type=FileCellType.VIDEO
        )
        row.set_file_cell("video", attachment)

        # Link to Google Docs
        attachment = ExternalAttachment(
            url="https://docs.google.com/document/d/example",
            filename="Project Plan"
        )
        row.set_file_cell("document", attachment)
    """

    def __init__(
        self,
        url: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        file_type: FileCellType = FileCellType.FILE,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an ExternalAttachment.

        Args:
            url: External URL to the file
            filename: Display name for the file
            content_type: MIME type
            file_type: Type of file (IMAGE, VIDEO, AUDIO, FILE)
            metadata: Additional metadata
        """
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
        """
        Attach the external URL to a dataset cell.

        Args:
            http_client: HTTP client for API communication
            dataset_slug: Dataset identifier
            row_id: Row identifier
            column_slug: Column identifier

        Returns:
            AttachmentReference for the external file
        """
        request_data = ExternalURLRequest(
            type=self.file_type,
            url=self.url,
            metadata=self.metadata,
        )

        result = http_client.post(
            f"datasets/{dataset_slug}/rows/{row_id}/cells/{column_slug}/external-url",
            request_data.model_dump(),
        )

        if result is None:
            raise Exception(f"Failed to set external URL for cell {column_slug}")

        return AttachmentReference(
            storage_type=FileStorageType.EXTERNAL,
            url=self.url,
            file_type=self.file_type,
            metadata=self.metadata,
            http_client=http_client,
            dataset_slug=dataset_slug,
        )


class AttachmentReference:
    """
    Reference to an attachment in a dataset cell.
    Provides methods to download or access the attachment data.

    Examples:
        # Get attachment reference from a row
        ref = row.get_file_cell("image")

        # Download to file
        ref.download("/path/to/save.png")

        # Get as bytes
        data = ref.data

        # Get download URL
        url = ref.get_url()
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
        """
        Initialize an AttachmentReference.

        Args:
            storage_type: INTERNAL or EXTERNAL storage
            storage_key: S3 storage key (for internal files)
            url: External URL (for external files)
            file_type: Type of file
            metadata: File metadata
            http_client: HTTP client for downloading
            dataset_slug: Dataset identifier
        """
        self.storage_type = storage_type
        self.storage_key = storage_key
        self.url = url
        self.file_type = file_type
        self.metadata = metadata or {}
        self.http_client = http_client
        self.dataset_slug = dataset_slug
        self._cached_data = None

    @property
    def data(self) -> bytes:
        """
        Download and return the attachment data as bytes.
        Data is cached after first download.
        """
        if self._cached_data is None:
            self._cached_data = self._download_data()
        return self._cached_data

    def _download_data(self) -> bytes:
        """Download the attachment data."""
        download_url = self.get_url()
        if not download_url:
            raise Exception("No download URL available")

        response = requests.get(download_url)
        response.raise_for_status()
        return response.content

    def download(self, file_path: Optional[str] = None) -> Union[bytes, None]:
        """
        Download the attachment.

        Args:
            file_path: If provided, save to this file path.
                      If not provided, return bytes.

        Returns:
            Bytes if file_path not provided, None otherwise
        """
        data = self.data

        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(data)
            return None
        else:
            return data

    def get_url(self) -> Optional[str]:
        """
        Get the download URL for the attachment.

        For internal files, this would typically return a presigned S3 URL.
        For external files, returns the original URL.
        """
        if self.storage_type == FileStorageType.EXTERNAL:
            return self.url
        elif self.storage_type == FileStorageType.INTERNAL and self.storage_key:
            # TODO: Implement getting presigned download URL from backend
            # This would require a new API endpoint like:
            # GET /datasets/{slug}/files/{storage_key}/download-url
            # For now, return None as download URL generation needs backend support
            return None
        else:
            return None

    def __repr__(self) -> str:
        """String representation of the attachment reference."""
        if self.storage_type == FileStorageType.EXTERNAL:
            return f"<AttachmentReference(external, url={self.url})>"
        else:
            return f"<AttachmentReference(internal, key={self.storage_key})>"
