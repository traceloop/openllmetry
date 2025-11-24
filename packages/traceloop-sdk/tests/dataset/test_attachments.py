"""Tests for the new Attachment API in datasets."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from traceloop.sdk.dataset import (
    Attachment,
    ExternalAttachment,
    AttachmentReference,
    FileCellType,
    FileStorageType,
)


@pytest.mark.vcr
def test_external_attachment(datasets):
    """Test setting external URL using ExternalAttachment."""
    from traceloop.sdk.dataset.model import CreateDatasetRequest, ColumnDefinition, ColumnType
    from traceloop.sdk.dataset.dataset import Dataset

    dataset_request = CreateDatasetRequest(
        slug="test-external-attachment",
        name="Test External Attachment",
        columns=[
            ColumnDefinition(slug="name", name="Name", type=ColumnType.STRING),
            ColumnDefinition(slug="video", name="Video", type=ColumnType.FILE),
        ],
        rows=[{"name": "Test Video", "video": None}],
    )

    dataset_response = datasets._create_dataset(dataset_request)
    dataset = Dataset.from_create_dataset_response(dataset_response, datasets._http)
    row = dataset.rows[0]

    # Create external attachment
    attachment = ExternalAttachment(
        url="https://www.youtube.com/watch?v=example",
        file_type=FileCellType.VIDEO,
        metadata={"title": "Demo Video"},
    )

    # Attach to row
    ref = attachment.attach(
        datasets._http,
        dataset.slug,
        row.id,
        "video"
    )

    # Verify reference
    assert ref is not None
    assert ref.storage_type == FileStorageType.EXTERNAL
    assert ref.url == "https://www.youtube.com/watch?v=example"
    assert ref.file_type == FileCellType.VIDEO
    assert ref.metadata.get("title") == "Demo Video"


def test_attachment_validation():
    """Test Attachment class parameter validation."""
    # Test: Both file_path and data provided
    with pytest.raises(ValueError, match="Cannot provide both"):
        Attachment(
            file_path="/path/to/file",
            data=b"test data"
        )

    # Test: Neither provided
    with pytest.raises(ValueError, match="Must provide either"):
        Attachment()

    # Test: Valid file_path
    att = Attachment(file_path="/test/file.txt")
    assert att.file_path == "/test/file.txt"
    assert att.filename == "file.txt"

    # Test: Valid data
    att = Attachment(data=b"test", filename="test.txt")
    assert att.data == b"test"
    assert att.filename == "test.txt"


@pytest.mark.vcr
def test_attachment_upload_with_mock_s3(datasets):
    """Test file upload using Attachment with mocked S3."""
    from traceloop.sdk.dataset.model import CreateDatasetRequest, ColumnDefinition, ColumnType
    from traceloop.sdk.dataset.dataset import Dataset

    dataset_request = CreateDatasetRequest(
        slug="test-attachment-upload",
        name="Test Attachment Upload",
        columns=[
            ColumnDefinition(slug="name", name="Name", type=ColumnType.STRING),
            ColumnDefinition(slug="file", name="File", type=ColumnType.FILE),
        ],
        rows=[{"name": "Test File", "file": None}],
    )

    dataset_response = datasets._create_dataset(dataset_request)
    dataset = Dataset.from_create_dataset_response(dataset_response, datasets._http)
    row = dataset.rows[0]

    # Create test file
    test_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    test_file.write(b"test content")
    test_file.close()

    # Create attachment
    attachment = Attachment(
        file_path=test_file.name,
        file_type=FileCellType.FILE,
        metadata={"source": "test"},
    )

    # Mock S3 upload
    with patch.object(attachment, "_upload_to_s3", return_value=True):
        ref = attachment.upload(
            datasets._http,
            dataset.slug,
            row.id,
            "file"
        )

    # Verify reference
    assert ref is not None
    assert ref.storage_type == FileStorageType.INTERNAL
    assert ref.file_type == FileCellType.FILE
    assert "storage_key" in ref.__dict__ and ref.storage_key

    # Clean up
    os.unlink(test_file.name)


def test_attachment_from_memory():
    """Test creating Attachment from in-memory data."""
    data = b"Hello World"
    attachment = Attachment(
        data=data,
        filename="hello.txt",
        content_type="text/plain",
        file_type=FileCellType.FILE,
    )

    assert attachment.data == data
    assert attachment.filename == "hello.txt"
    assert attachment.content_type == "text/plain"
    assert attachment.file_type == FileCellType.FILE
    assert attachment._get_file_data() == data
    assert attachment._get_file_size() == len(data)


def test_attachment_type_detection():
    """Test automatic file type detection based on content type."""
    # Image
    att = Attachment(data=b"fake", filename="test.jpg", content_type="image/jpeg")
    assert att.file_type == FileCellType.IMAGE

    # Video
    att = Attachment(data=b"fake", filename="test.mp4", content_type="video/mp4")
    assert att.file_type == FileCellType.VIDEO

    # Audio
    att = Attachment(data=b"fake", filename="test.mp3", content_type="audio/mp3")
    assert att.file_type == FileCellType.AUDIO

    # Default to FILE
    att = Attachment(data=b"fake", filename="test.pdf", content_type="application/pdf")
    assert att.file_type == FileCellType.FILE


def test_attachment_reference():
    """Test AttachmentReference creation and properties."""
    # External reference
    ref = AttachmentReference(
        storage_type=FileStorageType.EXTERNAL,
        url="https://example.com/file.pdf",
        file_type=FileCellType.FILE,
        metadata={"size": 1024},
    )

    assert ref.storage_type == FileStorageType.EXTERNAL
    assert ref.url == "https://example.com/file.pdf"
    assert ref.get_url() == "https://example.com/file.pdf"
    assert str(ref) == "<AttachmentReference(external, url=https://example.com/file.pdf)>"

    # Internal reference
    ref = AttachmentReference(
        storage_type=FileStorageType.INTERNAL,
        storage_key="bucket/key.pdf",
        file_type=FileCellType.FILE,
    )

    assert ref.storage_type == FileStorageType.INTERNAL
    assert ref.storage_key == "bucket/key.pdf"
    assert ref.get_url() is None  # No download URL implementation yet
    assert str(ref) == "<AttachmentReference(internal, key=bucket/key.pdf)>"