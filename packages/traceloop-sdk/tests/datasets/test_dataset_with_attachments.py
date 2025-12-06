"""Tests for creating datasets with initial attachments."""

import os
import tempfile
from unittest.mock import patch

import pytest
from traceloop.sdk.datasets import (
    Attachment,
    ExternalAttachment,
    FileCellType,
)
from traceloop.sdk.datasets.model import (
    ColumnDefinition,
    ColumnType,
    CreateDatasetRequest,
)


@pytest.mark.vcr
def test_create_dataset_with_external_attachments(datasets):
    """Test creating dataset with ExternalAttachment objects in rows."""

    # Create dataset request with external attachments
    dataset_request = CreateDatasetRequest(
        slug="test-dataset-external-attachments",
        name="Products with External Media",
        description="Test dataset with external URLs",
        columns=[
            ColumnDefinition(slug="name", name="Product Name", type=ColumnType.STRING),
            ColumnDefinition(slug="price", name="Price", type=ColumnType.NUMBER),
            ColumnDefinition(slug="video", name="Demo Video", type=ColumnType.FILE),
            ColumnDefinition(slug="manual", name="Manual", type=ColumnType.FILE),
        ],
        rows=[
            {
                "name": "Widget Pro",
                "price": 99.99,
                "video": ExternalAttachment(
                    url="https://www.youtube.com/watch?v=demo1",
                    file_type=FileCellType.VIDEO,
                    metadata={"title": "Widget Pro Demo", "duration": "3:45"},
                ),
                "manual": ExternalAttachment(
                    url="https://docs.google.com/document/d/widget-manual",
                    file_type=FileCellType.FILE,
                    metadata={"pages": 25},
                ),
            },
            {
                "name": "Gadget Plus",
                "price": 149.99,
                "video": ExternalAttachment(
                    url="https://vimeo.com/demo2", file_type=FileCellType.VIDEO
                ),
                "manual": None,  # No manual for this product
            },
        ],
    )

    # Create the dataset
    dataset = datasets.create(dataset_request)

    # Verify dataset was created
    assert dataset.slug == "test-dataset-external-attachments"
    assert len(dataset.rows) == 2

    # Verify first row attachments
    row1 = dataset.rows[0]
    assert row1.values["name"] == "Widget Pro"
    assert row1.values["price"] == 99.99

    # Check video attachment
    video_cell = row1.values.get("video")
    assert video_cell is not None
    assert video_cell["storage"] == "external"
    assert video_cell["url"] == "https://www.youtube.com/watch?v=demo1"
    assert video_cell["type"] == "video"
    assert video_cell["status"] == "success"

    # Check manual attachment
    manual_cell = row1.values.get("manual")
    assert manual_cell is not None
    assert manual_cell["storage"] == "external"
    assert manual_cell["url"] == "https://docs.google.com/document/d/widget-manual"

    # Verify second row
    row2 = dataset.rows[1]
    assert row2.values["name"] == "Gadget Plus"
    video_cell2 = row2.values.get("video")
    assert video_cell2 is not None
    assert video_cell2["url"] == "https://vimeo.com/demo2"
    assert row2.values.get("manual") is None  # Should remain None


@pytest.mark.vcr
def test_create_dataset_with_file_attachments_mocked(datasets):
    """Test creating dataset with Attachment objects (file uploads) using mocked S3."""

    # Create test files
    test_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    test_image.write(b"fake image data")
    test_image.close()

    test_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    test_pdf.write(b"fake pdf data")
    test_pdf.close()

    # Create dataset request with file attachments
    dataset_request = CreateDatasetRequest(
        slug="test-dataset-file-attachments",
        name="Products with Files",
        columns=[
            ColumnDefinition(slug="name", name="Name", type=ColumnType.STRING),
            ColumnDefinition(slug="image", name="Image", type=ColumnType.FILE),
            ColumnDefinition(slug="manual", name="Manual", type=ColumnType.FILE),
        ],
        rows=[
            {
                "name": "Product A",
                "image": Attachment(
                    file_path=test_image.name,
                    file_type=FileCellType.IMAGE,
                    metadata={"alt_text": "Product A image"},
                ),
                "manual": Attachment(
                    file_path=test_pdf.name,
                    file_type=FileCellType.FILE,
                    metadata={"version": "1.0"},
                ),
            }
        ],
    )

    # Mock the S3 upload to succeed
    with patch.object(Attachment, "_upload_to_s3", return_value=True):
        dataset = datasets.create(dataset_request)

    # Verify dataset was created
    assert dataset.slug == "test-dataset-file-attachments"
    assert len(dataset.rows) == 1

    # Verify attachments were processed
    row = dataset.rows[0]
    assert row.values["name"] == "Product A"

    image_cell = row.values.get("image")
    assert image_cell is not None
    assert image_cell["storage"] == "internal"
    assert image_cell["type"] == "image"
    assert image_cell["status"] == "success"
    assert "storage_key" in image_cell

    manual_cell = row.values.get("manual")
    assert manual_cell is not None
    assert manual_cell["storage"] == "internal"
    assert manual_cell["type"] == "file"
    assert manual_cell["status"] == "success"

    # Clean up
    os.unlink(test_image.name)
    os.unlink(test_pdf.name)


@pytest.mark.vcr
def test_create_dataset_with_mixed_attachments(datasets):
    """Test creating dataset with both Attachment and ExternalAttachment objects."""

    # Create a test file
    test_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    test_file.write(b"test content")
    test_file.close()

    dataset_request = CreateDatasetRequest(
        slug="test-dataset-mixed-attachments",
        name="Mixed Attachments Dataset",
        columns=[
            ColumnDefinition(slug="id", name="ID", type=ColumnType.STRING),
            ColumnDefinition(slug="file", name="File", type=ColumnType.FILE),
            ColumnDefinition(slug="video", name="Video", type=ColumnType.FILE),
        ],
        rows=[
            {
                "id": "001",
                "file": Attachment(
                    file_path=test_file.name, file_type=FileCellType.FILE
                ),
                "video": ExternalAttachment(
                    url="https://youtube.com/watch?v=test", file_type=FileCellType.VIDEO
                ),
            },
            {
                "id": "002",
                "file": None,  # No file for this row
                "video": ExternalAttachment(
                    url="https://vimeo.com/test", file_type=FileCellType.VIDEO
                ),
            },
        ],
    )

    # Mock S3 upload
    with patch.object(Attachment, "_upload_to_s3", return_value=True):
        dataset = datasets.create(dataset_request)

    # Verify both rows
    assert len(dataset.rows) == 2

    # First row should have both attachments
    row1 = dataset.rows[0]
    assert row1.values["file"]["storage"] == "internal"
    assert row1.values["video"]["storage"] == "external"
    assert row1.values["video"]["url"] == "https://youtube.com/watch?v=test"

    # Second row should have only video
    row2 = dataset.rows[1]
    assert row2.values["file"] is None
    assert row2.values["video"]["storage"] == "external"

    # Clean up
    os.unlink(test_file.name)


@pytest.mark.vcr
def test_create_dataset_with_in_memory_attachment(datasets):
    """Test creating dataset with Attachment from in-memory data."""

    # Create attachment from bytes
    image_data = b"fake image bytes"

    dataset_request = CreateDatasetRequest(
        slug="test-dataset-memory-attachment",
        name="Memory Attachment Dataset",
        columns=[
            ColumnDefinition(slug="name", name="Name", type=ColumnType.STRING),
            ColumnDefinition(slug="image", name="Image", type=ColumnType.FILE),
        ],
        rows=[
            {
                "name": "Test Item",
                "image": Attachment(
                    data=image_data,
                    filename="test.jpg",
                    content_type="image/jpeg",
                    file_type=FileCellType.IMAGE,
                ),
            }
        ],
    )

    # Mock S3 upload
    with patch.object(Attachment, "_upload_to_s3", return_value=True):
        dataset = datasets.create(dataset_request)

    # Verify
    assert len(dataset.rows) == 1
    row = dataset.rows[0]
    assert row.values["image"]["storage"] == "internal"
    assert row.values["image"]["type"] == "image"
    assert row.values["image"]["status"] == "success"


@pytest.mark.vcr
def test_create_dataset_without_attachments(datasets):
    """Test that create() method works normally for datasets without attachments."""

    dataset_request = CreateDatasetRequest(
        slug="test-dataset-no-attachments",
        name="Regular Dataset",
        columns=[
            ColumnDefinition(slug="id", name="ID", type=ColumnType.STRING),
            ColumnDefinition(slug="value", name="Value", type=ColumnType.NUMBER),
        ],
        rows=[
            {"id": "A", "value": 100},
            {"id": "B", "value": 200},
        ],
    )

    dataset = datasets.create(dataset_request)

    # Verify normal dataset creation
    assert dataset.slug == "test-dataset-no-attachments"
    assert len(dataset.rows) == 2
    assert dataset.rows[0].values["id"] == "A"
    assert dataset.rows[0].values["value"] == 100
    assert dataset.rows[1].values["id"] == "B"
    assert dataset.rows[1].values["value"] == 200
