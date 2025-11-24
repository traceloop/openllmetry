"""Minimal tests for file cell operations in datasets."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from traceloop.sdk.dataset.model import FileCellType


@pytest.mark.vcr
def test_external_url_file_cell(datasets):
    """Test setting external URL for file cell."""
    from traceloop.sdk.dataset.model import CreateDatasetRequest, ColumnDefinition, ColumnType
    from traceloop.sdk.dataset.dataset import Dataset

    dataset_request = CreateDatasetRequest(
        slug="test-external-url",
        name="Test External URL",
        columns=[
            ColumnDefinition(slug="video", name="Video", type=ColumnType.FILE),
        ],
        rows=[{"video": None}],
    )

    dataset_response = datasets._create_dataset(dataset_request)
    dataset = Dataset.from_create_dataset_response(dataset_response, datasets._http)
    row = dataset.rows[0]

    # Set external URL
    row.set_file_cell(
        column_slug="video",
        url="https://www.youtube.com/watch?v=example",
        file_type=FileCellType.VIDEO,
    )

    # Verify
    assert row.values["video"]["storage"] == "external"
    assert row.values["video"]["url"] == "https://www.youtube.com/watch?v=example"
    assert row.values["video"]["type"] == "video"


def test_file_cell_validation():
    """Test parameter validation for set_file_cell method."""
    from traceloop.sdk.dataset.row import Row
    from traceloop.sdk.client.http import HTTPClient

    # Create mock row
    http_client = Mock(spec=HTTPClient)
    mock_dataset = Mock()
    mock_dataset.slug = "test"

    row = Row(http=http_client, dataset=mock_dataset, id="row-1", values={})

    # Test: Both file_path and url provided
    with pytest.raises(ValueError, match="Cannot provide both"):
        row.set_file_cell(
            column_slug="test",
            file_path="/path/to/file",
            url="https://example.com/file",
        )

    # Test: Neither provided
    with pytest.raises(ValueError, match="Must provide either"):
        row.set_file_cell(column_slug="test")


@pytest.mark.vcr
def test_file_upload_with_mock_s3(datasets):
    """Test file upload with mocked S3 to bypass permission issues."""
    from traceloop.sdk.dataset.model import CreateDatasetRequest, ColumnDefinition, ColumnType
    from traceloop.sdk.dataset.dataset import Dataset

    dataset_request = CreateDatasetRequest(
        slug="test-mock-upload",
        name="Test Mock Upload",
        columns=[
            ColumnDefinition(slug="file", name="File", type=ColumnType.FILE),
        ],
        rows=[{"file": None}],
    )

    dataset_response = datasets._create_dataset(dataset_request)
    dataset = Dataset.from_create_dataset_response(dataset_response, datasets._http)
    row = dataset.rows[0]

    # Create test file
    test_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    test_file.write(b"test content")
    test_file.close()

    # Mock S3 upload to succeed
    with patch.object(row, "_upload_file_to_presigned_url", return_value=True):
        row.set_file_cell(
            column_slug="file",
            file_path=test_file.name,
            file_type=FileCellType.FILE,
        )

    # Verify
    assert row.values["file"]["storage"] == "internal"
    assert row.values["file"]["status"] == "success"
    assert "storage_key" in row.values["file"]

    # Clean up
    os.unlink(test_file.name)
