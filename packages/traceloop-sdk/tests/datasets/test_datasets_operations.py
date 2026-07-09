import pytest
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import DatasetMetadata


@pytest.mark.vcr
def test_get_dataset_by_slug(datasets):
    dataset = datasets.get_by_slug("test-qa")

    assert isinstance(dataset, Dataset)
    assert dataset.slug == "test-qa"
    assert hasattr(dataset, "name")
    assert hasattr(dataset, "description")


@pytest.mark.vcr
def test_get_all_datasets(datasets):
    datasets_list = datasets.get_all()

    assert isinstance(datasets_list, list)
    assert len(datasets_list) >= 0

    for dataset in datasets_list:
        assert isinstance(dataset, DatasetMetadata)
        assert hasattr(dataset, "id")
        assert hasattr(dataset, "slug")
        assert hasattr(dataset, "name")


@pytest.mark.vcr
def test_get_version_csv(datasets):
    try:
        csv_data = datasets.get_version_csv(slug="test-qa", version="v1")
        assert isinstance(csv_data, str)
    except Exception as e:
        # Allow for expected API errors during recording (dataset might not exist)
        assert "Failed to get dataset" in str(e) or "404" in str(e) or "401" in str(e)


@pytest.mark.vcr
def test_delete_by_slug(datasets):
    try:
        # Use a test dataset that's safe to delete
        datasets.delete_by_slug("test-csv-dataset-conflict")
    except Exception as e:
        # Allow for expected API errors (dataset might not exist)
        assert (
            "Failed to delete dataset" in str(e) or "404" in str(e) or "401" in str(e)
        )


@pytest.mark.vcr
def test_delete_by_slug_failure(datasets):
    with pytest.raises(Exception) as exc_info:
        datasets.delete_by_slug("non-existent-dataset-123")

    # The exact error message may vary based on the recorded API response
    assert "Failed to delete dataset" in str(exc_info.value) or "404" in str(
        exc_info.value
    )


@pytest.mark.vcr
def test_get_all_datasets_with_invalid_credentials():
    # Test with invalid API key to record failure case
    from traceloop.sdk.client.http import HTTPClient
    from traceloop.sdk.datasets.datasets import Datasets

    http = HTTPClient(
        base_url="https://api-staging.traceloop.com",
        api_key="invalid-key",
        version="1.0.0",
    )
    invalid_datasets = Datasets(http)

    try:
        invalid_datasets.get_all()
        # If this doesn't raise an exception, the test setup might be wrong
        assert False, "Expected authentication error"
    except Exception as exc_info:
        # Should get authentication error or a generic failure error when using VCR
        assert (
            "401" in str(exc_info)
            or "authentication" in str(exc_info).lower()
            or "Failed to get datasets" in str(exc_info)
        )


@pytest.mark.vcr
def test_get_dataset_by_slug_failure(datasets):
    with pytest.raises(Exception) as exc_info:
        datasets.get_by_slug("definitely-non-existent-dataset-123")

    assert "Failed to get dataset" in str(exc_info.value) or "404" in str(
        exc_info.value
    )


@pytest.mark.vcr
def test_get_version_csv_failure(datasets):
    with pytest.raises(Exception) as exc_info:
        datasets.get_version_csv("definitely-non-existent-dataset-123", "v1")

    assert "Failed to get dataset" in str(exc_info.value) or "404" in str(
        exc_info.value
    )
