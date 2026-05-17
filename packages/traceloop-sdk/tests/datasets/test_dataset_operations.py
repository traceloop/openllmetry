import pytest


@pytest.mark.vcr
def test_get_dataset_by_version(datasets):
    csv_data = datasets.get_version_csv(slug="test-qa", version="v1")
    assert isinstance(csv_data, str)


@pytest.mark.vcr
def test_publish_dataset(datasets):
    unique_slug = "test-publish-dataset"

    # Create a simple CSV for the dataset
    import tempfile
    import os

    csv_content = """Name,Value
Test,123"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

        dataset = datasets.from_csv(
            file_path=csv_path,
            slug=unique_slug,
            name="Test Publish Dataset",
            description="Dataset for testing publish functionality",
        )

        # Try to publish it
        version = dataset.publish()
        assert isinstance(version, str)

    os.unlink(csv_path)
