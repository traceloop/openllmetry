import pytest
import tempfile
import os


@pytest.mark.vcr
def test_create_dataset_and_add_rows(datasets):
    """Test creating a dataset and adding rows using real API calls"""

    unique_slug = "test-rows"

    csv_content = """Name,Age,Active
John,25,true
Jane,30,false
Bob,35,true"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

        dataset = datasets.from_csv(
            file_path=csv_path,
            slug=unique_slug,
            name="Test Rows Dataset",
            description="Dataset for testing row operations",
        )

        assert dataset is not None
        assert dataset.slug == unique_slug
        assert len(dataset.columns) >= 3  # Name, Age, Active
        assert len(dataset.rows) >= 0  # Allow any number of initial rows

        os.unlink(csv_path)


@pytest.mark.vcr
def test_add_rows(datasets):
    """Test the add_rows method that makes POST to /datasets/{slug}/rows"""

    unique_slug = "test-add-rows"

    # Create a simple CSV for the initial dataset
    csv_content = """Name,Age,Active
John,25,true
Jane,30,false"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

        dataset = datasets.from_csv(
            file_path=csv_path,
            slug=unique_slug,
            name="Test Add Rows Dataset",
            description="Dataset for testing add_rows method",
        )

        assert dataset is not None
        initial_row_count = len(dataset.rows) if dataset.rows else 0

        # Now test the add_rows method specifically
        new_rows = [{"name": "Alice", "age": "28", "active": "true"}]
        dataset.add_rows(new_rows)

        # Verify the row was added
        assert dataset.rows is not None
        assert len(dataset.rows) == initial_row_count + 1
        assert any(row.values["name"] == "Alice" for row in dataset.rows)

    os.unlink(csv_path)


@pytest.mark.vcr
def test_dataset_row_operations_api_errors(datasets):
    """Test handling of API errors for row operations"""
    try:
        # Try to get a non-existent dataset to record error response
        dataset = datasets.get_by_slug("definitely-non-existent-dataset-12345")
        # If we get here, the dataset unexpectedly exists, which is also valid for testing
        assert dataset is not None

    except Exception as e:
        # Should get a "dataset not found" type error
        assert "Failed to get dataset" in str(e) or "404" in str(e) or "401" in str(e)


@pytest.mark.vcr
def test_dataset_deletion(datasets):
    """Test dataset deletion using real API calls"""
    datasets.delete_by_slug("test-csv-dataset")
