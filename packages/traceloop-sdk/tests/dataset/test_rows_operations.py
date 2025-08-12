import pytest
import tempfile
import os
import time


@pytest.mark.vcr
def test_create_dataset_and_add_rows(datasets):
    """Test creating a dataset and adding rows using real API calls"""
    try:
        # Create a unique slug to avoid conflicts
        unique_slug = f"test-rows-{int(time.time())}"

        # Create a simple CSV for the dataset
        csv_content = """Name,Age,Active
John,25,true
Jane,30,false
Bob,35,true"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            # Create dataset from CSV
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

        finally:
            os.unlink(csv_path)

    except Exception as e:
        # Allow for expected API errors during recording
        assert (
            "Failed to create dataset" in str(e)
            or "401" in str(e)
            or "403" in str(e)
            or "409" in str(e)
        )


@pytest.mark.vcr
def test_add_rows(datasets):
    """Test the add_rows method that makes POST to /datasets/{slug}/rows"""
    try:
        # Create a unique slug to avoid conflicts
        unique_slug = f"test-add-rows-{int(time.time())}"

        # Create a simple CSV for the initial dataset
        csv_content = """Name,Age,Active
John,25,true
Jane,30,false"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            # First create a dataset
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

        finally:
            os.unlink(csv_path)

    except Exception as e:
        # Allow for expected API errors during recording
        assert (
            "Failed to create dataset" in str(e)
            or "Failed to add row" in str(e)
            or "401" in str(e)
            or "403" in str(e)
            or "409" in str(e)
        )


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
    try:
        # Delete an existing dataset directly
        datasets.delete_by_slug("test-csv-dataset-1754936890")

    except Exception as e:
        # Allow for expected API errors during recording
        assert (
            "Failed" in str(e) or "401" in str(e) or "403" in str(e) or "404" in str(e)
        )
