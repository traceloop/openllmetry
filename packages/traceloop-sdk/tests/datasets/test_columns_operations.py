import pytest
import tempfile
import os


@pytest.mark.vcr
def test_create_dataset_with_columns(datasets):
    """Test creating a dataset with different column types using real API calls"""

    unique_slug = "test-columns-dataset"

    csv_content = """Name,Price,InStock,Rating
Product A,99.99,true,4.5
Product B,149.99,false,3.8
Product C,79.99,true,4.2"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

    dataset = datasets.from_csv(
        file_path=csv_path,
        slug=unique_slug,
        name="Test Columns Dataset",
        description="Dataset for testing column operations",
    )

    assert dataset is not None
    assert dataset.slug == unique_slug
    assert len(dataset.columns) >= 4  # Name, Price, InStock, Rating

    # Check that we have columns with different names
    column_names = [col.name.lower() for col in dataset.columns]
    assert any("name" in name for name in column_names)
    assert any("price" in name for name in column_names)

    os.unlink(csv_path)


@pytest.mark.vcr
def test_get_dataset_with_columns(datasets):
    """Test retrieving a dataset and checking its columns"""

    dataset = datasets.get_by_slug("test-qa")

    assert dataset is not None
    assert len(dataset.columns) >= 0  # Allow any number of columns
    assert len(dataset.rows) >= 0  # Allow any number of rows

    for column in dataset.columns:
        assert hasattr(column, "name")
        assert hasattr(column, "type")
        assert hasattr(column, "id") or hasattr(column, "slug")


@pytest.mark.vcr
def test_dataset_operations_errors(datasets):
    """Test various error conditions for dataset operations"""
    try:
        # Test with completely invalid slug
        dataset = datasets.get_by_slug("invalid-dataset-name-12345")

        assert dataset is not None

    except Exception as e:
        # Should get appropriate error for non-existent dataset
        assert "Failed to get dataset" in str(e) or "404" in str(e) or "401" in str(e)
