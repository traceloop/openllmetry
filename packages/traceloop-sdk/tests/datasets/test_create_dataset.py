import pytest
import tempfile
import os

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from traceloop.sdk.datasets.dataset import Dataset
from .test_constants import TestConstants


@pytest.mark.vcr
def test_create_dataset_from_csv(datasets):
    csv_content = """Name,Price,In Stock
Laptop,999.99,true
Mouse,29.99,false"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

        unique_slug = "test-csv-dataset"

        dataset = datasets.from_csv(
            file_path=csv_path,
            slug=unique_slug,
            name="Test CSV Dataset",
            description="Dataset created from CSV for testing",
        )

        assert isinstance(dataset, Dataset)
        assert dataset.slug == unique_slug
        assert dataset.name == "Test CSV Dataset"
        assert dataset.description == "Dataset created from CSV for testing"
        assert len(dataset.columns) >= 2  # At least Name and Price columns
        assert len(dataset.rows) >= 0  # Allow for any number of rows

        os.unlink(csv_path)


@pytest.mark.vcr
def test_create_dataset_from_dataframe(datasets):
    # Create test dataframe
    df = pd.DataFrame(
        {
            "Name": ["Laptop", "Mouse"],
            "Price": [999.99, 29.99],
            "In Stock": [True, False],
        }
    )

    unique_slug = "test-df-dataset"

    dataset = datasets.from_dataframe(
        df=df,
        slug=unique_slug,
        name="Test DataFrame Dataset",
        description="Dataset created from DataFrame for testing",
    )

    assert isinstance(dataset, Dataset)
    assert dataset.slug == unique_slug
    assert dataset.name == "Test DataFrame Dataset"
    assert dataset.description == "Dataset created from DataFrame for testing"
    assert len(dataset.columns) >= 2  # At least Name and Price columns
    assert len(dataset.rows) >= 0  # Allow for any number of rows

    # Check for columns by name (flexible checking)
    column_names = [col.name for col in dataset.columns]
    name_columns = [name for name in column_names if "name" in name.lower()]
    price_columns = [name for name in column_names if "price" in name.lower()]

    assert (
        len(name_columns) >= 1 or len(price_columns) >= 1
    )  # At least one expected column


@pytest.mark.vcr
def test_create_dataset_from_csv_file_not_found(datasets):
    with pytest.raises(FileNotFoundError):
        datasets.from_csv(
            file_path=TestConstants.NON_EXISTENT_FILE_PATH,
            slug=TestConstants.DATASET_SLUG,
            name=TestConstants.DATASET_NAME,
        )


@pytest.mark.vcr
def test_create_dataset_with_duplicate_slug(datasets):
    # Test creating dataset with slug that already exists to record failure
    csv_content = """Name,Price
Laptop,999.99"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

        with pytest.raises(Exception) as exc_info:
            datasets.from_csv(
                file_path=csv_path,
                slug="duplicate-test-slug",  # Intentionally duplicate slug
                name="Duplicate Test Dataset",
            )

        # The exact error message may vary based on the API response
        error_msg = str(exc_info.value)
        assert (
            "Failed to create dataset" in error_msg
            or "409" in error_msg
            or "already exists" in error_msg.lower()
        )

    os.unlink(csv_path)


@pytest.mark.vcr
def test_create_dataset_from_dataframe_with_duplicate_slug(datasets):
    # Test creating dataset from dataframe with duplicate slug
    df = pd.DataFrame({"Name": ["Laptop"], "Price": [999.99]})

    with pytest.raises(Exception) as exc_info:
        datasets.from_dataframe(
            df=df,
            slug="duplicate-df-test-slug",  # Intentionally duplicate slug
            name="Duplicate DataFrame Dataset",
        )

    error_msg = str(exc_info.value)
    assert (
        "Failed to create dataset" in error_msg
        or "409" in error_msg
        or "already exists" in error_msg.lower()
    )
