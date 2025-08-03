import csv
import os
import tempfile
from unittest.mock import Mock, patch
import pytest

from traceloop.sdk.datasets import Dataset, ColumnType


class TestDatasetFromCSV:
    """Test Dataset.from_csv() method"""

    def test_from_csv_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file"""
        with pytest.raises(FileNotFoundError):
            Dataset.from_csv("non_existent.csv", slug="test")

    @patch('traceloop.sdk.datasets.client.DatasetClient')
    def test_from_csv_basic(self, mock_client_class):
        """Test basic CSV import functionality"""
        # Mock the client and API responses
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.create_dataset.return_value = {
            "id": "dataset_123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        mock_client.add_column.return_value = {"id": "col_123"}
        mock_client.add_row.return_value = {"id": "row_123"}

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'email'])
            writer.writerow(['John Doe', '30', 'john@example.com'])
            writer.writerow(['Jane Smith', '25', 'jane@example.com'])
            csv_path = f.name

        try:
            # Test the from_csv method
            dataset = Dataset.from_csv(
                csv_path,
                slug="test-dataset",
                name="Test Dataset",
                description="A test dataset"
            )

            # Verify dataset properties
            assert dataset.name == "Test Dataset"
            assert dataset.slug == "test-dataset"
            assert dataset.description == "A test dataset"
            assert dataset.id == "dataset_123"
            assert len(dataset.columns) == 3
            assert len(dataset.rows) == 2

            # Verify column properties
            assert dataset.columns[0].name == "name"
            assert dataset.columns[0].type == ColumnType.STRING
            assert dataset.columns[1].name == "age"
            assert dataset.columns[2].name == "email"

            # Verify API calls
            mock_client.create_dataset.assert_called_once_with("Test Dataset", "A test dataset")
            assert mock_client.add_column.call_count == 3
            assert mock_client.add_row.call_count == 2

        finally:
            # Clean up temporary file
            os.unlink(csv_path)

    @patch('traceloop.sdk.datasets.client.DatasetClient')
    def test_from_csv_defaults(self, mock_client_class):
        """Test CSV import with default values"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.create_dataset.return_value = {
            "id": "dataset_123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        mock_client.add_column.return_value = {"id": "col_123"}

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['col1'])
            writer.writerow(['value1'])
            csv_path = f.name

        try:
            dataset = Dataset.from_csv(csv_path, slug="test")

            # Should use filename (without extension) as default name
            filename_without_ext = os.path.splitext(os.path.basename(csv_path))[0]
            assert dataset.name == filename_without_ext
            assert dataset.slug == "test"
            assert dataset.description is None

        finally:
            os.unlink(csv_path)


class TestDatasetFromDataFrame:
    """Test Dataset.from_dataframe() method"""

    @patch('traceloop.sdk.datasets.client.DatasetClient')
    def test_from_dataframe_basic(self, mock_client_class):
        """Test basic DataFrame import functionality"""
        # Skip if pandas not available
        pd = pytest.importorskip("pandas")
        
        # Mock the client and API responses
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.create_dataset.return_value = {
            "id": "dataset_123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        mock_client.add_column.return_value = {"id": "col_123"}
        mock_client.add_row.return_value = {"id": "row_123"}

        # Create test DataFrame
        df = pd.DataFrame({
            'name': ['John', 'Jane'],
            'age': [30, 25],
            'active': [True, False]
        })

        # Test the from_dataframe method
        dataset = Dataset.from_dataframe(
            df,
            slug="test-dataset",
            name="Test Dataset",
            description="A test dataset"
        )

        # Verify dataset properties
        assert dataset.name == "Test Dataset"
        assert dataset.slug == "test-dataset"
        assert dataset.description == "A test dataset"
        assert dataset.id == "dataset_123"
        assert len(dataset.columns) == 3
        assert len(dataset.rows) == 2

        # Verify column types are inferred correctly
        name_col = next(col for col in dataset.columns if col.name == "name")
        age_col = next(col for col in dataset.columns if col.name == "age")
        active_col = next(col for col in dataset.columns if col.name == "active")
        
        assert name_col.type == ColumnType.STRING
        assert age_col.type == ColumnType.NUMBER
        assert active_col.type == ColumnType.BOOLEAN

        # Verify API calls
        mock_client.create_dataset.assert_called_once_with("Test Dataset", "A test dataset")
        assert mock_client.add_column.call_count == 3
        assert mock_client.add_row.call_count == 2

    def test_from_dataframe_no_pandas(self):
        """Test that ImportError is raised when pandas not available"""
        with patch.dict('sys.modules', {'pandas': None}):
            with pytest.raises(ImportError, match="pandas is required"):
                Dataset.from_dataframe(None, slug="test")

    @patch('traceloop.sdk.datasets.client.DatasetClient')
    def test_from_dataframe_invalid_input(self, mock_client_class):
        """Test that TypeError is raised for invalid input"""
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            Dataset.from_dataframe("not a dataframe", slug="test")

    @patch('traceloop.sdk.datasets.client.DatasetClient')
    def test_from_dataframe_defaults(self, mock_client_class):
        """Test DataFrame import with default values"""
        pd = pytest.importorskip("pandas")
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.create_dataset.return_value = {
            "id": "dataset_123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        mock_client.add_column.return_value = {"id": "col_123"}

        # Create simple DataFrame
        df = pd.DataFrame({'col1': ['value1']})

        dataset = Dataset.from_dataframe(df, slug="test")

        # Should use default name
        assert dataset.name == "Dataset from DataFrame"
        assert dataset.slug == "test"
        assert dataset.description is None


class TestDatasetClient:
    """Test DatasetClient functionality"""

    def test_client_singleton(self):
        """Test that DatasetClient follows singleton pattern"""
        with patch.dict(os.environ, {'TRACELOOP_API_KEY': 'test_key'}):
            from traceloop.sdk.datasets.client import DatasetClient
            
            client1 = DatasetClient()
            client2 = DatasetClient()
            
            assert client1 is client2

    def test_client_no_api_key(self):
        """Test that ValueError is raised when API key not provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="TRACELOOP_API_KEY environment variable is required"):
                from traceloop.sdk.datasets.client import DatasetClient
                DatasetClient()